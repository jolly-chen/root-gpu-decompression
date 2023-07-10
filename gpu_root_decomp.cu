////////////////////////////////////////////
// Decompress ROOT compressed files.
////////////////////////////////////////////

#include <random>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <unistd.h>
#include <string>

#include "RZip.h"
#include "TError.h"
#include "nvcomp/zstd.h"
#include "nvcomp/lz4.h"
#include "nvcomp/deflate.h"

bool verbose = false;

// The size of the ROOT block framing headers for compression:
// - 3 bytes to identify the compression algorithm and version.
// - 3 bytes to identify the deflated buffer size.
// - 3 bytes to identify the inflated buffer size.
#define HDRSIZE 9

#define ERRCHECK(err) __checkCudaErrors((err), __func__, __FILE__, __LINE__)
inline static void __checkCudaErrors(cudaError_t error, std::string func, std::string file, int line)
{
   if (error != cudaSuccess) {
      fprintf(stderr, (func + "(), " + file + ":" + std::to_string(line)).c_str(), "%s\n", cudaGetErrorString(error));
      throw std::bad_alloc();
   }
}

__global__ void PrintBatch(void **chunk_pointers, size_t *chunk_sizes, char *data, size_t num_chunks)
{
   printf("------------------------- BATCH ----------------------------\n");
   printf("num_chunks: %li\n", num_chunks);

   printf("chunk_sizes:\n");
   size_t total_size = 0;
   for (int i = 0; i < num_chunks; i++) {
      printf("%li ", chunk_sizes[i]);
      total_size += chunk_sizes[i];
   }
   printf("\n");

   printf("total_size: %li\n", total_size);

   printf("data:\n");
   for (int i = 0; i < 300; i++) {
      printf("%c ", data[i]);
   }
   printf("\n");

   printf("chunk_pointers:\n");
   for (int i = 0; i < num_chunks; i++) {
      printf("\tchunk %d:\n\t\t", i);
      for (int j = 0; j < min(100, (int)chunk_sizes[i]); j++) {
         printf("%c ", ((char *)chunk_pointers[i])[j]);
      }
      printf("\n");
   }
   printf("\n");
}

class GPUDecompressor {
private:
   cudaStream_t stream;
   size_t num_chunks;

   // Host dDecompressed
   std::vector<char> hCompressed, hDecompressed;
   std::vector<size_t> hCompSizes, hDecompSizes;

   // Device dDecompressed;
   char *dCompressed, *dDecompressed;
   void **dCompressedChunkPointers, **dDecompressedChunkPointers;
   size_t *dCompSizes, *dDecompSizes;
   size_t tempBufSize;
   void *dTempBuf;
   nvcompStatus_t *dStatusPtrs;

   // CUDA events to measure decompression time
   cudaEvent_t start, end;

   inline std::vector<void *> GetCompressedChunkPtrs()
   {
      std::vector<void *> ptrs(num_chunks);
      size_t offset = HDRSIZE;

      for (size_t i = 0; i < num_chunks; ++i) {
         ptrs[i] = static_cast<void *>(dCompressed + offset);
         offset += hCompSizes[i];
      }
      return ptrs;
   }

   inline std::vector<void *> GetDecompressedChunkPtrs()
   {
      std::vector<void *> ptrs(num_chunks);

      size_t offset = 0;
      for (size_t i = 0; i < num_chunks; ++i) {
         ptrs[i] = static_cast<void *>(dDecompressed + offset);
         offset += hDecompSizes[i];
      }
      return ptrs;
   }

   // Allocate and setup various host/device buffers for decompressing the input data.
   template <typename GetDecompressSizeFunc, typename GetTempSizeFunc>
   void Configure(GetDecompressSizeFunc nvcompGetDecompressSize, GetTempSizeFunc nvcompGetDecompressTempSize)
   {
      size_t remainder = hCompressed.size();
      unsigned char *source = const_cast<unsigned char *>(reinterpret_cast<const unsigned char *>(hCompressed.data()));

      // Loop over the chunks to determine their sizes from the header
      int decompressedTotalSize = 0;
      int maxUncompressedChunkSize = 0;
      do {
         int szSource;
         int szTarget;
         int retval = R__unzip_header(&szSource, source, &szTarget);
         R__ASSERT(retval == 0);
         R__ASSERT(szSource > 0);
         R__ASSERT(szTarget > szSource);
         R__ASSERT(static_cast<unsigned int>(szSource) <= hCompressed.size());

         num_chunks++;
         hCompSizes.push_back(szSource);
         hDecompSizes.push_back(szTarget);

         decompressedTotalSize += szTarget;
         if (szTarget > maxUncompressedChunkSize)
            maxUncompressedChunkSize = szTarget;

         // Move to next chunk
         source += szSource;
         remainder -= szSource;
      } while (remainder > 0);
      R__ASSERT(remainder == 0);

      std::cout << "chunks: " << num_chunks << std::endl;
      hDecompressed.resize(decompressedTotalSize);

      // Set up buffers for the compressed and decompressed data on the device.
      ERRCHECK(cudaMallocAsync(&dCompressed, hCompressed.size() * sizeof(char), stream));
      ERRCHECK(cudaMemcpyAsync(dCompressed, hCompressed.data(), hCompressed.size() * sizeof(char),
                               cudaMemcpyHostToDevice, stream));
      ERRCHECK(cudaMallocAsync(&dDecompressed, decompressedTotalSize * sizeof(char), stream));

      // Copy compressed and decompressed sizes of each chunk
      ERRCHECK(cudaMallocAsync(&dCompSizes, hCompSizes.size() * sizeof(size_t), stream));
      ERRCHECK(cudaMemcpyAsync(dCompSizes, hCompSizes.data(), hCompSizes.size() * sizeof(size_t),
                               cudaMemcpyHostToDevice, stream));
      ERRCHECK(cudaMallocAsync(&dDecompSizes, hDecompSizes.size() * sizeof(size_t), stream));
      ERRCHECK(cudaMemcpyAsync(dDecompSizes, hDecompSizes.data(), hDecompSizes.size() * sizeof(size_t),
                               cudaMemcpyHostToDevice, stream));

      cudaStreamSynchronize(stream);

      // Set up pointers to each chunk in the device buffer for the compressed data.
      auto cPtrs = GetCompressedChunkPtrs();
      ERRCHECK(cudaMallocAsync(&dCompressedChunkPointers, num_chunks * sizeof(void *), stream));
      ERRCHECK(cudaMemcpyAsync(dCompressedChunkPointers, cPtrs.data(), num_chunks * sizeof(void *),
                               cudaMemcpyHostToDevice, stream));

      // Set up pointers to each chunk in the device buffer for the decompressed data.
      auto dcPtrs = GetDecompressedChunkPtrs();
      ERRCHECK(cudaMallocAsync(&dDecompressedChunkPointers, num_chunks * sizeof(void *), stream));
      ERRCHECK(cudaMemcpyAsync(dDecompressedChunkPointers, dcPtrs.data(), num_chunks * sizeof(void *),
                               cudaMemcpyHostToDevice, stream));

      // Allocate temp space
      nvcompStatus_t status = nvcompGetDecompressTempSize(num_chunks, maxUncompressedChunkSize, &tempBufSize);
      if (status != nvcompSuccess) {
         throw std::runtime_error("nvcompBatched*DecompressGetTempSize() failed.");
      }
      ERRCHECK(cudaMallocAsync(&dTempBuf, tempBufSize, stream));

      // Status pointers
      ERRCHECK(cudaMallocAsync(&dStatusPtrs, num_chunks * sizeof(nvcompStatus_t), stream));

      // For measuring runtime
      cudaEventCreate(&start);
      cudaEventCreate(&end);

      if (verbose) {
         PrintBatch<<<1, 1, 0, stream>>>(dCompressedChunkPointers, dCompSizes, dCompressed, num_chunks);
         PrintBatch<<<1, 1, 0, stream>>>(dDecompressedChunkPointers, dDecompSizes, dDecompressed, num_chunks);
      }
   }

   template <typename GetDecompressSizeFunc, typename GetTempSizeFunc, typename DecompressFunc>
   void DecompressInternal(GetDecompressSizeFunc nvcompGetDecompressSize, GetTempSizeFunc nvcompGetDecompressTempSize,
                           DecompressFunc nvcompDecompress)
   {
      Configure(nvcompGetDecompressSize, nvcompGetDecompressTempSize);

      // Run decompression
      nvcompStatus_t status =
         nvcompDecompress(dCompressedChunkPointers, dCompSizes, dDecompSizes, dDecompSizes, num_chunks, dTempBuf,
                          tempBufSize, dDecompressedChunkPointers, dStatusPtrs, stream);
      if (status != nvcompSuccess) {
         throw std::runtime_error("ERROR: nvcompBatched*DecompressAsync() not successful");
      }

      if (verbose) {
         PrintBatch<<<1, 1, 0, stream>>>(dDecompressedChunkPointers, dDecompSizes, dDecompressed, num_chunks);
      }
   }

public:
   GPUDecompressor(const std::vector<char> &data) : hCompressed(data)
   {
      num_chunks = 0;
      cudaStreamCreate(&stream);
   }

   ~GPUDecompressor()
   {
      cudaFree(dDecompressed);
      cudaFree(dCompressed);
      cudaFree(dCompressedChunkPointers);
      cudaFree(dDecompressedChunkPointers);
      cudaFree(dCompSizes);
      cudaFree(dDecompSizes);
      cudaFree(dTempBuf);
      cudaFree(dStatusPtrs);

      cudaEventDestroy(start);
      cudaEventDestroy(end);
      cudaStreamDestroy(stream);
   }

   bool Decompress(std::string type)
   {
      if (type == "zstd") {
         std::cout << "method: zstd" << std::endl;
         DecompressInternal(nvcompBatchedZstdGetDecompressSizeAsync, nvcompBatchedZstdDecompressGetTempSize,
                            nvcompBatchedZstdDecompressAsync);
      } else if (type == "lz4") {
         std::cout << "method: lz4" << std::endl;
         DecompressInternal(nvcompBatchedLZ4GetDecompressSizeAsync, nvcompBatchedLZ4DecompressGetTempSize,
                            nvcompBatchedLZ4DecompressAsync);
      } else if (type == "zlib") {
         std::cout << "method: zlib" << std::endl;
         DecompressInternal(nvcompBatchedDeflateGetDecompressSizeAsync, nvcompBatchedDeflateDecompressGetTempSize,
                            nvcompBatchedDeflateDecompressAsync);
      } else {
         fprintf(stderr, "Unknown decompression type\n");
         return false;
      }

      return true;
   }

   std::vector<char> &GetResult()
   {
      // Retrieve resuts
      ERRCHECK(cudaMemcpyAsync(hDecompressed.data(), dDecompressed, hDecompressed.size() * sizeof(char),
                               cudaMemcpyDeviceToHost, stream));
      cudaStreamSynchronize(stream);
      return hDecompressed;
   }
};

/**
 * File reading
 */

std::vector<char> readFile(const std::string &filename)
{
   std::vector<char> buffer(4096);
   std::vector<char> hCompressed;

   std::ifstream fin(filename, std::ifstream::binary);
   fin.exceptions(std::ifstream::failbit | std::ifstream::badbit);

   size_t num;
   do {
      num = fin.readsome(buffer.data(), buffer.size());
      hCompressed.insert(hCompressed.end(), buffer.begin(), buffer.begin() + num);
   } while (num > 0);

   return hCompressed;
}

/**
 * Main
 */

int main(int argc, char *argv[])
{
   std::string file_name, type, output_file;

   int c;
   while ((c = getopt(argc, argv, "f:t:o:v")) != -1) {
      switch (c) {
      case 'f': file_name = optarg; break;
      case 't': type = optarg; break;
      case 'o': output_file = optarg; break;
      case 'v': verbose = true; break;
      default: std::cout << "Got unknown parse returns: " << char(c) << std::endl; return 1;
      }
   }

   if (file_name.empty() || type.empty()) {
      std::cerr << "Must specify a file (-f) and decompression type (-t)" << std::endl;
      return 1;
   }

   const std::vector<char> &dDecompressed = readFile(file_name);
   size_t input_buffer_len = dDecompressed.size();

   std::cout << "--------------------- INPUT INFORMATION ---------------------" << std::endl;
   std::cout << "file name: " << file_name.c_str() << std::endl;
   std::cout << "compressed (B): " << input_buffer_len << std::endl;

   GPUDecompressor decompressor(dDecompressed);
   decompressor.Decompress(type);
   auto &result = decompressor.GetResult();

   if (!output_file.empty()) {
      std::cout << "output file: " << output_file.c_str() << std::endl;
      auto fp = fopen(output_file.c_str(), "w");
      for (auto i = 0; i < result.size(); i++) {
         fprintf(fp, "%c", result[i]);
      }
   }

   return 0;
}
