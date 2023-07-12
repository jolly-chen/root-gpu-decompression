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
#include <chrono>

#include "RZip.h"
#include "TError.h"
#include "nvcomp/zstd.h"
#include "nvcomp/lz4.h"
#include "nvcomp/deflate.h"

using Clock = std::chrono::high_resolution_clock;

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

__global__ void PrintBatch(void **chunk_pointers, size_t *chunk_sizes, char *data, size_t nChunks)
{
   printf("------------------------- BATCH ----------------------------\n");
   printf("nChunks: %li\n", nChunks);

   printf("chunk_sizes:\n");
   size_t total_size = 0;
   for (int i = 0; i < nChunks; i++) {
      printf("%li ", chunk_sizes[i]);
      total_size += chunk_sizes[i];
   }
   printf("\n");

   printf("total_size: %li\n", total_size);

   printf("data:\n");
   for (int i = 0; i < 100; i++) {
      printf("%c ", data[i]);
   }
   printf("\n");

   printf("chunk_pointers:\n");
   for (int i = 0; i < nChunks; i++) {
      printf("\tchunk %d:\n\t\t", i);
      for (int j = 0; j < min(100, (int)chunk_sizes[i]); j++) {
         printf("%c ", ((char *)chunk_pointers[i])[j]);
      }
      printf("\n");
   }
   printf("\n");
}

__global__ void CheckStatuses(nvcompStatus_t *statusPtrs, size_t nChunks)
{
   for (int i = 0; i < nChunks; i++) {
      if (statusPtrs[i] != nvcompSuccess) {
         printf("Decompression of chunk %d has FAILED with status: %d\n", i, statusPtrs[i]);
      }
   }
}

struct Result {
   float setupTime, decompTime;
   std::vector<char> decompressed;
};

class GPUDecompressor {
private:
   cudaStream_t stream;
   size_t nChunks;

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
   cudaEvent_t decompStart, decompEnd;

   float setupTime, decompTime;

   inline std::vector<void *> GetCompressedChunkPtrs()
   {
      std::vector<void *> ptrs(nChunks);
      size_t offset = HDRSIZE;

      for (size_t i = 0; i < nChunks; ++i) {
         ptrs[i] = static_cast<void *>(dCompressed + offset);
         offset += hCompSizes[i];
      }
      return ptrs;
   }

   inline std::vector<void *> GetDecompressedChunkPtrs()
   {
      std::vector<void *> ptrs(nChunks);

      size_t offset = 0;
      for (size_t i = 0; i < nChunks; ++i) {
         ptrs[i] = static_cast<void *>(dDecompressed + offset);
         offset += hDecompSizes[i];
      }
      return ptrs;
   }

   // Allocate and setup various host/device buffers for decompressing the input data.
   template <typename GetDecompressSizeFunc, typename GetTempSizeFunc>
   void Configure(GetDecompressSizeFunc nvcompGetDecompressSize, GetTempSizeFunc nvcompGetDecompressTempSize)
   {
      // For measuring decompression runtime
      ERRCHECK(cudaEventCreate(&decompStart));
      ERRCHECK(cudaEventCreate(&decompEnd));

      // For measuring memory allocation and transfer times on the GPU
      cudaEvent_t memStart, memEnd;
      ERRCHECK(cudaEventCreate(&memStart));
      ERRCHECK(cudaEventCreate(&memEnd));

      // For measuring setup time on the CPU
      auto cpuConfigureStart = Clock::now();
      auto cpuConfigureEnd = Clock::now();

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

         nChunks++;
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

      hDecompressed.resize(decompressedTotalSize);
      cpuConfigureEnd = Clock::now();

      // Set up buffers for the compressed and decompressed data on the device.
      cudaEventRecord(memStart, stream);
      ERRCHECK(cudaMallocAsync(&dCompressed, hCompressed.size() * sizeof(char), stream));
      ERRCHECK(cudaMemcpyAsync(dCompressed, hCompressed.data(), hCompressed.size() * sizeof(char),
                               cudaMemcpyHostToDevice, stream));
      ERRCHECK(cudaMallocAsync(&dDecompressed, decompressedTotalSize * sizeof(char), stream));

      ERRCHECK(cudaStreamSynchronize(stream));

      // Set up pointers to each chunk in the device buffer for the compressed data.
      cpuConfigureStart = Clock::now();
      auto cPtrs = GetCompressedChunkPtrs();
      cpuConfigureEnd = Clock::now();
      ERRCHECK(cudaMallocAsync(&dCompressedChunkPointers, nChunks * sizeof(void *), stream));
      ERRCHECK(cudaMemcpyAsync(dCompressedChunkPointers, cPtrs.data(), nChunks * sizeof(void *), cudaMemcpyHostToDevice,
                               stream));

      // Set up pointers to each chunk in the device buffer for the decompressed data.
      cpuConfigureStart = Clock::now();
      auto dcPtrs = GetDecompressedChunkPtrs();
      cpuConfigureEnd = Clock::now();
      ERRCHECK(cudaMallocAsync(&dDecompressedChunkPointers, nChunks * sizeof(void *), stream));
      ERRCHECK(cudaMemcpyAsync(dDecompressedChunkPointers, dcPtrs.data(), nChunks * sizeof(void *),
                               cudaMemcpyHostToDevice, stream));

      // Copy compressed and decompressed sizes of each chunk
      cpuConfigureStart = Clock::now();
      std::transform(hCompSizes.begin(), hCompSizes.end(), hCompSizes.begin(), [&](auto x) { return x - HDRSIZE; });
      cpuConfigureEnd = Clock::now();
      ERRCHECK(cudaMallocAsync(&dCompSizes, hCompSizes.size() * sizeof(size_t), stream));
      ERRCHECK(cudaMemcpyAsync(dCompSizes, hCompSizes.data(), hCompSizes.size() * sizeof(size_t),
                               cudaMemcpyHostToDevice, stream));
      ERRCHECK(cudaMallocAsync(&dDecompSizes, hDecompSizes.size() * sizeof(size_t), stream));
      ERRCHECK(cudaMemcpyAsync(dDecompSizes, hDecompSizes.data(), hDecompSizes.size() * sizeof(size_t),
                               cudaMemcpyHostToDevice, stream));

      // Allocate temp space
      nvcompStatus_t status =
         nvcompGetDecompressTempSize(nChunks, maxUncompressedChunkSize, &tempBufSize, decompressedTotalSize);
      if (status != nvcompSuccess) {
         throw std::runtime_error("nvcompBatched*DecompressGetTempSize() failed.");
      }
      ERRCHECK(cudaMallocAsync(&dTempBuf, tempBufSize, stream));

      // Status pointers
      ERRCHECK(cudaMallocAsync(&dStatusPtrs, nChunks * sizeof(nvcompStatus_t), stream));

      ERRCHECK(cudaEventRecord(memEnd, stream));
      ERRCHECK(cudaEventSynchronize(memEnd));
      ERRCHECK(cudaEventElapsedTime(&setupTime, memStart, memEnd));
      setupTime +=
         std::chrono::duration_cast<std::chrono::nanoseconds>(cpuConfigureEnd - cpuConfigureStart).count() / 1e6;
      ERRCHECK(cudaEventDestroy(memStart));
      ERRCHECK(cudaEventDestroy(memEnd));

      if (verbose) {
         std::cout << "chunks        : " << nChunks << std::endl;
         PrintBatch<<<1, 1, 0, stream>>>(dCompressedChunkPointers, dCompSizes, dCompressed, nChunks);
         ERRCHECK(cudaPeekAtLastError());
         PrintBatch<<<1, 1, 0, stream>>>(dDecompressedChunkPointers, dDecompSizes, dDecompressed, nChunks);
         ERRCHECK(cudaPeekAtLastError());
      }
   }

   template <typename GetDecompressSizeFunc, typename GetTempSizeFunc, typename DecompressFunc>
   void DecompressInternal(GetDecompressSizeFunc nvcompGetDecompressSize, GetTempSizeFunc nvcompGetDecompressTempSize,
                           DecompressFunc nvcompDecompress)
   {
      Configure(nvcompGetDecompressSize, nvcompGetDecompressTempSize);

      // Run decompression
      ERRCHECK(cudaEventRecord(decompStart, stream));
      nvcompStatus_t status =
         nvcompDecompress(dCompressedChunkPointers, dCompSizes, dDecompSizes, dDecompSizes, nChunks, dTempBuf,
                          tempBufSize, dDecompressedChunkPointers, dStatusPtrs, stream);
      if (status != nvcompSuccess) {
         throw std::runtime_error("ERROR: nvcompBatched*DecompressAsync() not successful");
      }
      ERRCHECK(cudaEventRecord(decompEnd, stream));
      ERRCHECK(cudaEventSynchronize(decompEnd));
      ERRCHECK(cudaEventElapsedTime(&decompTime, decompStart, decompEnd));

      if (verbose) {
         CheckStatuses<<<1, 1, 0, stream>>>(dStatusPtrs, nChunks);
         ERRCHECK(cudaPeekAtLastError());
         PrintBatch<<<1, 1, 0, stream>>>(dDecompressedChunkPointers, dDecompSizes, dDecompressed, nChunks);
         ERRCHECK(cudaPeekAtLastError());
      }
   }

public:
   GPUDecompressor(const std::vector<char> &data) : hCompressed(data)
   {
      nChunks = 0;
      ERRCHECK(cudaStreamCreate(&stream));
   }

   ~GPUDecompressor()
   {
      ERRCHECK(cudaFree(dDecompressed));
      ERRCHECK(cudaFree(dCompressed));
      ERRCHECK(cudaFree(dCompressedChunkPointers));
      ERRCHECK(cudaFree(dDecompressedChunkPointers));
      ERRCHECK(cudaFree(dCompSizes));
      ERRCHECK(cudaFree(dDecompSizes));
      ERRCHECK(cudaFree(dTempBuf));
      ERRCHECK(cudaFree(dStatusPtrs));

      ERRCHECK(cudaEventDestroy(decompStart));
      ERRCHECK(cudaEventDestroy(decompEnd));
      ERRCHECK(cudaStreamDestroy(stream));
   }

   bool Decompress(std::string type)
   {
      if (type == "zstd") {
         DecompressInternal(nvcompBatchedZstdGetDecompressSizeAsync, nvcompBatchedZstdDecompressGetTempSizeEx,
                            nvcompBatchedZstdDecompressAsync);
      } else if (type == "lz4") {
         DecompressInternal(nvcompBatchedLZ4GetDecompressSizeAsync, nvcompBatchedLZ4DecompressGetTempSizeEx,
                            nvcompBatchedLZ4DecompressAsync);
      } else if (type == "zlib") {
         DecompressInternal(nvcompBatchedDeflateGetDecompressSizeAsync, nvcompBatchedDeflateDecompressGetTempSizeEx,
                            nvcompBatchedDeflateDecompressAsync);
      } else {
         fprintf(stderr, "Unknown decompression type\n");
         return false;
      }

      return true;
   }

   Result GetResult()
   {
      Result result;

      // Retrieve resuts
      ERRCHECK(cudaMemcpyAsync(hDecompressed.data(), dDecompressed, hDecompressed.size() * sizeof(char),
                               cudaMemcpyDeviceToHost, stream));
      ERRCHECK(cudaStreamSynchronize(stream));

      result.decompTime = decompTime;
      result.setupTime = setupTime;
      result.decompressed = hDecompressed;

      return result;
   }
};

float GetMean(const std::vector<float> &vec)
{
   return std::accumulate(vec.begin(), vec.end(), 0.0) / vec.size();
}

float GetStdDev(const std::vector<float> &vec)
{
   auto mean = GetMean(vec);
   std::vector<double> diff(vec.size());
   std::transform(vec.begin(), vec.end(), diff.begin(), [mean](double x) { return x - mean; });
   double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
   return std::sqrt(sq_sum / vec.size());
}

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
   int repetitions = 1;

   int c;
   while ((c = getopt(argc, argv, "f:t:o:vn:")) != -1) {
      switch (c) {
      case 'f': file_name = optarg; break;
      case 't': type = optarg; break;
      case 'o': output_file = optarg; break;
      case 'v': verbose = true; break;
      case 'n': repetitions = atoi(optarg); break;
      default: std::cout << "Got unknown parse returns: " << char(c) << std::endl; return 1;
      }
   }

   if (file_name.empty() || type.empty()) {
      std::cerr << "Must specify a file (-f) and decompression type (-t)" << std::endl;
      return 1;
   }

   const std::vector<char> &data = readFile(file_name);
   size_t input_buffer_len = data.size();

   std::cout << "--------------------- INPUT INFORMATION ---------------------" << std::endl;
   std::cout << "file name     : " << file_name.c_str() << std::endl;
   std::cout << "compressed (B): " << input_buffer_len << std::endl;
   std::cout << "type          : " << type.c_str() << std::endl;

   std::vector<float> setupTimes, decompTimes;
   Result result;
   for (int i = 0; i < repetitions; i++) {
      GPUDecompressor decompressor(data);
      decompressor.Decompress(type);
      result = decompressor.GetResult();
      setupTimes.push_back(result.setupTime);
      decompTimes.push_back(result.decompTime);
   }

   std::cout << "--------------------- OUTPUT INFORMATION ---------------------" << std::endl;
   std::cout << "decompressed (B): " << result.decompressed.size() << std::endl;
   std::cout << "Avg setup (ms)\tStdDev\t\tAvg decomp (ms)\t\tStdDev\t\tRatio\t\tRepetitions" << std::endl;
   std::cout << GetMean(setupTimes) << "\t\t" << GetStdDev(setupTimes) << "\t\t" << GetMean(decompTimes) << "\t\t\t"
             << GetStdDev(decompTimes) << "\t\t" << result.decompressed.size() / (double) data.size() << "\t\t" << repetitions << std::endl;

   if (!output_file.empty()) {
      std::cout << "output file: " << output_file.c_str() << std::endl;
      auto fp = fopen(output_file.c_str(), "w");
      for (auto i = 0; i < result.decompressed.size(); i++) {
         fprintf(fp, "%c", result.decompressed[i]);
      }
   }

   return 0;
}
