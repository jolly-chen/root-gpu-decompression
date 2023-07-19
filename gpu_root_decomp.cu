////////////////////////////////////////////
// Decompress ROOT compressed files.
////////////////////////////////////////////

#include <random>
#include <assert.h>
#include <iostream>
#include <vector>
#include <unistd.h>
#include <string>
#include <chrono>

#include "utils.h"
#include "pack.cuh"
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
   for (int k = 0; k < 100; k++) {
      printf("%c ", data[k]);
   }
   printf("\n");

   printf("chunk_pointers:\n");
   for (int c = 0; c < nChunks; c++) {
      printf("\tchunk %d:\n\t\t", c);
      for (int j = 0; j < min(100, (int)chunk_sizes[c]); j++) {
         printf("%c ", ((char *)chunk_pointers[c])[j]);
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
   float setupTime, decompTime, unpackTime;
   std::vector<char> decompressed;
};

class GPUDecompressor {
private:
   cudaStream_t stream;
   size_t nChunks;
   size_t compTotalSize, decompTotalSize;
   bool packed;

   // Host buffers
   std::vector<std::vector<char>> hCompressed;
   std::vector<char> hDecompressed;
   std::vector<size_t> hCompSizes, hDecompSizes;

   // Device buffers;
   char *dCompressed, *dDecompressed, *dUnpackOut;
   void **dCompressedChunkPointers, **dDecompressedChunkPointers;
   size_t *dCompSizes, *dDecompSizes;
   size_t tempBufSize;
   void *dTempBuf;
   nvcompStatus_t *dStatusPtrs;

   float setupTime, decompTime, unpackTime;

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
      // For measuring setup time
      auto configureStart = Clock::now();

      decompTotalSize = 0;
      int maxUncompressedChunkSize = 0;
      for (int i = 0; i < hCompressed.size(); i++) {
         size_t remainder = hCompressed[i].size();
         auto source = const_cast<unsigned char *>(reinterpret_cast<const unsigned char *>(hCompressed[i].data()));

         // Loop over the chunks to determine their sizes from the header
         do {
            int szSource;
            int szTarget;
            int retval = R__unzip_header(&szSource, source, &szTarget);
            R__ASSERT(retval == 0);
            R__ASSERT(szSource > 0);
            R__ASSERT(szTarget > szSource);
            R__ASSERT(static_cast<unsigned char>(szSource) <= hCompressed[i].size());

            nChunks++;
            hCompSizes.push_back(szSource);
            hDecompSizes.push_back(szTarget);

            decompTotalSize += szTarget;
            if (szTarget > maxUncompressedChunkSize)
               maxUncompressedChunkSize = szTarget;

            // Move to next chunk
            source += szSource;
            remainder -= szSource;
         } while (remainder > 0);
         R__ASSERT(remainder == 0);
      }

      hDecompressed.resize(decompTotalSize);

      // Set up buffers for the compressed and decompressed data on the device.
      ERRCHECK(cudaMallocAsync(&dCompressed, compTotalSize * sizeof(char), stream));
      int offset = 0;
      for (int i = 0; i < hCompressed.size(); i++) {
         ERRCHECK(cudaMemcpyAsync(&dCompressed[offset], hCompressed[i].data(), hCompressed[i].size() * sizeof(char),
                                  cudaMemcpyHostToDevice, stream));
         offset += hCompressed[i].size();
      }
      ERRCHECK(cudaMallocAsync(&dDecompressed, decompTotalSize * sizeof(char), stream));

      // Wait for the buffers to be allocated to create chunk pointers.
      ERRCHECK(cudaStreamSynchronize(stream));

      // Set up pointers to each chunk in the device buffer for the compressed data.
      auto cPtrs = GetCompressedChunkPtrs();
      ERRCHECK(cudaMallocAsync(&dCompressedChunkPointers, nChunks * sizeof(void *), stream));
      ERRCHECK(cudaMemcpyAsync(dCompressedChunkPointers, cPtrs.data(), nChunks * sizeof(void *), cudaMemcpyHostToDevice,
                               stream));

      // Set up pointers to each chunk in the device buffer for the decompressed data.
      auto dcPtrs = GetDecompressedChunkPtrs();
      ERRCHECK(cudaMallocAsync(&dDecompressedChunkPointers, nChunks * sizeof(void *), stream));
      ERRCHECK(cudaMemcpyAsync(dDecompressedChunkPointers, dcPtrs.data(), nChunks * sizeof(void *),
                               cudaMemcpyHostToDevice, stream));

      // Copy compressed and decompressed sizes of each chunk
      std::transform(hCompSizes.begin(), hCompSizes.end(), hCompSizes.begin(), [&](auto x) { return x - HDRSIZE; });
      ERRCHECK(cudaMallocAsync(&dCompSizes, hCompSizes.size() * sizeof(size_t), stream));
      ERRCHECK(cudaMemcpyAsync(dCompSizes, hCompSizes.data(), hCompSizes.size() * sizeof(size_t),
                               cudaMemcpyHostToDevice, stream));
      ERRCHECK(cudaMallocAsync(&dDecompSizes, hDecompSizes.size() * sizeof(size_t), stream));
      ERRCHECK(cudaMemcpyAsync(dDecompSizes, hDecompSizes.data(), hDecompSizes.size() * sizeof(size_t),
                               cudaMemcpyHostToDevice, stream));

      // Allocate temp space
      nvcompStatus_t status =
         nvcompGetDecompressTempSize(nChunks, maxUncompressedChunkSize, &tempBufSize, decompTotalSize);
      if (status != nvcompSuccess) {
         throw std::runtime_error("nvcompBatched*DecompressGetTempSize() failed.");
      }
      ERRCHECK(cudaMallocAsync(&dTempBuf, tempBufSize, stream));

      // Status pointers
      ERRCHECK(cudaMallocAsync(&dStatusPtrs, nChunks * sizeof(nvcompStatus_t), stream));

      if (packed) {
         ERRCHECK(cudaMallocAsync(&dUnpackOut, decompTotalSize * sizeof(size_t), stream));
      }

      ERRCHECK(cudaDeviceSynchronize());
      setupTime = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - configureStart).count() / 1e6;

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

      // CUDA events to measure time
      cudaEvent_t decompStart, decompEnd, unpackStart, unpackEnd;

      // For measuring decompression runtime
      ERRCHECK(cudaEventCreate(&decompStart));
      ERRCHECK(cudaEventCreate(&decompEnd));

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
      ERRCHECK(cudaEventDestroy(decompStart));
      ERRCHECK(cudaEventDestroy(decompEnd));

      // Unpack data if necessary
      if (packed) {
         ERRCHECK(cudaEventCreate(&unpackStart));
         ERRCHECK(cudaEventCreate(&unpackEnd));
         ERRCHECK(cudaEventRecord(unpackStart, stream));

         Unpack1<float, float><<<ceil(decompTotalSize / 256.), 256, 0, stream>>>(
            dUnpackOut, dDecompressed, dDecompSizes, nChunks, decompTotalSize);
         ERRCHECK(cudaPeekAtLastError());
         dDecompressed = dUnpackOut;

         ERRCHECK(cudaEventRecord(unpackEnd, stream));
         ERRCHECK(cudaEventSynchronize(unpackEnd));
         ERRCHECK(cudaEventElapsedTime(&unpackTime, unpackStart, unpackEnd));
         ERRCHECK(cudaEventDestroy(unpackStart));
         ERRCHECK(cudaEventDestroy(unpackEnd));
      }

      if (verbose) {
         CheckStatuses<<<1, 1, 0, stream>>>(dStatusPtrs, nChunks);
         ERRCHECK(cudaPeekAtLastError());
         PrintBatch<<<1, 1, 0, stream>>>(dDecompressedChunkPointers, dDecompSizes, dDecompressed, nChunks);
         ERRCHECK(cudaPeekAtLastError());
      }
   }

public:
   GPUDecompressor(const std::vector<std::vector<char>> &data, const size_t totalSize, bool _packed)
      : hCompressed(data), packed(_packed)
   {
      nChunks = 0;
      compTotalSize = totalSize;
      setupTime = 0;
      decompTime = 0;
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
      result.unpackTime = unpackTime;
      result.decompressed = hDecompressed;

      return result;
   }
};

/**
 * Main
 */

int main(int argc, char *argv[])
{
   std::string fileName, type, outputFile;
   int repetitions = 1;
   int multiFileSize = 1;
   int warmUp = 10;
   bool packed = false;

   int c;
   while ((c = getopt(argc, argv, "f:t:o:vn:m:w:p")) != -1) {
      switch (c) {
      case 'f': fileName = optarg; break;
      case 't': type = optarg; break;
      case 'o': outputFile = optarg; break;
      case 'v': verbose = true; break;
      case 'n': repetitions = atoi(optarg); break;
      case 'm': multiFileSize = atoi(optarg); break;
      case 'w': warmUp = atoi(optarg); break;
      case 'p': packed = true; break;
      default: std::cout << "Ignoring unknown parse returns: " << char(c) << std::endl; ;
      }
   }

   if (fileName.empty() || type.empty()) {
      std::cerr << "Must specify a file (-f) and decompression type (-t)" << std::endl;
      return 1;
   }

   auto files = GenerateMultiFile(fileName, multiFileSize);
   size_t totalSize = 0;
   for (int i = 0; i < files.size(); i++) {
      totalSize += files[i].size();
   }

   std::cout << "--------------------- INPUT INFORMATION ---------------------" << std::endl;
   std::cout << "file name      : " << fileName.c_str() << std::endl;
   std::cout << "type           : " << type.c_str() << std::endl;
   std::cout << "compressed (B) : " << totalSize << std::endl;
   std::cout << "repetitions    : " << repetitions << std::endl;
   std::cout << "warmup         : " << warmUp << std::endl;
   std::cout << "packed         : " << (packed ? "yes" : "no") << std::endl;

   std::vector<float> setupTimes, decompTimes, unpackTimes;
   Result result;
   for (int i = 0; i < repetitions + warmUp; i++) {
      GPUDecompressor decompressor(files, totalSize, packed);
      decompressor.Decompress(type);
      result = decompressor.GetResult();

      if (i >= warmUp) {
         setupTimes.push_back(result.setupTime);
         decompTimes.push_back(result.decompTime);
         unpackTimes.push_back(result.unpackTime);
      }
   }

   std::cout << "--------------------- OUTPUT INFORMATION ---------------------" << std::endl;
   std::cout << "decompressed (B): " << result.decompressed.size() << std::endl;
   std::cout << "Ratio\t\tAvg setup (ms)\tStdDev\t\tAvg decomp (ms)\t\tStdDev\t\tAvg unpack (ms)\t\tStdDev" << std::endl;
   std::cout << result.decompressed.size() / (double)totalSize << "\t\t" << GetMean(setupTimes) << "\t"
             << GetStdDev(setupTimes) << "\t\t" << GetMean(decompTimes) << "\t\t" << GetStdDev(decompTimes) << "\t\t"
             << GetMean(unpackTimes) << "\t" << GetStdDev(unpackTimes) << std::endl;

   if (!outputFile.empty()) {
      std::cout << "output file: " << outputFile.c_str() << std::endl;
      auto fp = fopen(outputFile.c_str(), "w");
      for (auto i = 0; i < result.decompressed.size(); i++) {
         fprintf(fp, "%c", result.decompressed[i]);
      }
   }

   return 0;
}
