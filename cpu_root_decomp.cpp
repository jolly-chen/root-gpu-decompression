////////////////////////////////////////////
// Compress files using RNTupleDecompressor //
////////////////////////////////////////////

#include <random>
#include <assert.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <unistd.h>
#include <string>
#include <chrono>
#include <thread>
#include <atomic>

#include "utils.h"
#include "ROOT/RNTupleZip.hxx"

using ROOT::Experimental::Detail::RNTupleDecompressor;
using Clock = std::chrono::high_resolution_clock;

bool verbose = false;

struct result_t {
   std::vector<char> decompressed;
   std::vector<char> unpacked;
   float decompTime;

   result_t() {}
   result_t(size_t decompSize) : decompressed(decompSize), unpacked(decompSize) {}
};

void Decompress(const int tid, const int nThreads, const std::vector<std::vector<char>> &data, result_t *result,
                const int decompSize, std::atomic<bool> &startRunning)
{
   while (!startRunning) {
   }

   // Distribute files round-robin
   for (int i = tid; i < data.size(); i += nThreads) {
      RNTupleDecompressor decompressor;
      decompressor.Unzip(data[i].data(), data[i].size(), decompSize, &result->decompressed[i * decompSize]);
   }
}

void Unpack(const int tid, const int nThreads, const std::vector<std::vector<char>> &data, result_t *result,
            const int decompSize, std::atomic<bool> &startRunning)
{
   while (!startRunning) {
   }

   for (int i = tid; i < data.size(); i += nThreads) {
      CastSplitUnpack<float, float>(&result->unpacked[i * decompSize], &result->decompressed[i * decompSize],
                                    decompSize / sizeof(float));
   }
}

int main(int argc, char *argv[])
{
   std::string fileName, type, outputFile;
   int decompSize = -1;
   int repetitions = 1;
   int multiFileSize = 1;
   int nThreads = std::max(1, (int)std::thread::hardware_concurrency());
   int warmUp = 10;
   bool packed = false;

   int c;
   while ((c = getopt(argc, argv, "f:o:dvs:n:m:c:w:p")) != -1) {
      switch (c) {
      case 'f': fileName = optarg; break;
      case 'o': outputFile = optarg; break;
      case 'v': verbose = true; break;
      case 's': decompSize = atoi(optarg); break;
      case 'm': multiFileSize = atoi(optarg); break;
      case 'n': repetitions = atoi(optarg); break;
      case 'c': nThreads = atoi(optarg); break;
      case 'w': warmUp = atoi(optarg); break;
      case 'p': packed = true; break;
      default: std::cout << "Ignoring unknown parse returns: " << char(c) << std::endl;
      }
   }

   if (fileName.empty() || decompSize < 0) {
      std::cerr << "Must specify a file (-f) and size of decompressed data (-s)" << std::endl;
      return 1;
   }

   auto files = GenerateMultiFile(fileName, multiFileSize);
   size_t compTotalSize = 0;
   for (int i = 0; i < files.size(); i++) {
      compTotalSize += files[i].size();
   }

   std::cout << "--------------------- INPUT INFORMATION ---------------------" << std::endl;
   std::cout << "file name       : " << fileName.c_str() << std::endl;
   std::cout << "compressed (B)  : " << compTotalSize << std::endl;
   std::cout << "packed          : " << (packed ? "yes" : "no") << std::endl;
   std::cout << "repetitions     : " << repetitions << std::endl;
   std::cout << "warmup          : " << warmUp << std::endl;
   std::cout << "threads         : " << nThreads << std::endl;

   std::thread threadPool[nThreads];
   std::vector<float> decompTimes, unpackTimes;
   result_t result(decompSize * multiFileSize);
   for (int i = 0; i < repetitions + warmUp; i++) {
      std::atomic<bool> startRunning(false);
      for (int t = 0; t < nThreads; t++) {
         threadPool[t] =
            std::thread(Decompress, t, nThreads, std::ref(files), &result, decompSize, std::ref(startRunning));
      }

      auto startDecomp = Clock::now();
      startRunning.store(true);
      for (int t = 0; t < nThreads; t++) {
         threadPool[t].join();
      }
      auto endDecomp = Clock::now();

      auto startUnpack = Clock::now();
      auto endUnpack = Clock::now();
      if (packed) {
         std::atomic<bool> startRunning(false);
         for (int t = 0; t < nThreads; t++) {
            threadPool[t] =
               std::thread(Unpack, t, nThreads, std::ref(files), &result, decompSize, std::ref(startRunning));
         }

         startUnpack = Clock::now();
         startRunning.store(true);
         for (int t = 0; t < nThreads; t++) {
            threadPool[t].join();
         }
         endUnpack = Clock::now();
      }

      if (i >= warmUp) {
         decompTimes.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(endDecomp - startDecomp).count() /
                               1e6);
         unpackTimes.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(endUnpack - startUnpack).count() /
                               1e6);
      }
   }

   std::cout << "--------------------- OUTPUT INFORMATION ---------------------" << std::endl;
   std::cout << "decompressed (B)     : " << result.decompressed.size() << std::endl;
   std::cout << "ratio                : " << compTotalSize / (double)decompSize << std::endl;
   std::cout << "avg decomp time (ms) : " << GetMean(decompTimes) << std::endl;
   std::cout << "std deviation        : " << GetStdDev(decompTimes) << std::endl;
   if (packed) {
      std::cout << "avg unpack time (ms) : " << GetMean(unpackTimes) << std::endl;
      std::cout << "std deviation        : " << GetStdDev(unpackTimes) << std::endl;
   }

   if (!outputFile.empty()) {
      std::cout << "output file     : " << outputFile.c_str() << std::endl;
      auto fp = fopen(outputFile.c_str(), "w");
      for (auto i = 0; i < result.decompressed.size(); i++) {
         fprintf(fp, "%c", packed ? result.unpacked[i] : result.decompressed[i]);
      }
   }

   return 0;
}
