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
using Clock = std::chrono::steady_clock;

bool verbose = false;

struct result_t {
   const int padding = 0;
   std::vector<char> decompressed;
   std::vector<char> unpacked;
   float decompTime;

   result_t() {}
   result_t(size_t decompSize, const int nThreads)
      : decompressed(decompSize + nThreads * padding), unpacked(decompSize + nThreads * padding)
   {
   }
};

void Decompress(const int tid, const int nThreads, std::vector<std::vector<char>> &data, result_t *result,
                const int decompSize, std::atomic<int> &running, std::atomic<int> &completed)
{
   RNTupleDecompressor decompressor;
   running++;
   while (running != nThreads);

   // Distribute files round-robin
   for (int i = tid; i < data.size(); i += nThreads) {
      decompressor.Unzip(data[i].data(), data[i].size(), decompSize,
                         &result->decompressed[i * (decompSize + result->padding)]);
   }

   completed++;
}

void Unpack(const int tid, const int nThreads, const std::vector<std::vector<char>> &data, result_t *result,
            const int decompSize, std::atomic<int> &running, std::atomic<int> &completed)
{
   running++;
   while (running != nThreads);

   for (int i = tid; i < data.size(); i += nThreads) {
      CastSplitUnpack<float, float>(&result->unpacked[0 * (decompSize + result->padding)],
                                    &result->decompressed[0 * (decompSize + result->padding)],
                                    decompSize / sizeof(float));
   }
   completed++;
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
   result_t result(decompSize * multiFileSize, nThreads);
   for (int i = 0; i < repetitions + warmUp; i++) {
      // Decompression
      std::atomic<int> running(0), completed(0);
      for (int t = 0; t < nThreads; t++) {
         threadPool[t] = std::thread(Decompress, t, nThreads, std::ref(files), &result, decompSize, std::ref(running),
                                     std::ref(completed));
      }

      while (running != nThreads);
      auto startDecomp = Clock::now();
      while (completed != nThreads);
      auto endDecomp = Clock::now();
      for (int t = 0; t < nThreads; t++) {
         threadPool[t].join();
      }

      if (i >= warmUp) {
         decompTimes.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(endDecomp - startDecomp).count() /
                               1e6);
      }

      // Unpacking
      if (packed) {
         running.store(0);
         completed.store(0);
         for (int t = 0; t < nThreads; t++) {
            threadPool[t] = std::thread(Unpack, t, nThreads, std::ref(files), &result, decompSize, std::ref(running),
                                        std::ref(completed));
         }

         while (running != nThreads);
         auto startUnpack = Clock::now();
         while (completed != nThreads);
         auto endUnpack = Clock::now();
         for (int t = 0; t < nThreads; t++) {
            threadPool[t].join();
         }

         if (i >= warmUp) {
            unpackTimes.push_back(
               std::chrono::duration_cast<std::chrono::nanoseconds>(endUnpack - startUnpack).count() / 1e6);
         }
      }
   }

   std::cout << "--------------------- OUTPUT INFORMATION ---------------------" << std::endl;
   std::cout << "decompressed (B)     : " << result.decompressed.size() << std::endl;
   std::cout << "ratio                : " << compTotalSize / (double)decompSize << std::endl;
   std::cout << "avg decomp time (ms) : " << GetMean(decompTimes) << std::endl;
   std::cout << "std deviation        : " << GetStdDev(decompTimes) << std::endl;
   std::cout << "avg unpack time (ms) : " << GetMean(unpackTimes) << std::endl;
   std::cout << "std deviation        : " << GetStdDev(unpackTimes) << std::endl;

   if (!outputFile.empty()) {
      std::cout << "output file     : " << outputFile.c_str() << std::endl;
      auto fp = fopen(outputFile.c_str(), "w");
      for (auto i = 0; i < result.decompressed.size(); i++) {
         fprintf(fp, "%c", packed ? result.unpacked[i] : result.decompressed[i]);
      }
   }

   return 0;
}
