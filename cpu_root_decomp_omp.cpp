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
#include <omp.h>
#include <atomic>

#include "utils.h"
#include "ROOT/RNTupleZip.hxx"

using ROOT::Experimental::Detail::RNTupleDecompressor;
using Clock = std::chrono::high_resolution_clock;

bool verbose = false;

struct result_t {
   std::vector<char> decompressed;
   std::vector<char> unpacked;

   result_t() {}
   result_t(size_t decompSize) : decompressed(decompSize), unpacked(decompSize) {}
};

float Decompress(const std::vector<std::vector<char>> &data, result_t *result, const int decompSize)
{
   float decompTime;
   auto startDecomp = Clock::now();

#pragma omp parallel shared(data, result, decompSize)
   {
#pragma omp single
      startDecomp = Clock::now();

#pragma omp for nowait
      for (int i = 0; i < data.size(); i++) {
         RNTupleDecompressor decompressor;
         decompressor.Unzip(data[i].data(), data[i].size(), decompSize, &result->decompressed[i * decompSize]);
      }

      decompTime = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - startDecomp).count() / 1e6;
   }

   return decompTime;
}

float Unpack(const std::vector<std::vector<char>> &data, result_t *result, const int decompSize)
{
   float unpackTime;
   auto startUnpack = Clock::now();

#pragma omp parallel shared(data, result, decompSize)
   {
#pragma omp single

      startUnpack = Clock::now();

#pragma omp for nowait
      for (int i = 0; i < data.size(); i++) {
         CastSplitUnpack<float, float>(&result->unpacked[i * decompSize], &result->decompressed[i * decompSize],
                                       decompSize / sizeof(float));
      }

      unpackTime = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - startUnpack).count() / 1e6;
   }

   return unpackTime;
}

int main(int argc, char *argv[])
{
   std::string fileName, type, outputFile;
   int decompSize = -1;
   int repetitions = 1;
   int multiFileSize = 1;
   int nThreads = std::max(1, omp_get_max_threads());
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

   omp_set_num_threads(nThreads);
   std::vector<float> decompTimes, unpackTimes;
   result_t result(decompSize * multiFileSize);
   for (int i = 0; i < repetitions + warmUp; i++) {
      auto dt = Decompress(files, &result, decompSize);

      float pt = 0;
      if (packed) {
         pt = Unpack(files, &result, decompSize);
      }

      if (i >= warmUp) {
         decompTimes.push_back(dt);
         unpackTimes.push_back(pt);
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
