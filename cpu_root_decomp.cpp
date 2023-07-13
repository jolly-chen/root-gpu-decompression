////////////////////////////////////////////
// Compress files using RNTupleDeompressor //
////////////////////////////////////////////

#include <random>
#include <assert.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <unistd.h>
#include <string>
#include <chrono>

#include "utils.h"
#include "ROOT/RNTupleZip.hxx"

using ROOT::Experimental::Detail::RNTupleDecompressor;
using Clock = std::chrono::high_resolution_clock;

bool verbose = false;

struct result_t {
   std::vector<char> data;
   float decompTime;

   result_t() {}
   result_t(size_t compressed_size) : data(compressed_size) {}
};

result_t Decompress(const std::vector<char> &data, size_t decompSize)
{
   RNTupleDecompressor decompressor;
   result_t result(decompSize);

   auto start = Clock::now();
   auto end = Clock::now();
   decompressor.Unzip(data.data(), data.size(), decompSize, result.data.data());
   end = Clock::now();

   result.decompTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1e6;

   return result;
}

int main(int argc, char *argv[])
{
   std::string file_name, type, output_file;
   int decompSize = -1;
   int repetitions = 1;

   int c;
   while ((c = getopt(argc, argv, "f:o:dvs:n:")) != -1) {
      switch (c) {
      case 'f': file_name = optarg; break;
      case 'o': output_file = optarg; break;
      case 'v': verbose = true; break;
      case 's': decompSize = atoi(optarg); break;
      case 'n': repetitions = atoi(optarg); break;
      default: std::cout << "Got unknown parse returns: " << char(c) << std::endl; return 1;
      }
   }

   if (file_name.empty() || decompSize < 0) {
      std::cerr << "Must specify a file (-f), deompression type (-t), and size of decompressed data (-s)" << std::endl;
      return 1;
   }

   const std::vector<char> &data = readFile(file_name);
   size_t input_buffer_len = data.size();

   std::cout << "--------------------- INPUT INFORMATION ---------------------" << std::endl;
   std::cout << "file name       : " << file_name.c_str() << std::endl;
   std::cout << "compressed (B)  : " << input_buffer_len << std::endl;
   std::cout << "repetitions     : " << repetitions << std::endl;

   result_t result;
   std::vector<float> decompTimes;
   for (int i = 0; i < repetitions; i++) {
      result = Decompress(data, decompSize);
      decompTimes.push_back(result.decompTime);
   }

   std::cout << "decompressed (B): " << result.data.size() << std::endl;
   std::cout << "avg time (ms)   : " << GetMean(decompTimes) << std::endl;
   std::cout << "std deviation   : " << GetStdDev(decompTimes) << std::endl;
   std::cout << "ratio           : " << data.size() / (double) decompSize << std::endl;

   if (!output_file.empty()) {
      std::cout << "output file     : " << output_file.c_str() << std::endl;
      auto fp = fopen(output_file.c_str(), "w");
      for (auto i = 0; i < result.data.size(); i++) {
         fprintf(fp, "%c", result.data[i]);
      }
   }

   return 0;
}
