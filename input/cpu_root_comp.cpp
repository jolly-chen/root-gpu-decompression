////////////////////////////////////////////
// Compress files using RNTupleCompressor //
////////////////////////////////////////////

#include <random>
#include <assert.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <unistd.h>
#include <string>

#include "utils.h"
#include "ROOT/RNTupleZip.hxx"

using ROOT::Experimental::Detail::RNTupleCompressor;

bool verbose = false;

struct result_t {
   std::vector<unsigned char> data;

   result_t() {}
   result_t(size_t compressed_size) : data(compressed_size) {}
};

result_t Compress(const std::vector<char> &data, int compression_code, bool pack)
{
   RNTupleCompressor compressor;
   result_t result(data.size());
   size_t result_size;

   // Pack the data
   if (pack) {
      std::vector<char> packed(data.size());
      CastSplitPack<float, float>(packed.data(), data.data(), data.size() / 4);
      result_size = compressor.Zip(packed.data(), packed.size(), compression_code, result.data.data());
   } else {
      result_size = compressor.Zip(data.data(), data.size(), compression_code, result.data.data());
   }

   // Compress
   result.data.resize(result_size);

   return result;
}

int main(int argc, char *argv[])
{
   std::string file_name, type, output_file;
   bool pack = false;

   int c;
   while ((c = getopt(argc, argv, "f:t:o:dvp")) != -1) {
      switch (c) {
      case 'f': file_name = optarg; break;
      case 't': type = optarg; break;
      case 'o': output_file = optarg; break;
      case 'v': verbose = true; break;
      case 'p': pack = true; break;
      default: std::cout << "Got unknown parse returns: " << char(c) << std::endl; return 1;
      }
   }

   if (file_name.empty() || type.empty()) {
      std::cerr << "Must specify a file (-f) and compression type (-t)" << std::endl;
      return 1;
   }

   const std::vector<char> &data = ReadFile(file_name);
   size_t input_buffer_len = data.size();

   std::cout << "--------------------- INPUT INFORMATION ---------------------" << std::endl;
   std::cout << "file name: " << file_name.c_str() << std::endl;
   std::cout << "uncompressed (B): " << input_buffer_len << std::endl;
   std::cout << "packed          : " << (pack ? "yes" : "no") << std::endl;
   std::cout << "method          : " << type << std::endl;

   result_t result;

   if (type == "zstd") {
      result = Compress(data, 505, pack);
   } else if (type == "lz4") {
      result = Compress(data, 404, pack);
   } else if (type == "zlib") {
      result = Compress(data, 101, pack);
   } else {
      fprintf(stderr, "Unknown compression type :%s\n", type.c_str());
      return 1;
   }

   std::cout << "compressed (B)  : " << result.data.size() << std::endl;
   std::cout << "ratio           : " << data.size() / (double)result.data.size() << std::endl;

   if (!output_file.empty()) {
      std::cout << "output file: " << output_file.c_str() << std::endl;
      auto fp = fopen(output_file.c_str(), "w");
      for (auto i = 0; i < result.data.size(); i++) {
         fprintf(fp, "%c", result.data[i]);
      }
   }

   return 0;
}
