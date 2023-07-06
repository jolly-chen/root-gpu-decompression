////////////////////////////////////////////
// Compress files using RNTupleCompressor
////////////////////////////////////////////

#include <random>
#include <assert.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <unistd.h>
#include <string>

#include "ROOT/RNTupleZip.hxx"

using ROOT::Experimental::Detail::RNTupleCompressor;

bool debug = false;

struct result_t {
   std::vector<char> compressed;

   result_t() {}
   result_t(size_t compressed_size) : compressed(compressed_size) {}
};

result_t Compress(const std::vector<char> &data, int compression_code)
{
   RNTupleCompressor compressor;
   result_t result(data.size());

   auto size = compressor.Zip(data.data(), data.size(), compression_code, result.compressed.data());
   result.compressed.resize(size);

   return result;
}

/**
 * File reading
 */

std::vector<char> readFile(const std::string &filename)
{
   std::vector<char> buffer(4096);
   std::vector<char> host_data;

   std::ifstream fin(filename, std::ifstream::binary);
   fin.exceptions(std::ifstream::failbit | std::ifstream::badbit);

   size_t num;
   do {
      num = fin.readsome(buffer.data(), buffer.size());
      host_data.insert(host_data.end(), buffer.begin(), buffer.begin() + num);
   } while (num > 0);

   return host_data;
}

int main(int argc, char *argv[])
{
   std::string file_name, type, output_file;

   int c;
   while ((c = getopt(argc, argv, "f:t:o:d")) != -1) {
      switch (c) {
      case 'f': file_name = optarg; break;
      case 't': type = optarg; break;
      case 'o': output_file = optarg; break;
      case 'd': debug = true; break;
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
   std::cout << "file name: " << file_name.c_str() << std::endl;
   std::cout << "uncompressed (B): " << input_buffer_len << std::endl;

   result_t result;

   if (type == "zstd") {
      std::cout << "method: zstd" << std::endl;
      result = Compress(data, 505);
   } else if (type == "lz4") {
      std::cout << "method: lz4" << std::endl;
      result = Compress(data, 404);
   } else if (type == "zlib") {
      std::cout << "method: zlib" << std::endl;
      result = Compress(data, 101);
   } else {
      fprintf(stderr, "Unknown decompression type :%s\n", type.c_str());
      return 1;
   }

   std::cout << "compressed (B): " << result.compressed.size() << std::endl;

   if (!output_file.empty()) {
      std::cout << "output file: " << output_file.c_str() << std::endl;
      auto fp = fopen(output_file.c_str(), "w");
      for (auto i = 0; i < result.compressed.size(); i++) {
         fprintf(fp, "%c", result.compressed[i]);
      }
   }

   return 0;
}
