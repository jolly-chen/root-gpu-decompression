////////////////////////////////////////////
// Decompress zstd compressed files.
////////////////////////////////////////////

#include <random>
#include <assert.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <unistd.h>
#include <string>

#include "nvcomp/zstd.h"
#include "nvcomp/lz4.h"
#include "nvcomp/deflate.h"
#include "BatchData.cuh"

bool debug = false;

struct result_t {
   std::vector<char> decompressed;
   double decomp_time, transfer_time, total_time;

   result_t() {}
   result_t(size_t decompressed_size) : decompressed(decompressed_size) {}
};

template <typename GetDecompressSizeFunc, typename GetTempSizeFunc, typename DecompressFunc>
result_t Decompress(GetDecompressSizeFunc nvcompGetDecompressSize, GetTempSizeFunc nvcompGetDecompressTempSize,
                    DecompressFunc nvcompDecompress, const std::vector<char> &data, const size_t in_bytes,
                    const size_t chunk_size)
{
   // Create CUDA stream
   cudaStream_t stream;
   cudaStreamCreate(&stream);

   // Copy compressed data to GPU
   BatchData comp_data(data, chunk_size, in_bytes, stream);
   std::cout << "chunks: " << comp_data.num_chunks << std::endl;

   if (debug) {
      PrintBatch<<<1, 1, 0, stream>>>(comp_data.chunk_pointers, comp_data.chunk_sizes, comp_data.data,
                                      comp_data.num_chunks);
   }

   // CUDA events to measure decompression time
   cudaEvent_t start, end;
   cudaEventCreate(&start);
   cudaEventCreate(&end);

   // Determine the size after decompression per chunk
   size_t *d_decomp_sizes = NULL;
   ERRCHECK(cudaMallocAsync(&d_decomp_sizes, comp_data.num_chunks * sizeof(size_t), stream));
   nvcompStatus_t status = nvcompGetDecompressSize(comp_data.chunk_pointers, comp_data.chunk_sizes, d_decomp_sizes,
                                                   comp_data.num_chunks, stream);
   if (status != nvcompSuccess) {
      throw std::runtime_error("nvcompBatched*DecompressGetSize() failed.");
   }

   // Allocate decompression buffer
   size_t *h_decomp_sizes = (size_t *)malloc(comp_data.num_chunks * sizeof(size_t));
   ERRCHECK(cudaMemcpyAsync(h_decomp_sizes, d_decomp_sizes, comp_data.num_chunks * sizeof(size_t),
                            cudaMemcpyDeviceToHost, stream));
   size_t total_decomp_size = 0;
   for (size_t i = 0; i < comp_data.num_chunks; ++i) {
      total_decomp_size += h_decomp_sizes[i];
   }
   BatchData decomp_data(h_decomp_sizes, d_decomp_sizes, comp_data.num_chunks, total_decomp_size, stream);

   // Allocate temp space
   size_t d_decomp_temp_bytes;
   status = nvcompGetDecompressTempSize(comp_data.num_chunks, chunk_size, &d_decomp_temp_bytes);
   if (status != nvcompSuccess) {
      throw std::runtime_error("nvcompBatched*DecompressGetTempSize() failed.");
   }
   void *d_decomp_temp = NULL;
   ERRCHECK(cudaMallocAsync(&d_decomp_temp, d_decomp_temp_bytes, stream));

   // Status pointers
   nvcompStatus_t *d_status_ptrs = NULL;
   ERRCHECK(cudaMallocAsync(&d_status_ptrs, decomp_data.num_chunks * sizeof(nvcompStatus_t), stream));

   if (debug) {
      PrintBatch<<<1, 1, 0, stream>>>(decomp_data.chunk_pointers, decomp_data.chunk_sizes, decomp_data.data,
                                      decomp_data.num_chunks);
   }

   // Run decompression
   status = nvcompDecompress(comp_data.chunk_pointers, comp_data.chunk_sizes, decomp_data.chunk_sizes, d_decomp_sizes,
                             comp_data.num_chunks, d_decomp_temp, d_decomp_temp_bytes, decomp_data.chunk_pointers,
                             d_status_ptrs, stream);
   if (status != nvcompSuccess) {
      throw std::runtime_error("ERROR: nvcompBatched*DecompressAsync() not successful");
   }

   // Retrieve resuts
   result_t result(total_decomp_size);
   ERRCHECK(cudaMemcpyAsync(result.decompressed.data(), decomp_data.data, total_decomp_size * sizeof(char),
                            cudaMemcpyDeviceToHost, stream));

   if (debug) {
      PrintBatch<<<1, 1, 0, stream>>>(decomp_data.chunk_pointers, decomp_data.chunk_sizes, decomp_data.data,
                                      decomp_data.num_chunks);
   }

   cudaStreamSynchronize(stream);

   cudaFree(d_decomp_sizes);
   cudaFree(d_decomp_temp);
   cudaFree(d_status_ptrs);

   cudaEventDestroy(start);
   cudaEventDestroy(end);
   cudaStreamDestroy(stream);

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

/**
 * Main
 */

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
   std::cout << "compressed (B): " << input_buffer_len << std::endl;

   result_t result;

   if (type == "zstd") {
      std::cout << "method: zstd" << std::endl;
      result = Decompress(nvcompBatchedZstdGetDecompressSizeAsync, nvcompBatchedZstdDecompressGetTempSize,
                          nvcompBatchedZstdDecompressAsync, data, input_buffer_len, 1 << 16);
   } else if (type == "lz4") {
      std::cout << "method: lz4" << std::endl;
      result = Decompress(nvcompBatchedLZ4GetDecompressSizeAsync, nvcompBatchedLZ4DecompressGetTempSize,
                          nvcompBatchedLZ4DecompressAsync, data, input_buffer_len, 1 << 16);
   } else if (type == "zlib") {
      std::cout << "method: zlib" << std::endl;
      result = Decompress(nvcompBatchedDeflateGetDecompressSizeAsync, nvcompBatchedDeflateDecompressGetTempSize,
                          nvcompBatchedDeflateDecompressAsync, data, input_buffer_len, 1 << 16);
   } else {
      fprintf(stderr, "Unknown decompression type\n");
      return 1;
   }

   if (!output_file.empty()) {
      auto fp = fopen(output_file.c_str(), "w");
      for (auto i = 0; i < result.decompressed.size(); i++) {
         fprintf(fp, "%c", result.decompressed[i]);
      }
   }

   return 0;
}
