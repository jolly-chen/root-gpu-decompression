#include <random>
#include <assert.h>
#include <iostream>
#include <vector>
#include <fstream>

#include "nvcomp/zstd.h"
#include "BatchData.cuh"

void Decompress(const std::vector<char> &data, const size_t in_bytes)
{
   const size_t chunk_size = 1 << 16;

   // Copy compressed data to GPU
   BatchData comp_data(data, chunk_size, in_bytes);
   std::cout << "chunks: " << comp_data.num_chunks << std::endl;

   // Create CUDA stream
   cudaStream_t stream;
   cudaStreamCreate(&stream);

   // CUDA events to measure decompression time
   cudaEvent_t start, end;
   cudaEventCreate(&start);
   cudaEventCreate(&end);

   // Determine the size after decompression per chunk
   size_t *d_decomp_sizes = NULL;
   ERRCHECK(cudaMallocAsync(&d_decomp_sizes, comp_data.num_chunks * sizeof(size_t), stream));
   nvcompStatus_t status = nvcompBatchedZstdGetDecompressSizeAsync(comp_data.chunk_pointers, comp_data.chunk_sizes,
                                                                   d_decomp_sizes, comp_data.num_chunks, stream);
   if (status != nvcompSuccess) {
      throw std::runtime_error("nvcompBatchedZstdDecompressGetTempSize() failed.");
   }

   // Allocate decompression buffer
   size_t *h_decomp_sizes = (size_t *)malloc(comp_data.num_chunks * sizeof(size_t));
   ERRCHECK(cudaMemcpyAsync(h_decomp_sizes, d_decomp_sizes, comp_data.num_chunks * sizeof(size_t),
                            cudaMemcpyDeviceToHost, stream));
   size_t total_decomp_size = 0;
   for (size_t i = 0; i < comp_data.num_chunks; ++i) {
      total_decomp_size += h_decomp_sizes[i];
   }
   BatchData decomp_data(h_decomp_sizes, d_decomp_sizes, comp_data.num_chunks, total_decomp_size);

   // Allocate temp space
   size_t d_decomp_temp_bytes;
   status = nvcompBatchedZstdDecompressGetTempSize(comp_data.num_chunks, chunk_size, &d_decomp_temp_bytes);
   if (status != nvcompSuccess) {
      throw std::runtime_error("nvcompBatchedZstdDecompressGetTempSize() failed.");
   }
   void *d_decomp_temp = NULL;
   ERRCHECK(cudaMallocAsync(&d_decomp_temp, d_decomp_temp_bytes, stream));

   // Status pointers
   nvcompStatus_t *d_status_ptrs = NULL;
   ERRCHECK(cudaMallocAsync(&d_status_ptrs, decomp_data.num_chunks * sizeof(nvcompStatus_t), stream));

   // Run decompression
   status = nvcompBatchedZstdDecompressAsync(comp_data.chunk_pointers, comp_data.chunk_sizes, decomp_data.chunk_sizes,
                                             d_decomp_sizes, comp_data.num_chunks, d_decomp_temp, d_decomp_temp_bytes,
                                             decomp_data.chunk_pointers, d_status_ptrs, stream);
   if (status != nvcompSuccess) {
      throw std::runtime_error("ERROR: nvcompBatchedGzipDecompressAsync() not successful");
   }

   // Retrieve resuts
   std::vector<char> uncompressed(total_decomp_size);
   ERRCHECK(cudaMemcpyAsync(uncompressed.data(), decomp_data.data, total_decomp_size * sizeof(char),
                            cudaMemcpyDeviceToHost, stream));
   cudaStreamSynchronize(stream);

   std::cout << "---------------  RESULTS ------------------" << std::endl;
   for (auto i = 0; i < uncompressed.size(); i++) {
      fprintf(stdout, "%c", uncompressed[i]);
   }
   fprintf(stdout, "\n\n");

   cudaFree(d_decomp_temp);

   cudaEventDestroy(start);
   cudaEventDestroy(end);
   cudaStreamDestroy(stream);
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
   std::string file_name;

   if (argc == 1) {
      std::cerr << "Must specify a file." << std::endl;
      return 1;
   }

   file_name = argv[1];

   const std::vector<char> &data = readFile(file_name);
   size_t input_buffer_len = data.size();

   std::cout << "----------" << std::endl;
   std::cout << "compressed (B): " << input_buffer_len << std::endl;

   Decompress(data, input_buffer_len);

   return 0;
}
