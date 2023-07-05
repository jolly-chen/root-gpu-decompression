#include <random>
#include <assert.h>
#include <iostream>
#include <vector>
#include <fstream>

#include "nvcomp/zstd.h"
#include "BatchData.cuh"

// For debugging
__global__ void PrintBatch(void **chunk_pointers, size_t *chunk_sizes, uint8_t *data, size_t num_chunks)
{
   printf("------------------------- BATCH ----------------------------\n");
   printf("num_chunks: %li\n", num_chunks);

   printf("chunk_sizes:\n");
   size_t total_size = 0;
   for (int i = 0; i < num_chunks; i++) {
      printf("%li ", chunk_sizes[i]);
      total_size += chunk_sizes[i];
   }
   printf("\n");

   printf("total_size: %li\n", total_size);

   printf("data:\n");
   for (int i = 0; i < total_size; i++) {
      printf("%c ", data[i]);
   }
   printf("\n");

   printf("chunk_pointers:\n");
   for (int i = 0; i < num_chunks; i++) {
      printf("\tchunk %d:\n\t\t", i);
      for (int j = 0; j < chunk_sizes[i]; j++) {
         printf("%c ", ((char *)chunk_pointers[i])[j]);
      }
      printf("\n");
   }
   printf("\n");
}

void Decompress(const std::vector<std::vector<char>> &data, const size_t in_bytes)
{
   const size_t chunk_size = 1 << 16;

   // Copy compressed data to GPU
   BatchData compress_data(data, chunk_size, in_bytes);
   std::cout << "chunks: " << compress_data.num_chunks << std::endl;

   // Create CUDA stream
   cudaStream_t stream;
   cudaStreamCreate(&stream);

   // CUDA events to measure decompression time
   cudaEvent_t start, end;
   cudaEventCreate(&start);
   cudaEventCreate(&end);

   // Determine the size after decompression per chunk
   size_t *d_decomp_sizes = NULL;
   ERRCHECK(cudaMalloc(&d_decomp_sizes, compress_data.num_chunks * sizeof(size_t)));
   nvcompStatus_t status = nvcompBatchedZstdGetDecompressSizeAsync(
      compress_data.chunk_pointers, compress_data.chunk_sizes, d_decomp_sizes, compress_data.num_chunks, stream);
   if (status != nvcompSuccess) {
      throw std::runtime_error("nvcompBatchedZstdDecompressGetTempSize() failed.");
   }
   ERRCHECK(cudaStreamSynchronize(stream));

   // Allocate decompression buffer
   size_t *decomp_sizes = (size_t *)malloc(compress_data.num_chunks * sizeof(size_t));
   ERRCHECK(
      cudaMemcpy(decomp_sizes, d_decomp_sizes, compress_data.num_chunks * sizeof(size_t), cudaMemcpyDeviceToHost));
   size_t total_decomp_size = 0;
   for (size_t i = 0; i < compress_data.num_chunks; ++i) {
      total_decomp_size += decomp_sizes[i];
   }
   BatchData decomp_data(total_decomp_size, compress_data.num_chunks);

   // Allocate temp space
   size_t decomp_temp_bytes;
   status = nvcompBatchedZstdDecompressGetTempSize(compress_data.num_chunks, chunk_size, &decomp_temp_bytes);
   if (status != nvcompSuccess) {
      throw std::runtime_error("nvcompBatchedZstdDecompressGetTempSize() failed.");
   }
   void *d_decomp_temp = NULL;
   ERRCHECK(cudaMalloc(&d_decomp_temp, decomp_temp_bytes));

   // Status pointers
   nvcompStatus_t *d_status_ptrs = NULL;
   ERRCHECK(cudaMalloc(&d_status_ptrs, decomp_data.num_chunks * sizeof(nvcompStatus_t)));

   // Run decompression
   status = nvcompBatchedZstdDecompressAsync(
      compress_data.chunk_pointers, compress_data.chunk_sizes, decomp_data.chunk_sizes, d_decomp_sizes,
      compress_data.num_chunks, d_decomp_temp, decomp_temp_bytes, decomp_data.chunk_pointers, d_status_ptrs, stream);
   if (status != nvcompSuccess) {
      throw std::runtime_error("ERROR: nvcompBatchedGzipDecompressAsync() not successful");
   }
   ERRCHECK(cudaStreamSynchronize(stream));

   std::vector<std::vector<char>> uncompressed;

   size_t offset = 0;
   for (size_t i = 0; i < decomp_data.num_chunks; ++i) {
      std::vector<char> result(decomp_sizes[i]);
      ERRCHECK(cudaMemcpy(result.data(), &decomp_data.data[offset], decomp_sizes[i] * sizeof(char),
                          cudaMemcpyDeviceToHost));
      offset += decomp_sizes[i];
      uncompressed.push_back(result);
   }

   for (auto file = 0; file < uncompressed.size(); file++) {
      for (auto i = 0; i < uncompressed[file].size(); i++) {
         fprintf(stdout, "%c", uncompressed[file][i]);
      }
   }

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

std::vector<std::vector<char>> multi_file(const std::vector<std::string> &filenames)
{
   std::vector<std::vector<char>> split_data;

   for (auto const &filename : filenames) {
      split_data.emplace_back(readFile(filename));
   }

   return split_data;
}

/**
 * Main
 */

int main(int argc, char *argv[])
{
   std::vector<std::string> file_names(argc - 1);

   if (argc == 1) {
      std::cerr << "Must specify at least one file." << std::endl;
      return 1;
   }

   // multi-file mode
   for (int i = 1; i < argc; ++i) {
      file_names[i - 1] = argv[i];
   }

   const std::vector<std::vector<char>> &data = multi_file(file_names);
   size_t input_buffer_len = data.size();
   for (const std::vector<char> &part : data) {
      input_buffer_len += part.size();
   }

   std::cout << "----------" << std::endl;
   std::cout << "files: " << data.size() << std::endl;
   std::cout << "compressed (B): " << input_buffer_len << std::endl;

   Decompress(data, input_buffer_len);

   return 0;
}
