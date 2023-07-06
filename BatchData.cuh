#pragma once

#include <stdio.h>
#include <string>
#include <error.h>
#include <vector>

#define ERRCHECK(err) __checkCudaErrors((err), __func__, __FILE__, __LINE__)
inline static void __checkCudaErrors(cudaError_t error, std::string func, std::string file, int line)
{
   if (error != cudaSuccess) {
      fprintf(stderr, (func + "(), " + file + ":" + std::to_string(line)).c_str(), "%s\n", cudaGetErrorString(error));
      throw std::bad_alloc();
   }
}

inline size_t compute_batch_size(const std::vector<char> &data, const size_t chunk_size)
{
   return (data.size() + chunk_size - 1) / chunk_size;
}

inline std::vector<size_t>
compute_chunk_sizes(const std::vector<char> &data, const size_t batch_size, const size_t chunk_size)
{
   std::vector<size_t> sizes(batch_size, chunk_size);

   if (data.size() % chunk_size != 0) {
      sizes[sizes.size() - 1] = data.size() % chunk_size;
   }

   return sizes;
}

inline std::vector<void *> get_chunk_ptrs(char *data, const size_t *chunk_sizes, const size_t num_chunks)
{
   std::vector<void *> ptrs(num_chunks);
   ptrs[0] = static_cast<void *>(data);
   size_t offset = 0;
   for (size_t i = 1; i < num_chunks; ++i) {
      offset += chunk_sizes[i-1];
      ptrs[i] = static_cast<void *>(data + offset);
   }
   return ptrs;
}

// For debugging
__global__ void PrintBatch(void **chunk_pointers, size_t *chunk_sizes, char *data, size_t num_chunks)
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

class BatchData {
public:
   void **chunk_pointers;
   size_t *chunk_sizes;
   char *data;
   size_t num_chunks;

   BatchData(const std::vector<char> &host_data, const size_t chunk_size, const size_t in_bytes, cudaStream_t stream)
      : chunk_pointers(), chunk_sizes(), data(), num_chunks(0)
   {
      // Compute number of chunks
      num_chunks = compute_batch_size(host_data, chunk_size);
      ERRCHECK(cudaMallocAsync(&data, chunk_size * num_chunks * sizeof(char), stream));

      // Compute size of each chunk
      std::vector<size_t> sizes = compute_chunk_sizes(host_data, num_chunks, chunk_size);
      ERRCHECK(cudaMallocAsync(&chunk_sizes, sizes.size() * sizeof(size_t), stream));
      ERRCHECK(
         cudaMemcpyAsync(chunk_sizes, sizes.data(), sizes.size() * sizeof(size_t), cudaMemcpyHostToDevice, stream));

      // Copy data to GPU
      ERRCHECK(cudaMallocAsync(&data, in_bytes * sizeof(char), stream));
      ERRCHECK(cudaMemcpyAsync(data, host_data.data(), host_data.size(), cudaMemcpyHostToDevice, stream));

      // Set up chunk pointers
      auto ptrs = get_chunk_ptrs(data, sizes.data(), num_chunks);
      ERRCHECK(cudaMallocAsync(&chunk_pointers, num_chunks * sizeof(void *), stream));
      ERRCHECK(
         cudaMemcpyAsync(chunk_pointers, ptrs.data(), num_chunks * sizeof(void *), cudaMemcpyHostToDevice, stream));
   }

   BatchData(size_t *h_chunk_sizes, size_t *d_chunk_sizes, const size_t num_chunks, const size_t total_size,
             cudaStream_t stream)
      : chunk_pointers(), chunk_sizes(d_chunk_sizes), data(), num_chunks(num_chunks)
   {
      // Allocate space for data
      ERRCHECK(cudaMallocAsync(&data, total_size * sizeof(char), stream));

      // Set up chunk pointers
      auto ptrs = get_chunk_ptrs(data, h_chunk_sizes, num_chunks);
      ERRCHECK(cudaMallocAsync(&chunk_pointers, ptrs.size() * sizeof(void *), stream));
      ERRCHECK(
         cudaMemcpyAsync(chunk_pointers, ptrs.data(), ptrs.size() * sizeof(void *), cudaMemcpyHostToDevice, stream));
   }

   BatchData(BatchData &&other) = default;

   // disable copying
   BatchData(const BatchData &other) = delete;
   BatchData &operator=(const BatchData &other) = delete;
};
