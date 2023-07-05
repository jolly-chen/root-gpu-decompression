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

static size_t compute_batch_size(const std::vector<std::vector<char>> &data, const size_t chunk_size)
{
   size_t batch_size = 0;
   for (size_t i = 0; i < data.size(); ++i) {
      const size_t num_chunks = (data[i].size() + chunk_size - 1) / chunk_size;
      batch_size += num_chunks;
   }

   return batch_size;
}

std::vector<size_t>
compute_chunk_sizes(const std::vector<std::vector<char>> &data, const size_t batch_size, const size_t chunk_size)
{
   std::vector<size_t> sizes(batch_size, chunk_size);

   size_t offset = 0;
   for (size_t i = 0; i < data.size(); ++i) {
      const size_t num_chunks = (data[i].size() + chunk_size - 1) / chunk_size;
      offset += num_chunks;
      if (data[i].size() % chunk_size != 0) {
         sizes[offset - 1] = data[i].size() % chunk_size;
      }
   }
   return sizes;
}

class BatchData {
public:
   void **chunk_pointers;
   size_t *chunk_sizes;
   uint8_t *data;
   size_t num_chunks;

   BatchData(const std::vector<std::vector<char>> &host_data, const size_t chunk_size, const size_t in_bytes)
      : chunk_pointers(), chunk_sizes(), data(), num_chunks(0)
   {
      // Compute number of chunks
      num_chunks = compute_batch_size(host_data, chunk_size);
      ERRCHECK(cudaMalloc(&data, chunk_size * num_chunks * sizeof(uint8_t)));

      // Compute size of each chunk
      std::vector<size_t> sizes = compute_chunk_sizes(host_data, num_chunks, chunk_size);
      ERRCHECK(cudaMalloc(&chunk_sizes, sizes.size() * sizeof(size_t)));
      ERRCHECK(cudaMemcpy(chunk_sizes, sizes.data(), sizes.size() * sizeof(size_t), cudaMemcpyHostToDevice));

      // Copy data to GPU
      ERRCHECK(cudaMalloc(&data, in_bytes * sizeof(uint8_t)));
      size_t offset = 0;
      for (size_t i = 0; i < host_data.size(); ++i) {
         ERRCHECK(cudaMemcpy(&data[offset], host_data[i].data(), host_data[i].size(),
                             cudaMemcpyHostToDevice));

         const size_t num_chunks = (host_data[i].size() + chunk_size - 1) / chunk_size;
         offset += num_chunks;
      }

      // Set up chunk pointers
      std::vector<void *> ptrs(num_chunks);
      for (size_t i = 0; i < num_chunks; ++i) {
         ptrs[i] = static_cast<void *>(data + chunk_size * i);
      }
      ERRCHECK(cudaMalloc(&chunk_pointers, num_chunks * sizeof(void *)));
      ERRCHECK(cudaMemcpy(chunk_pointers, ptrs.data(), num_chunks * sizeof(void *), cudaMemcpyHostToDevice));
   }

   BatchData(const size_t max_output_size, const size_t batch_size)
      : chunk_pointers(), chunk_sizes(), data(), num_chunks(batch_size)
   {
      // Allocate space for data
      ERRCHECK(cudaMalloc(&data, max_output_size * num_chunks * sizeof(uint8_t)));

      // Compute size of each chunk
      std::vector<size_t> sizes(num_chunks, max_output_size);
      ERRCHECK(cudaMalloc(&chunk_sizes, sizes.size() * sizeof(size_t)));
      ERRCHECK(cudaMemcpy(chunk_sizes, sizes.data(), sizes.size() * sizeof(size_t), cudaMemcpyHostToDevice));

      // Set up chunk pointers
      std::vector<void *> ptrs(batch_size);
      for (size_t i = 0; i < batch_size; ++i) {
         ptrs[i] = data + max_output_size * i;
      }
      ERRCHECK(cudaMalloc(&chunk_pointers, ptrs.size() * sizeof(void *)));
      ERRCHECK(cudaMemcpy(chunk_pointers, ptrs.data(), ptrs.size() * sizeof(void *), cudaMemcpyHostToDevice));
   }

   BatchData(BatchData &&other) = default;

   // disable copying
   BatchData(const BatchData &other) = delete;
   BatchData &operator=(const BatchData &other) = delete;
};
