#pragma once

#include <stdio.h>
#include <string>
#include <error.h>
#include <vector>

#include "RZip.h"
#include <TError.h>

/**
 * Debug helpers
 */

#define ERRCHECK(err) __checkCudaErrors((err), __func__, __FILE__, __LINE__)
inline static void __checkCudaErrors(cudaError_t error, std::string func, std::string file, int line)
{
   if (error != cudaSuccess) {
      fprintf(stderr, (func + "(), " + file + ":" + std::to_string(line)).c_str(), "%s\n", cudaGetErrorString(error));
      throw std::bad_alloc();
   }
}

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
   for (int i = 0; i < 100; i++) {
      printf("%c ", data[i]);
   }
   printf("\n");

   printf("chunk_pointers:\n");
   for (int i = 0; i < num_chunks; i++) {
      printf("\tchunk %d:\n\t\t", i);
      for (int j = 0; j < min(100, (int)chunk_sizes[i]); j++) {
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
      std::vector<size_t> sizes;
      size_t remainder = host_data.size();
      unsigned char *source = const_cast<unsigned char *>(reinterpret_cast<const unsigned char *>(host_data.data()));

      // Loop over the chunks to determine their sizes from the header
      do {
         int szSource;
         int szTarget;
         int retval = R__unzip_header(&szSource, source, &szTarget);
         R__ASSERT(retval == 0);
         R__ASSERT(szSource > 0);
         R__ASSERT(szTarget > szSource);
         R__ASSERT(static_cast<unsigned int>(szSource) <= host_data.size());

         num_chunks++;
         sizes.push_back(szSource);
         source += szSource;
         remainder -= szSource;
      } while (remainder > 0);
      R__ASSERT(remainder == 0);

      // Copy data to GPU
      ERRCHECK(cudaMallocAsync(&data, in_bytes * sizeof(char), stream));
      ERRCHECK(cudaMemcpyAsync(data, host_data.data(), host_data.size(), cudaMemcpyHostToDevice, stream));

      // Copy size of each chunk
      ERRCHECK(cudaMallocAsync(&chunk_sizes, sizes.size() * sizeof(size_t), stream));
      ERRCHECK(
         cudaMemcpyAsync(chunk_sizes, sizes.data(), sizes.size() * sizeof(size_t), cudaMemcpyHostToDevice, stream));

      // Copy chunk pointers
      auto ptrs = get_compressed_chunk_ptrs(data, sizes.data(), num_chunks);
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
      auto ptrs = get_decompress_chunk_ptrs(data, h_chunk_sizes, num_chunks);
      ERRCHECK(cudaMallocAsync(&chunk_pointers, ptrs.size() * sizeof(void *), stream));
      ERRCHECK(
         cudaMemcpyAsync(chunk_pointers, ptrs.data(), ptrs.size() * sizeof(void *), cudaMemcpyHostToDevice, stream));
   }

   BatchData(BatchData &&other) = default;

   // disable copying
   BatchData(const BatchData &other) = delete;
   BatchData &operator=(const BatchData &other) = delete;

protected:
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

   inline std::vector<void *> get_compressed_chunk_ptrs(char *data, const size_t *chunk_sizes, const size_t num_chunks)
   {
      std::vector<void *> ptrs(num_chunks);
      size_t offset = 9;

      ptrs[0] = static_cast<void *>(data + offset);
      for (size_t i = 1; i < num_chunks; ++i) {
         offset += chunk_sizes[i - 1];
         ptrs[i] = static_cast<void *>(data + offset); // The ROOT header block is 9 bytes
      }
      return ptrs;
   }

   inline std::vector<void *> get_decompress_chunk_ptrs(char *data, const size_t *chunk_sizes, const size_t num_chunks)
   {
      std::vector<void *> ptrs(num_chunks);
      ptrs[0] = static_cast<void *>(data);
      size_t offset = 0;
      for (size_t i = 1; i < num_chunks; ++i) {
         offset += chunk_sizes[i - 1];
         ptrs[i] = static_cast<void *>(data + offset);
      }
      return ptrs;
   }
};
