#include <vector>
#include <iostream>
#include <cassert>
#include "utils.h"

#define ERRCHECK(err) __checkCudaErrors((err), __func__, __FILE__, __LINE__)
inline static void __checkCudaErrors(cudaError_t error, std::string func, std::string file, int line)
{
   if (error != cudaSuccess) {
      fprintf(stderr, (func + "(), " + file + ":" + std::to_string(line)).c_str(), "%s\n", cudaGetErrorString(error));
      throw std::bad_exception();
   }
}

/**
 * @brief
 *
 * @tparam DestT
 * @tparam SourceT
 * @param dest output buffer
 * @param src input buffer
 * @param chunkSizes size of each chunk in src
 * @param totalSize total size of src in bytes
 * @return void
 */
template <typename DestT, typename SourceT>
__global__ void
Unpack1(void *destination, const void *source, const size_t *chunkSizes, const size_t nChunks, const size_t totalSize)
{
   unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
   unsigned int stride = blockDim.x * gridDim.x; // total number of threads

   constexpr size_t N = sizeof(SourceT);
   auto dst = reinterpret_cast<DestT *>(destination);
   auto splitArray = reinterpret_cast<const char *>(source);
   size_t offset = 0;

   for (auto c = 0; c < nChunks; c++) {
      size_t nElem = chunkSizes[c] / N;

      for (auto i = tid; i < nElem; i += stride) {
         SourceT val = 0;
         for (size_t b = 0; b < N; ++b) {
            reinterpret_cast<char *>(&val)[b] = splitArray[b * nElem + i + offset];
         }
         dst[i + offset / sizeof(DestT)] = val;
      }

      offset += chunkSizes[c];
   }
}

/**
 * @brief
 *
 * @tparam DestT
 * @tparam SourceT
 * @param dest output buffer
 * @param src input buffer
 * @param chunkSizes size of each chunk in src
 * @param totalSize total size of src in bytes
 * @return void
 */
template <typename DestT, typename SourceT>
__global__ void
Unpack2(void *destination, const void *source, const size_t *chunkSizes, const size_t nChunks, const size_t totalSize)
{
   extern __shared__ size_t chunkOffsets[];

   unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
   unsigned int stride = blockDim.x * gridDim.x; // total number of threads

   constexpr size_t N = sizeof(SourceT);
   auto dst = reinterpret_cast<DestT *>(destination);
   auto splitArray = reinterpret_cast<const char *>(source);
   size_t offset = 0;

   for (auto c = 0; c < nChunks; c++) {
      size_t nElem = chunkSizes[c] / N;

      for (auto i = tid; i < nElem; i += stride) {
         SourceT val = 0;
         for (size_t b = 0; b < N; ++b) {
            reinterpret_cast<char *>(&val)[b] = splitArray[b * nElem + i];
         }
         dst[i] = val;
      }

      offset += chunkSizes[c];
   }
}

int main(int argc, char const *argv[])
{
   std::vector<float> in(64, 123456.);
   std::vector<float> pack(in.size());
   std::vector<float> out(in.size());
   std::vector<float> unpack(in.size());
   std::vector<size_t> sizes{};

   // pack input buffer and unpack to test correctness
   size_t offset = 0;
   for (auto c = 0; c < sizes.size(); c++) {
      CastSplitPack<float, float>(&pack.data()[offset], &in.data()[offset], sizes[c]);
      offset += sizes[c];
   }
   offset = 0;
   for (auto c = 0; c < sizes.size(); c++) {
      CastSplitUnpack<float, float>(&unpack.data()[offset], &pack.data()[offset], sizes[c]);
      offset += sizes[c];
   }
   assert(in == unpack);

   // allocate device buffers
   float *dOut = NULL;
   ERRCHECK(cudaMalloc(&dOut, out.size() * sizeof(float)));

   float *dPack = NULL;
   ERRCHECK(cudaMalloc(&dPack, pack.size() * sizeof(float)));
   ERRCHECK(cudaMemcpy(dPack, pack.data(), pack.size() * sizeof(float), cudaMemcpyHostToDevice));

   size_t *dSizes = NULL;
   std::transform(sizes.begin(), sizes.end(), sizes.begin(), [&](auto &x) { return x * sizeof(float); });
   ERRCHECK(cudaMalloc(&dSizes, sizes.size() * sizeof(size_t)));
   ERRCHECK(cudaMemcpy(dSizes, sizes.data(), sizes.size() * sizeof(size_t), cudaMemcpyHostToDevice));

   // run unpack kernels
   Unpack1<float, float>
      <<<ceil(pack.size() / 256.), 256>>>(dOut, dPack, dSizes, sizes.size(), pack.size() * sizeof(float));
   ERRCHECK(cudaPeekAtLastError());
   ERRCHECK(cudaMemcpy(out.data(), dOut, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
   assert(in == out && "Unpack1 failed");

   Unpack2<float, float>
      <<<ceil(pack.size() / 256.), 256, sizes.size()>>>(dOut, dPack, dSizes, sizes.size(), pack.size() * sizeof(float));
   ERRCHECK(cudaPeekAtLastError());
   ERRCHECK(cudaMemcpy(out.data(), dOut, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
   assert(in == out && "Unpack2 failed");

   return 0;
}
