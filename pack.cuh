#include <thrust/binary_search.h>
#include <thrust/functional.h>

// Number of bytes
#define TILE_SIZE 8
#define BLOCK_SIZE 32

/**
 * Unpacking on GPU
 */

#define ERRCHECK(err) __checkCudaErrors((err), __func__, __FILE__, __LINE__)
inline static void __checkCudaErrors(cudaError_t error, std::string func, std::string file, int line)
{
   if (error != cudaSuccess) {
      fprintf(stderr, (func + "(), " + file + ":" + std::to_string(line)).c_str(), "%s\n", cudaGetErrorString(error));
      throw std::bad_exception();
   }
}

///
/// VERSION 1
/// Every thread unpacks the a designated float in each chunk
///

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

///
/// VERSION 1.1
/// Also parallelize the chunks
///

template <typename T>
__device__ long long BinarySearch(long long n, const T *array, T value)
{
   const T *pind;

   pind = thrust::lower_bound(thrust::seq, array, array + n, value);

   if ((pind != array + n) && (*pind == value))
      return (pind - array);
   else
      return (pind - array - 1);
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
__global__ void Unpack1_1(void *destination, const void *source, const size_t *chunkSizes, const size_t *chunkOffsets,
                          const size_t nChunks, const size_t totalSize)
{
   size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
   size_t stride = blockDim.x * gridDim.x; // total number of threads

   constexpr size_t N = sizeof(SourceT);
   auto dst = reinterpret_cast<DestT *>(destination);
   auto splitArray = reinterpret_cast<const char *>(source);
   size_t nElem = totalSize / N;

   for (auto i = tid; i < nElem; i += stride) {
      auto chunk = BinarySearch(nChunks, chunkOffsets, i);
      auto offset = chunkOffsets[chunk];
      size_t nElem = chunkSizes[chunk] / N;

      printf("tid: %li chunk: %li offset: %li nElem: %li\n", i, chunk, offset, nElem);
      SourceT val = 0;
      for (size_t b = 0; b < N; ++b) {
         reinterpret_cast<char *>(&val)[b] = splitArray[b * nElem];
      }
      dst[i + offset] = val;
   }
}

///
/// VERSION 2
/// Every thread assigns the current source byte to the correct destination byte per chunk
///

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
   unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
   unsigned int stride = blockDim.x * gridDim.x; // total number of threads

   constexpr size_t N = sizeof(SourceT);
   auto dst = reinterpret_cast<char *>(destination);
   auto src = reinterpret_cast<const char *>(source);
   size_t chunkBeginIdx = 0;

   for (auto c = 0; c < nChunks; c++) {
      size_t nElem = chunkSizes[c] / N;

      for (auto b = tid; b < chunkSizes[c]; b += stride) {
         size_t nFloat = b % nElem;
         size_t nFloatByte = b / nElem;
         dst[chunkBeginIdx + nFloat * N + nFloatByte] = src[chunkBeginIdx + b];
      }

      chunkBeginIdx += chunkSizes[c];
   }
}

///
/// VERSION 2_1
/// Every thread assigns the current source byte to the correct destination byte + chunks in parallel
///

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
__global__ void Unpack2_1(void *destination, const void *source, const size_t *chunkSizes, const size_t *chunkOffsets,
                          const size_t nChunks, const size_t totalSize)
{
   size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
   size_t stride = blockDim.x * gridDim.x; // total number of threads

   constexpr size_t N = sizeof(SourceT);
   auto dst = reinterpret_cast<char *>(destination);
   auto src = reinterpret_cast<const char *>(source);

   for (auto b = tid; b < totalSize; b += stride) {
      auto chunk = BinarySearch(nChunks, chunkOffsets, b);
      auto chunkBeginIdx = chunkOffsets[chunk];
      size_t nElem = chunkSizes[chunk] / N;

      size_t chunkByte = b - chunkBeginIdx;
      size_t nFloat = chunkByte % nElem;
      size_t nFloatByte = chunkByte / nElem;
      // printf("tid: %li totalsize: %li chunk: %li begin: %li nElem: %li chunkByte: %li nFloat: %li nFloatByte %li\n",
      // b,
      //        totalSize, chunk, chunkBeginIdx, nElem, chunkByte, nFloat, nFloatByte);
      dst[chunkBeginIdx + nFloat * N + nFloatByte] = src[chunkBeginIdx + chunkByte];
   }
}

///
/// VERSION 3
/// Every thread retrieves the current dest byte from the correct source byte per chunk
///

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
Unpack3(void *destination, const void *source, const size_t *chunkSizes, const size_t nChunks, const size_t totalSize)
{
   unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
   unsigned int stride = blockDim.x * gridDim.x; // total number of threads

   constexpr size_t N = sizeof(SourceT);
   auto dst = reinterpret_cast<char *>(destination);
   auto src = reinterpret_cast<const char *>(source);
   size_t chunkBeginIdx = 0;

   for (auto c = 0; c < nChunks; c++) {
      size_t nElem = chunkSizes[c] / N;

      for (auto b = tid; b < chunkSizes[c]; b += stride) {
         size_t nFloat = b % N;
         size_t nFloatByte = b / N;
         dst[chunkBeginIdx + b] = src[chunkBeginIdx + nFloat * nElem + nFloatByte];
      }

      chunkBeginIdx += chunkSizes[c];
   }
}

///
/// VERSION 4
/// Every thread retrieves the current dest byte from the correct source byte per chunk, per TILE
///

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
Unpack4(void *destination, const void *source, const size_t *chunkSizes, const size_t nChunks, const size_t totalSize)
{
   constexpr size_t N = sizeof(SourceT);
   __shared__ char tile[TILE_SIZE * N];

   unsigned int x = threadIdx.x + TILE_SIZE * N * blockIdx.x;

   auto dst = reinterpret_cast<char *>(destination);
   auto src = reinterpret_cast<const char *>(source);
   size_t chunkBeginIdx = 0;

   for (auto chunk = 0; chunk < nChunks; chunk++) {
      size_t nElem = chunkSizes[chunk] / N;

      for (int offset = 0; offset < TILE_SIZE * N; offset += BLOCK_SIZE) {
         tile[x + offset] = src[chunkBeginIdx + x + offset];
      }

      __syncthreads();

      for (int offset = 0; offset < TILE_SIZE * N; offset += BLOCK_SIZE) {
         int elem = (x + offset) / N;
         int byte = x % N;
         // dst[chunkBeginIdx + x + offset] = src[chunkBeginIdx + nFloat * nElem + nFloatByte];
         dst[chunkBeginIdx + x + offset] = tile[byte * nElem + elem];
      }

      chunkBeginIdx += chunkSizes[chunk];
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
Unpack5(void *destination, const void *source, const size_t *chunkSizes, const size_t nChunks, const size_t totalSize)
{
   constexpr size_t N = sizeof(SourceT);
   // __shared__ char tile[TILE_SIZE][N];
   __shared__ char tile[TILE_SIZE * N];

   unsigned int byte = threadIdx.x;
   unsigned int elem = blockIdx.y * N + threadIdx.y;

   auto dst = reinterpret_cast<char *>(destination);
   auto src = reinterpret_cast<const char *>(source);
   size_t chunkBeginIdx = 0;

   for (auto chunk = 0; chunk < nChunks; chunk++) {
      for (int offset = 0; offset < TILE_SIZE * N; offset += BLOCK_SIZE)
         tile[(threadIdx.y + offset) * N + byte] = src[chunkBeginIdx + (elem + offset) * N + byte];

      __syncthreads();

      size_t nElem = chunkSizes[chunk] / N;
      for (int offset = 0; offset < TILE_SIZE * N; offset += BLOCK_SIZE) {
         dst[chunkBeginIdx + (elem + offset) * N + byte] = tile[byte * nElem + (elem + offset)];
      }

      chunkBeginIdx += chunkSizes[chunk];
   }
}