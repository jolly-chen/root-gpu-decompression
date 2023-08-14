#include <thrust/binary_search.h>
#include <thrust/functional.h>

// Number of elem
#define TILE_SIZE 512

// Number of bytes
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
/// TODO: VERSION 4
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

   unsigned int globalX = threadIdx.x + TILE_SIZE * N * blockIdx.x;
   unsigned int localX = threadIdx.x;

   auto dst = reinterpret_cast<char *>(destination);
   auto src = reinterpret_cast<const char *>(source);
   size_t chunkBeginIdx = 0;

   for (auto chunk = 0; chunk < nChunks; chunk++) {
      size_t nElem = chunkSizes[chunk] / N;

      for (int offset = 0; offset < TILE_SIZE * N; offset += BLOCK_SIZE) {
         int elem = globalX % TILE_SIZE;
         int byte = (globalX + offset) / TILE_SIZE;
         printf("global: %d elem: %d byte: %d\n", globalX, elem, byte);
         tile[localX + offset] = src[chunkBeginIdx + elem * nElem + byte];
      }

      __syncthreads();

      for (int offset = 0; offset < TILE_SIZE * N; offset += BLOCK_SIZE) {
         int elem = (localX + offset) / N;
         int byte = localX % N;
         dst[chunkBeginIdx + globalX + offset] = tile[byte * nElem + elem];
      }

      chunkBeginIdx += chunkSizes[chunk];
   }
}

/**
 * src: array of packed bytes, with the diffrent elements along the columns and
 *      the bytes for each element along the rows. Needs at least TILE_SIZE of
 *      elements to unpack.
 * dst: array of unpacked bytes, with the different elements along the rows and the
 *      bytes for each element along the columns.
 * src array is split into blocks of TILE_SIZE elements -> TILE_SIZE * sizeof(SourceT)
 * bytes. The size of a cuda block is equal to the TILE_SIZE and each thread in a block
 * transposes sizeof(SourceT) bytes. The read/writes to src and dst are in row-major
 * order by transposing the tile within sharec memory to achieve coalesced accesses
 */
template <typename DestT, typename SourceT>
__global__ void
Unpack4_1(void *destination, const void *source, const size_t *chunkSizes, const size_t nChunks, const size_t totalSize)
{
   constexpr size_t N = sizeof(SourceT);
   __shared__ char tile[TILE_SIZE * N];

   unsigned int globalElem = blockIdx.x * TILE_SIZE + threadIdx.x;
   unsigned int localElem = threadIdx.x;

   auto dst = reinterpret_cast<char *>(destination);
   auto src = reinterpret_cast<const char *>(source);
   size_t chunkBeginIdx = 0;

   for (auto chunk = 0; chunk < nChunks; chunk++) {
      size_t nElem = chunkSizes[chunk] / N;

      for (int byte = 0; byte < N; byte++)
         tile[localElem * N + byte] = src[chunkBeginIdx + byte * nElem + globalElem];

      __syncthreads();

      for (int offset = 0; offset < TILE_SIZE * N; offset += TILE_SIZE) {
         dst[chunkBeginIdx + blockIdx.x * TILE_SIZE * N + threadIdx.x + offset] = tile[localElem + offset];
      }

      chunkBeginIdx += chunkSizes[chunk];
   }
}

/**
 * @brief TODO:
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

   auto dst = reinterpret_cast<char *>(destination);
   auto src = reinterpret_cast<const char *>(source);
   size_t chunkBeginIdx = 0;

   for (auto chunk = 0; chunk < nChunks; chunk++) {
      size_t nElem = chunkSizes[chunk] / N;
      unsigned int byte = threadIdx.y;
      unsigned int elem = blockIdx.x * TILE_SIZE + threadIdx.x;

      printf("y: %u block: %u x: %u tid: %lu elem: %d byte: %d\n", threadIdx.y, blockIdx.x, threadIdx.x,
             threadIdx.y * nElem + blockIdx.x * blockDim.x + threadIdx.x, elem, byte);

      for (int offset = 0; offset < TILE_SIZE * N; offset += BLOCK_SIZE)
         tile[offset + threadIdx.y * TILE_SIZE + threadIdx.x] = src[chunkBeginIdx + byte * nElem + elem];

      __syncthreads();

      for (int offset = 0; offset < TILE_SIZE * N; offset += BLOCK_SIZE) {
         dst[chunkBeginIdx + (elem + offset) * N + byte] = tile[byte * nElem + (elem + offset)];
      }

      chunkBeginIdx += chunkSizes[chunk];
   }
}

/**
 * @brief TODO:
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
Unpack6(void *destination, const void *source, const size_t *chunkSizes, const size_t nChunks, const size_t totalSize)
{
   constexpr size_t N = sizeof(SourceT);
   // __shared__ char tile[TILE_SIZE][N];
   __shared__ char tile[TILE_SIZE * N];

   auto dst = reinterpret_cast<char *>(destination);
   auto src = reinterpret_cast<const char *>(source);
   size_t chunkBeginIdx = 0;

   for (auto chunk = 0; chunk < nChunks; chunk++) {
      size_t nElem = chunkSizes[chunk] / N;
      unsigned int elem = blockIdx.x * TILE_SIZE + threadIdx.x;

      printf("y: %u block: %u x: %u tid: %lu elem: %d byte: %d\n", threadIdx.y, blockIdx.x, threadIdx.x,
             threadIdx.y * nElem + blockIdx.x * blockDim.x + threadIdx.x, elem, blockIdx.y);

      for (int byte = 0; byte < N; byte++)
         tile[byte * TILE_SIZE + threadIdx.x] = src[chunkBeginIdx + byte * nElem + elem];

      __syncthreads();

      // unsigned int elem =
      // for (int byte = 0; byte < N; byte++)
      //    dst[chunkBeginIdx + elem * N + byte] = tile[byte * nElem + (elem + offset)];

      chunkBeginIdx += chunkSizes[chunk];
   }
}