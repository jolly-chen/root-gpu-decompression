/**
 * Unpacking on GPU
 */

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
Unpack(void *destination, const void *source, const size_t *chunkSizes, const size_t nChunks, const size_t totalSize)
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
