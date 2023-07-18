#include <vector>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cmath>

/**
 * (Un)packing on CPU
 */

/// \brief Split encoding of elements, possibly into narrower column
///
/// Used to first cast and then split-encode in-memory values to the on-disk column. Swap bytes if necessary.
template <typename DestT, typename SourceT>
static void CastSplitPack(void *destination, const void *source, std::size_t count)
{
   constexpr std::size_t N = sizeof(DestT);
   auto splitArray = reinterpret_cast<char *>(destination);
   auto src = reinterpret_cast<const SourceT *>(source);
   for (std::size_t i = 0; i < count; ++i) {
      DestT val = src[i];
      for (std::size_t b = 0; b < N; ++b) {
         splitArray[b * count + i] = reinterpret_cast<const char *>(&val)[b];
      }
   }
}

/// \brief Reverse split encoding of elements
///
/// Used to first unsplit a column, possibly storing elements in wider C++ types. Swaps bytes if necessary
template <typename DestT, typename SourceT>
static void CastSplitUnpack(void *destination, const void *source, std::size_t count)
{
   constexpr std::size_t N = sizeof(SourceT);
   auto dst = reinterpret_cast<DestT *>(destination);
   auto splitArray = reinterpret_cast<const char *>(source);
   for (std::size_t i = 0; i < count; ++i) {
      SourceT val = 0;
      for (std::size_t b = 0; b < N; ++b) {
         reinterpret_cast<char *>(&val)[b] = splitArray[b * count + i];
      }
      dst[i] = val;
   }
}

/**
 * File reading
 */

std::vector<char> ReadFile(const std::string &filename)
{
   std::vector<char> buffer(4096);
   std::vector<char> hCompressed;

   std::ifstream fin(filename, std::ifstream::binary);
   fin.exceptions(std::ifstream::failbit | std::ifstream::badbit);

   size_t num;
   do {
      num = fin.readsome(buffer.data(), buffer.size());
      hCompressed.insert(hCompressed.end(), buffer.begin(), buffer.begin() + num);
   } while (num > 0);

   return hCompressed;
}

std::vector<std::vector<char>> GenerateMultiFile(const std::string &filename, int n)
{
   auto file = ReadFile(filename);
   std::vector<std::vector<char>> multiFile(n, file);
   return multiFile;
}

/**
 * Results processing
 */

float GetMean(const std::vector<float> &vec)
{
   return std::accumulate(vec.begin(), vec.end(), 0.0) / vec.size();
}

float GetStdDev(const std::vector<float> &vec)
{
   auto mean = GetMean(vec);
   std::vector<double> diff(vec.size());
   std::transform(vec.begin(), vec.end(), diff.begin(), [mean](double x) { return x - mean; });
   double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
   return std::sqrt(sq_sum / vec.size());
}
