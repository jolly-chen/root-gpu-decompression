#include <vector>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cmath>

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
