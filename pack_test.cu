#include <vector>
#include <iostream>
#include <cassert>
#include <random>
#include <algorithm>
#include <iterator>

#include "utils.h"
#include "pack.cuh"

std::vector<float> get_random_floats(size_t size)
{
   std::default_random_engine generator(123);
   std::uniform_int_distribution<int> dist{-1234567, 1234567};
   auto gen = [&]() { return dist(generator); };

   std::vector<float> vec(size);
   std::generate(std::begin(vec), std::end(vec), gen);
   return vec;
}

int main(int argc, char const *argv[])
{
   int nFloats = 16;
   int nChunks = 100;
   auto in = get_random_floats(nFloats * nChunks);
   std::vector<float> pack(in.size());
   std::vector<float> out(in.size());
   std::vector<float> unpack(in.size());
   std::vector<size_t> sizes(nChunks, nFloats);
   // std::vector<size_t> sizes{256, 3, 1};
   // std::vector<size_t> sizes{1, 1, 1, 1};
   std::vector<size_t> chunkOffsets(nChunks);

   // pack input buffer and unpack to test correctness
   size_t offset = 0;
   for (auto c = 0; c < nChunks; c++) {
      CastSplitPack<float, float>(&pack.data()[offset], &in.data()[offset], sizes[c]);
      offset += sizes[c];
   }

   offset = 0;
   for (auto c = 0; c < nChunks; c++) {
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
   ERRCHECK(cudaMalloc(&dSizes, nChunks * sizeof(size_t)));
   ERRCHECK(cudaMemcpy(dSizes, sizes.data(), nChunks * sizeof(size_t), cudaMemcpyHostToDevice));

   chunkOffsets[0] = 0;
   for (auto c = 1; c < nChunks; c++) {
      chunkOffsets[c] += sizes[c - 1] + chunkOffsets[c - 1];
   }
   size_t *dOffsets = NULL;
   ERRCHECK(cudaMalloc(&dOffsets, chunkOffsets.size() * sizeof(size_t)));
   ERRCHECK(cudaMemcpy(dOffsets, chunkOffsets.data(), chunkOffsets.size() * sizeof(size_t), cudaMemcpyHostToDevice));

   // run unpack kernels
   Unpack1<float, float>
      <<<ceil(pack.size() / 256.), 256>>>(dOut, dPack, dSizes, nChunks, pack.size() * sizeof(float));
   ERRCHECK(cudaPeekAtLastError());
   ERRCHECK(cudaMemcpy(out.data(), dOut, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
   assert(in == out && "Unpack1 failed");

   ERRCHECK(cudaMemset(dOut, 0, out.size() * sizeof(float)));
   Unpack1_1<float, float><<<ceil(pack.size() / 256.), 256, nChunks>>>(dOut, dPack, dSizes, dOffsets,
   nChunks,
                                                                          pack.size() * sizeof(float));
   ERRCHECK(cudaPeekAtLastError());
   ERRCHECK(cudaMemcpy(out.data(), dOut, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
   assert(in == out && "Unpack2 failed");

   ERRCHECK(cudaMemset(dOut, 0, out.size() * sizeof(float)));
   Unpack2<float, float>
      <<<ceil(pack.size() / 256.), 256>>>(dOut, dPack, dSizes, nChunks, pack.size() * sizeof(float));
   ERRCHECK(cudaPeekAtLastError());
   ERRCHECK(cudaMemcpy(out.data(), dOut, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
   assert(in == out && "Unpack2 failed");

   // ERRCHECK(cudaMemset(dOut, 0, out.size() * sizeof(float)));
   // Unpack2_1<float, float><<<ceil(pack.size() / 256.), 256, nChunks>>>(dOut, dPack, dSizes, dOffsets, nChunks,
   //                                                                          pack.size() * sizeof(float));
   // ERRCHECK(cudaPeekAtLastError());
   // ERRCHECK(cudaMemcpy(out.data(), dOut, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
   // assert(in == out && "Unpack2.1 failed");

   ERRCHECK(cudaMemset(dOut, 0, out.size() * sizeof(float)));
   Unpack3<float, float>
      <<<ceil(pack.size() / 256.), 256>>>(dOut, dPack, dSizes, nChunks, pack.size() * sizeof(float));
   ERRCHECK(cudaPeekAtLastError());
   ERRCHECK(cudaMemcpy(out.data(), dOut, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
   assert(in == out && "Unpack3 failed");

   // ERRCHECK(cudaMemset(dOut, 0, out.size() * sizeof(float)));
   // Unpack4<float, float><<<ceil(nFloats / TILE_SIZE), BLOCK_SIZE>>>(
   //    dOut, dPack, dSizes, nChunks, pack.size() * sizeof(float));
   // ERRCHECK(cudaPeekAtLastError());
   // ERRCHECK(cudaMemcpy(out.data(), dOut, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
   // assert(in == out && "Unpack4 failed");

   ERRCHECK(cudaMemset(dOut, 0, out.size() * sizeof(float)));
   Unpack4_1<float, float>
      <<<ceil(float(nFloats) / TILE_SIZE), TILE_SIZE>>>(dOut, dPack, dSizes, nChunks, pack.size() * sizeof(float));
   ERRCHECK(cudaPeekAtLastError());
   ERRCHECK(cudaMemcpy(out.data(), dOut, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
   assert(in == out && "Unpack4_1 failed");

   // dim3 dimGrid(nFloats / TILE_SIZE, 1, 1);
   // dim3 dimBlock(BLOCK_SIZE / sizeof(float), sizeof(float), 1);
   // ERRCHECK(cudaMemset(dOut, 0, out.size() * sizeof(float)));
   // Unpack5<float, float>
   //    <<<dimGrid, dimBlock>>>(dOut, dPack, dSizes, nChunks, pack.size() * sizeof(float));
   // ERRCHECK(cudaPeekAtLastError());
   // ERRCHECK(cudaMemcpy(out.data(), dOut, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
   // assert(in == out && "Unpack5 failed");

   printf("Test finished\n");

   return 0;
}
