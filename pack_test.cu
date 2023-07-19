#include <vector>
#include <iostream>
#include <cassert>

#include "utils.h"
#include "pack.cuh"


int main(int argc, char const *argv[])
{
   std::vector<float> in(64000*10, 123456.);
   std::vector<float> pack(in.size());
   std::vector<float> out(in.size());
   std::vector<float> unpack(in.size());
   std::vector<size_t> sizes(10, 64000);
   // std::vector<size_t> sizes{256, 3, 1};
   // std::vector<size_t> sizes{1, 1, 1, 1};
   std::vector<size_t> chunkOffsets(sizes.size());

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

   chunkOffsets[0] = 0;
   for (auto c = 1; c < sizes.size(); c++) {
      chunkOffsets[c] += sizes[c - 1] + chunkOffsets[c - 1];
   }
   size_t *dOffsets = NULL;
   ERRCHECK(cudaMalloc(&dOffsets, chunkOffsets.size() * sizeof(size_t)));
   ERRCHECK(cudaMemcpy(dOffsets, chunkOffsets.data(), chunkOffsets.size() * sizeof(size_t), cudaMemcpyHostToDevice));

   // run unpack kernels
   Unpack1<float, float>
      <<<ceil(pack.size() / 256.), 256>>>(dOut, dPack, dSizes, sizes.size(), pack.size() * sizeof(float));
   ERRCHECK(cudaPeekAtLastError());
   ERRCHECK(cudaMemcpy(out.data(), dOut, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
   assert(in == out && "Unpack1 failed");

   // ERRCHECK(cudaMemset(dOut, 0, out.size() * sizeof(float)));
   // Unpack1_1<float, float><<<ceil(pack.size() / 256.), 256, sizes.size()>>>(dOut, dPack, dSizes, dOffsets,
   // sizes.size(),
   //                                                                        pack.size() * sizeof(float));
   // ERRCHECK(cudaPeekAtLastError());
   // ERRCHECK(cudaMemcpy(out.data(), dOut, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
   // assert(in == out && "Unpack2 failed");

   ERRCHECK(cudaMemset(dOut, 0, out.size() * sizeof(float)));
   Unpack2<float, float>
      <<<ceil(pack.size() / 256.), 256>>>(dOut, dPack, dSizes, sizes.size(), pack.size() * sizeof(float));
   ERRCHECK(cudaPeekAtLastError());
   ERRCHECK(cudaMemcpy(out.data(), dOut, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
   assert(in == out && "Unpack2 failed");

   ERRCHECK(cudaMemset(dOut, 0, out.size() * sizeof(float)));
   Unpack2_1<float, float><<<ceil(pack.size() / 256.), 256, sizes.size()>>>(dOut, dPack, dSizes, dOffsets, sizes.size(),
                                                                            pack.size() * sizeof(float));
   ERRCHECK(cudaPeekAtLastError());
   ERRCHECK(cudaMemcpy(out.data(), dOut, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
   assert(in == out && "Unpack2.1 failed");

   ERRCHECK(cudaMemset(dOut, 0, out.size() * sizeof(float)));
   Unpack3<float, float>
      <<<ceil(pack.size() / 256.), 256>>>(dOut, dPack, dSizes, sizes.size(), pack.size() * sizeof(float));
   ERRCHECK(cudaPeekAtLastError());
   ERRCHECK(cudaMemcpy(out.data(), dOut, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
   assert(in == out && "Unpack3 failed");

   return 0;
}
