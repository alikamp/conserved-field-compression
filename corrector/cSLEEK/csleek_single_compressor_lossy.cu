// cSLEEK modification of SLEEK
// nvcc -O3 -arch=sm_86 csleek_single_compressor_lossy.cu -o cSLEEKcompr -DSTATS

/*
This file is part of SLEEK, a set of ultra-fast lossless and guaranteed-error-bounded lossy main-memory compression algorithms for floating-point data on GPUs.

BSD 3-Clause License

Copyright (c) 2026, Anju Mongandampulath Akathoott, Andrew Rodriguez, and Martin Burtscher

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

URL: The latest version of this code is available at https://github.com/burtscher/SLEEK/.

Publication: This work is described in detail in the following paper.
Anju Mongandampulath Akathoott, Andrew Rodriguez, and Martin Burtscher. "SLEEK: Compressing Memory Copies for Floating-Point Data on GPUs." Proceedings of the 40th IEEE International Parallel and Distributed Processing Symposium (IPDPS'26). May 2026.

Sponsor: This material is based upon work supported by the U.S. National Science Foundation under Grant Number 2403380 and by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Research (ASCR), under Award Number DE-SC0022223.
*/


#include <string>
#include <cmath>
#include <cassert>
#include <cuda.h>
#include "sleek_lossy_gpu.h"
#include "../conserve_cuda.h"


static const float EBfactor = 0.9f;


struct GPUTimer
{
  cudaEvent_t beg, end;
  GPUTimer() {cudaEventCreate(&beg); cudaEventCreate(&end);}
  ~GPUTimer() {cudaEventDestroy(beg); cudaEventDestroy(end);}
  void start() {cudaEventRecord(beg, 0);}
  double stop() {cudaEventRecord(end, 0); cudaEventSynchronize(end); float ms; cudaEventElapsedTime(&ms, beg, end); return 0.001 * ms;}
};

/*
#if defined(STATS)
  static inline float computeNOAeb(const float* const input, const long long size, const float eb_param)
  {
    float min = input[0], max = input[0];
    #pragma omp parallel for default(none) shared(size, input) reduction(max: max) reduction(min: min)
    for (long long i = 1; i < size; i++) {
      const float val = input[i];
      min = std::min(val, min);
      max = std::max(val, max);
    }

    printf("min_val: %.10f\n", min);
    printf("max_val: %.10f\n", max);
    printf("diff between max and min values: %.10f\n", (max - min));
    return (max - min) * eb_param;
  }
#endif
*/

static inline void get_conservation_params(const int mode, const int blocks, const float* const __restrict__ d_input, const long long N, double& p1, double& p2)
{
  if (mode != 0) {
    const int extra_bytes = __builtin_popcount(mode) * sizeof(double);
    double* d_res;
    cudaMalloc(&d_res, extra_bytes);
    cudaMemsetAsync(d_res, 0, extra_bytes);
    if (mode == CONS_MASS) {
      c_sum<<<blocks, SLEEK_TPB>>>(d_input, N, d_res);  // compute L0
      cudaMemcpy(&p1, d_res, sizeof(double), cudaMemcpyDeviceToHost);
    } else if (mode == CONS_ENERGY) {
      c_sum2<<<blocks, SLEEK_TPB>>>(d_input, N, d_res);  // compute S0
      cudaMemcpy(&p2, d_res, sizeof(double), cudaMemcpyDeviceToHost);
    } else {  // mode == (CONS_MASS | CONS_ENERGY)
      c_sum_sum2<<<blocks, SLEEK_TPB>>>(d_input, N, d_res);  // compute M0 and S0
      cudaMemcpy(&p1, &d_res[0], sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(&p2, &d_res[1], sizeof(double), cudaMemcpyDeviceToHost);
    }
    cudaFree(d_res);
  }
}


int main(int argc, char* argv [])
{
  printf("GPU cSLEEK 1.0: single-precision lossy property-conserving compressor\n");
  printf("Copyright 2026 Texas State University\n\n");

  if (argc != 5) {printf("USAGE: %s input_file_name compressed_file_name error_bound mode\n\n", argv[0]); return -1;}

  // system check
  if (sleek_system_checks() != 0) {return -1;}

  const int mode = std::stoi(argv[4]);
  const int all = CONS_MASS | CONS_ENERGY;
  if ((mode < 0) || (mode > all)) {fprintf(stderr, "ERROR: mode must be between 0 and %d\n", all); return -1;}
  // 0: conserve nothing
  // 1: conserve mass
  // 2: conserve energy
  // 3: conserve both

  // read input file
  FILE* const fin = fopen(argv[1], "rb");
  fseek(fin, 0, SEEK_END);
  const long long fsize = ftell(fin);
  if (fsize <= 0) {fprintf(stderr, "ERROR: input file too small\n\n"); return -1;}
  const long long num_elements = fsize / sizeof(float);
  float* const input = new float [num_elements];
  fseek(fin, 0, SEEK_SET);
  // all "*size" variables in bytes
  const long long insize = fread(input, 1, fsize, fin);  assert(insize == fsize);
  fclose(fin);
  printf("original size: %lld bytes\n", insize);

  if (insize % sizeof(float) != 0) {fprintf(stderr, "ERROR: size of input must be a multiple of %ld bytes\n", sizeof(float)); return -1;}

  // allocate GPU memory
  const long long maxsize = sleek_upper_bound_size(insize);
  byte* dencoded;
  cudaMallocHost((void **)&dencoded, maxsize);
  float* d_input;
  cudaMalloc((void **)&d_input, insize);
  cudaMemcpy(d_input, input, insize, cudaMemcpyHostToDevice);
  byte* d_encoded;
  cudaMalloc((void **)&d_encoded, maxsize);
  long long* d_encsize;
  cudaMalloc((void **)&d_encsize, sizeof(long long));
  CheckCuda(__LINE__);

  const int blocks = sleek_thread_blocks();
  const long long N = insize / sizeof(float);

  const int extra_bytes = __builtin_popcount(mode) * sizeof(double);

  #if defined(STATS)

    // eb variables
//    const float parameter = std::stof(argv[3]);
//    const float eb = computeNOAeb(input, num_elements, parameter);
    const float eb = std::stof(argv[3]);
    const auto eb_info = sleek_error_bound(eb * EBfactor);

    // warm up
    double p1, p2;  // conservation parameters
    get_conservation_params(mode, blocks, d_input, N, p1, p1);
    sleek_compress_gpu(blocks, eb_info, d_input, num_elements, d_encoded, d_encsize);
    cudaDeviceSynchronize();
    CheckCuda(__LINE__);

    GPUTimer dtimer;
    dtimer.start();
    get_conservation_params(mode, blocks, d_input, N, p1, p2);
    const double constime = dtimer.stop();
    dtimer.start();
    sleek_compress_gpu(blocks, eb_info, d_input, num_elements, d_encoded, d_encsize);
    const double runtime = dtimer.stop();

    long long dencsize = 0;
    cudaMemcpy(&dencsize, d_encsize, sizeof(long long), cudaMemcpyDeviceToHost);

    // get encoded GPU result
    cudaMemcpy(dencoded, d_encoded, dencsize, cudaMemcpyDeviceToHost);
    printf("encoded size: %lld bytes\n", dencsize + extra_bytes);
    CheckCuda(__LINE__);

    const float CR = (100.0 * (dencsize + extra_bytes)) / insize;
    printf("ratio: %6.2f%% %7.3fx\n", CR, 100.0 / CR);

    if (mode != 0) {
      printf("cpreproc time: %.6f s\n", constime);
      const double throughput = insize * 0.000000001 / constime;
      printf("cpreproc throughput: %8.3f Gbytes/s\n", throughput);
    }

    printf("encoding time: %.6f s\n", runtime);
    const double throughput = insize * 0.000000001 / runtime;
    printf("encoding throughput: %8.3f Gbytes/s\n", throughput);

    if (mode != 0) {
      printf("over-all time: %.6f s\n", constime + runtime);
      const double throughput = insize * 0.000000001 / (constime + runtime);
      printf("over-all throughput: %8.3f Gbytes/s\n", throughput);
    }
    CheckCuda(__LINE__);

  #else

    // eb variables
    const float eb = std::stof(argv[3]);
    const auto eb_info = sleek_error_bound(eb * EBfactor);

    double p1, p2;  // conservation parameters
    get_conservation_params(mode, blocks, d_input, N, p1, p2);
    sleek_compress_gpu(blocks, eb_info, d_input, num_elements, d_encoded, d_encsize);
    long long dencsize = 0;
    cudaMemcpy(&dencsize, d_encsize, sizeof(long long), cudaMemcpyDeviceToHost);

    // get encoded GPU result
    cudaMemcpy(dencoded, d_encoded, dencsize, cudaMemcpyDeviceToHost);
    printf("encoded size: %lld bytes\n", dencsize + extra_bytes);
    CheckCuda(__LINE__);

    const float CR = (100.0 * (dencsize + extra_bytes)) / insize;
    printf("ratio: %6.2f%% %7.3fx\n", CR, 100.0 / CR);

  #endif

  // write to file
  FILE* const fout = fopen(argv[2], "wb");
  fwrite(dencoded, 1, dencsize, fout);
  if (mode & CONS_MASS) fwrite(&p1, 1, sizeof(double), fout);
  if (mode & CONS_ENERGY) fwrite(&p2, 1, sizeof(double), fout);
  fclose(fout);

  // clean up GPU memory
  cudaFree(d_input);
  cudaFree(d_encoded);
  cudaFree(d_encsize);
  CheckCuda(__LINE__);

  // clean up
  delete [] input;
  cudaFreeHost(dencoded);
  return 0;
}
