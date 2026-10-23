// cPFPL modification of PFPL
// nvcc -O3 -arch=sm_86 cPFPL_single_compressor.cu -o cPFPLcompr -DSTATS

/*
This file is part of the LC framework for synthesizing high-speed parallel lossless and error-bounded lossy data compression and decompression algorithms for CPUs and GPUs.

BSD 3-Clause License

Copyright (c) 2021-2025, Noushin Azami, Alex Fallin, Brandon Burtchell, Andrew Rodriguez, Benila Jerald, Yiqian Liu, and Martin Burtscher
All rights reserved.

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

URL: The latest version of this code is available at https://github.com/burtscher/LC-framework.

Sponsor: This code is based upon work supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Research (ASCR), under contract DE-SC0022223.
*/


#define NDEBUG

using byte = unsigned char;
static const int CS = 1024 * 16;  // chunk size (in bytes) [must be multiple of 8]
//static const int TPB = 512;  // threads per block [must be power of 2 and at least 128]
#define WS 32


#include <string>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <cuda.h>
#include <sys/time.h>
#include "../conserve_cuda.h"
#include "include/macros.h"
#include "include/sum_reduction.h"
#include "include/max_scan.h"
#include "include/prefix_sum.h"
#include "components/d_DIFFNB_4.h"
#include "components/d_BIT_4.h"
#include "components/d_RZE_1.h"


static const float EBfactor = 0.9f;


const int NUM_RUNS = 2;  // one warmup run


// copy (len) bytes from shared memory (source) to global memory (destination)
// source must we word aligned
static inline __device__ void s2g(void* const __restrict__ destination, const void* const __restrict__ source, const int len)
{
  const int tid = threadIdx.x;
  const byte* const __restrict__ input = (byte*)source;
  byte* const __restrict__ output = (byte*)destination;
  if (len < 128) {
    if (tid < len) output[tid] = input[tid];
  } else {
    const int nonaligned = (int)(size_t)output;
    const int wordaligned = (nonaligned + 3) & ~3;
    const int linealigned = (nonaligned + 127) & ~127;
    const int bcnt = wordaligned - nonaligned;
    const int wcnt = (linealigned - wordaligned) / 4;
    const int* const __restrict__ in_w = (int*)input;
    if (bcnt == 0) {
      int* const __restrict__ out_w = (int*)output;
      if (tid < wcnt) out_w[tid] = in_w[tid];
      for (int i = tid + wcnt; i < len / 4; i += TPB) {
        out_w[i] = in_w[i];
      }
      if (tid < (len & 3)) {
        const int i = len - 1 - tid;
        output[i] = input[i];
      }
    } else {
      const int shift = bcnt * 8;
      const int rlen = len - bcnt;
      int* const __restrict__ out_w = (int*)&output[bcnt];
      if (tid < bcnt) output[tid] = input[tid];
      if (tid < wcnt) out_w[tid] = __funnelshift_r(in_w[tid], in_w[tid + 1], shift);
      for (int i = tid + wcnt; i < rlen / 4; i += TPB) {
        out_w[i] = __funnelshift_r(in_w[i], in_w[i + 1], shift);
      }
      if (tid < (rlen & 3)) {
        const int i = len - 1 - tid;
        output[i] = input[i];
      }
    }
  }
}


static __device__ int g_chunk_counter;


static __global__ void d_reset()
{
  g_chunk_counter = 0;
}


static inline __device__ void propagate_carry(const int value, const int chunkID, volatile int* const __restrict__ fullcarry, int* const __restrict__ s_fullc)
{
  if (threadIdx.x == TPB - 1) {  // last thread
    fullcarry[chunkID] = (chunkID == 0) ? value : -value;
  }

  if (chunkID != 0) {
    if (threadIdx.x + WS >= TPB) {  // last warp
      const int lane = threadIdx.x % WS;
      const int cidm1ml = chunkID - 1 - lane;
      int val = -1;
      __syncwarp();  // not optional
      do {
        if (cidm1ml >= 0) {
          val = fullcarry[cidm1ml];
        }
      } while ((__any_sync(~0, val == 0)) || (__all_sync(~0, val <= 0)));
#if defined(WS) && (WS == 64)
      const long long mask = __ballot_sync(~0, val > 0);
      const int pos = __ffsll(mask) - 1;
#else
      const int mask = __ballot_sync(~0, val > 0);
      const int pos = __ffs(mask) - 1;
#endif
      int partc = (lane < pos) ? -val : 0;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
      partc = __reduce_add_sync(~0, partc);
#else
      partc += __shfl_xor_sync(~0, partc, 1);
      partc += __shfl_xor_sync(~0, partc, 2);
      partc += __shfl_xor_sync(~0, partc, 4);
      partc += __shfl_xor_sync(~0, partc, 8);
      partc += __shfl_xor_sync(~0, partc, 16);
#endif
      if (lane == pos) {
        const int fullc = partc + val;
        fullcarry[chunkID] = fullc + value;
        *s_fullc = fullc;
      }
    }
  }
}


static __device__ inline void d_QABS_4(int& csize, byte in [CS], byte out [CS], const float errorbound, const float threshold)
{
  using ftype = float;
  using itype = int;
  const int size = csize / sizeof(ftype);
  const int tid = threadIdx.x;

  const int mantissabits = 23;
  const itype maxbin = 1 << (mantissabits - 1);  // leave 1 bit for sign

  const ftype eb2 = 2 * errorbound;
  const ftype inv_eb2 = 0.5f / errorbound;

  const ftype* const data_f = (ftype*)in;
  itype* const data_i = (itype*)out;

  for (int i = tid; i < size; i += TPB) {
    const ftype orig_f = data_f[i];
    const ftype scaled = orig_f * inv_eb2;
    const itype bin = (itype)roundf(scaled);
    const ftype recon = bin * eb2;

    itype val;
    if ((bin >= maxbin) || (bin <= -maxbin) || (fabsf(orig_f) >= threshold) || (fabsf(orig_f - recon) > errorbound) || (orig_f != orig_f)) {  // last check is to handle NaNs
      val = *((itype*)&orig_f);
    } else {
      val = (bin << 1) ^ (bin >> 31);  // TCMS encoding, 'sign' and 'exponent' fields are zero
    }
    data_i[i] = val;
  }
}


#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800)
static __global__ __launch_bounds__(TPB, 3)
#else
static __global__ __launch_bounds__(TPB, 2)
#endif
void d_encode(const byte* const __restrict__ input, const int insize, byte* const __restrict__ output, int* const __restrict__ outsize, int* const __restrict__ fullcarry, const float errorbound, const float threshold)
{
  // allocate shared memory buffer
  __shared__ long long chunk [3 * (CS / sizeof(long long))];

  // split into 3 shared memory buffers
  byte* in = (byte*)&chunk[0 * (CS / sizeof(long long))];
  byte* out = (byte*)&chunk[1 * (CS / sizeof(long long))];
  byte* const temp = (byte*)&chunk[2 * (CS / sizeof(long long))];

  // initialize
  const int tid = threadIdx.x;
  const int last = 3 * (CS / sizeof(long long)) - 2 - WS;
  const int chunks = (insize + CS - 1) / CS;  // round up
  long long* const head_out = (long long*)output;
  unsigned short* const size_out = (unsigned short*)&head_out[2];
  byte* const data_out = (byte*)&size_out[chunks];

  // loop over chunks
  do {
    // assign work dynamically
    if (tid == 0) chunk[last] = atomicAdd(&g_chunk_counter, 1);
    __syncthreads();  // chunk[last] produced, chunk consumed

    // terminate if done
    const int chunkID = chunk[last];
    const int base = chunkID * CS;
    if (base >= insize) break;

    // load chunk
    const int osize = min(CS, insize - base);
    long long* const input_l = (long long*)&input[base];
    long long* const out_l = (long long*)out;
    for (int i = tid; i < osize / 8; i += TPB) {
      out_l[i] = input_l[i];
    }
    const int extra = osize % 8;
    if (tid < extra) out[osize - extra + tid] = input[base + osize - extra + tid];

    // encode chunk
    __syncthreads();  // chunk produced, chunk[last] consumed
    int csize = osize;
    bool good = true;
    if (good) {
      byte* tmp = in; in = out; out = tmp;
      d_QABS_4(csize, in, out, errorbound, threshold);
      __syncthreads();
    }
    if (good) {
      byte* tmp = in; in = out; out = tmp;
      d_DIFFNB_4(csize, in, out, temp);
      __syncthreads();
    }
    if (good) {
      byte* tmp = in; in = out; out = tmp;
      d_BIT_4(csize, in, out, temp);
      __syncthreads();
    }
    if (good) {
      byte* tmp = in; in = out; out = tmp;
      good = d_RZE_1(csize, in, out, temp);
      __syncthreads();
    }

    // handle carry
    if (!good || (csize >= osize)) csize = osize;
    propagate_carry(csize, chunkID, fullcarry, (int*)temp);

    // reload chunk if incompressible
    if (tid == 0) size_out[chunkID] = csize;
    if (csize == osize) {
      // store original data
      long long* const out_l = (long long*)out;
      for (int i = tid; i < osize / 8; i += TPB) {
        out_l[i] = input_l[i];
      }
      const int extra = osize % 8;
      if (tid < extra) out[osize - extra + tid] = input[base + osize - extra + tid];
    }
    __syncthreads();  // "out" done, temp produced

    // store chunk
    const int offs = (chunkID == 0) ? 0 : *((int*)temp);
    s2g(&data_out[offs], out, csize);

    // finalize if last chunk
    if ((tid == 0) && (base + CS >= insize)) {
      // output header
      head_out[0] = (long long)insize;
      float* const head_out_f = (float*)&head_out[1];
      head_out_f[0] = errorbound;
      // compute compressed size
      *outsize = &data_out[fullcarry[chunkID]] - output;
    }
  } while (true);
}


struct GPUTimer
{
  cudaEvent_t beg, end;
  GPUTimer() {cudaEventCreate(&beg); cudaEventCreate(&end);}
  ~GPUTimer() {cudaEventDestroy(beg); cudaEventDestroy(end);}
  void start() {cudaEventRecord(beg, 0);}
  double stop() {cudaEventRecord(end, 0); cudaEventSynchronize(end); float ms; cudaEventElapsedTime(&ms, beg, end); return 0.001 * ms;}
};


static void CheckCuda(const int line)
{
  cudaError_t e;
  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "CUDA error %d on line %d: %s\n\n", e, line, cudaGetErrorString(e));
    exit(-1);
  }
}


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


static inline void get_conservation_params(const int mode, const int blocks, const float* const __restrict__ d_input, const long long N, double& p1, double& p2)
{
  if (mode != 0) {
    const int extra_bytes = __builtin_popcount(mode) * sizeof(double);
    double* d_res;
    cudaMalloc(&d_res, extra_bytes);
    cudaMemsetAsync(d_res, 0, extra_bytes);
    if (mode == CONS_MASS) {
      c_sum<<<blocks, TPB>>>(d_input, N, d_res);  // compute L0
      cudaMemcpy(&p1, d_res, sizeof(double), cudaMemcpyDeviceToHost);
    } else if (mode == CONS_ENERGY) {
      c_sum2<<<blocks, TPB>>>(d_input, N, d_res);  // compute S0
      cudaMemcpy(&p2, d_res, sizeof(double), cudaMemcpyDeviceToHost);
    } else {  // mode == (CONS_MASS | CONS_ENERGY)
      c_sum_sum2<<<blocks, TPB>>>(d_input, N, d_res);  // compute M0 and S0
      cudaMemcpy(&p1, &d_res[0], sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(&p2, &d_res[1], sizeof(double), cudaMemcpyDeviceToHost);
    }
    cudaFree(d_res);
  }
}


int main(int argc, char* argv [])
{
  printf("GPU cPFPL: single-precision lossy property-preserving compressor\n");
  printf("Copyright 2026 Texas State University\n\n");

  // read input from file
  if (argc < 5 || argc > 6) {printf("USAGE: %s input_file_name compressed_file_name error_bound mode [threshold]\n\n", argv[0]);  exit(-1);}

  const int mode = std::stoi(argv[4]);
  const int all = CONS_MASS | CONS_ENERGY;
  if ((mode < 0) || (mode > all)) {fprintf(stderr, "ERROR: mode must be between 0 and %d\n", all); return -1;}
  // 0: conserve nothing
  // 1: conserve mass
  // 2: conserve energy
  // 3: conserve both

  FILE* const fin = fopen(argv[1], "rb");
  fseek(fin, 0, SEEK_END);
  const long long fsize = ftell(fin);
  if (fsize <= 0) {fprintf(stderr, "ERROR: input file too small\n\n"); exit(-1);}
  if (fsize >= 2147221529) {fprintf(stderr, "ERROR: input file too large\n\n"); exit(-1);}
  byte* const input = new byte [fsize];
  fseek(fin, 0, SEEK_SET);
  const int insize = fread(input, 1, fsize, fin);  assert(insize == fsize);
  fclose(fin);
  printf("original size: %d bytes\n", insize);

  // get GPU info
  cudaSetDevice(0);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {fprintf(stderr, "ERROR: no CUDA capable device detected\n\n"); exit(-1);}
  const int SMs = deviceProp.multiProcessorCount;
  const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
  const int blocks = SMs * (mTpSM / TPB);
  const int chunks = (insize + CS - 1) / CS;  // round up
  CheckCuda(__LINE__);
  const int maxsize = 3 * sizeof(int) + chunks * sizeof(short) + chunks * CS;
  const long long N = insize / sizeof(float);
  const int extra_bytes = __builtin_popcount(mode) * sizeof(double);

  // allocate GPU memory
  byte* dencoded;
  cudaMallocHost((void **)&dencoded, maxsize);
  byte* d_input;
  cudaMalloc((void **)&d_input, insize);
  cudaMemcpy(d_input, input, insize, cudaMemcpyHostToDevice);
  byte* d_encoded;
  cudaMalloc((void **)&d_encoded, maxsize);
  int* d_encsize;
  cudaMalloc((void **)&d_encsize, sizeof(int));
  int* d_fullcarry;
  cudaMalloc((void**)&d_fullcarry, chunks * sizeof(int));
  CheckCuda(__LINE__);

  #if defined(STATS)
    const float parameter = std::stof(argv[3]);
    const float eb = computeNOAeb((float*)input, N, parameter);
    const float errorbound = eb * EBfactor;
    printf("requested NOA EB: %.6f\n", parameter);
    printf("resulting ABS EB: %.6f\n", eb);
    printf("utilized  ABS EB: %.6f\n", errorbound);
  #else
    const float errorbound = atof(argv[3]) * EBfactor;
  #endif

  const float threshold = (argc == 6) ? atof(argv[5]) : std::numeric_limits<float>::infinity();
  if (threshold < std::numeric_limits<float>::min()) {printf("ERROR: threshold must be a positive, normal, floating-point value\n");  exit(-1);}

  double p1, p2;  // conservation parameters
  double constime, runtime;
  GPUTimer dtimer;
  for (int i = 0; i < NUM_RUNS; i++) {
    cudaDeviceSynchronize();
    dtimer.start();
    get_conservation_params(mode, blocks, (float*)d_input, N, p1, p2);
    constime = dtimer.stop();
    dtimer.start();
    d_reset<<<1, 1>>>();
    cudaMemset(d_fullcarry, 0, chunks * sizeof(int));
    d_encode<<<blocks, TPB>>>(d_input, insize, d_encoded, d_encsize, d_fullcarry, errorbound, threshold);
//    cudaDeviceSynchronize();
    runtime = dtimer.stop();

    if (i < NUM_RUNS - 1) {
      cudaMemset(d_encsize, 0, sizeof(int));
      cudaMemset(d_encoded, 0, maxsize);
    }
    CheckCuda(__LINE__);
  }

  // get encoded GPU result
  int dencsize = 0;
  cudaMemcpy(&dencsize, d_encsize, sizeof(int), cudaMemcpyDeviceToHost);
    CheckCuda(__LINE__);
  cudaMemcpy(dencoded, d_encoded, dencsize, cudaMemcpyDeviceToHost);
    CheckCuda(__LINE__);

  printf("encoded size: %d bytes\n", dencsize + extra_bytes);

  #if defined(STATS)
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
  #else
    constime = constime;  // to suppress warning
    printf("compression ratio: %.2f\n", 1.0 * insize / (dencsize + extra_bytes));
  #endif
  CheckCuda(__LINE__);

  // write to file
  FILE* const fout = fopen(argv[2], "wb");
  fwrite(dencoded, 1, dencsize, fout);
  if (mode & CONS_MASS) fwrite(&p1, 1, sizeof(double), fout);
  if (mode & CONS_ENERGY) fwrite(&p2, 1, sizeof(double), fout);
  fclose(fout);

  if (mode & CONS_MASS) printf("saved p1: %e\n", p1);
  if (mode & CONS_ENERGY) printf("saved p2: %e\n", p2);

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
