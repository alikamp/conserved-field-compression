// checks EB and conservation
// g++ -O3 -march=native -mno-fma -ffp-contract=off check_conservation.cpp -o check_conservation -I../cSLEEK


using byte = unsigned char;


#include <string>
#include <cmath>
#include <cassert>
#include "MAXABS_f32.h"


static const float EBfactor = 0.5f;


static inline float computeNOAeb(const float* const input, const long long size, const float eb_param)
{
  float min = input[0], max = input[0];
  for (long long i = 1; i < size; i++) {
    const float val = input[i];
    min = std::min(val, min);
    max = std::max(val, max);
  }
/*
  printf("min_val: %.10f\n", min);
  printf("max_val: %.10f\n", max);
  printf("diff between max and min values: %.10f\n", (max - min));
*/
  return (max - min) * eb_param;
}


int main(int argc, char* argv [])
{
  printf("Check Conservation: single-precision lossy property conservation\n");

  if (argc != 4) {printf("USAGE: %s original_file_name decompressed_file_name error_bound\n\n", argv[0]); return -1;}

  // read original file
  FILE* const fin1 = fopen(argv[1], "rb");
  fseek(fin1, 0, SEEK_END);
  const long long size1 = ftell(fin1);  assert(size1 > 0);
  byte* const data1 = new byte [size1];
  fseek(fin1, 0, SEEK_SET);
  // all "*size" variables in bytes
  long long insize = fread(data1, 1, size1, fin1);  assert(insize == size1);
  fclose(fin1);

  // read decoded file
  FILE* const fin2 = fopen(argv[2], "rb");
  fseek(fin2, 0, SEEK_END);
  const long long size2 = ftell(fin2);  assert(size2 > 0);
  byte* const data2 = new byte [size2];
  fseek(fin2, 0, SEEK_SET);
  insize = fread(data2, 1, size2, fin2);  assert(insize == size2);
  fclose(fin2);

  if (size1 != size2) {printf("ERROR: both files must have the same length\n\n"); return -1;}
  const long long N = size1 / sizeof(float);

  // eb variables
  const float parameter = std::stof(argv[3]);
  const float eb = computeNOAeb((float*)data1, N, parameter);
//  const float eb = std::stof(argv[3]);
  printf("EB factor: %.6f\n", EBfactor);
  printf("requested NOA EB: %.6f\n", parameter);
  printf("resulting ABS EB: %.6f\n", eb);
  printf("utilized  ABS EB: %.6f\n", eb * EBfactor);

  // verify point-wise EB
  MAXABS_f32(size1, data2, data1, eb);

  // check invariants
  float* float1 = (float*)data1;
  float* float2 = (float*)data2;
  double M1 = 0.0, S1 = 0.0;
  for (long long i = 0; i < N; i++) {
    const double val = float1[i];
    M1 += val;
    S1 += val * val;
  }
  double M2 = 0.0, S2 = 0.0;
  for (long long i = 0; i < N; i++) {
    const double val = float2[i];
    M2 += val;
    S2 += val * val;
  }
  printf("[invariants] mass:   %.3e\n", fabs(M2 - M1) / fabs(M1));
  printf("[invariants] energy: %.3e\n", fabs(S2 - S1) / fabs(S1));

  // clean up memory
  delete [] data1;
  delete [] data2;
  return 0;
}
