# GPU results: cSLEEK and cPFPL on an RTX 4090

Runs by Martin Burtscher with `cSLEEK/cSLEEKrun.py` and `cPFPL/cPFPLrun.py` (raw sheets in this folder, received 2026-10-26). Inputs: CESM-ATM `RELHUM_1_26_1800_3600.f32`, ISABEL `TCf48.bin.f32`, NYX `velocity_x.f32`. NOA bounds 1e-2 to 1e-4; modes 1-3 compress at 0.9 x the bound. Mode 0 = codec alone, 1 = mass, 2 = energy, 3 = mass+energy. Drift is |invariant change| / |invariant| as reported by `check_conservation.cpp` (for NYX mass that is relative to |sum x|, see note 3). Throughput: encode = preprocessing + compression, decode = decompression + postprocessing.

## Summary (mode 3, mass+energy, over 3 inputs x 5 bounds)

| codec | ratio kept | compression throughput kept | decompression throughput kept | mass drift reduction (geo. mean / max) | energy drift reduction (geo. mean / max) | bound violations |
|---|---|---|---|---|---|---|
| PFPL | 95.3% | 55.2% | 41.4% | 2201x / 5.9e+06x | 2101x / 1.2e+06x | 1 |
| SLEEK | 97.8% | 53.0% | 29.4% | 1100x / 4.7e+05x | 2179x / 4.6e+05x | 0 |

The sheets' own summary row uses arithmetic means of the reduction factors (PFPL 5.5e5x mass, 1.5e5x energy; SLEEK 8.0e4x, 8.7e4x). Those are dominated by the largest cases; the geometric mean above is the more representative single number.

## PFPL: mode 0 (codec alone) vs mode 3 (mass+energy)

| input | NOA eps | ratio | ratio +layer | mass drift | mass +layer | energy drift | energy +layer | comp GB/s | encode GB/s +layer | decomp GB/s | decode GB/s +layer | viol. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CESM RELHUM | 0.01 | 59.77 | 56.25 | 7.0e-4 | 1.2e-10 | 1.4e-4 | 1.6e-10 | 449 | 248 | 348 | 140 | 0 |
| CESM RELHUM | 0.003 | 33.94 | 32.69 | 1.1e-4 | 3.7e-9 | 1.2e-5 | 7.6e-9 | 443 | 247 | 343 | 140 | 0 |
| CESM RELHUM | 0.001 | 23.15 | 22.10 | 1.7e-5 | 7.1e-10 | 1.6e-6 | 4.2e-9 | 441 | 245 | 340 | 139 | 0 |
| CESM RELHUM | 0.0003 | 14.92 | 14.10 | 1.7e-6 | 1.2e-10 | 1.3e-7 | 3.0e-10 | 436 | 242 | 338 | 139 | 0 |
| CESM RELHUM | 0.0001 | 10.09 | 9.41 | 2.1e-6 | 3.7e-10 | 7.1e-8 | 8.2e-10 | 426 | 239 | 333 | 138 | 0 |
| ISABEL TC | 0.01 | 54.73 | 50.95 | 2.7e-5 | 1.8e-8 | 1.2e-4 | 3.6e-8 | 404 | 217 | 319 | 132 | 0 |
| ISABEL TC | 0.003 | 26.00 | 24.59 | 1.4e-5 | 8.0e-9 | 6.7e-6 | 1.7e-8 | 400 | 214 | 317 | 133 | 0 |
| ISABEL TC | 0.001 | 15.58 | 14.98 | 2.6e-7 | 2.3e-9 | 2.8e-6 | 4.8e-9 | 400 | 216 | 317 | 132 | 0 |
| ISABEL TC | 0.0003 | 10.23 | 9.95 | 3.2e-8 | 5.5e-9 | 1.8e-7 | 1.2e-8 | 392 | 215 | 315 | 132 | 0 |
| ISABEL TC | 0.0001 | 6.96 | 6.74 | 4.7e-8 | 2.6e-8 | 5.4e-8 | 5.3e-8 | 390 | 210 | 312 | 131 | 0 |
| NYX vel_x | 0.01 | 50.19 | 47.33 | 2.9e+0 | 9.4e-5 | 5.5e-3 | 4.7e-9 | 438 | 243 | 334 | 138 | 1 |
| NYX vel_x | 0.003 | 26.28 | 25.00 | 2.3e-1 | 1.0e-7 | 5.0e-4 | 5.1e-9 | 435 | 244 | 333 | 138 | 0 |
| NYX vel_x | 0.001 | 17.05 | 16.47 | 2.5e-2 | 8.6e-6 | 5.3e-5 | 1.6e-9 | 435 | 242 | 333 | 138 | 0 |
| NYX vel_x | 0.0003 | 11.11 | 10.66 | 2.2e-5 | 2.3e-6 | 4.6e-6 | 2.9e-10 | 432 | 241 | 332 | 138 | 0 |
| NYX vel_x | 0.0001 | 7.42 | 7.18 | 9.7e-4 | 1.3e-5 | 4.0e-7 | 8.7e-11 | 424 | 238 | 326 | 137 | 0 |

## SLEEK: mode 0 (codec alone) vs mode 3 (mass+energy)

| input | NOA eps | ratio | ratio +layer | mass drift | mass +layer | energy drift | energy +layer | comp GB/s | encode GB/s +layer | decomp GB/s | decode GB/s +layer | viol. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CESM RELHUM | 0.01 | 5.64 | 5.64 | 3.8e-4 | 8.1e-10 | 2.2e-5 | 1.5e-9 | 731 | 376 | 699 | 192 | 0 |
| CESM RELHUM | 0.003 | 4.88 | 4.88 | 1.2e-4 | 6.4e-10 | 5.2e-6 | 1.2e-9 | 714 | 370 | 686 | 189 | 0 |
| CESM RELHUM | 0.001 | 3.82 | 3.82 | 2.8e-5 | 2.7e-10 | 4.5e-7 | 4.5e-10 | 684 | 363 | 660 | 188 | 0 |
| CESM RELHUM | 0.0003 | 3.12 | 3.12 | 7.3e-6 | 9.5e-11 | 1.1e-7 | 3.0e-10 | 660 | 353 | 633 | 186 | 0 |
| CESM RELHUM | 0.0001 | 2.84 | 2.84 | 1.4e-6 | 1.5e-10 | 1.1e-7 | 3.9e-10 | 649 | 351 | 621 | 185 | 0 |
| ISABEL TC | 0.01 | 5.96 | 5.02 | 1.1e-4 | 3.3e-10 | 2.1e-4 | 7.8e-10 | 610 | 334 | 622 | 186 | 0 |
| ISABEL TC | 0.003 | 4.34 | 4.34 | 3.5e-6 | 3.0e-9 | 1.1e-5 | 7.1e-9 | 610 | 332 | 604 | 183 | 0 |
| ISABEL TC | 0.001 | 3.41 | 3.41 | 1.2e-9 | 1.2e-8 | 3.3e-7 | 2.6e-8 | 595 | 314 | 583 | 182 | 0 |
| ISABEL TC | 0.0003 | 3.08 | 2.81 | 5.5e-8 | 9.2e-9 | 2.5e-7 | 1.8e-8 | 581 | 309 | 560 | 181 | 0 |
| ISABEL TC | 0.0001 | 2.59 | 2.59 | 5.4e-8 | 3.4e-8 | 1.2e-7 | 6.8e-8 | 558 | 307 | 553 | 179 | 0 |
| NYX vel_x | 0.01 | 7.75 | 7.75 | 4.1e-1 | 2.7e-5 | 2.3e-3 | 8.9e-9 | 743 | 378 | 716 | 193 | 0 |
| NYX vel_x | 0.003 | 5.23 | 5.23 | 2.6e-2 | 3.5e-5 | 1.4e-4 | 3.1e-10 | 709 | 369 | 689 | 191 | 0 |
| NYX vel_x | 0.001 | 4.50 | 4.50 | 1.7e-3 | 1.3e-5 | 3.4e-5 | 8.1e-10 | 701 | 365 | 670 | 190 | 0 |
| NYX vel_x | 0.0003 | 3.51 | 3.51 | 3.5e-3 | 6.3e-6 | 2.1e-6 | 8.4e-12 | 673 | 357 | 646 | 188 | 0 |
| NYX vel_x | 0.0001 | 3.17 | 2.88 | 2.4e-4 | 1.1e-4 | 6.7e-7 | 5.7e-9 | 652 | 348 | 632 | 185 | 0 |

## Across invariants: correcting one invariant vs. both

Energy drift with mass-only correction (mode 1) and mass drift with energy-only correction (mode 2), relative to the codec alone. A factor above 1 means the uncorrected invariant got worse.

| codec | input | NOA eps | energy drift: alone -> mass-only (factor) | mass drift: alone -> energy-only (factor) |
|---|---|---|---|---|
| PFPL | CESM RELHUM | 0.01 | 1.4e-4 -> 9.2e-4 (6.6x) | 7.0e-4 -> 7.1e-4 (1x) |
| PFPL | CESM RELHUM | 0.003 | 1.2e-5 -> 1.6e-4 (13x) | 1.1e-4 -> 1.2e-4 (1.2x) |
| PFPL | CESM RELHUM | 0.001 | 1.6e-6 -> 2.4e-5 (15x) | 1.7e-5 -> 1.9e-5 (1.1x) |
| PFPL | CESM RELHUM | 0.0003 | 1.3e-7 -> 3.1e-6 (24x) | 1.7e-6 -> 2.5e-6 (1.5x) |
| PFPL | CESM RELHUM | 0.0001 | 7.1e-8 -> 1.4e-6 (20x) | 2.1e-6 -> 1.1e-6 (0.52x) |
| PFPL | ISABEL TC | 0.01 | 1.2e-4 -> 3.3e-5 (0.27x) | 2.7e-5 -> 3.6e-5 (1.3x) |
| PFPL | ISABEL TC | 0.003 | 6.7e-6 -> 1.3e-5 (2x) | 1.4e-5 -> 4.5e-6 (0.33x) |
| PFPL | ISABEL TC | 0.001 | 2.8e-6 -> 1.4e-6 (0.49x) | 2.6e-7 -> 8.4e-7 (3.2x) |
| PFPL | ISABEL TC | 0.0003 | 1.8e-7 -> 2.3e-7 (1.2x) | 3.2e-8 -> 1.4e-7 (4.5x) |
| PFPL | ISABEL TC | 0.0001 | 5.4e-8 -> 7.2e-8 (1.3x) | 4.7e-8 -> 3.8e-8 (0.81x) |
| PFPL | NYX vel_x | 0.01 | 5.5e-3 -> 4.4e-3 (0.8x) | 2.9e+0 -> 4.7e-1 (0.16x) |
| PFPL | NYX vel_x | 0.003 | 5.0e-4 -> 3.8e-4 (0.75x) | 2.3e-1 -> 2.1e-1 (0.9x) |
| PFPL | NYX vel_x | 0.001 | 5.3e-5 -> 4.5e-5 (0.84x) | 2.5e-2 -> 5.2e-3 (0.21x) |
| PFPL | NYX vel_x | 0.0003 | 4.6e-6 -> 3.2e-6 (0.69x) | 2.2e-5 -> 3.2e-5 (1.4x) |
| PFPL | NYX vel_x | 0.0001 | 4.0e-7 -> 4.7e-7 (1.2x) | 9.7e-4 -> 8.3e-4 (0.86x) |
| SLEEK | CESM RELHUM | 0.01 | 2.2e-5 -> 5.0e-4 (22x) | 3.8e-4 -> 3.9e-4 (1x) |
| SLEEK | CESM RELHUM | 0.003 | 5.2e-6 -> 1.5e-4 (30x) | 1.2e-4 -> 1.2e-4 (1x) |
| SLEEK | CESM RELHUM | 0.001 | 4.5e-7 -> 3.5e-5 (77x) | 2.8e-5 -> 2.8e-5 (1x) |
| SLEEK | CESM RELHUM | 0.0003 | 1.1e-7 -> 9.3e-6 (86x) | 7.3e-6 -> 7.4e-6 (1x) |
| SLEEK | CESM RELHUM | 0.0001 | 1.1e-7 -> 1.6e-6 (14x) | 1.4e-6 -> 1.3e-6 (0.94x) |
| SLEEK | ISABEL TC | 0.01 | 2.1e-4 -> 6.0e-6 (0.029x) | 1.1e-4 -> 7.6e-6 (0.066x) |
| SLEEK | ISABEL TC | 0.003 | 1.1e-5 -> 1.4e-5 (1.3x) | 3.5e-6 -> 8.9e-6 (2.6x) |
| SLEEK | ISABEL TC | 0.001 | 3.3e-7 -> 3.3e-7 (1x) | 1.2e-9 -> 1.6e-7 (1.3e+02x) |
| SLEEK | ISABEL TC | 0.0003 | 2.5e-7 -> 1.4e-7 (0.55x) | 5.5e-8 -> 1.3e-8 (0.24x) |
| SLEEK | ISABEL TC | 0.0001 | 1.2e-7 -> 1.2e-7 (0.98x) | 5.4e-8 -> 4.3e-8 (0.8x) |
| SLEEK | NYX vel_x | 0.01 | 2.3e-3 -> 2.3e-3 (1x) | 4.1e-1 -> 4.1e-1 (1x) |
| SLEEK | NYX vel_x | 0.003 | 1.4e-4 -> 1.4e-4 (1x) | 2.6e-2 -> 2.6e-2 (1x) |
| SLEEK | NYX vel_x | 0.001 | 3.4e-5 -> 3.4e-5 (1x) | 1.7e-3 -> 1.8e-3 (1x) |
| SLEEK | NYX vel_x | 0.0003 | 2.1e-6 -> 2.1e-6 (1x) | 3.5e-3 -> 3.5e-3 (1x) |
| SLEEK | NYX vel_x | 0.0001 | 6.7e-7 -> 1.8e-7 (0.26x) | 2.4e-4 -> 2.9e-4 (1.2x) |

Worst energy increase from correcting mass alone: PFPL 24x (CESM RELHUM, eps 0.0003); SLEEK 86x (CESM RELHUM, eps 0.0003). Mode 3 keeps both at or below 7e-8 in every case (NYX mass on the paper's sum |x| scale, note 3).

## Notes

1. **Bound violations:** PFPL NYX vel_x eps 0.01 mode 2; PFPL NYX vel_x eps 0.01 mode 3 (one value each). The sheets give the count, not the size; a max-error/bound column would show by how much.
2. **Mass-only (mode 1) decoding is slower than mass+energy** in both codecs, by roughly 1 ms per call (postprocessing on ISABEL: PFPL ~65 vs ~226 GB/s, SLEEK ~70 vs ~260 GB/s). Likely cause: in `conserve_cuda.h`, `conserve_linear_w` calls `conserve_linear(d_x, N, L0)` without passing `blocks`, so `setBlocks` calls `cudaGetDeviceProperties` on every call. Passing `blocks` through should remove the overhead.
3. **NYX mass drift** is relative to |sum x|; velocity nearly cancels, so the paper reports it relative to sum |x|. Multiply the NYX mass values by |sum x| / sum |x| = 6.76e-05 for the paper's scale (reduction factors are unchanged). Example, PFPL eps 1e-2 mode 3: 9.38e-05 -> 6.3e-09.
4. **At the float32 floor.** Where the codec's own drift is already near float32 precision, mode 3 can leave it slightly higher, still below 1e-7: SLEEK ISABEL TC eps 0.001 mass 1.2e-9 -> 1.2e-8.
