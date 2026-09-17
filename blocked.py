"""
Block-based Viewer-First quantizer (the #3 + #4 improvements).

Motivation: the original tiered quantizer uses ONE global min/max per tier,
so the same bitwidth has to span the whole dynamic range of a tier's voxels
-- wasteful. This version cuts the grid into small blocks and:

  #3  quantizes each voxel against its BLOCK's local min/max, and picks the
      block's bitwidth from the max attention inside that block. Small local
      range => the same bitwidth resolves much finer detail => lower error at
      equal bits (directly attacking where VFC loses to ZFP).

  #4  optionally DOWNSAMPLES the smoothest (lowest-attention) blocks: instead
      of storing B^3 codes, store the block mean (constant reconstruction).
      Smooth laminar background has tiny in-block variance, so this is nearly
      free in error but large in ratio (that background is ~1/3 of voxels).

Per-block overhead is tiny: a 1-byte mode + 1-byte bitwidth + two float32
(local min/max) per block, negligible next to the payload.
"""
from __future__ import annotations
import numpy as np
import struct
from dataclasses import dataclass
from typing import List, Tuple

from bitpack import pack_bits_fast, unpack_bits_fast, TierSpec, DEFAULT_TIERS

MODE_FULL = 0        # per-voxel codes at block bitwidth, block-local min/max
MODE_CONST = 1        # single block mean (constant reconstruction)
MODE_RAW32 = 2        # lossless float32 passthrough (highest tier)
MODE_DOWN2 = 3        # 2x-coarsened block, trilinear-upsampled on decode (#4)

MAGIC = b"VFB2"       # Viewer-First Blocked, v2 (adds trilinear downsample)


def _bitwidth_for_attention(a: float, tier_specs: List[TierSpec]) -> int:
    for spec in tier_specs:
        if spec.lo <= a < spec.hi:
            return spec.bitwidth
    return tier_specs[-1].bitwidth


def auto_tiers(attention_map: np.ndarray,
               bitwidths=(4, 8, 16, 32),
               quantiles=(0.60, 0.90, 0.99)) -> List[TierSpec]:
    """#2: set tier cut points from the attention histogram instead of
    hardcoding them. Each cut is a quantile of the actual attention values,
    so the split adapts per-dataset. len(quantiles) == len(bitwidths) - 1."""
    assert len(quantiles) == len(bitwidths) - 1
    a = attention_map.ravel()
    cuts = [0.0] + [float(np.quantile(a, q)) for q in quantiles] + [1.0001]
    # ensure strictly increasing (degenerate/flat attention -> nudge)
    for i in range(1, len(cuts)):
        if cuts[i] <= cuts[i - 1]:
            cuts[i] = cuts[i - 1] + 1e-6
    return [TierSpec(cuts[i], cuts[i + 1], bitwidths[i]) for i in range(len(bitwidths))]


def _coarsen_2x(blk: np.ndarray) -> np.ndarray:
    """Average-pool a (B,B,B) block by 2 -> (B/2,B/2,B/2). B must be even."""
    B = blk.shape[0]
    h = B // 2
    return blk.reshape(h, 2, h, 2, h, 2).mean(axis=(1, 3, 5))


def _upsample_2x(coarse: np.ndarray, B: int) -> np.ndarray:
    """Trilinear upsample (B/2)^3 -> B^3."""
    from scipy.ndimage import zoom
    factor = B / coarse.shape[0]
    up = zoom(coarse.astype(np.float64), factor, order=1, mode="nearest")
    # zoom can be off-by-one on exact factors; crop/pad to B
    if up.shape[0] != B:
        out = np.zeros((B, B, B), dtype=np.float64)
        s = tuple(slice(0, min(B, up.shape[d])) for d in range(3))
        out[s] = up[s]
        up = out
    return up.astype(np.float32)


def _pad_to_blocks(field: np.ndarray, B: int):
    pads = [(0, (B - (s % B)) % B) for s in field.shape]
    return np.pad(field, pads, mode="edge"), pads


def _block_view(padded: np.ndarray, B: int):
    """Return (nb0,nb1,nb2, B,B,B) block view for a 3D array."""
    n0, n1, n2 = padded.shape
    nb0, nb1, nb2 = n0 // B, n1 // B, n2 // B
    return (padded.reshape(nb0, B, nb1, B, nb2, B)
                  .transpose(0, 2, 4, 1, 3, 5)), (nb0, nb1, nb2)


@dataclass
class BlockedField:
    orig_shape: Tuple[int, ...]
    block_size: int
    stream: bytes          # fully serialized payload
    alpha: float           # energy-conservation scale on correctable blocks
    lowest_bitwidth: int


def quantize_blocked(field: np.ndarray, attention_map: np.ndarray,
                      tier_specs: List[TierSpec] = None,
                      block_size: int = 8,
                      downsample: str = "none",   # "none" | "const" | "tri"
                      downsample_low: bool = None,  # back-compat: True -> "const"
                      lowest_bitwidth: int = None,
                      correctable_max_bitwidth: int = 16,
                      conserve_energy: bool = True) -> BlockedField:
    tier_specs = tier_specs or DEFAULT_TIERS
    B = block_size
    if downsample_low is not None:      # back-compat with earlier bool API
        downsample = "const" if downsample_low else "none"
    if downsample == "tri" and B % 2 != 0:
        downsample = "none"            # trilinear needs even block size
    if lowest_bitwidth is None:
        lowest_bitwidth = min(s.bitwidth for s in tier_specs)
    # `lowest_bitwidth` = downsample threshold; `correctable_max_bitwidth` =
    # which lossy blocks absorb the energy correction (decode scales these).

    f = field.astype(np.float32)
    fpad, pads = _pad_to_blocks(f, B)
    apad, _ = _pad_to_blocks(attention_map.astype(np.float32), B)

    fblocks, (nb0, nb1, nb2) = _block_view(fpad, B)
    ablocks, _ = _block_view(apad, B)

    # Pass 1: build block records + accumulate energies (over the ORIGINAL,
    # unpadded voxels only, so padding never enters the invariant).
    block_records = []            # (mode, bitw, vmin, vmax, packed_or_none)
    correctable_energy = 0.0      # sum of squares of quantized correctable voxels
    fixed_energy = 0.0            # sum of squares of quantized fixed voxels
    # valid-voxel mask per block (drops edge padding)
    vv, _ = _block_view(_pad_to_blocks(np.ones_like(f), B)[0], B)

    def recon_block(mode, bitw, vmin, vmax, codes, blk_shape):
        if mode == MODE_RAW32:
            return codes.reshape(blk_shape)  # codes holds raw floats here
        if mode == MODE_CONST:
            return np.full(blk_shape, vmin, dtype=np.float32)
        if mode == MODE_DOWN2:
            h = blk_shape[0] // 2
            levels = (1 << bitw) - 1
            if levels == 0 or vmax - vmin < 1e-12:
                coarse = np.full((h, h, h), vmin, dtype=np.float32)
            else:
                coarse = (vmin + (codes.astype(np.float64) / levels) * (vmax - vmin)).reshape(h, h, h)
            return _upsample_2x(coarse.astype(np.float32), blk_shape[0])
        levels = (1 << bitw) - 1
        if levels == 0 or vmax - vmin < 1e-12:
            return np.full(blk_shape, vmin, dtype=np.float32)
        return (vmin + (codes.astype(np.float64) / levels) * (vmax - vmin)).reshape(blk_shape).astype(np.float32)

    for i in range(nb0):
        for j in range(nb1):
            for k in range(nb2):
                blk = fblocks[i, j, k]
                mask = vv[i, j, k].astype(bool)
                a_max = float(ablocks[i, j, k].max())
                bitw = _bitwidth_for_attention(a_max, tier_specs)
                vmin = float(blk.min()); vmax = float(blk.max())

                if bitw >= 32:
                    rec = blk.astype(np.float32)
                    block_records.append((MODE_RAW32, 0, vmin, vmax, blk.astype(np.float32)))
                    fixed_energy += float(np.sum((rec[mask].astype(np.float64)) ** 2))
                    continue

                if downsample != "none" and bitw <= lowest_bitwidth:
                    if downsample == "const":
                        mean = float(blk.mean())
                        block_records.append((MODE_CONST, 0, mean, mean, None))
                        rec = np.full(blk.shape, mean, dtype=np.float32)
                    else:  # "tri": 2x-coarsen, quantize coarse at block bitwidth
                        coarse = _coarsen_2x(blk)
                        cmin, cmax = float(coarse.min()), float(coarse.max())
                        if cmax - cmin < 1e-12:
                            ccodes = np.zeros(coarse.size, dtype=np.uint32)
                        else:
                            levels = (1 << bitw) - 1
                            ccodes = np.round((coarse.ravel() - cmin) / (cmax - cmin) * levels).astype(np.uint32)
                            ccodes = np.clip(ccodes, 0, levels)
                        block_records.append((MODE_DOWN2, bitw, cmin, cmax, ccodes))
                        rec = recon_block(MODE_DOWN2, bitw, cmin, cmax, ccodes, blk.shape)
                    correctable_energy += float(np.sum((rec[mask].astype(np.float64)) ** 2))
                    continue

                if vmax - vmin < 1e-12:
                    codes = np.zeros(blk.size, dtype=np.uint32)
                else:
                    levels = (1 << bitw) - 1
                    codes = np.round((blk.ravel() - vmin) / (vmax - vmin) * levels).astype(np.uint32)
                    codes = np.clip(codes, 0, levels)
                block_records.append((MODE_FULL, bitw, vmin, vmax, codes))
                rec = recon_block(MODE_FULL, bitw, vmin, vmax, codes, blk.shape)
                if bitw <= correctable_max_bitwidth:
                    correctable_energy += float(np.sum((rec[mask].astype(np.float64)) ** 2))
                else:
                    fixed_energy += float(np.sum((rec[mask].astype(np.float64)) ** 2))

    # Solve alpha so total energy matches the original (over valid voxels).
    alpha = 1.0
    if conserve_energy:
        target = float(np.sum(f.astype(np.float64) ** 2))
        remainder = target - fixed_energy
        if correctable_energy > 1e-30 and remainder > 0:
            alpha = float(np.sqrt(remainder / correctable_energy))

    # Pass 2: serialize (alpha stored in header; decode applies it).
    parts = [MAGIC]
    parts.append(struct.pack("<B", len(field.shape)))
    for s in field.shape:
        parts.append(struct.pack("<Q", s))
    parts.append(struct.pack("<B", B))
    parts.append(struct.pack("<Bf", correctable_max_bitwidth, alpha))
    parts.append(struct.pack("<QQQ", nb0, nb1, nb2))

    for (mode, bitw, vmin, vmax, payload) in block_records:
        if mode == MODE_RAW32:
            parts.append(struct.pack("<Bff", MODE_RAW32, vmin, vmax))
            parts.append(payload.tobytes())
        elif mode == MODE_CONST:
            parts.append(struct.pack("<Bff", MODE_CONST, vmin, vmax))
        elif mode == MODE_DOWN2:
            parts.append(struct.pack("<BBff", MODE_DOWN2, bitw, vmin, vmax))
            parts.append(pack_bits_fast(payload, bitw))
        else:
            parts.append(struct.pack("<BBff", MODE_FULL, bitw, vmin, vmax))
            parts.append(pack_bits_fast(payload, bitw))

    return BlockedField(orig_shape=field.shape, block_size=B,
                        stream=b"".join(parts), alpha=alpha,
                        lowest_bitwidth=lowest_bitwidth)


def dequantize_blocked(bf: BlockedField) -> np.ndarray:
    buf = bf.stream
    off = 0
    assert buf[:4] == MAGIC
    off += 4
    ndim = buf[off]; off += 1
    shape = []
    for _ in range(ndim):
        (s,) = struct.unpack_from("<Q", buf, off); off += 8
        shape.append(s)
    B = buf[off]; off += 1
    correctable_max_bitwidth, alpha = struct.unpack_from("<Bf", buf, off); off += 5
    nb0, nb1, nb2 = struct.unpack_from("<QQQ", buf, off); off += 24

    padded = np.zeros((nb0 * B, nb1 * B, nb2 * B), dtype=np.float32)
    out_blocks, _ = _block_view(padded, B)  # a view we can write into

    for i in range(nb0):
        for j in range(nb1):
            for k in range(nb2):
                mode = buf[off]
                if mode == MODE_RAW32:
                    _, vmin, vmax = struct.unpack_from("<Bff", buf, off); off += 9
                    n = B ** 3
                    vals = np.frombuffer(buf, dtype=np.float32, count=n, offset=off)
                    off += n * 4
                    out_blocks[i, j, k] = vals.reshape(B, B, B)  # fixed: no alpha
                elif mode == MODE_CONST:
                    _, mean, _m2 = struct.unpack_from("<Bff", buf, off); off += 9
                    out_blocks[i, j, k] = mean * alpha  # correctable
                elif mode == MODE_DOWN2:
                    _, bitw, vmin, vmax = struct.unpack_from("<BBff", buf, off); off += 10
                    h = B // 2
                    nc = h ** 3
                    levels = (1 << bitw) - 1
                    nbytes = _packed_len(nc, bitw)
                    packed = buf[off:off + nbytes]; off += nbytes
                    ccodes = unpack_bits_fast(packed, bitw, nc)
                    if levels == 0 or vmax - vmin < 1e-12:
                        coarse = np.full((h, h, h), vmin, dtype=np.float32)
                    else:
                        coarse = (vmin + (ccodes.astype(np.float64) / levels) * (vmax - vmin)).reshape(h, h, h).astype(np.float32)
                    up = _upsample_2x(coarse, B)
                    scale = alpha if bitw <= correctable_max_bitwidth else 1.0
                    out_blocks[i, j, k] = (up * scale).astype(np.float32)
                else:  # MODE_FULL
                    _, bitw, vmin, vmax = struct.unpack_from("<BBff", buf, off); off += 10
                    n = B ** 3
                    levels = (1 << bitw) - 1
                    nbytes = _packed_len(n, bitw)
                    packed = buf[off:off + nbytes]; off += nbytes
                    codes = unpack_bits_fast(packed, bitw, n)
                    if levels == 0 or vmax - vmin < 1e-12:
                        vals = np.full(n, vmin, dtype=np.float32)
                    else:
                        vals = vmin + (codes.astype(np.float64) / levels) * (vmax - vmin)
                    scale = alpha if bitw <= correctable_max_bitwidth else 1.0
                    out_blocks[i, j, k] = (vals * scale).reshape(B, B, B).astype(np.float32)

    # crop padding
    sl = tuple(slice(0, s) for s in shape)
    return padded[sl]


def _packed_len(n: int, bitw: int) -> int:
    if bitw == 8:
        return n
    if bitw == 16:
        return n * 2
    if bitw == 32:
        return n * 4
    if bitw == 4:
        return (n + 1) // 2
    # generic
    return (n * bitw + 7) // 8
