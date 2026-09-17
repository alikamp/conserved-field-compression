"""
Low-level arbitrary-bitwidth packing (1-32 bits per value) and the
adaptive, attention-routed quantization engine built on top of it.

Attention-map value -> tier -> bitwidth:
    high attention  (structure, shocks, vortex cores) -> wide bitwidth
                                                          (up to lossless float32)
    low attention   (smooth / laminar background)      -> narrow bitwidth (down to 4 bits)
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# --------------------------------------------------------------------------
# Generic bit packer: packs an array of unsigned ints (each < 2**bitwidth)
# into a tightly packed byte buffer, MSB-first within a 64-bit accumulator.
# --------------------------------------------------------------------------

def pack_bits(values: np.ndarray, bitwidth: int) -> bytes:
    if bitwidth <= 0:
        return b""
    values = values.astype(np.uint64)
    assert values.min() >= 0
    assert values.max() < (1 << bitwidth), "value exceeds bitwidth range"

    acc = 0
    acc_bits = 0
    out = bytearray()
    for v in values.tolist():
        acc = (acc << bitwidth) | v
        acc_bits += bitwidth
        while acc_bits >= 8:
            acc_bits -= 8
            out.append((acc >> acc_bits) & 0xFF)
    if acc_bits > 0:
        out.append((acc << (8 - acc_bits)) & 0xFF)
    return bytes(out)


def unpack_bits(buf: bytes, bitwidth: int, count: int) -> np.ndarray:
    if bitwidth <= 0 or count == 0:
        return np.zeros(count, dtype=np.uint32)

    mask = (1 << bitwidth) - 1
    acc = 0
    acc_bits = 0
    out = np.empty(count, dtype=np.uint64)
    idx = 0
    for byte in buf:
        acc = (acc << 8) | byte
        acc_bits += 8
        while acc_bits >= bitwidth and idx < count:
            acc_bits -= bitwidth
            out[idx] = (acc >> acc_bits) & mask
            idx += 1
        if idx >= count:
            break
    return out[:count].astype(np.uint32)


# Vectorized fast paths for the common power-of-two-friendly bitwidths used
# by this engine (4, 8, 16, 32) avoid the slow Python loop above.

def pack_bits_fast(values: np.ndarray, bitwidth: int) -> bytes:
    values = values.astype(np.uint32)
    if bitwidth == 8:
        return values.astype(np.uint8).tobytes()
    if bitwidth == 16:
        return values.astype(np.uint16).tobytes()
    if bitwidth == 32:
        return values.astype(np.uint32).tobytes()
    if bitwidth == 4:
        if values.size % 2 == 1:
            values = np.concatenate([values, [0]])
        hi = values[0::2].astype(np.uint8) << 4
        lo = values[1::2].astype(np.uint8) & 0x0F
        return (hi | lo).tobytes()
    # Fallback: generic (slow) path
    return pack_bits(values, bitwidth)


def unpack_bits_fast(buf: bytes, bitwidth: int, count: int) -> np.ndarray:
    if bitwidth == 8:
        return np.frombuffer(buf, dtype=np.uint8, count=count).astype(np.uint32)
    if bitwidth == 16:
        return np.frombuffer(buf, dtype=np.uint16, count=count).astype(np.uint32)
    if bitwidth == 32:
        return np.frombuffer(buf, dtype=np.uint32, count=count).astype(np.uint32)
    if bitwidth == 4:
        packed = np.frombuffer(buf, dtype=np.uint8)
        hi = (packed >> 4) & 0x0F
        lo = packed & 0x0F
        out = np.empty(hi.size + lo.size, dtype=np.uint32)
        out[0::2] = hi
        out[1::2] = lo
        return out[:count]
    return unpack_bits(buf, bitwidth, count)


# --------------------------------------------------------------------------
# Tiered quantization engine
# --------------------------------------------------------------------------

@dataclass
class TierSpec:
    lo: float          # attention lower bound (inclusive)
    hi: float           # attention upper bound (exclusive, 1.0 is inclusive on last tier)
    bitwidth: int        # 4, 8, 16, or 32 (32 == lossless float32 passthrough)


DEFAULT_TIERS: List[TierSpec] = [
    TierSpec(0.0, 0.15, 4),
    TierSpec(0.15, 0.45, 8),
    TierSpec(0.45, 0.75, 16),
    TierSpec(0.75, 1.0001, 32),
]


@dataclass
class TierBlock:
    bitwidth: int
    indices: np.ndarray       # flat indices into the original array (int64)
    vmin: float
    vmax: float
    packed: bytes


@dataclass
class QuantizedField:
    shape: Tuple[int, ...]
    tiers: List[TierBlock]
    tier_specs: List[TierSpec]


def assign_tiers(attention_map: np.ndarray, tier_specs: List[TierSpec] = None) -> np.ndarray:
    """Return an int8 array (same shape) giving the tier index for each element."""
    tier_specs = tier_specs or DEFAULT_TIERS
    tier_idx = np.zeros(attention_map.shape, dtype=np.int8)
    for i, spec in enumerate(tier_specs):
        mask = (attention_map >= spec.lo) & (attention_map < spec.hi)
        tier_idx[mask] = i
    return tier_idx


def quantize_tiered(field: np.ndarray, attention_map: np.ndarray,
                     tier_specs: List[TierSpec] = None) -> QuantizedField:
    tier_specs = tier_specs or DEFAULT_TIERS
    tier_idx = assign_tiers(attention_map, tier_specs)
    flat_field = field.astype(np.float32).ravel()
    flat_tier = tier_idx.ravel()

    blocks: List[TierBlock] = []
    for i, spec in enumerate(tier_specs):
        idx = np.nonzero(flat_tier == i)[0]
        if idx.size == 0:
            blocks.append(TierBlock(spec.bitwidth, idx, 0.0, 0.0, b""))
            continue
        vals = flat_field[idx]

        if spec.bitwidth == 32:
            # Lossless passthrough: store raw float32 bytes.
            packed = vals.tobytes()
            blocks.append(TierBlock(32, idx, float(vals.min()), float(vals.max()), packed))
            continue

        vmin, vmax = float(vals.min()), float(vals.max())
        if vmax - vmin < 1e-12:
            codes = np.zeros(vals.shape, dtype=np.uint32)
        else:
            levels = (1 << spec.bitwidth) - 1
            codes = np.round((vals - vmin) / (vmax - vmin) * levels).astype(np.uint32)
            codes = np.clip(codes, 0, levels)
        packed = pack_bits_fast(codes, spec.bitwidth)
        blocks.append(TierBlock(spec.bitwidth, idx, vmin, vmax, packed))

    return QuantizedField(shape=field.shape, tiers=blocks, tier_specs=tier_specs)


def dequantize_tiered(qf: QuantizedField) -> np.ndarray:
    flat = np.zeros(int(np.prod(qf.shape)), dtype=np.float32)
    for block in qf.tiers:
        if block.indices.size == 0:
            continue
        if block.bitwidth == 32:
            vals = np.frombuffer(block.packed, dtype=np.float32, count=block.indices.size)
            flat[block.indices] = vals
            continue
        levels = (1 << block.bitwidth) - 1
        codes = unpack_bits_fast(block.packed, block.bitwidth, block.indices.size)
        if levels == 0 or block.vmax - block.vmin < 1e-12:
            vals = np.full(block.indices.size, block.vmin, dtype=np.float32)
        else:
            vals = block.vmin + (codes.astype(np.float64) / levels) * (block.vmax - block.vmin)
        flat[block.indices] = vals.astype(np.float32)
    return flat.reshape(qf.shape)


def tier_report(qf: QuantizedField) -> str:
    total = int(np.prod(qf.shape))
    lines = ["Tier allocation:"]
    for i, block in enumerate(qf.tiers):
        spec = qf.tier_specs[i]
        n = block.indices.size
        pct = 100.0 * n / total if total else 0.0
        lines.append(
            f"  tier {i}  attn[{spec.lo:.2f},{spec.hi:.2f})  "
            f"bits={block.bitwidth:>2}  n={n:>8} ({pct:5.1f}%)  "
            f"packed_bytes={len(block.packed):>9}"
        )
    return "\n".join(lines)
