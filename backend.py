"""
Backend entropy coding: serialize the multi-precision tiered bitstream into
a single container and run it through zstandard (ANS-family entropy coder),
compared against gzip/zlib and plain zstd-on-raw-float baselines.
"""
from __future__ import annotations
import struct
import zlib
import numpy as np
import zstandard as zstd

from bitpack import QuantizedField, TierBlock


MAGIC = b"VFC1"  # Viewer-First Compression, v1


def serialize_quantized_field(qf: QuantizedField) -> bytes:
    """Pack the tiered, multi-precision representation into one bitstream."""
    parts = [MAGIC]
    parts.append(struct.pack("<B", len(qf.shape)))
    for dim in qf.shape:
        parts.append(struct.pack("<Q", dim))
    parts.append(struct.pack("<B", len(qf.tiers)))

    for spec, block in zip(qf.tier_specs, qf.tiers):
        parts.append(struct.pack("<ffB", spec.lo, spec.hi, block.bitwidth))
        parts.append(struct.pack("<Qdd", block.indices.size, block.vmin, block.vmax))
        # indices: delta + varint-ish via plain uint32 (fine for prototype scale)
        idx = block.indices.astype(np.uint32).tobytes()
        parts.append(struct.pack("<Q", len(idx)))
        parts.append(idx)
        parts.append(struct.pack("<Q", len(block.packed)))
        parts.append(block.packed)

    return b"".join(parts)


def deserialize_quantized_field(buf: bytes):
    from bitpack import TierSpec  # local import to avoid cycle at module load
    off = 0
    assert buf[:4] == MAGIC
    off += 4
    ndim = buf[off]; off += 1
    shape = []
    for _ in range(ndim):
        (dim,) = struct.unpack_from("<Q", buf, off); off += 8
        shape.append(dim)
    n_tiers = buf[off]; off += 1

    tier_specs = []
    tiers = []
    for _ in range(n_tiers):
        lo, hi, bitwidth = struct.unpack_from("<ffB", buf, off); off += 9
        n, vmin, vmax = struct.unpack_from("<Qdd", buf, off); off += 24
        (idx_len,) = struct.unpack_from("<Q", buf, off); off += 8
        idx_bytes = buf[off:off + idx_len]; off += idx_len
        indices = np.frombuffer(idx_bytes, dtype=np.uint32).astype(np.int64)
        (packed_len,) = struct.unpack_from("<Q", buf, off); off += 8
        packed = buf[off:off + packed_len]; off += packed_len

        tier_specs.append(TierSpec(lo, hi, bitwidth))
        tiers.append(TierBlock(bitwidth, indices, vmin, vmax, packed))

    return QuantizedField(shape=tuple(shape), tiers=tiers, tier_specs=tier_specs)


def compress_zstd(data: bytes, level: int = 19) -> bytes:
    cctx = zstd.ZstdCompressor(level=level)
    return cctx.compress(data)


def compress_gzip(data: bytes, level: int = 9) -> bytes:
    return zlib.compress(data, level)


def evaluate_compression(original_field: np.ndarray, qf: QuantizedField,
                          zstd_level: int = 19) -> dict:
    raw_bytes = original_field.astype(np.float32).tobytes()

    vfc_stream = serialize_quantized_field(qf)
    vfc_zstd = compress_zstd(vfc_stream, level=zstd_level)

    baseline_zstd_on_raw = compress_zstd(raw_bytes, level=zstd_level)
    baseline_gzip_on_raw = compress_gzip(raw_bytes)

    return {
        "raw_bytes": len(raw_bytes),
        "vfc_stream_bytes": len(vfc_stream),
        "vfc_stream_plus_zstd_bytes": len(vfc_zstd),
        "baseline_zstd_on_raw_bytes": len(baseline_zstd_on_raw),
        "baseline_gzip_on_raw_bytes": len(baseline_gzip_on_raw),
        "ratio_vfc_vs_raw": len(raw_bytes) / len(vfc_zstd),
        "ratio_baseline_zstd_vs_raw": len(raw_bytes) / len(baseline_zstd_on_raw),
        "ratio_baseline_gzip_vs_raw": len(raw_bytes) / len(baseline_gzip_on_raw),
        "vfc_vs_baseline_zstd_improvement": len(baseline_zstd_on_raw) / len(vfc_zstd),
    }
