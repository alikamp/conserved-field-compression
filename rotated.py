"""
Per-block component-frame rotation (the "rotating frame" idea, vector-space
version).

Within each spatial block we compute the 3x3 second-moment matrix of the
velocity vectors and rotate every vector into that block's principal-axis
frame. Because velocity is locally coherent, this piles most of the block's
energy into component 0; components 1 and 2 collapse toward zero and become
highly compressible.

Key property: the rotation is ORTHOGONAL, so it preserves sum-of-squares
(energy) exactly -- it composes cleanly with the conservation guarantee and
needs no spatial resampling (unlike rotating the grid itself).

We store one rotation per block. A 3x3 is redundant for an orthonormal
matrix; a quaternion (4 floats) suffices, so overhead is ~16 bytes/block.
"""
from __future__ import annotations
import numpy as np

from blocked import _pad_to_blocks, _block_view, quantize_blocked, dequantize_blocked
from backend import compress_zstd


def _unblock(blocks: np.ndarray, B: int) -> np.ndarray:
    """Inverse of _block_view: (nb0,nb1,nb2,B,B,B) -> (nb0*B,nb1*B,nb2*B)."""
    nb0, nb1, nb2 = blocks.shape[:3]
    return (blocks.transpose(0, 3, 1, 4, 2, 5)
                  .reshape(nb0 * B, nb1 * B, nb2 * B))


def _make_proper(R):
    """Ensure each 3x3 in (N,3,3) is a proper rotation (det +1) by flipping
    the last principal axis where det < 0. Done BEFORE the vectors are
    rotated, so the stored quaternion matches the transform exactly."""
    R = R.copy()
    det = np.linalg.det(R)
    R[det < 0, :, 2] *= -1.0
    return R


def _mat_to_quat(R):
    """Batch (N,3,3) proper rotations -> quaternions (N,4), scalar-first.
    Uses scipy's numerically robust converter (handles the 180-deg case)."""
    from scipy.spatial.transform import Rotation
    q_xyzw = Rotation.from_matrix(R).as_quat()      # (N,4) x,y,z,w
    q = np.roll(q_xyzw, 1, axis=1)                   # -> w,x,y,z
    return q.astype(np.float32)


def _quat_to_mat(q):
    """Batch quaternions (N,4) scalar-first -> (N,3,3) rotations."""
    from scipy.spatial.transform import Rotation
    q_xyzw = np.roll(q.astype(np.float64), -1, axis=1)  # w,x,y,z -> x,y,z,w
    return Rotation.from_quat(q_xyzw).as_matrix()


def block_pca_rotate(u, v, w, B):
    """Rotate each block's vectors into local principal axes.
    Returns (u',v',w', quats, grid_meta)."""
    upad, _ = _pad_to_blocks(u.astype(np.float32), B)
    vpad, _ = _pad_to_blocks(v.astype(np.float32), B)
    wpad, _ = _pad_to_blocks(w.astype(np.float32), B)

    ub, (nb0, nb1, nb2) = _block_view(upad, B)
    vb, _ = _block_view(vpad, B)
    wb, _ = _block_view(wpad, B)
    Nb = nb0 * nb1 * nb2
    m = B ** 3

    U = ub.reshape(Nb, m); V = vb.reshape(Nb, m); W = wb.reshape(Nb, m)
    Vec = np.stack([U, V, W], axis=2)            # (Nb, m, 3)

    M = np.einsum('nmi,nmj->nij', Vec, Vec)       # (Nb,3,3) second-moment
    evals, evecs = np.linalg.eigh(M)              # ascending
    R = evecs[:, :, ::-1]                          # columns = principal axes, desc
    R = _make_proper(R)                            # det +1 so quat is exact

    Vrot = np.einsum('nmi,nij->nmj', Vec, R)      # (Nb, m, 3) in principal frame
    quats = _mat_to_quat(R)

    up = _unblock(Vrot[:, :, 0].reshape(nb0, nb1, nb2, B, B, B), B)
    vp = _unblock(Vrot[:, :, 1].reshape(nb0, nb1, nb2, B, B, B), B)
    wp = _unblock(Vrot[:, :, 2].reshape(nb0, nb1, nb2, B, B, B), B)

    meta = (nb0, nb1, nb2, u.shape)
    # crop rotated comps back to original shape
    sl = tuple(slice(0, s) for s in u.shape)
    return up[sl], vp[sl], wp[sl], quats, meta


def block_pca_unrotate(up, vp, wp, quats, meta, B):
    nb0, nb1, nb2, shape = meta
    R = _quat_to_mat(quats)                        # (Nb,3,3)
    upad, _ = _pad_to_blocks(up.astype(np.float32), B)
    vpad, _ = _pad_to_blocks(vp.astype(np.float32), B)
    wpad, _ = _pad_to_blocks(wp.astype(np.float32), B)
    ub, _ = _block_view(upad, B); vb, _ = _block_view(vpad, B); wb, _ = _block_view(wpad, B)
    Nb = nb0 * nb1 * nb2; m = B ** 3
    Vrot = np.stack([ub.reshape(Nb, m), vb.reshape(Nb, m), wb.reshape(Nb, m)], axis=2)
    # inverse rotation: Vec = Vrot @ R^T
    Vec = np.einsum('nmj,nij->nmi', Vrot, R)
    u = _unblock(Vec[:, :, 0].reshape(nb0, nb1, nb2, B, B, B), B)
    v = _unblock(Vec[:, :, 1].reshape(nb0, nb1, nb2, B, B, B), B)
    w = _unblock(Vec[:, :, 2].reshape(nb0, nb1, nb2, B, B, B), B)
    sl = tuple(slice(0, s) for s in shape)
    return u[sl], v[sl], w[sl]
