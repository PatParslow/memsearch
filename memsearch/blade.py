"""k-blade region math -- memsearch's own self-contained copy of the
technique validated in the parslow-soft-editorial repo's
tools/site/concept_experiments/ga_kblade_hierarchy.py. Ported rather than
imported: the two projects are independent and shouldn't share a runtime
path dependency, and the functions themselves are tiny, pure numpy.

A "blade" here is an orthonormal basis (dim x r) spanning a genuine
r-dimensional region of embedding space, not a single direction --
`cluster_basis` builds one from a set of vectors (r = min(n-1,
MAX_BLADE_RANK, dim)), and `generalized_sine` measures the volume of new
subspace between two blades via the product of sines of their principal
angles (the k-dimensional generalization of the k=2 case, which reduces
exactly to the ordinary bivector norm / sin(angle) between two vectors).

Used by graph.py's extrapolation-gap detector to represent a cluster's
outward spread as a region (several boundary members' directions) rather
than collapsing it to one arbitrary frontier-member ray.
"""

from __future__ import annotations

import numpy as np

MAX_BLADE_RANK = 3


def cluster_basis(members: np.ndarray) -> np.ndarray:
    """Orthonormal basis (dim x r) of the dominant directions spanned by
    `members`, r = min(n_members - 1, MAX_BLADE_RANK, dim). A single
    member degenerates to its own unit vector (rank-1 blade)."""
    n_members, dim = members.shape
    if n_members == 1:
        v = members[0]
        norm = np.linalg.norm(v)
        return (v / norm if norm > 1e-9 else v).reshape(dim, 1)
    r = min(n_members - 1, MAX_BLADE_RANK, dim)
    _, _, vt = np.linalg.svd(members, full_matrices=False)
    return vt[:r].T  # (dim, r), orthonormal columns


def generalized_sine(basis_a: np.ndarray, basis_b: np.ndarray) -> float:
    """Product of sin(principal angles) between two subspaces, via the
    singular values of A^T @ B (= cos of principal angles). Reduces
    exactly to the ordinary single-vector sine when both bases are rank 1."""
    cos_angles = np.linalg.svd(basis_a.T @ basis_b, compute_uv=False)
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    sin_angles = np.sqrt(np.clip(1.0 - cos_angles ** 2, 0.0, None))
    r = min(basis_a.shape[1], basis_b.shape[1])
    return float(np.prod(sin_angles[:r]))


def sample_blade_directions(basis: np.ndarray, n_samples: int) -> np.ndarray:
    """n_samples unit vectors spanning `basis`'s column space (dim x r),
    used to explore a rank>1 blade as a region rather than a single ray.

    Rank 1: the single basis direction, `n_samples` ignored (nothing else
    to sample -- same behaviour as today's single-frontier extrapolation).

    Rank 2: `n_samples` directions evenly spaced around the 2D disc the
    basis spans (cos/sin combinations of the two basis vectors) -- a fan
    covering the whole plane the cluster is spreading into, not just its
    two defining edges.

    Rank 3: combinations of the three basis vectors' signs (up to 8,
    i.e. every vertex of the +/-1 cube in that 3D span, normalized back
    to unit length) capped at `n_samples` -- deliberately not a dense
    sphere sampling, since MAX_BLADE_RANK caps rank at 3 and a handful of
    directions already covers the octants a real spread could occupy."""
    r = basis.shape[1]
    if r == 1:
        return basis[:, :1].T

    if r == 2:
        n = max(n_samples, 1)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        dirs = np.cos(angles)[:, None] * basis[:, 0] + np.sin(angles)[:, None] * basis[:, 1]
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1e-9
        return dirs / norms

    # r == 3 (MAX_BLADE_RANK's ceiling)
    signs = np.array([[sx, sy, sz] for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)], dtype=np.float64)
    dirs = signs @ basis.T  # (8, dim)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1e-9
    dirs = dirs / norms
    return dirs[:n_samples] if n_samples < len(dirs) else dirs
