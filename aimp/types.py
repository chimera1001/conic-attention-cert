"""Core data types: the head, the token sets, the certification instance and the solver result."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

SetKind = Literal["box", "ball", "ellipsoid", "hull"]


@dataclass
class HeadSpec:
    """One attention head in bilinear form: ``s_j = z_i^T M z_j``, ``y = sum_j a_j N z_j``."""

    M: np.ndarray
    N: np.ndarray
    d: int
    d_k: int
    d_v: int
    raw: Optional[dict] = None  # W_Q, W_K, W_V, W_O if available
    name: str = "head"

    @staticmethod
    def from_weights(W_Q, W_K, W_V, W_O, name: str = "head") -> "HeadSpec":
        """``M = W_Q W_K^T / sqrt(d_k)``, ``N = (W_V W_O)^T`` (W_Q, W_K: d x d_k; W_V: d x d_v; W_O: d_v x d)."""
        d, d_k = W_Q.shape
        d_v = W_V.shape[1]
        M = W_Q @ W_K.T / np.sqrt(d_k)
        N = (W_V @ W_O).T
        raw = {"W_Q": W_Q, "W_K": W_K, "W_V": W_V, "W_O": W_O}
        return HeadSpec(M=M, N=N, d=d, d_k=d_k, d_v=d_v, raw=raw, name=name)


@dataclass
class SetSpec:
    """Input set of one token.

    box: ``l <= z <= u``; ball: ``||z - center|| <= rho``;
    ellipsoid: ``{center + E w : ||w|| <= rho}``; hull: ``conv(points)`` (rows of ``points``).
    """

    kind: SetKind
    center: np.ndarray
    rho: Optional[float] = None
    l: Optional[np.ndarray] = None
    u: Optional[np.ndarray] = None
    E: Optional[np.ndarray] = None
    points: Optional[np.ndarray] = None


@dataclass
class Instance:
    """One head at one query position.

    Tokens are numbered ``0..T-1``; ``sets[t]`` is token ``t``'s set. ``keys`` is the key set J
    (list of token ids, order = key order j=0..n-1); ``query`` is the query token id (it may be
    one of the keys — self-attention — in which case query and key share one variable).
    """

    head: HeadSpec
    keys: list[int]
    query: int
    sets: dict[int, SetSpec]
    c: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.keys)

    @property
    def d(self) -> int:
        return self.head.d

    @property
    def T(self) -> int:
        return len(self.sets)

    @property
    def query_in_keys(self) -> bool:
        return self.query in self.keys

    @property
    def tokens(self) -> list[int]:
        """Tokens that enter the head: the query first, then keys not equal to the query."""
        return [self.query] + [t for t in self.keys if t != self.query]

    def centers(self) -> np.ndarray:
        """(T x d) array of set centers."""
        return np.stack([self.sets[t].center for t in range(self.T)])


@dataclass
class SolveResult:
    method: str
    status: str  # optimal | time_limit | suboptimal | infeasible | error
    value: float  # r_X (for exact: p_star_ub = ObjBound)
    value_lb: Optional[float] = None  # exact: incumbent objective
    value_lb_verified: Optional[float] = None  # exact: numpy re-evaluation at incumbent
    recovered_point: Optional[np.ndarray] = None  # (T x d) feasible point
    recovered_value: Optional[float] = None  # true head at recovered point (valid LB on p_star)
    diagnostics: dict = field(default_factory=dict)
    approximations: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    wall_time_s: float = 0.0
    cpu_time_s: Optional[float] = None  # process CPU time of the solve (all solver threads)
