"""Meta-path / meta-graph library chi_1..chi_20 and commuting matrices - manuscript Sec. 3.3 / 3.4.

SOURCE OF TRUTH.  The node topology of every structure is taken from Fig. 3 (``fig3.png``); relation / triplet names
come from Table 2 wherever the figure's labels are inconsistent.  Conventions (all consistent with Algorithm 1):

  Q_k      binary biadjacency matrix of triplet T_k in its stored direction (source type x target type);
  C_k      = Q_k Q_k^T  for the base symmetric meta-paths chi_1..chi_11, i.e. "two nodes share a target";
  (.)      Hadamard product (parallel branches of a meta-graph that share source AND target);
  chains   outer relation matrices wrap an inner commuting matrix (``Q_TAD C Q_TAD^T``).

  chi_1  TA-T1->V<-T1-TA          C1 = Q_T1 Q_T1^T                      chi_12  C1 (.) C8
  chi_2  D -T2->V<-T2- D          C2 = Q_T2 Q_T2^T                      chi_13  C2 (.) C7
  chi_3  P -T3->V<-T3- P          C3 = Q_T3 Q_T3^T                      chi_14  C11 (.) C8
  chi_4  F -T4->V<-T4- F          C4 = Q_T4 Q_T4^T                      chi_15  Q_DP C3 Q_DP^T          (D-P-V-P-D)
  chi_5  AT-T5->V<-T5-AT          C5 = Q_T5 Q_T5^T                      chi_16  Q_TAD C2 Q_TAD^T        (TA-D-V-D-TA)
  chi_6  V -T6->V<-T6- V          C6 = Q_T6 Q_T6^T                      chi_17  Q_TAD C7 Q_TAD^T        (TA-D-P-D-TA)
  chi_7  D -T7->P<-T7- D          C7 = Q_T7 Q_T7^T                      chi_18  Q_TAM C10 Q_TAM^T       (TA-M-V-M-TA)
  chi_8  TA-T8->D<-T8-TA          C8 = Q_T8 Q_T8^T                      chi_19  Q_TAD (C2 (.) C7) Q_TAD^T  (Algorithm 1)
  chi_9  TA-T9->TA<-T9-TA         C9 = Q_T9 Q_T9^T                      chi_20  C18 (.) C19
  chi_10 M -T10->V<-T10- M        C10 = Q_T10 Q_T10^T
  chi_11 TA-T11->M<-T11-TA        C11 = Q_T11 Q_T11^T

Fig. 3 label inconsistencies resolved with Table 2 (topology is unambiguous in every case; see docs/CHI_DEFINITIONS.md):
  chi_2 right edge "T'3" -> T2;  chi_6 "R7"/"T'6" -> T6;  chi_7 "R8"/"T'7" -> T7;  chi_9 "R10"/"R'10" -> T9 (the only
  TA->TA triplet; T10 is M->V);  chi_10 node "AM" -> M (attack method).

Everything downstream (MeGiTS, per-fold adjacency, ablations, path templates) is driven by this registry.  All 20
structures are the default set, so the uniform MeGiTS weight is 1/20 (Sec. 3.4).  Instance counting = entries of the
commuting matrix (Eq. 2-3).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from .tikg import TIKG, EntityType, TA, V, M, F, AT, D, P

Step = Tuple[EntityType, str, EntityType, str]  # (from_type, relation, to_type, 'fwd'|'rev')


class RelationBlocks:
    """Lazily-built biadjacency matrices Q_Tk (Table 2, stored direction) used by the commuting-matrix expressions."""

    def __init__(self, tikg: TIKG, exclude: Optional[np.ndarray] = None):
        self.g = tikg
        self.exclude = exclude
        self._c: Dict[str, sp.csr_matrix] = {}

    def _get(self, key: str, src, rel, tgt) -> sp.csr_matrix:
        if key not in self._c:
            self._c[key] = self.g.Q(src, rel, tgt, exclude=self.exclude)
        return self._c[key]

    @property
    def T1(self): return self._get("T1", TA, "exploits", V)
    @property
    def T2(self): return self._get("T2", D, "affected_by", V)
    @property
    def T3(self): return self._get("T3", P, "affected_by", V)
    @property
    def T4(self): return self._get("T4", F, "contains", V)
    @property
    def T5(self): return self._get("T5", AT, "exploits", V)
    @property
    def T6(self): return self._get("T6", V, "evolves_to", V)
    @property
    def T7(self): return self._get("T7", D, "runs_on", P)
    @property
    def T8(self): return self._get("T8", TA, "unauthorised_access", D)       # Q_TAD
    @property
    def T9(self): return self._get("T9", TA, "assists", TA)
    @property
    def T10(self): return self._get("T10", M, "exploits", V)
    @property
    def T11(self): return self._get("T11", TA, "uses", M)                    # Q_TAM


def _gram(q: sp.spmatrix) -> sp.csr_matrix:
    return (q @ q.T).tocsr()


def _wrap(outer: sp.spmatrix, inner: sp.spmatrix) -> sp.csr_matrix:
    """Q C Q^T: outer relation matrix around an inner commuting matrix."""
    return (outer @ inner @ outer.T).tocsr()


def _had(a: sp.spmatrix, b: sp.spmatrix) -> sp.csr_matrix:
    return a.multiply(b).tocsr()


# ---- commuting-matrix builders (one per chi_k); composites reuse the base builders -----------------------------
def _c1(b): return _gram(b.T1)
def _c2(b): return _gram(b.T2)
def _c3(b): return _gram(b.T3)
def _c4(b): return _gram(b.T4)
def _c5(b): return _gram(b.T5)
def _c6(b): return _gram(b.T6)
def _c7(b): return _gram(b.T7)
def _c8(b): return _gram(b.T8)
def _c9(b): return _gram(b.T9)
def _c10(b): return _gram(b.T10)
def _c11(b): return _gram(b.T11)
def _c12(b): return _had(_c1(b), _c8(b))
def _c13(b): return _had(_c2(b), _c7(b))                    # = C_Pr of Algorithm 1
def _c14(b): return _had(_c11(b), _c8(b))
def _c15(b): return _wrap(b.T7, _c3(b))
def _c16(b): return _wrap(b.T8, _c2(b))
def _c17(b): return _wrap(b.T8, _c7(b))
def _c18(b): return _wrap(b.T11, _c10(b))
def _c19(b): return _wrap(b.T8, _c13(b))                    # Algorithm 1
def _c20(b): return _had(_c18(b), _c19(b))


def _s(a, r, b, d="fwd") -> Step:
    return (a, r, b, d)


def _sym(*hops) -> Tuple[Step, ...]:
    """Schema traversal template of a symmetric branch: hops (a, r, b) towards the centre, then mirrored back."""
    return tuple(_s(a, r, b) for a, r, b in hops) + tuple(_s(b, r, a, "rev") for a, r, b in reversed(hops))


@dataclass(frozen=True)
class StructureSpec:
    id: int
    kind: str                       # "path" (chi_1..11) | "graph" (chi_12..20)
    node_type: EntityType           # endpoint type of the symmetric structure
    schema: str                     # Fig. 3 topology
    formula: str                    # commuting-matrix formula
    builder: Callable[[RelationBlocks], sp.csr_matrix]
    basis: str = ""                 # where the definition comes from (Fig. 3 / Table 2 / Table 13 / Alg. 1)
    # schema-level traversal templates (used by attack-path tracing); one per source-to-source branch of the structure
    templates: Tuple[Tuple[Step, ...], ...] = ()
    uses: Tuple[int, ...] = ()      # Table-13 constituents (documentation / cross-check only)

    @property
    def name(self) -> str:
        return f"chi{self.id}"

    @property
    def provenance(self) -> str:
        return "manuscript_fig3"


_SH = "Fig. 3"
_T13 = "Fig. 3 (= Table 13 row)"

STRUCTURES: Tuple[StructureSpec, ...] = (
    StructureSpec(1, "path", TA, "TA -T1-> V <-T1- TA", "Q_T1 Q_T1^T", _c1, _SH, (_sym((TA, "exploits", V)),)),
    StructureSpec(2, "path", D, "D -T2-> V <-T2- D", "Q_T2 Q_T2^T", _c2,
                  _SH + " (right-edge label T'3 read as T'2) + Alg. 1 step 1", (_sym((D, "affected_by", V)),)),
    StructureSpec(3, "path", P, "P -T3-> V <-T3- P", "Q_T3 Q_T3^T", _c3, _SH, (_sym((P, "affected_by", V)),)),
    StructureSpec(4, "path", F, "F -T4-> V <-T4- F", "Q_T4 Q_T4^T", _c4, _SH, (_sym((F, "contains", V)),)),
    StructureSpec(5, "path", AT, "AT -T5-> V <-T5- AT", "Q_T5 Q_T5^T", _c5, _SH, (_sym((AT, "exploits", V)),)),
    StructureSpec(6, "path", V, "V -T6-> V <-T6- V", "Q_T6 Q_T6^T", _c6,
                  _SH + " (labels R7 / T'6 read as T6, Table 2: V evolves_to V)", (_sym((V, "evolves_to", V)),)),
    StructureSpec(7, "path", D, "D -T7-> P <-T7- D", "Q_T7 Q_T7^T", _c7,
                  _SH + " (labels R8 / T'7 read as T7) + Alg. 1 step 2", (_sym((D, "runs_on", P)),)),
    StructureSpec(8, "path", TA, "TA -T8-> D <-T8- TA", "Q_T8 Q_T8^T", _c8, _SH, (_sym((TA, "unauthorised_access", D)),)),
    StructureSpec(9, "path", TA, "TA -T9-> TA <-T9- TA", "Q_T9 Q_T9^T", _c9,
                  _SH + " (labels R10 / R'10 read as T9: the only TA->TA triplet in Table 2)", (_sym((TA, "assists", TA)),)),
    StructureSpec(10, "path", M, "M -T10-> V <-T10- M", "Q_T10 Q_T10^T", _c10,
                  _SH + " (node label AM read as M, attack method)", (_sym((M, "exploits", V)),)),
    StructureSpec(11, "path", TA, "TA -T11-> M <-T11- TA", "Q_T11 Q_T11^T", _c11, _SH, (_sym((TA, "uses", M)),)),
    StructureSpec(12, "graph", TA, "TA => {V via T1, D via T8} => TA", "C1 (.) C8", _c12, _T13,
                  (_sym((TA, "exploits", V)), _sym((TA, "unauthorised_access", D))), (1, 8)),
    StructureSpec(13, "graph", D, "D => {V via T2, P via T7} => D", "C2 (.) C7  (= C_Pr of Alg. 1)", _c13, _T13,
                  (_sym((D, "affected_by", V)), _sym((D, "runs_on", P))), (2, 7)),
    StructureSpec(14, "graph", TA, "TA => {M via T11, D via T8} => TA", "C11 (.) C8", _c14, _T13,
                  (_sym((TA, "uses", M)), _sym((TA, "unauthorised_access", D))), (8, 11)),
    StructureSpec(15, "graph", D, "D -T7-> P -T3-> V <-T3- P <-T7- D", "Q_T7 C3 Q_T7^T", _c15, _T13,
                  (_sym((D, "runs_on", P), (P, "affected_by", V)),), (3, 7)),
    StructureSpec(16, "graph", TA, "TA -T8-> D -T2-> V <-T2- D <-T8- TA", "Q_T8 C2 Q_T8^T", _c16, _T13,
                  (_sym((TA, "unauthorised_access", D), (D, "affected_by", V)),), (2, 8)),
    StructureSpec(17, "graph", TA, "TA -T8-> D -T7-> P <-T7- D <-T8- TA", "Q_T8 C7 Q_T8^T", _c17, _T13,
                  (_sym((TA, "unauthorised_access", D), (D, "runs_on", P)),), (7, 8)),
    StructureSpec(18, "graph", TA, "TA -T11-> M -T10-> V <-T10- M <-T11- TA", "Q_T11 C10 Q_T11^T", _c18, _T13,
                  (_sym((TA, "uses", M), (M, "exploits", V)),), (10, 11)),
    StructureSpec(19, "graph", TA, "TA -T8-> D => {V via T2, P via T7} => D <-T8- TA", "Q_T8 (C2 (.) C7) Q_T8^T", _c19,
                  _T13 + " + Algorithm 1",
                  (_sym((TA, "unauthorised_access", D), (D, "affected_by", V)),
                   _sym((TA, "unauthorised_access", D), (D, "runs_on", P))), (2, 7, 8)),
    StructureSpec(20, "graph", TA, "TA => {chi_18 branch via M, chi_19 branch via D} => TA", "C18 (.) C19", _c20, _T13,
                  (_sym((TA, "uses", M), (M, "exploits", V)),
                   _sym((TA, "unauthorised_access", D), (D, "affected_by", V)),
                   _sym((TA, "unauthorised_access", D), (D, "runs_on", P))), (2, 7, 8, 10, 11)),
)
N_STRUCTURES = len(STRUCTURES)
assert N_STRUCTURES == 20
_BY_ID = {s.id: s for s in STRUCTURES}


def get_structure(k: int) -> StructureSpec:
    return _BY_ID[k]


def default_structures() -> List[StructureSpec]:
    """The manuscript's full set chi_1..chi_20 (uniform MeGiTS weight w_k = 1/20)."""
    return list(STRUCTURES)


def structure_templates(k: int) -> List[Tuple[Step, ...]]:
    """Schema traversal templates of chi_k (source-to-source branches drawn in Fig. 3)."""
    return list(_BY_ID[k].templates)


def select_structures(kinds: Optional[Sequence[str]] = None, ids: Optional[Sequence[int]] = None) -> List[StructureSpec]:
    out = [_BY_ID[i] for i in sorted(set(ids))] if ids is not None else default_structures()
    if kinds is not None:
        out = [s for s in out if s.kind in kinds]
    return out


def commuting_matrices(tikg: TIKG, structures: Optional[Sequence[StructureSpec]] = None,
                       exclude: Optional[np.ndarray] = None) -> Dict[int, sp.csr_matrix]:
    """Commuting matrices C_chi_k (type-local, |nodes of type| square) - one separate matrix per structure.
    Default: all 20 structures."""
    structures = list(structures) if structures is not None else default_structures()
    blocks = RelationBlocks(tikg, exclude=exclude)
    return {s.id: s.builder(blocks).astype(np.float32).tocsr() for s in structures}


def global_matrix(tikg: TIKG, spec: StructureSpec, C: sp.spmatrix) -> sp.csr_matrix:
    """Embed a type-local commuting matrix into the N x N node index space (valid node mapping)."""
    idx = tikg.type_nodes[spec.node_type]
    coo = sp.coo_matrix(C)
    return sp.csr_matrix((coo.data, (idx[coo.row], idx[coo.col])), shape=(tikg.n, tikg.n))


def structure_report(tikg: TIKG, exclude: Optional[np.ndarray] = None) -> List[dict]:
    """Per-structure shape / non-zero counts."""
    rows = []
    for s in STRUCTURES:
        C = commuting_matrices(tikg, [s], exclude)[s.id]
        off = C - sp.diags(C.diagonal())
        off.eliminate_zeros()
        rows.append({"id": s.id, "kind": s.kind, "schema": s.schema, "formula": s.formula,
                     "node_type": s.node_type.value, "shape": tuple(C.shape), "nnz": int(C.nnz),
                     "offdiag_nnz": int(off.nnz)})
    return rows
