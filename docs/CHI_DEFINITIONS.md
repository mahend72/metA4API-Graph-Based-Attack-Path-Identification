# chi_1..chi_20 - definitions implemented (source: Fig. 3)

Source of truth: the node topology of `fig3.png`; relation names from Table 2 where the figure's labels are inconsistent.
Registry: `iocevaluator/metagraphs.py`. Notation: `Q_k` = biadjacency matrix of triplet `T_k` in its stored direction;
`C_k = Q_k Q_k^T` (two nodes sharing a target); `(.)` = Hadamard product; `Q C Q^T` wraps an inner commuting matrix.

| chi | kind | endpoints | Fig. 3 topology | commuting matrix |
|---|---|---|---|---|
| 1 | path | TA | TA -T1-> V <-T1- TA | `Q_T1 Q_T1^T` |
| 2 | path | D | D -T2-> V <-T2- D | `Q_T2 Q_T2^T` (Alg. 1 step 1) |
| 3 | path | P | P -T3-> V <-T3- P | `Q_T3 Q_T3^T` |
| 4 | path | F | F -T4-> V <-T4- F | `Q_T4 Q_T4^T` |
| 5 | path | AT | AT -T5-> V <-T5- AT | `Q_T5 Q_T5^T` |
| 6 | path | V | V -T6-> V <-T6- V | `Q_T6 Q_T6^T` |
| 7 | path | D | D -T7-> P <-T7- D | `Q_T7 Q_T7^T` (Alg. 1 step 2) |
| 8 | path | TA | TA -T8-> D <-T8- TA | `Q_T8 Q_T8^T` |
| 9 | path | TA | TA -T9-> TA <-T9- TA | `Q_T9 Q_T9^T` |
| 10 | path | M | M -T10-> V <-T10- M | `Q_T10 Q_T10^T` |
| 11 | path | TA | TA -T11-> M <-T11- TA | `Q_T11 Q_T11^T` |
| 12 | graph | TA | TA => {V via T1, D via T8} => TA | `C1 (.) C8` |
| 13 | graph | D | D => {V via T2, P via T7} => D | `C2 (.) C7` (= `C_Pr`, Alg. 1 step 3) |
| 14 | graph | TA | TA => {M via T11, D via T8} => TA | `C11 (.) C8` |
| 15 | graph | D | D -T7-> P -T3-> V <-T3- P <-T7- D | `Q_T7 C3 Q_T7^T` |
| 16 | graph | TA | TA -T8-> D -T2-> V <-T2- D <-T8- TA | `Q_T8 C2 Q_T8^T` |
| 17 | graph | TA | TA -T8-> D -T7-> P <-T7- D <-T8- TA | `Q_T8 C7 Q_T8^T` |
| 18 | graph | TA | TA -T11-> M -T10-> V <-T10- M <-T11- TA | `Q_T11 C10 Q_T11^T` |
| 19 | graph | TA | TA -T8-> D => {V via T2, P via T7} => D <-T8- TA | `Q_T8 (C2 (.) C7) Q_T8^T` (Alg. 1) |
| 20 | graph | TA | TA => {chi_18 branch, chi_19 branch} => TA | `C18 (.) C19` |

chi_12, 14, 15 were derived from the figure (the Hadamard diamonds of chi_12 / chi_14 share both end nodes; chi_15 is the
linear chain D-P-V-P-D). chi_20 uses two distinct V nodes (one per branch), so the branches are independent and the
instance count is the product `C18[a,b] * C19[a,b]`. chi_15..chi_18 are drawn as linear chains in Fig. 3; they are kept in the
"graph" group (chi_12..chi_20) as in Table 13.

All 20 structures are the default MeGiTS set with uniform `w_k = 1/20` (Sec. 3.4). There is no provisional / unresolved
status any more. No structure uses the non-Table-2 triplet X1 = `<TA, uses, F>`.

## Fig. 3 vs Table 2 notation inconsistencies (resolved with Table 2; topology is unambiguous)
1. chi_2: right edge labelled `T'3`; the left edge and the node types (D, V) give T2.
2. chi_6: edges `R7` / `T'6`; Table 2 T6 = `<V, evolves_to, V>` matches the V->V<-V topology.
3. chi_7: edges `R8` / `T'7`; Table 2 T7 = `<D, runs_on, P>`.
4. chi_9: edges `R10` / `R'10`; no `R*` relations exist in Table 2, and T9 = `<TA, assists, TA>` is the only TA->TA triplet
   (T10 is M->V). This is the least certain of the four: it relies on node types alone.
5. chi_10: node `AM` read as M (attack method, Table 1); the legend uses M.

## Remaining conflicts
No conflict between Fig. 3 topology and Table 2 triplet signatures remains. Conflicts that are *manuscript text* vs
figure/Table 13 (not changed here, `manuscript.tex` is untouched):
- Sec. 3.3 prose describes chi_13 (actors sharing hashes / e-mails) and chi_18 (actors, domains, IPs, e-mails); the figure
  and Table 13 give chi_13 = D=>{V,P}=>D and chi_18 = TA-M-V-M-TA.
- Table 13 lists chi_8 as a constituent of chi_19, while Algorithm 1 uses only `Q_TAD` (as the outer wrapper of chi_19, as
  in chi_16 / chi_17). Fig. 3 agrees with Algorithm 1.
- The Sec. 3.4 text calls chi_19 a meta-path in Algorithm 1's output line; it is a meta-graph.
- Table 10's "Full metA4API" row equals Table 13's chi_19 row (0.7551 / 0.7696) although the full model is stated to use all 20.
