//! Cover-Tree Dual-Tree Traversal
//!
//! Implements dual-tree traversal for cover trees using a recursive descent
//! approach. Both query and reference trees are traversed depth-first, with
//! subtree pairs pruned when no useful neighbors can exist.
//!
//! # Distance-Sorted Traversal
//!
//! Children are sorted by distance before recursion, following the same
//! principle as Algorithm 4 (Gray 2003) from Curtin et al. Visiting closer
//! subtrees first tightens pruning bounds faster, enabling more aggressive
//! pruning of farther subtrees. This mirrors the single-tree k-NN approach
//! in `knn_impl.rs`.
//!
//! # Cover Trees Without Self-Children
//!
//! Our cover tree does NOT have explicit self-children. When expanding one side,
//! the held-constant node's point must still reach all descendants of the
//! expanded side. We achieve this by also recursing the held-constant node's
//! point against the expanded side's children (via `point_vs_subtree` and
//! `subtree_vs_point`) to simulate the missing self-child.
//!
//! # Parent Bound Propagation
//!
//! Following Curtin et al., the pruning bound for a query node includes
//! the parent's bound: B(Nq) = min(B1, B2, B(Par(Nq))). The parent bound
//! is always valid for children (a child's query region is a subset of the
//! parent's), providing "free" tightening of bounds.

#[cfg(not(feature = "no-smallvec"))]
use smallvec::SmallVec;

use crate::distance::Distance;
use crate::node::Node;
use super::knn_rules::KnnRules;

/// Inline capacity for sorted child arrays. Cover trees at base 1.3 typically have
/// 3-5 children; 8 handles the common case without heap allocation while keeping
/// per-recursion stack frames small for better cache locality.
#[cfg(not(feature = "no-smallvec"))]
type ScoredVec = SmallVec<[(f64, usize); 8]>;
#[cfg(feature = "no-smallvec")]
type ScoredVec = Vec<(f64, usize)>;

/// Inline capacity for child-child pair arrays in the both-internal case.
/// With max ~5 children per side at base 1.3, up to 25 pairs typical.
#[cfg(not(feature = "no-smallvec"))]
type PairVec = SmallVec<[(f64, usize, usize); 16]>;
#[cfg(feature = "no-smallvec")]
type PairVec = Vec<(f64, usize, usize)>;

/// Inline capacity for per-child distance arrays used in triangle inequality
/// pre-pruning. Stores d(qp, rc_j) or d(qc_i, rp) values indexed by child index.
#[cfg(not(feature = "no-smallvec"))]
type DistVec = SmallVec<[f64; 8]>;
#[cfg(feature = "no-smallvec")]
type DistVec = Vec<f64>;

/// Inline capacity for index arrays (child indices, etc.).
#[cfg(not(feature = "no-smallvec"))]
type IdxVec = SmallVec<[usize; 8]>;
#[cfg(feature = "no-smallvec")]
type IdxVec = Vec<usize>;

/// Inline capacity for pre-computed reference-child data arrays.
#[cfg(not(feature = "no-smallvec"))]
type RefDataVec = SmallVec<[RefChildData; 8]>;
#[cfg(feature = "no-smallvec")]
type RefDataVec = Vec<RefChildData>;

/// Inline capacity for pre-computed query-child data arrays.
#[cfg(not(feature = "no-smallvec"))]
type QueryDataVec = SmallVec<[QueryChildData; 8]>;
#[cfg(feature = "no-smallvec")]
type QueryDataVec = Vec<QueryChildData>;

/// Halfsort: O(n) quickselect partition + O(n/2 log n/2) sort of the closer half.
/// For small arrays (<= 8 elements), falls back to full sort since the overhead
/// of select_nth_unstable_by isn't worthwhile. The farther half is still visited
/// but in arbitrary order — the closer half tightens bounds first, which is what
/// matters for pruning.
///
/// With `no-halfsort` feature, always uses full sort for A/B benchmarking.
#[inline]
fn halfsort<T>(slice: &mut [T], cmp: impl Fn(&T, &T) -> std::cmp::Ordering) {
    #[cfg(feature = "no-halfsort")]
    {
        slice.sort_unstable_by(cmp);
    }
    #[cfg(not(feature = "no-halfsort"))]
    {
        if slice.len() <= 8 {
            slice.sort_unstable_by(cmp);
        } else {
            let mid = slice.len() / 2;
            slice.select_nth_unstable_by(mid, &cmp);
            slice[..mid].sort_unstable_by(cmp);
        }
    }
}

/// Pre-computed per-reference-child data, collected during Phase 2.
/// Consolidates two parallel arrays into one for better locality.
/// (Unlike PackedDualTree, unpacked traversal accesses rc.maxdist directly.)
#[derive(Clone, Copy)]
struct RefChildData {
    /// d(query point, this ref child) if `qp_rc_exact`; otherwise only a lower bound
    /// (the early-exit threshold the true distance exceeded).
    d_qp_rc: f64,
    qp_rc_exact: bool,
    d_rp_rc: f64,    // distance from ref parent to this ref child (d_parent)
}

/// Pre-computed per-query-child data, collected during Phase 3.
/// Consolidates two parallel arrays into one for better locality.
#[derive(Clone, Copy)]
struct QueryChildData {
    /// d(this query child, ref parent) if `qc_rp_exact`; otherwise only a lower bound.
    d_qc_rp: f64,
    qc_rp_exact: bool,
    d_qp_qc: f64,   // distance from query parent to this query child (d_parent)
}

/// Dual-tree traversal for cover trees.
pub(crate) struct DualTreeTraversal;

impl DualTreeTraversal {
    /// Run the dual-tree traversal.
    pub fn traverse<'a, T: Clone, D: Distance<T>>(
        query_root: &Node<T>,
        ref_root: &'a Node<T>,
        rules: &mut KnnRules<'a, T, D>,
    ) {
        // Root is always index 0 (assigned first during DFS init_parent_map)
        let q_idx = 0;
        Self::dual_recurse(query_root, q_idx, ref_root, f64::INFINITY, None, rules);
    }

    /// Recursive dual-tree descent.
    ///
    /// Children/pairs are sorted by distance before recursion so that closer
    /// subtrees are visited first, tightening pruning bounds faster.
    ///
    /// # Arguments
    ///
    /// * `parent_bound` - The pruning bound from the parent query node,
    ///   propagated per Curtin et al. B(Nq) = min(B1, B2, B(Par(Nq))).
    /// * `precomputed_dist` - If Some, the distance d(query_node.point, ref_node.point)
    ///   was already computed by the caller (e.g., during sorting). Avoids redundant
    ///   metric evaluation.
    fn dual_recurse<'a, T: Clone, D: Distance<T>>(
        query_node: &Node<T>,
        q_idx: usize,
        ref_node: &'a Node<T>,
        parent_bound: f64,
        precomputed_dist: Option<f64>,
        rules: &mut KnnRules<'a, T, D>,
    ) {
        let _guard = crate::core::utils::StackGuard::enter();
        // Step 1: Base case using pre-resolved index (no HashMap)
        let bound = rules.state.bound_with_idx(q_idx).min(parent_bound);
        let dist = match precomputed_dist {
            Some(d) => {
                rules.base_case_by_idx(&query_node.point, q_idx, &ref_node.point, ref_node.is_duplicate, d);
                d
            }
            None => {
                let ub = bound + query_node.maxdist + ref_node.maxdist;
                let d = rules.metric.distance_with_bound(&query_node.point, &ref_node.point, ub);
                // An early exit returns some value > ub, not the distance; only an
                // exact distance may be recorded. A pair with d > ub is pruned below.
                if d <= ub {
                    rules.base_case_by_idx(&query_node.point, q_idx, &ref_node.point, ref_node.is_duplicate, d);
                }
                d
            }
        };

        // Step 2: Pruning check using pre-resolved index (no HashMap)
        let dmin = dist - query_node.maxdist - ref_node.maxdist;
        if dmin > bound {
            return;
        }

        let q_is_leaf = query_node.children.is_empty();
        let r_is_leaf = ref_node.children.is_empty();

        match (q_is_leaf, r_is_leaf) {
            (true, true) => {
                // Both leaves — base case already computed
            }
            (true, false) => {
                // Query is leaf, expand reference children sorted by distance.
                // `bound` from above is still valid (base case only touched
                // the query_node.point vs ref_node.point pair, not query children).
                let mut scored: ScoredVec = ScoredVec::new();
                for (i, rc) in ref_node.children.iter().enumerate() {
                    // Self-child optimization: reuse parent distance when d_parent == 0.0
                    let d = if rc.d_parent == 0.0 {
                        dist
                    } else {
                        // Shell pre-filter: triangle inequality lower bound
                        if cfg!(not(feature = "no-triangle-filter")) {
                            let shell_lb = (dist - rc.d_parent).max(0.0);
                            if shell_lb - query_node.maxdist - rc.maxdist > bound {
                                continue;
                            }
                        }
                        let ub = bound + query_node.maxdist + rc.maxdist;
                        rules.metric.distance_with_bound(&query_node.point, &rc.point, ub)
                    };
                    let child_dmin = d - query_node.maxdist - rc.maxdist;
                    if child_dmin <= bound {
                        scored.push((d, i));
                    }
                }

                halfsort(&mut scored, |a, b| a.0.total_cmp(&b.0));

                for (d, idx) in scored {
                    let rc = &ref_node.children[idx];
                    let child_dmin = d - query_node.maxdist - rc.maxdist;
                    let current_bound = rules.state.bound_with_idx(q_idx).min(parent_bound);
                    if child_dmin > current_bound {
                        continue;
                    }
                    Self::dual_recurse(query_node, q_idx, rc, parent_bound, Some(d), rules);
                }
            }
            (false, true) => {
                // Reference is leaf, expand query children sorted by distance.
                // Build children indices and bounds without cloning.
                // Reading individual elements via child_at releases the borrow
                // between iterations, allowing mutable bound_with_idx calls.
                let num_qc = rules.state.num_children(q_idx);
                let mut qc_idxs: IdxVec = IdxVec::with_capacity(num_qc);
                let mut qc_bounds: DistVec = DistVec::with_capacity(num_qc);
                for ci in 0..num_qc {
                    let child_idx = rules.state.child_at(q_idx, ci);
                    qc_idxs.push(child_idx);
                    qc_bounds.push(rules.state.bound_with_idx(child_idx).min(bound));
                }
                let mut scored: ScoredVec = ScoredVec::new();
                for (i, qc) in query_node.children.iter().enumerate() {
                    // Self-child optimization: reuse parent distance when d_parent == 0.0
                    let d = if qc.d_parent == 0.0 {
                        dist
                    } else {
                        // Shell pre-filter: triangle inequality lower bound
                        if cfg!(not(feature = "no-triangle-filter")) {
                            let shell_lb = (dist - qc.d_parent).max(0.0);
                            if shell_lb - qc.maxdist - ref_node.maxdist > qc_bounds[i] {
                                continue;
                            }
                        }
                        let ub = qc_bounds[i] + qc.maxdist + ref_node.maxdist;
                        rules.metric.distance_with_bound(&qc.point, &ref_node.point, ub)
                    };
                    let child_dmin = d - qc.maxdist - ref_node.maxdist;
                    if child_dmin <= qc_bounds[i] {
                        scored.push((d, i));
                    }
                }

                halfsort(&mut scored, |a, b| a.0.total_cmp(&b.0));

                for (d, idx) in scored {
                    let qc = &query_node.children[idx];
                    let child_dmin = d - qc.maxdist - ref_node.maxdist;
                    // Re-read bound: previous sibling's recursion may have tightened it
                    let child_bound = rules.state.bound_with_idx(qc_idxs[idx]).min(bound);
                    if child_dmin > child_bound {
                        continue;
                    }
                    Self::dual_recurse(qc, qc_idxs[idx], ref_node, bound, Some(d), rules);
                }
            }
            (false, false) => {
                // Both have children — three phases, reordered so that
                // Phases 2 & 3 run first. Their distances feed triangle-
                // inequality lower bounds that let Phase 1 skip expensive
                // d(qc, rc) computations.
                let q_children = &query_node.children;
                let r_children = &ref_node.children;

                // Build children indices without cloning.
                let num_qc = rules.state.num_children(q_idx);
                let mut qc_idxs: IdxVec = IdxVec::with_capacity(num_qc);
                for ci in 0..num_qc {
                    qc_idxs.push(rules.state.child_at(q_idx, ci));
                }

                // --- Phase 2: query_node.point vs ref children ---
                // Computes d(qp, rc_j) for each ref child, used later for
                // triangle-inequality pre-pruning in Phase 1.
                // q_idx already resolved above — use directly for kth_distance_by_idx.
                let mut ref_data: RefDataVec = RefDataVec::new();
                let mut ref_scored: ScoredVec = ScoredVec::new();
                let qp_kth = rules.kth_distance_by_idx(q_idx);
                for (i, rc) in r_children.iter().enumerate() {
                    // Self-child optimization: reuse parent distance when d_parent == 0.0
                    // (`dist` is exact: an early-exited pair would have been pruned).
                    let (d, exact, lower) = if rc.d_parent == 0.0 {
                        (dist, true, dist) // d(qp, rp) == d(qp, rc) for self-child
                    } else {
                        let ub = qp_kth + rc.maxdist;
                        let d = rules.metric.distance_with_bound(&query_node.point, &rc.point, ub);
                        (d, d <= ub, if d <= ub { d } else { ub })
                    };
                    ref_data.push(RefChildData {
                        d_qp_rc: lower,
                        qp_rc_exact: exact,
                        d_rp_rc: rc.d_parent,
                    });
                    let child_dmin = d - rc.maxdist;
                    if child_dmin <= qp_kth {
                        ref_scored.push((d, i));
                    }
                }

                halfsort(&mut ref_scored, |a, b| a.0.total_cmp(&b.0));

                for (d, idx) in ref_scored {
                    let rc = &r_children[idx];
                    let child_dmin = d - rc.maxdist;
                    let current_kth = rules.kth_distance_by_idx(q_idx);
                    if child_dmin > current_kth {
                        continue;
                    }
                    Self::point_vs_subtree(&query_node.point, q_idx, rc, Some(d), rules);
                }

                // --- Phase 3: query children vs ref_node.point ---
                // Computes d(qc_i, rp) for each query child, used later for
                // triangle-inequality pre-pruning in Phase 1.
                //
                // Pre-compute bounds using pre-resolved child indices.
                let mut qc_bounds: DistVec = qc_idxs.iter()
                    .map(|&ci| rules.state.bound_with_idx(ci).min(bound))
                    .collect();

                let mut query_data: QueryDataVec = QueryDataVec::new();
                let mut query_scored: ScoredVec = ScoredVec::new();
                for (i, qc) in q_children.iter().enumerate() {
                    // Self-child optimization: reuse parent distance when d_parent == 0.0
                    let (d, exact, lower) = if qc.d_parent == 0.0 {
                        (dist, true, dist) // d(qp, rp) == d(qc, rp) for self-child
                    } else {
                        let ub = qc_bounds[i] + qc.maxdist;
                        let d = rules.metric.distance_with_bound(&qc.point, &ref_node.point, ub);
                        (d, d <= ub, if d <= ub { d } else { ub })
                    };
                    query_data.push(QueryChildData {
                        d_qc_rp: lower,
                        qc_rp_exact: exact,
                        d_qp_qc: qc.d_parent,
                    });
                    let child_dmin = d - qc.maxdist;
                    if child_dmin <= qc_bounds[i] {
                        query_scored.push((d, i));
                    }
                }

                halfsort(&mut query_scored, |a, b| a.0.total_cmp(&b.0));

                for (d, idx) in query_scored {
                    let qc = &q_children[idx];
                    let child_dmin = d - qc.maxdist;
                    // Re-read bound: Phase 2 recursion may have tightened it
                    let child_bound = rules.state.bound_with_idx(qc_idxs[idx]).min(bound);
                    if child_dmin > child_bound {
                        continue;
                    }
                    Self::subtree_vs_point(qc, qc_idxs[idx], &ref_node.point, ref_node.is_duplicate, bound, Some(d), rules);
                }

                // --- Phase 1: child-child pairs with triangle-inequality pre-pruning ---
                // ref_data was collected during Phase 2 loop above.
                // query_data was collected during Phase 3 loop above.
                // Two valid lower bounds on d(qc_i, rc_j):
                //   lb1 = |d(qp, qc_i) - d(qp, rc_j)|   (pivot = qp)
                //   lb2 = |d(rp, qc_i) - d(rp, rc_j)|   (pivot = rp)
                // Take max(lb1, lb2) for the tightest cheap lower bound.

                // Refresh query child bounds after Phase 2/3 recursion
                for (i, &ci) in qc_idxs.iter().enumerate() {
                    qc_bounds[i] = rules.state.bound_with_idx(ci).min(bound);
                }

                let mut pairs: PairVec = PairVec::new();
                for (qi, qc) in q_children.iter().enumerate() {
                    let qc_bound = qc_bounds[qi];
                    for (ri, rc) in r_children.iter().enumerate() {
                        // Triangle inequality lower bounds (avoid expensive metric call).
                        // lb1 needs d(qp, rc) exactly; lb2 only needs a lower bound on
                        // d(qc, rp), which the stored value always is.
                        if cfg!(not(feature = "no-triangle-filter")) {
                            let lb1 = if ref_data[ri].qp_rc_exact {
                                (query_data[qi].d_qp_qc - ref_data[ri].d_qp_rc).max(0.0) // pivot = qp
                            } else {
                                0.0
                            };
                            let lb2 = (query_data[qi].d_qc_rp - ref_data[ri].d_rp_rc).max(0.0); // pivot = rp
                            let lb = lb1.max(lb2);
                            let lb_dmin = lb - qc.maxdist - rc.maxdist;
                            if lb_dmin > qc_bound {
                                continue; // Pre-pruned by triangle inequality
                            }
                        }

                        // Self-child optimization: if rc is a self-child, d(qc, rc) == d(qc, rp);
                        // if qc is a self-child, d(qc, rc) == d(qp, rc). Reuse them when
                        // exact; an early-exited value is only a lower bound, so prune on
                        // it or compute the distance.
                        let reuse = if rc.d_parent == 0.0 {
                            Some((query_data[qi].d_qc_rp, query_data[qi].qc_rp_exact))
                        } else if qc.d_parent == 0.0 {
                            Some((ref_data[ri].d_qp_rc, ref_data[ri].qp_rc_exact))
                        } else {
                            None
                        };
                        let d = match reuse {
                            Some((d, true)) => d,
                            Some((lower, false)) if lower - qc.maxdist - rc.maxdist > qc_bound => continue,
                            _ => {
                                let ub = qc_bound + qc.maxdist + rc.maxdist;
                                let d = rules.metric.distance_with_bound(&qc.point, &rc.point, ub);
                                if d > ub {
                                    continue; // early exit: the pair is pruned
                                }
                                d
                            }
                        };
                        let pair_dmin = d - qc.maxdist - rc.maxdist;
                        if pair_dmin <= qc_bound {
                            pairs.push((pair_dmin, qi, ri));
                        }
                    }
                }

                halfsort(&mut pairs, |a, b| a.0.total_cmp(&b.0));

                let mut last_qi: usize = usize::MAX;
                let mut cached_bound: f64 = 0.0;
                for &(pair_dmin, qi, ri) in &pairs {
                    if qi != last_qi {
                        cached_bound = rules.state.bound_with_idx(qc_idxs[qi]).min(bound);
                        last_qi = qi;
                    }
                    if pair_dmin > cached_bound {
                        continue;
                    }
                    let qc = &q_children[qi];
                    let rc = &r_children[ri];
                    let d = pair_dmin + qc.maxdist + rc.maxdist;
                    Self::dual_recurse(qc, qc_idxs[qi], rc, bound, Some(d), rules);
                    // Invalidate: recursion may have tightened the bound
                    last_qi = usize::MAX;
                }
            }
        }
    }

    /// Compare a single query POINT against all points in a reference subtree.
    fn point_vs_subtree<'a, T: Clone, D: Distance<T>>(
        query_point: &T,
        q_idx: usize,
        ref_node: &'a Node<T>,
        precomputed_dist: Option<f64>,
        rules: &mut KnnRules<'a, T, D>,
    ) {
        let _guard = crate::core::utils::StackGuard::enter();
        let kth = rules.kth_distance_by_idx(q_idx);
        let dist = match precomputed_dist {
            Some(d) => {
                rules.base_case_by_idx(query_point, q_idx, &ref_node.point, ref_node.is_duplicate, d);
                d
            }
            None => {
                let ub = kth + ref_node.maxdist;
                let d = rules.metric.distance_with_bound(query_point, &ref_node.point, ub);
                // An early exit returns some value > ub, not the distance; only an
                // exact distance may be recorded. A pair with d > ub is pruned below.
                if d <= ub {
                    rules.base_case_by_idx(query_point, q_idx, &ref_node.point, ref_node.is_duplicate, d);
                }
                d
            }
        };

        let dmin = dist - ref_node.maxdist;
        let kth = rules.kth_distance_by_idx(q_idx);
        if dmin > kth {
            return;
        }

        if ref_node.children.is_empty() {
            return;
        }

        let mut scored: ScoredVec = ScoredVec::new();
        let current_kth = rules.kth_distance_by_idx(q_idx);
        for (i, rc) in ref_node.children.iter().enumerate() {
            // Self-child optimization: reuse parent distance when d_parent == 0.0
            let d = if rc.d_parent == 0.0 {
                dist
            } else {
                // Shell pre-filter: triangle inequality lower bound
                if cfg!(not(feature = "no-triangle-filter")) {
                    let shell_lb = (dist - rc.d_parent).max(0.0);
                    if shell_lb - rc.maxdist > current_kth {
                        continue;
                    }
                }
                let ub = current_kth + rc.maxdist;
                rules.metric.distance_with_bound(query_point, &rc.point, ub)
            };
            let child_dmin = d - rc.maxdist;
            if child_dmin <= current_kth {
                scored.push((d, i));
            }
        }

        halfsort(&mut scored, |a, b| a.0.total_cmp(&b.0));

        for (d, idx) in scored {
            let rc = &ref_node.children[idx];
            let child_dmin = d - rc.maxdist;
            if child_dmin > rules.kth_distance_by_idx(q_idx) {
                continue;
            }
            Self::point_vs_subtree(query_point, q_idx, rc, Some(d), rules);
        }
    }

    /// Compare all points in a query subtree against a single reference POINT.
    fn subtree_vs_point<'a, T: Clone, D: Distance<T>>(
        query_node: &Node<T>,
        q_idx: usize,
        ref_point: &'a T,
        ref_is_duplicate: bool,
        parent_bound: f64,
        precomputed_dist: Option<f64>,
        rules: &mut KnnRules<'a, T, D>,
    ) {
        let _guard = crate::core::utils::StackGuard::enter();
        let bound = rules.state.bound_with_idx(q_idx).min(parent_bound);
        let dist = match precomputed_dist {
            Some(d) => {
                rules.base_case_by_idx(&query_node.point, q_idx, ref_point, ref_is_duplicate, d);
                d
            }
            None => {
                let ub = bound + query_node.maxdist;
                let d = rules.metric.distance_with_bound(&query_node.point, ref_point, ub);
                // An early exit returns some value > ub, not the distance; only an
                // exact distance may be recorded. A pair with d > ub is pruned below.
                if d <= ub {
                    rules.base_case_by_idx(&query_node.point, q_idx, ref_point, ref_is_duplicate, d);
                }
                d
            }
        };

        let dmin = dist - query_node.maxdist;
        if dmin > bound {
            return;
        }

        if query_node.children.is_empty() {
            return;
        }

        // Build children indices and bounds without cloning.
        let num_qc = rules.state.num_children(q_idx);
        let mut qc_idxs: IdxVec = IdxVec::with_capacity(num_qc);
        let mut qc_bounds: DistVec = DistVec::with_capacity(num_qc);
        for ci in 0..num_qc {
            let child_idx = rules.state.child_at(q_idx, ci);
            qc_idxs.push(child_idx);
            qc_bounds.push(rules.state.bound_with_idx(child_idx).min(bound));
        }
        let mut scored: ScoredVec = ScoredVec::new();
        for (i, qc) in query_node.children.iter().enumerate() {
            // Self-child optimization: reuse parent distance when d_parent == 0.0
            let d = if qc.d_parent == 0.0 {
                dist
            } else {
                // Shell pre-filter: triangle inequality lower bound
                if cfg!(not(feature = "no-triangle-filter")) {
                    let shell_lb = (dist - qc.d_parent).max(0.0);
                    if shell_lb - qc.maxdist > qc_bounds[i] {
                        continue;
                    }
                }
                let ub = qc_bounds[i] + qc.maxdist;
                rules.metric.distance_with_bound(&qc.point, ref_point, ub)
            };
            let child_dmin = d - qc.maxdist;
            if child_dmin <= qc_bounds[i] {
                scored.push((d, i));
            }
        }

        halfsort(&mut scored, |a, b| a.0.total_cmp(&b.0));

        for (d, idx) in scored {
            let qc = &query_node.children[idx];
            let child_dmin = d - qc.maxdist;
            // Re-read bound: previous sibling's recursion may have tightened it
            let child_bound = rules.state.bound_with_idx(qc_idxs[idx]).min(bound);
            if child_dmin > child_bound {
                continue;
            }
            Self::subtree_vs_point(qc, qc_idxs[idx], ref_point, ref_is_duplicate, bound, Some(d), rules);
        }
    }
}
