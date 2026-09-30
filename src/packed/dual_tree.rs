//! Packed Dual-Tree Traversal for k-NN Search
//!
//! Implements dual-tree traversal where the **reference tree is a packed cover tree**
//! (cache-friendly contiguous memory layout) and the **query tree is an unpacked cover tree**
//! (standard pointer-based `Node<T>`).
//!
//! # Why Mixed (Unpacked Query + Packed Reference)?
//!
//! - The **reference tree is large** (the dataset). Keeping it packed preserves the
//!   15-25% cache performance gain — this is the hot path.
//! - The **query tree is small** (batch of queries). Building it unpacked with
//!   `SimplifiedCoverTree::insert` is trivial. The entire query tree likely fits
//!   in L2 cache regardless.
//! - `DualKnnState` and `KnnRules` already work with unpacked query trees via
//!   `*const Node<T>` pointer indexing — zero changes needed.
//!
//! # Structural Mapping from Unpacked to Packed (Reference Side)
//!
//! | Unpacked (`traversal.rs`)              | Packed (this module)                             |
//! |----------------------------------------|--------------------------------------------------|
//! | `ref_node: &Node<T>`                   | `ref_idx: usize` + `packed: &PackedCoverTree`    |
//! | `ref_node.point`                       | `packed.node(ref_idx).point`                     |
//! | `ref_node.maxdist`                     | `packed.node(ref_idx).maxdist`                   |
//! | `ref_node.is_duplicate`                | `packed.node(ref_idx).is_duplicate`              |
//! | `ref_node.children.is_empty()`         | `packed.node(ref_idx).child_index_count == 0`    |
//! | `ref_node.children.iter().enumerate()` | `packed.children_of(ref_idx).iter().enumerate()` |
//!
//! All query-side code remains unchanged (still `&Node<T>` pointer-based).

use smallvec::SmallVec;

use crate::distance::Distance;
use crate::node::Node;
use crate::core::dual_tree::knn_rules::KnnRules;
use super::tree::PackedCoverTree;

/// Inline capacity for sorted child arrays. Cover trees at base 1.3 typically have
/// 3-5 children; 8 handles the common case without heap allocation while keeping
/// per-recursion stack frames small for better cache locality.
type ScoredVec = SmallVec<[(f64, usize); 8]>;

/// Inline capacity for child-child pair arrays in the both-internal case.
/// With max ~5 children per side at base 1.3, up to 25 pairs typical.
type PairVec = SmallVec<[(f64, usize, usize); 16]>;

/// Inline capacity for per-child distance arrays used in triangle inequality
/// pre-pruning. Stores d(qp, rc_j) or d(qc_i, rp) values indexed by child index.
type DistVec = SmallVec<[f64; 8]>;

/// Halfsort: O(n) quickselect partition + O(n/2 log n/2) sort of the closer half.
/// For small arrays (<= 8 elements), falls back to full sort since the overhead
/// of select_nth_unstable_by isn't worthwhile.
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
/// Consolidates three parallel arrays into one for better locality.
#[derive(Clone, Copy)]
struct RefChildData {
    maxdist: f64,
    d_qp_rc: f64,   // distance from query point to this ref child
    d_rp_rc: f64,    // distance from ref parent to this ref child (d_parent)
}

/// Pre-computed per-query-child data, collected during Phase 3.
/// Consolidates two parallel arrays into one for better locality.
#[derive(Clone, Copy)]
struct QueryChildData {
    d_qc_rp: f64,   // distance from this query child to ref parent
    d_qp_qc: f64,   // distance from query parent to this query child (d_parent)
}

/// Dual-tree traversal with a packed reference tree.
///
/// Mirrors `DualTreeTraversal` from `src/core/dual_tree/traversal.rs` exactly,
/// with one systematic change: all reference-side node access is index-based
/// through `PackedCoverTree` accessors.
pub(crate) struct PackedDualTreeTraversal;

impl PackedDualTreeTraversal {
    /// Run the dual-tree traversal.
    ///
    /// # Arguments
    ///
    /// * `query_root` - Root of the unpacked query tree
    /// * `packed` - The packed reference tree
    /// * `ref_idx` - Index of the reference root in the packed tree
    /// * `rules` - k-NN rules for base case and bound computation
    pub fn traverse<'a, T: Clone, D: Distance<T>>(
        query_root: &Node<T>,
        packed: &'a PackedCoverTree<T, D>,
        ref_idx: usize,
        rules: &mut KnnRules<'a, T, D>,
    ) {
        // Root is always index 0 (assigned first during DFS init_parent_map)
        let q_idx = 0;
        Self::dual_recurse(query_root, q_idx, packed, ref_idx, f64::INFINITY, None, rules);
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
    ///   was already computed by the caller. Avoids redundant metric evaluation.
    fn dual_recurse<'a, T: Clone, D: Distance<T>>(
        query_node: &Node<T>,
        q_idx: usize,
        packed: &'a PackedCoverTree<T, D>,
        ref_idx: usize,
        parent_bound: f64,
        precomputed_dist: Option<f64>,
        rules: &mut KnnRules<'a, T, D>,
    ) {
        let ref_node = packed.node(ref_idx);

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
                rules.base_case_by_idx(&query_node.point, q_idx, &ref_node.point, ref_node.is_duplicate, d);
                d
            }
        };

        // Step 2: Pruning check using pre-resolved index (no HashMap)
        let dmin = dist - query_node.maxdist - ref_node.maxdist;
        if dmin > bound {
            return;
        }

        let q_is_leaf = query_node.children.is_empty();
        let r_is_leaf = ref_node.child_index_count == 0;

        match (q_is_leaf, r_is_leaf) {
            (true, true) => {
                // Both leaves — base case already computed
            }
            (true, false) => {
                // Query is leaf, expand reference children sorted by distance.
                // Bundle (dist, maxdist, child_local_index) into one array for locality.
                let ref_children = packed.children_of(ref_idx);

                let mut scored: SmallVec<[(f64, f64, usize); 8]> = SmallVec::new();
                for (i, &rc_idx) in ref_children.iter().enumerate() {
                    let rc = packed.node(rc_idx);
                    // Self-child optimization: reuse parent distance when d_parent == 0.0
                    let d = if rc.d_parent == 0.0 {
                        dist
                    } else {
                        // Shell pre-filter: triangle inequality lower bound
                        let shell_lb = (dist - rc.d_parent).max(0.0);
                        if shell_lb - query_node.maxdist - rc.maxdist > bound {
                            continue;
                        }
                        let ub = bound + query_node.maxdist + rc.maxdist;
                        rules.metric.distance_with_bound(&query_node.point, &rc.point, ub)
                    };
                    let child_dmin = d - query_node.maxdist - rc.maxdist;
                    if child_dmin <= bound {
                        scored.push((d, rc.maxdist, i));
                    }
                }

                halfsort(&mut scored, |a, b| a.0.total_cmp(&b.0));

                for &(d, maxdist, idx) in &scored {
                    let rc_idx = ref_children[idx];
                    let child_dmin = d - query_node.maxdist - maxdist;
                    let current_bound = rules.state.bound_with_idx(q_idx).min(parent_bound);
                    if child_dmin > current_bound {
                        continue;
                    }
                    Self::dual_recurse(query_node, q_idx, packed, rc_idx, parent_bound, Some(d), rules);
                }
            }
            (false, true) => {
                // Reference is leaf, expand query children sorted by distance.
                // Build children indices and bounds without cloning SmallVec.
                let num_qc = rules.state.num_children(q_idx);
                let mut qc_idxs: SmallVec<[usize; 8]> = SmallVec::with_capacity(num_qc);
                let mut qc_bounds: DistVec = SmallVec::with_capacity(num_qc);
                for ci in 0..num_qc {
                    let child_idx = rules.state.child_at(q_idx, ci);
                    qc_idxs.push(child_idx);
                    qc_bounds.push(rules.state.bound_with_idx(child_idx).min(bound));
                }
                let mut scored: ScoredVec = SmallVec::new();
                for (i, qc) in query_node.children.iter().enumerate() {
                    // Self-child optimization: reuse parent distance when d_parent == 0.0
                    let d = if qc.d_parent == 0.0 {
                        dist
                    } else {
                        // Shell pre-filter: triangle inequality lower bound
                        let shell_lb = (dist - qc.d_parent).max(0.0);
                        if shell_lb - qc.maxdist - ref_node.maxdist > qc_bounds[i] {
                            continue;
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
                    Self::dual_recurse(qc, qc_idxs[idx], packed, ref_idx, bound, Some(d), rules);
                }
            }
            (false, false) => {
                // Both have children — three phases.
                let q_children = &query_node.children;
                let ref_children = packed.children_of(ref_idx);

                // Build children indices without cloning SmallVec.
                let num_qc = rules.state.num_children(q_idx);
                let mut qc_idxs: SmallVec<[usize; 8]> = SmallVec::with_capacity(num_qc);
                for ci in 0..num_qc {
                    qc_idxs.push(rules.state.child_at(q_idx, ci));
                }

                // --- Phase 2: query_node.point vs ref children ---
                // q_idx already resolved above — use directly for kth_distance_by_idx.
                // Collect maxdist, d_parent alongside distance computation to avoid
                // a separate pre-load pass over the same packed nodes.
                let mut ref_data: SmallVec<[RefChildData; 8]> = SmallVec::new();
                let mut ref_scored: ScoredVec = SmallVec::new();
                let qp_kth = rules.kth_distance_by_idx(q_idx);
                for (i, &rc_idx) in ref_children.iter().enumerate() {
                    let rc = packed.node(rc_idx);
                    // Self-child optimization: reuse parent distance when d_parent == 0.0
                    let d = if rc.d_parent == 0.0 {
                        dist // d(qp, rp) == d(qp, rc) for self-child
                    } else {
                        let ub = qp_kth + rc.maxdist;
                        rules.metric.distance_with_bound(&query_node.point, &rc.point, ub)
                    };
                    ref_data.push(RefChildData {
                        maxdist: rc.maxdist,
                        d_qp_rc: d,
                        d_rp_rc: rc.d_parent,
                    });
                    let child_dmin = d - rc.maxdist;
                    if child_dmin <= qp_kth {
                        ref_scored.push((d, i));
                    }
                }

                halfsort(&mut ref_scored, |a, b| a.0.total_cmp(&b.0));

                for (d, idx) in ref_scored {
                    let rc_idx = ref_children[idx];
                    let child_dmin = d - ref_data[idx].maxdist;
                    let current_kth = rules.kth_distance_by_idx(q_idx);
                    if child_dmin > current_kth {
                        continue;
                    }
                    Self::point_vs_packed_subtree(&query_node.point, q_idx, packed, rc_idx, Some(d), rules);
                }

                // --- Phase 3: query children vs ref_node.point ---
                // Pre-compute bounds using pre-resolved child indices.
                let mut qc_bounds: DistVec = qc_idxs.iter()
                    .map(|&ci| rules.state.bound_with_idx(ci).min(bound))
                    .collect();

                let mut query_data: SmallVec<[QueryChildData; 8]> = SmallVec::new();
                let mut query_scored: ScoredVec = SmallVec::new();
                for (i, qc) in q_children.iter().enumerate() {
                    // Self-child optimization: reuse parent distance when d_parent == 0.0
                    let d = if qc.d_parent == 0.0 {
                        dist // d(qp, rp) == d(qc, rp) for self-child
                    } else {
                        let ub = qc_bounds[i] + qc.maxdist;
                        rules.metric.distance_with_bound(&qc.point, &ref_node.point, ub)
                    };
                    query_data.push(QueryChildData {
                        d_qc_rp: d,
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
                    Self::subtree_vs_packed_point(qc, qc_idxs[idx], packed, ref_idx, bound, Some(d), rules);
                }

                // --- Phase 1: child-child pairs with triangle-inequality pre-pruning ---
                // query_data was collected during Phase 3 loop above.
                // ref_data was collected during Phase 2 loop above.

                // Refresh query child bounds after Phase 2/3 recursion
                for (i, &ci) in qc_idxs.iter().enumerate() {
                    qc_bounds[i] = rules.state.bound_with_idx(ci).min(bound);
                }

                let mut pairs: PairVec = SmallVec::new();
                for (qi, qc) in q_children.iter().enumerate() {
                    let qc_bound = qc_bounds[qi];
                    for (ri, &rc_idx) in ref_children.iter().enumerate() {
                        // Triangle inequality lower bounds (avoid expensive metric call)
                        let lb1 = (query_data[qi].d_qp_qc - ref_data[ri].d_qp_rc).max(0.0); // pivot = qp
                        let lb2 = (query_data[qi].d_qc_rp - ref_data[ri].d_rp_rc).max(0.0); // pivot = rp
                        let lb = lb1.max(lb2);
                        let lb_dmin = lb - qc.maxdist - ref_data[ri].maxdist;
                        if lb_dmin > qc_bound {
                            continue; // Pre-pruned by triangle inequality
                        }

                        // Self-child optimization: if rc is a self-child, d(qc, rc) == d(qc, rp);
                        // if qc is a self-child, d(qc, rc) == d(qp, rc). Both already computed.
                        let rc = packed.node(rc_idx);
                        let d = if rc.d_parent == 0.0 {
                            query_data[qi].d_qc_rp
                        } else if qc.d_parent == 0.0 {
                            ref_data[ri].d_qp_rc
                        } else {
                            let ub = qc_bound + qc.maxdist + ref_data[ri].maxdist;
                            rules.metric.distance_with_bound(&qc.point, &rc.point, ub)
                        };
                        let pair_dmin = d - qc.maxdist - ref_data[ri].maxdist;
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
                    let rc_idx = ref_children[ri];
                    let d = pair_dmin + qc.maxdist + ref_data[ri].maxdist;
                    Self::dual_recurse(qc, qc_idxs[qi], packed, rc_idx, bound, Some(d), rules);
                    // Invalidate: recursion may have tightened the bound
                    last_qi = usize::MAX;
                }
            }
        }
    }

    /// Compare a single query POINT against all points in a packed reference subtree.
    fn point_vs_packed_subtree<'a, T: Clone, D: Distance<T>>(
        query_point: &T,
        q_idx: usize,
        packed: &'a PackedCoverTree<T, D>,
        ref_idx: usize,
        precomputed_dist: Option<f64>,
        rules: &mut KnnRules<'a, T, D>,
    ) {
        let ref_node = packed.node(ref_idx);

        let kth = rules.kth_distance_by_idx(q_idx);
        let dist = match precomputed_dist {
            Some(d) => {
                rules.base_case_by_idx(query_point, q_idx, &ref_node.point, ref_node.is_duplicate, d);
                d
            }
            None => {
                let ub = kth + ref_node.maxdist;
                let d = rules.metric.distance_with_bound(query_point, &ref_node.point, ub);
                rules.base_case_by_idx(query_point, q_idx, &ref_node.point, ref_node.is_duplicate, d);
                d
            }
        };

        let dmin = dist - ref_node.maxdist;
        let kth = rules.kth_distance_by_idx(q_idx);
        if dmin > kth {
            return;
        }

        if ref_node.child_index_count == 0 {
            return;
        }

        let ref_children = packed.children_of(ref_idx);

        // Bundle (dist, maxdist, child_local_index) into one array for locality.
        let mut scored: SmallVec<[(f64, f64, usize); 8]> = SmallVec::new();
        let current_kth = rules.kth_distance_by_idx(q_idx);
        for (i, &rc_idx) in ref_children.iter().enumerate() {
            let rc = packed.node(rc_idx);
            // Self-child optimization: reuse parent distance when d_parent == 0.0
            let d = if rc.d_parent == 0.0 {
                dist
            } else {
                // Shell pre-filter: triangle inequality lower bound
                let shell_lb = (dist - rc.d_parent).max(0.0);
                if shell_lb - rc.maxdist > current_kth {
                    continue;
                }
                let ub = current_kth + rc.maxdist;
                rules.metric.distance_with_bound(query_point, &rc.point, ub)
            };
            let child_dmin = d - rc.maxdist;
            if child_dmin <= current_kth {
                scored.push((d, rc.maxdist, i));
            }
        }

        halfsort(&mut scored, |a, b| a.0.total_cmp(&b.0));

        for &(d, maxdist, idx) in &scored {
            let rc_idx = ref_children[idx];
            let child_dmin = d - maxdist;
            if child_dmin > rules.kth_distance_by_idx(q_idx) {
                continue;
            }
            Self::point_vs_packed_subtree(query_point, q_idx, packed, rc_idx, Some(d), rules);
        }
    }

    /// Compare all points in a query subtree against a single packed reference POINT.
    fn subtree_vs_packed_point<'a, T: Clone, D: Distance<T>>(
        query_node: &Node<T>,
        q_idx: usize,
        packed: &'a PackedCoverTree<T, D>,
        ref_idx: usize,
        parent_bound: f64,
        precomputed_dist: Option<f64>,
        rules: &mut KnnRules<'a, T, D>,
    ) {
        let ref_node = packed.node(ref_idx);

        let bound = rules.state.bound_with_idx(q_idx).min(parent_bound);
        let dist = match precomputed_dist {
            Some(d) => {
                rules.base_case_by_idx(&query_node.point, q_idx, &ref_node.point, ref_node.is_duplicate, d);
                d
            }
            None => {
                let ub = bound + query_node.maxdist;
                let d = rules.metric.distance_with_bound(&query_node.point, &ref_node.point, ub);
                rules.base_case_by_idx(&query_node.point, q_idx, &ref_node.point, ref_node.is_duplicate, d);
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

        // Build children indices and bounds without cloning SmallVec.
        let num_qc = rules.state.num_children(q_idx);
        let mut qc_idxs: SmallVec<[usize; 8]> = SmallVec::with_capacity(num_qc);
        let mut qc_bounds: DistVec = SmallVec::with_capacity(num_qc);
        for ci in 0..num_qc {
            let child_idx = rules.state.child_at(q_idx, ci);
            qc_idxs.push(child_idx);
            qc_bounds.push(rules.state.bound_with_idx(child_idx).min(bound));
        }
        let mut scored: ScoredVec = SmallVec::new();
        for (i, qc) in query_node.children.iter().enumerate() {
            // Self-child optimization: reuse parent distance when d_parent == 0.0
            let d = if qc.d_parent == 0.0 {
                dist
            } else {
                // Shell pre-filter: triangle inequality lower bound
                let shell_lb = (dist - qc.d_parent).max(0.0);
                if shell_lb - qc.maxdist > qc_bounds[i] {
                    continue;
                }
                let ub = qc_bounds[i] + qc.maxdist;
                rules.metric.distance_with_bound(&qc.point, &ref_node.point, ub)
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
            Self::subtree_vs_packed_point(qc, qc_idxs[idx], packed, ref_idx, bound, Some(d), rules);
        }
    }
}
