//! Fully-Packed Dual-Tree Traversal for Self-Query k-NN
//!
//! Both query and reference sides use index-based access into the **same** packed
//! cover tree. This eliminates the need to build a separate unpacked query tree
//! for self-query (all-NN), which was the main performance bottleneck in
//! `PackedCoverTree::find_k_nearest_self`.
//!
//! # Structural Mapping from Mixed to Fully-Packed
//!
//! The mixed traversal (`dual_tree.rs`) uses `&Node<T>` for the query side and
//! packed indices for the reference side. Here, BOTH sides use packed indices:
//!
//! | Mixed (dual_tree.rs)                   | Fully-Packed (this module)                     |
//! |----------------------------------------|------------------------------------------------|
//! | `query_node: &Node<T>`                 | `q_state_idx` → `packed.node(packed_idx[q])`   |
//! | `query_node.point`                     | `packed.node(q_packed).point`                  |
//! | `query_node.maxdist`                   | `packed.node(q_packed).maxdist`                |
//! | `query_node.d_parent`                  | `packed.node(q_packed).d_parent`               |
//! | `query_node.children.is_empty()`       | `packed.node(q_packed).child_index_count == 0` |
//! | `query_node.children[i]`               | `packed.children_of(q_packed)[i]`              |
//!
//! The state index → packed index mapping is maintained in `DualKnnState::packed_idx`,
//! populated by `init_parent_map_packed`.

use smallvec::SmallVec;

use crate::distance::Distance;
use crate::core::dual_tree::knn_rules::KnnRules;
use super::tree::PackedCoverTree;

/// Inline capacity for sorted child arrays.
type ScoredVec = SmallVec<[(f64, usize); 8]>;

/// Inline capacity for child-child pair arrays.
type PairVec = SmallVec<[(f64, usize, usize); 16]>;

/// Inline capacity for per-child distance arrays.
type DistVec = SmallVec<[f64; 8]>;

/// Halfsort: O(n) partition + O(n/2 log n/2) sort of closer half.
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

/// Pre-computed per-reference-child data.
#[derive(Clone, Copy)]
struct RefChildData {
    maxdist: f64,
    d_qp_rc: f64,
    d_rp_rc: f64,
}

/// Pre-computed per-query-child data.
#[derive(Clone, Copy)]
struct QueryChildData {
    d_qc_rp: f64,
    d_qp_qc: f64,
}

/// Fully-packed dual-tree traversal where both query and reference are the same packed tree.
pub(crate) struct FullyPackedDualTreeTraversal;

impl FullyPackedDualTreeTraversal {
    /// Run the fully-packed self-query dual-tree traversal.
    ///
    /// Both query and reference sides index into the same `packed` tree.
    /// `DualKnnState::init_parent_map_packed` must have been called first.
    pub fn traverse<'a, T: Clone, D: Distance<T>>(
        packed: &'a PackedCoverTree<T, D>,
        root_state_idx: usize,
        ref_root_packed_idx: usize,
        rules: &mut KnnRules<'a, T, D>,
    ) {
        Self::dual_recurse(packed, root_state_idx, ref_root_packed_idx, f64::INFINITY, None, rules);
    }

    /// Recursive dual-tree descent — both sides are packed indices.
    fn dual_recurse<'a, T: Clone, D: Distance<T>>(
        packed: &'a PackedCoverTree<T, D>,
        q_state_idx: usize,
        ref_packed_idx: usize,
        parent_bound: f64,
        precomputed_dist: Option<f64>,
        rules: &mut KnnRules<'a, T, D>,
    ) {
        let q_packed_idx = rules.state.packed_node_idx(q_state_idx);
        let q_node = packed.node(q_packed_idx);
        let r_node = packed.node(ref_packed_idx);

        // Step 1: Base case
        let bound = rules.state.bound_with_idx(q_state_idx).min(parent_bound);
        let dist = match precomputed_dist {
            Some(d) => {
                rules.base_case_by_idx(&q_node.point, q_state_idx, &r_node.point, r_node.is_duplicate, d);
                d
            }
            None => {
                let ub = bound + q_node.maxdist + r_node.maxdist;
                let d = rules.metric.distance_with_bound(&q_node.point, &r_node.point, ub);
                rules.base_case_by_idx(&q_node.point, q_state_idx, &r_node.point, r_node.is_duplicate, d);
                d
            }
        };

        // Step 2: Pruning check
        let dmin = dist - q_node.maxdist - r_node.maxdist;
        if dmin > bound {
            return;
        }

        let q_is_leaf = q_node.child_index_count == 0;
        let r_is_leaf = r_node.child_index_count == 0;

        match (q_is_leaf, r_is_leaf) {
            (true, true) => {
                // Both leaves — base case already computed
            }
            (true, false) => {
                // Query is leaf, expand reference children
                let ref_children = packed.children_of(ref_packed_idx);

                let mut scored: SmallVec<[(f64, f64, usize); 8]> = SmallVec::new();
                for (i, &rc_idx) in ref_children.iter().enumerate() {
                    let rc = packed.node(rc_idx);
                    let d = if rc.d_parent == 0.0 {
                        dist
                    } else {
                        let shell_lb = (dist - rc.d_parent).max(0.0);
                        if shell_lb - q_node.maxdist - rc.maxdist > bound {
                            continue;
                        }
                        let ub = bound + q_node.maxdist + rc.maxdist;
                        rules.metric.distance_with_bound(&q_node.point, &rc.point, ub)
                    };
                    let child_dmin = d - q_node.maxdist - rc.maxdist;
                    if child_dmin <= bound {
                        scored.push((d, rc.maxdist, i));
                    }
                }

                halfsort(&mut scored, |a, b| a.0.total_cmp(&b.0));

                for &(d, maxdist, idx) in &scored {
                    let rc_idx = ref_children[idx];
                    let child_dmin = d - q_node.maxdist - maxdist;
                    let current_bound = rules.state.bound_with_idx(q_state_idx).min(parent_bound);
                    if child_dmin > current_bound {
                        continue;
                    }
                    Self::dual_recurse(packed, q_state_idx, rc_idx, parent_bound, Some(d), rules);
                }
            }
            (false, true) => {
                // Reference is leaf, expand query children
                let num_qc = rules.state.num_children(q_state_idx);
                let mut qc_state_idxs: SmallVec<[usize; 8]> = SmallVec::with_capacity(num_qc);
                let mut qc_bounds: DistVec = SmallVec::with_capacity(num_qc);
                for ci in 0..num_qc {
                    let child_state = rules.state.child_at(q_state_idx, ci);
                    qc_state_idxs.push(child_state);
                    qc_bounds.push(rules.state.bound_with_idx(child_state).min(bound));
                }

                let q_children = packed.children_of(q_packed_idx);
                let mut scored: ScoredVec = SmallVec::new();
                for (i, &qc_packed) in q_children.iter().enumerate() {
                    let qc = packed.node(qc_packed);
                    let d = if qc.d_parent == 0.0 {
                        dist
                    } else {
                        let shell_lb = (dist - qc.d_parent).max(0.0);
                        if shell_lb - qc.maxdist - r_node.maxdist > qc_bounds[i] {
                            continue;
                        }
                        let ub = qc_bounds[i] + qc.maxdist + r_node.maxdist;
                        rules.metric.distance_with_bound(&qc.point, &r_node.point, ub)
                    };
                    let child_dmin = d - qc.maxdist - r_node.maxdist;
                    if child_dmin <= qc_bounds[i] {
                        scored.push((d, i));
                    }
                }

                halfsort(&mut scored, |a, b| a.0.total_cmp(&b.0));

                for (d, idx) in scored {
                    let qc_packed = q_children[idx];
                    let qc = packed.node(qc_packed);
                    let child_dmin = d - qc.maxdist - r_node.maxdist;
                    let child_bound = rules.state.bound_with_idx(qc_state_idxs[idx]).min(bound);
                    if child_dmin > child_bound {
                        continue;
                    }
                    Self::dual_recurse(packed, qc_state_idxs[idx], ref_packed_idx, bound, Some(d), rules);
                }
            }
            (false, false) => {
                // Both have children — three phases
                let q_children = packed.children_of(q_packed_idx);
                let ref_children = packed.children_of(ref_packed_idx);

                // Build query children state indices
                let num_qc = rules.state.num_children(q_state_idx);
                let mut qc_state_idxs: SmallVec<[usize; 8]> = SmallVec::with_capacity(num_qc);
                for ci in 0..num_qc {
                    qc_state_idxs.push(rules.state.child_at(q_state_idx, ci));
                }

                // --- Phase 2: query point vs ref children ---
                let mut ref_data: SmallVec<[RefChildData; 8]> = SmallVec::new();
                let mut ref_scored: ScoredVec = SmallVec::new();
                let qp_kth = rules.kth_distance_by_idx(q_state_idx);
                for (i, &rc_idx) in ref_children.iter().enumerate() {
                    let rc = packed.node(rc_idx);
                    let d = if rc.d_parent == 0.0 {
                        dist
                    } else {
                        let ub = qp_kth + rc.maxdist;
                        rules.metric.distance_with_bound(&q_node.point, &rc.point, ub)
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
                    let current_kth = rules.kth_distance_by_idx(q_state_idx);
                    if child_dmin > current_kth {
                        continue;
                    }
                    Self::point_vs_packed_subtree(packed, q_state_idx, rc_idx, Some(d), rules);
                }

                // --- Phase 3: query children vs ref point ---
                let mut qc_bounds: DistVec = qc_state_idxs.iter()
                    .map(|&ci| rules.state.bound_with_idx(ci).min(bound))
                    .collect();

                let mut query_data: SmallVec<[QueryChildData; 8]> = SmallVec::new();
                let mut query_scored: ScoredVec = SmallVec::new();
                for (i, &qc_packed) in q_children.iter().enumerate() {
                    let qc = packed.node(qc_packed);
                    let d = if qc.d_parent == 0.0 {
                        dist
                    } else {
                        let ub = qc_bounds[i] + qc.maxdist;
                        rules.metric.distance_with_bound(&qc.point, &r_node.point, ub)
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
                    let qc_packed = q_children[idx];
                    let qc = packed.node(qc_packed);
                    let child_dmin = d - qc.maxdist;
                    let child_bound = rules.state.bound_with_idx(qc_state_idxs[idx]).min(bound);
                    if child_dmin > child_bound {
                        continue;
                    }
                    Self::subtree_vs_packed_point(packed, qc_state_idxs[idx], ref_packed_idx, bound, Some(d), rules);
                }

                // --- Phase 1: child-child pairs with triangle-inequality pre-pruning ---
                for (i, &ci) in qc_state_idxs.iter().enumerate() {
                    qc_bounds[i] = rules.state.bound_with_idx(ci).min(bound);
                }

                let mut pairs: PairVec = SmallVec::new();
                for (qi, &qc_packed) in q_children.iter().enumerate() {
                    let qc = packed.node(qc_packed);
                    let qc_bound = qc_bounds[qi];
                    for (ri, &rc_idx) in ref_children.iter().enumerate() {
                        let lb1 = (query_data[qi].d_qp_qc - ref_data[ri].d_qp_rc).max(0.0);
                        let lb2 = (query_data[qi].d_qc_rp - ref_data[ri].d_rp_rc).max(0.0);
                        let lb = lb1.max(lb2);
                        let lb_dmin = lb - qc.maxdist - ref_data[ri].maxdist;
                        if lb_dmin > qc_bound {
                            continue;
                        }

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
                        cached_bound = rules.state.bound_with_idx(qc_state_idxs[qi]).min(bound);
                        last_qi = qi;
                    }
                    if pair_dmin > cached_bound {
                        continue;
                    }
                    let qc_packed = q_children[qi];
                    let qc = packed.node(qc_packed);
                    let rc_idx = ref_children[ri];
                    let d = pair_dmin + qc.maxdist + ref_data[ri].maxdist;
                    Self::dual_recurse(packed, qc_state_idxs[qi], rc_idx, bound, Some(d), rules);
                    last_qi = usize::MAX;
                }
            }
        }
    }

    /// Compare a single query POINT against all points in a packed reference subtree.
    fn point_vs_packed_subtree<'a, T: Clone, D: Distance<T>>(
        packed: &'a PackedCoverTree<T, D>,
        q_state_idx: usize,
        ref_packed_idx: usize,
        precomputed_dist: Option<f64>,
        rules: &mut KnnRules<'a, T, D>,
    ) {
        let q_packed_idx = rules.state.packed_node_idx(q_state_idx);
        let q_node = packed.node(q_packed_idx);
        let r_node = packed.node(ref_packed_idx);

        let kth = rules.kth_distance_by_idx(q_state_idx);
        let dist = match precomputed_dist {
            Some(d) => {
                rules.base_case_by_idx(&q_node.point, q_state_idx, &r_node.point, r_node.is_duplicate, d);
                d
            }
            None => {
                let ub = kth + r_node.maxdist;
                let d = rules.metric.distance_with_bound(&q_node.point, &r_node.point, ub);
                rules.base_case_by_idx(&q_node.point, q_state_idx, &r_node.point, r_node.is_duplicate, d);
                d
            }
        };

        let dmin = dist - r_node.maxdist;
        let kth = rules.kth_distance_by_idx(q_state_idx);
        if dmin > kth {
            return;
        }

        if r_node.child_index_count == 0 {
            return;
        }

        let ref_children = packed.children_of(ref_packed_idx);
        let mut scored: SmallVec<[(f64, f64, usize); 8]> = SmallVec::new();
        let current_kth = rules.kth_distance_by_idx(q_state_idx);
        for (i, &rc_idx) in ref_children.iter().enumerate() {
            let rc = packed.node(rc_idx);
            let d = if rc.d_parent == 0.0 {
                dist
            } else {
                let shell_lb = (dist - rc.d_parent).max(0.0);
                if shell_lb - rc.maxdist > current_kth {
                    continue;
                }
                let ub = current_kth + rc.maxdist;
                rules.metric.distance_with_bound(&q_node.point, &rc.point, ub)
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
            if child_dmin > rules.kth_distance_by_idx(q_state_idx) {
                continue;
            }
            Self::point_vs_packed_subtree(packed, q_state_idx, rc_idx, Some(d), rules);
        }
    }

    /// Compare all points in a query subtree against a single packed reference POINT.
    fn subtree_vs_packed_point<'a, T: Clone, D: Distance<T>>(
        packed: &'a PackedCoverTree<T, D>,
        q_state_idx: usize,
        ref_packed_idx: usize,
        parent_bound: f64,
        precomputed_dist: Option<f64>,
        rules: &mut KnnRules<'a, T, D>,
    ) {
        let q_packed_idx = rules.state.packed_node_idx(q_state_idx);
        let q_node = packed.node(q_packed_idx);
        let r_node = packed.node(ref_packed_idx);

        let bound = rules.state.bound_with_idx(q_state_idx).min(parent_bound);
        let dist = match precomputed_dist {
            Some(d) => {
                rules.base_case_by_idx(&q_node.point, q_state_idx, &r_node.point, r_node.is_duplicate, d);
                d
            }
            None => {
                let ub = bound + q_node.maxdist;
                let d = rules.metric.distance_with_bound(&q_node.point, &r_node.point, ub);
                rules.base_case_by_idx(&q_node.point, q_state_idx, &r_node.point, r_node.is_duplicate, d);
                d
            }
        };

        let dmin = dist - q_node.maxdist;
        if dmin > bound {
            return;
        }

        if q_node.child_index_count == 0 {
            return;
        }

        let q_children = packed.children_of(q_packed_idx);
        let num_qc = rules.state.num_children(q_state_idx);
        let mut qc_state_idxs: SmallVec<[usize; 8]> = SmallVec::with_capacity(num_qc);
        let mut qc_bounds: DistVec = SmallVec::with_capacity(num_qc);
        for ci in 0..num_qc {
            let child_state = rules.state.child_at(q_state_idx, ci);
            qc_state_idxs.push(child_state);
            qc_bounds.push(rules.state.bound_with_idx(child_state).min(bound));
        }

        let mut scored: ScoredVec = SmallVec::new();
        for (i, &qc_packed) in q_children.iter().enumerate() {
            let qc = packed.node(qc_packed);
            let d = if qc.d_parent == 0.0 {
                dist
            } else {
                let shell_lb = (dist - qc.d_parent).max(0.0);
                if shell_lb - qc.maxdist > qc_bounds[i] {
                    continue;
                }
                let ub = qc_bounds[i] + qc.maxdist;
                rules.metric.distance_with_bound(&qc.point, &r_node.point, ub)
            };
            let child_dmin = d - qc.maxdist;
            if child_dmin <= qc_bounds[i] {
                scored.push((d, i));
            }
        }

        halfsort(&mut scored, |a, b| a.0.total_cmp(&b.0));

        for (d, idx) in scored {
            let qc_packed = q_children[idx];
            let qc = packed.node(qc_packed);
            let child_dmin = d - qc.maxdist;
            let child_bound = rules.state.bound_with_idx(qc_state_idxs[idx]).min(bound);
            if child_dmin > child_bound {
                continue;
            }
            Self::subtree_vs_packed_point(packed, qc_state_idxs[idx], ref_packed_idx, bound, Some(d), rules);
        }
    }
}
