//! Batch Single-Tree k-NN Algorithm
//!
//! Implements the batch nearest-neighbor query of Beygelzimer, Kakade & Langford,
//! "Cover Trees for Nearest Neighbor" (ICML 2006): walks the query tree hierarchy but
//! performs single-tree reference descent at each scale level. Each query child
//! receives an independently filtered COPY of the parent's reference candidates with
//! recomputed distances, avoiding the O(children_q × children_r) combinatorial
//! explosion that Curtin-style dual-tree traversal creates.
//!
//! # Algorithm Overview
//!
//! The algorithm maintains a stack of "cover sets" indexed by scale level, plus a
//! "zero set" for leaf-level reference nodes. Scales iterate from the root (max_scale)
//! down to the leaves (min_scale). At each step:
//!
//! 1. **current_scale < min_scale (base case)**: Score zero_set entries.
//!
//! 2. **Query has children at this scale (split)**: Each real query child gets a COPY
//!    of the parent's cover sets with distances recomputed to the new query point and
//!    filtered by the upper bound. The query node's own point acts as a "virtual
//!    self-child" inheriting the cover sets directly.
//!
//! 3. **Reference scale reduction (descend)**: Expand reference nodes at `current_scale`
//!    by replacing parents with their children, computing `d(query.point, child.point)`.
//!
//! # Adaptation for Simplified Cover Trees
//!
//! Our trees have no self-children (children[0] is NOT the parent point at level-1).
//! When entering the split case, the query node's OWN point acts as the "virtual
//! self-child" — it inherits the cover_sets directly and continues descending.
//! Each real child gets `copy_cover_sets` with recomputed distances.
//!
//! # k=1 Fast Path
//!
//! For k=1 (the common benchmark case), a lightweight `K1State` struct is threaded
//! through recursion by `&mut` reference, bypassing the HashMap-based `BatchResults`
//! entirely: one pointer dereference per access instead of HashMap hash+probe.

use std::collections::HashMap;

use crate::Distance;
use crate::knn::KnnState;
use crate::node::Node;

// ---------------------------------------------------------------------------
// K1State: lightweight k=1 state passed by &mut through recursion
// ---------------------------------------------------------------------------

/// Lightweight k=1 state passed by `&mut` through recursion.
///
/// Replaces HashMap + KnnState for the common k=1 case. Acts as both
/// `upper_bound` AND `kth_distance` — a single f64 field read replaces
/// HashMap hash+probe (~15-25 cycles saved per access).
struct K1State<'a, T> {
    best_dist: f64,
    best_point: Option<&'a T>,
    /// When true, skip distance-0 matches (for self-query where the query
    /// point appears in the reference tree at distance 0).
    skip_zero: bool,
}

impl<'a, T> K1State<'a, T> {
    #[inline(always)]
    fn new(initial_bound: f64) -> Self {
        K1State {
            best_dist: initial_bound,
            best_point: None,
            skip_zero: false,
        }
    }

    #[inline(always)]
    fn new_self_query(initial_bound: f64) -> Self {
        K1State {
            best_dist: initial_bound,
            best_point: None,
            skip_zero: true,
        }
    }

    /// Try to update best with a new candidate. Returns true if updated.
    #[inline(always)]
    fn update(&mut self, point: &'a T, dist: f64) -> bool {
        // In self-query mode, skip exact self-matches (distance == 0.0).
        // The only way a non-duplicate reference point has dist=0 to a query
        // point is if it IS the same point (self-match).
        if self.skip_zero && dist == 0.0 {
            return false;
        }
        if dist < self.best_dist {
            self.best_dist = dist;
            self.best_point = Some(point);
            true
        } else {
            false
        }
    }
}

// ---------------------------------------------------------------------------
// Unpacked (Node<T>) variant
// ---------------------------------------------------------------------------

/// An entry in a cover set: a reference node with its precomputed distance to
/// the current query point.
struct CoverSetEntry<'a, T: Clone> {
    node: &'a Node<T>,
    dist: f64, // d(current_query_point, this_ref_node.point)
}

/// Result collector: maps query nodes (by pointer identity) to their KnnState.
/// Uses HashMap for O(1) amortized lookup instead of O(n) linear scan.
struct BatchResults<'a, T: Clone> {
    states: Vec<KnnState<'a, T>>,
    index: HashMap<usize, usize>, // ptr-as-usize → states index
    k: usize,
}

impl<'a, T: Clone> BatchResults<'a, T> {
    fn new(k: usize) -> Self {
        BatchResults {
            states: Vec::new(),
            index: HashMap::new(),
            k,
        }
    }

    /// Get or create a KnnState for the given query point (identified by pointer).
    #[inline]
    fn get_state(&mut self, query_ptr: *const T) -> &mut KnnState<'a, T> {
        let key = query_ptr as usize;
        let k = self.k;
        let states = &mut self.states;
        let idx = *self.index.entry(key).or_insert_with(|| {
            let i = states.len();
            states.push(KnnState::new(k));
            i
        });
        &mut self.states[idx]
    }
}

/// Public entry point: batch single-tree k-NN using a query tree and reference tree.
///
/// Walks the query tree hierarchy, performing single-tree reference descent at each
/// scale level. Returns results indexed by non-duplicate query leaf in DFS order.
pub fn batch_single_tree_knn<'a, T, D>(
    query_root: &'a Node<T>,
    ref_root: &'a Node<T>,
    k: usize,
    metric: &D,
    _base: f64,
) -> Vec<(*const T, Vec<(&'a T, f64)>)>
where
    T: Clone,
    D: Distance<T>,
{
    // k=1 fast path: bypass HashMap entirely
    if k == 1 {
        return batch_single_tree_knn_k1(query_root, ref_root, metric);
    }

    let mut results = BatchResults::new(k);

    // Compute scale range for cover sets.
    // Scales iterate top-down: max_scale (root) → min_scale (deepest leaf).
    let max_scale = ref_root.level;
    let min_scale = find_min_level(ref_root);

    // cover_sets indexed by [scale - min_scale].
    let n_scales = (max_scale - min_scale + 1).max(1) as usize;
    let mut cover_sets: Vec<Vec<CoverSetEntry<'a, T>>> =
        (0..n_scales).map(|_| Vec::new()).collect();

    // Seed with root reference node at its level.
    let root_dist = metric.distance(&query_root.point, &ref_root.point);
    let scale_idx = (ref_root.level - min_scale) as usize;
    cover_sets[scale_idx].push(CoverSetEntry {
        node: ref_root,
        dist: root_dist,
    });

    // zero_set collects leaf-level references.
    let mut zero_set: Vec<CoverSetEntry<'a, T>> = Vec::new();
    if ref_root.children.is_empty() {
        zero_set.push(CoverSetEntry {
            node: ref_root,
            dist: root_dist,
        });
    }

    let mut upper_bound = root_dist;

    internal_batch_nn(
        query_root,
        &mut cover_sets,
        &mut zero_set,
        max_scale, // start at root level and descend
        min_scale,
        min_scale,
        &mut upper_bound,
        k,
        metric,
        &mut results,
    );

    results
        .index
        .into_iter()
        .map(|(ptr_key, state_idx)| {
            let state = std::mem::replace(
                &mut results.states[state_idx],
                KnnState::new(results.k),
            );
            (ptr_key as *const T, state.into_sorted_vec())
        })
        .collect()
}

/// Recursive core of the batch single-tree algorithm.
///
/// Scales iterate top-down: current_scale starts at max_scale (root) and decrements
/// toward min_scale (deepest leaf).
///
/// `upper_bound` is threaded by `&mut` through the entire recursion. After any insert
/// that improves the kth-distance, `*upper_bound` is updated immediately so subsequent
/// entries in the same loop iteration see the tighter bound — this cascading tightening
/// is the key to effective pruning for k>1.
///
/// Three cases:
/// 1. `current_scale < min_scale`: base case — score zero_set
/// 2. `query.level - 1 == current_scale` (query has children at this scale): split
/// 3. Otherwise: descend reference nodes at current_scale, then recurse
fn internal_batch_nn<'a, T, D>(
    query: &'a Node<T>,
    cover_sets: &mut Vec<Vec<CoverSetEntry<'a, T>>>,
    zero_set: &mut Vec<CoverSetEntry<'a, T>>,
    current_scale: i32,
    min_scale: i32,
    min_scale_offset: i32,
    upper_bound: &mut f64,
    _k: usize,
    metric: &D,
    results: &mut BatchResults<'a, T>,
) where
    T: Clone,
    D: Distance<T>,
{
    // Case 1: below min scale — base case, score zero_set
    if current_scale < min_scale {
        brute_nearest(query, zero_set, upper_bound, results);
        return;
    }

    // Check if query has children at this scale
    let query_splits = !query.children.is_empty() && query.level - 1 == current_scale;

    if query_splits {
        // Case 2: Query has children at this scale — SPLIT

        // Score the query node's own point against the current zero_set
        // (acts as "virtual self-child")
        {
            let state = results.get_state(&query.point as *const T);
            for entry in zero_set.iter() {
                if !entry.node.is_duplicate {
                    if state.insert(&entry.node.point, entry.dist) {
                        *upper_bound = state.kth_distance();
                    }
                }
            }
        }

        // Process each real query child with reused scratch buffers
        let n_scales = cover_sets.len();
        let mut scratch_cover: Vec<Vec<CoverSetEntry<'a, T>>> =
            (0..n_scales).map(|_| Vec::new()).collect();
        let mut scratch_zero: Vec<CoverSetEntry<'a, T>> = Vec::new();

        for child in &query.children {
            let child_d_parent = if child.d_parent > 0.0 {
                child.d_parent
            } else {
                metric.distance(&query.point, &child.point)
            };

            // Child's initial bound: parent's current bound + d(parent, child), which is
            // valid for the child by the triangle inequality.
            let mut new_ub = *upper_bound + child_d_parent;

            // Clear scratch (retains capacity!)
            for v in &mut scratch_cover { v.clear(); }
            scratch_zero.clear();

            copy_cover_sets_into(
                &child.point,
                &mut new_ub,
                cover_sets,
                child_d_parent,
                metric,
                results,
                &child.point as *const T,
                &mut scratch_cover,
            );

            copy_zero_set_into(
                &child.point,
                &mut new_ub,
                zero_set,
                child_d_parent,
                metric,
                results,
                &child.point as *const T,
                &mut scratch_zero,
            );

            internal_batch_nn(
                child,
                &mut scratch_cover,
                &mut scratch_zero,
                current_scale, // children continue at same scale (they'll descend refs)
                min_scale,
                min_scale_offset,
                &mut new_ub,
                _k,
                metric,
                results,
            );
        }

        // Self-child (the query node's own point) inherits cover_sets directly
        // and continues descending through remaining reference scales
        internal_batch_nn(
            query,
            cover_sets,
            zero_set,
            current_scale - 1,
            min_scale,
            min_scale_offset,
            upper_bound,
            _k,
            metric,
            results,
        );
    } else {
        // Case 3: Reference scale reduction — expand one reference level
        descend(
            query,
            current_scale,
            min_scale_offset,
            cover_sets,
            zero_set,
            upper_bound,
            metric,
            results,
        );

        // Recurse at next (lower) scale
        internal_batch_nn(
            query,
            cover_sets,
            zero_set,
            current_scale - 1,
            min_scale,
            min_scale_offset,
            upper_bound,
            _k,
            metric,
            results,
        );
    }
}

/// Single-tree reference expansion at one scale level.
///
/// Takes entries from cover_sets[current_scale] and expands them: for each entry's
/// children, compute d(query, child) and add to cover_sets[child.level] or zero_set.
///
/// `upper_bound` is updated immediately when any insert improves the kth-distance,
/// so later entries in the same iteration see a tighter bound (cascading tightening).
fn descend<'a, T, D>(
    query: &'a Node<T>,
    current_scale: i32,
    min_scale_offset: i32,
    cover_sets: &mut Vec<Vec<CoverSetEntry<'a, T>>>,
    zero_set: &mut Vec<CoverSetEntry<'a, T>>,
    upper_bound: &mut f64,
    metric: &D,
    results: &mut BatchResults<'a, T>,
) where
    T: Clone,
    D: Distance<T>,
{
    let scale_idx = (current_scale - min_scale_offset) as usize;
    if scale_idx >= cover_sets.len() {
        return;
    }

    // Take entries at this scale level (drain them)
    let mut entries = std::mem::take(&mut cover_sets[scale_idx]);

    // Halfsort: sort closer half, leave far half unsorted. Applied at every descend
    // call — with threaded upper_bound, processing closer nodes first tightens bounds
    // faster, enabling more pruning of later entries.
    if entries.len() > 1 {
        let mid = entries.len() / 2;
        entries.as_mut_slice()
            .select_nth_unstable_by(mid, |a, b| a.dist.total_cmp(&b.dist));
        entries[..mid].sort_by(|a, b| a.dist.total_cmp(&b.dist));
    }

    for entry in &entries {
        let query_ptr = &query.point as *const T;

        // Score the parent node itself
        if !entry.node.is_duplicate {
            let state = results.get_state(query_ptr);
            if state.insert(&entry.node.point, entry.dist) {
                *upper_bound = state.kth_distance();
            }
        }

        // Use threaded upper_bound for pruning — single f64 read, no HashMap lookup
        let kth = *upper_bound;

        // Expand children
        for child in &entry.node.children {
            // Shell test: triangle inequality pre-filter using parent distance
            if child.d_parent > 0.0 {
                let lower_bound = (entry.dist - child.d_parent).max(0.0);
                if (lower_bound - child.maxdist).max(0.0) > kth {
                    continue;
                }
            }

            let child_dist = if child.is_duplicate && child.d_parent == 0.0 {
                // Self-child optimization: reuse parent distance
                entry.dist
            } else {
                metric.distance_with_bound(
                    &query.point,
                    &child.point,
                    kth + child.maxdist,
                )
            };

            // Pruning: can this child's subtree contain a better neighbor?
            let min_possible = (child_dist - child.maxdist).max(0.0);
            if min_possible > *upper_bound {
                continue;
            }

            if child.children.is_empty() {
                // Leaf node: add to zero_set
                zero_set.push(CoverSetEntry {
                    node: child,
                    dist: child_dist,
                });
            } else {
                // Internal node: add to appropriate scale level
                let child_scale_idx = (child.level - min_scale_offset) as usize;
                if child_scale_idx < cover_sets.len() {
                    cover_sets[child_scale_idx].push(CoverSetEntry {
                        node: child,
                        dist: child_dist,
                    });
                }
            }
        }
    }
}

/// Copy and filter cover_sets for a new query child into pre-allocated output buffers.
/// Callers should .clear() each inner Vec before calling to reuse capacity.
///
/// `new_ub` is threaded by `&mut` — tightened when scoring improves kth-distance.
fn copy_cover_sets_into<'a, T, D>(
    new_query_point: &T,
    new_ub: &mut f64,
    parent_cover_sets: &[Vec<CoverSetEntry<'a, T>>],
    child_d_parent: f64,
    metric: &D,
    results: &mut BatchResults<'a, T>,
    query_ptr: *const T,
    output: &mut Vec<Vec<CoverSetEntry<'a, T>>>,
) where
    T: Clone,
    D: Distance<T>,
{
    for (i, parent_entries) in parent_cover_sets.iter().enumerate() {
        if parent_entries.is_empty() { continue; } // Skip empty scales
        for entry in parent_entries {
            // Shell test: triangle inequality pre-filter using parent distance
            // d(new_query, entry) >= entry.dist - child_d_parent (clamped to 0)
            let lower_bound = (entry.dist - child_d_parent).max(0.0);
            if lower_bound - entry.node.maxdist > *new_ub {
                continue; // Skip distance computation entirely
            }

            let new_dist = metric.distance_with_bound(
                new_query_point,
                &entry.node.point,
                *new_ub + entry.node.maxdist,
            );

            let min_possible = (new_dist - entry.node.maxdist).max(0.0);
            let state = results.get_state(query_ptr);
            if min_possible > state.kth_distance().min(*new_ub) {
                continue;
            }

            output[i].push(CoverSetEntry {
                node: entry.node,
                dist: new_dist,
            });
        }
    }
}

/// Copy and filter zero_set for a new query child into a pre-allocated output buffer.
/// Caller should .clear() the output Vec before calling to reuse capacity.
///
/// `new_ub` is threaded by `&mut` — tightened when scoring improves kth-distance.
fn copy_zero_set_into<'a, T, D>(
    new_query_point: &T,
    new_ub: &mut f64,
    parent_zero_set: &[CoverSetEntry<'a, T>],
    child_d_parent: f64,
    metric: &D,
    results: &mut BatchResults<'a, T>,
    query_ptr: *const T,
    output: &mut Vec<CoverSetEntry<'a, T>>,
) where
    T: Clone,
    D: Distance<T>,
{
    for entry in parent_zero_set {
        // Shell test: triangle inequality pre-filter using parent distance
        // d(new_query, entry) >= entry.dist - child_d_parent (clamped to 0)
        let lower_bound = (entry.dist - child_d_parent).max(0.0);
        if lower_bound > *new_ub {
            continue; // Skip distance computation entirely (zero_set entries are leaves, no maxdist)
        }

        let new_dist = metric.distance(new_query_point, &entry.node.point);

        // Score this entry immediately for the new query point
        let state = results.get_state(query_ptr);
        if !entry.node.is_duplicate {
            if state.insert(&entry.node.point, new_dist) {
                *new_ub = state.kth_distance();
            }
        }

        if new_dist <= state.kth_distance().min(*new_ub) {
            output.push(CoverSetEntry {
                node: entry.node,
                dist: new_dist,
            });
        }
    }
}

/// Base case: score all zero-set entries for the given query node.
fn brute_nearest<'a, T: Clone>(
    query: &'a Node<T>,
    zero_set: &[CoverSetEntry<'a, T>],
    upper_bound: &mut f64,
    results: &mut BatchResults<'a, T>,
) {
    let state = results.get_state(&query.point as *const T);
    for entry in zero_set {
        if !entry.node.is_duplicate {
            if state.insert(&entry.node.point, entry.dist) {
                *upper_bound = state.kth_distance();
            }
        }
    }
}

/// Find the minimum level in a subtree (for determining scale range).
fn find_min_level<T: Clone>(node: &Node<T>) -> i32 {
    let mut min = node.level;
    for child in &node.children {
        let child_min = find_min_level(child);
        if child_min < min {
            min = child_min;
        }
    }
    min
}

/// Public entry point for self-query: batch single-tree k-NN where query=reference.
///
/// For each non-duplicate point in the tree, finds k nearest neighbors excluding
/// itself (self-match exclusion by pointer identity).
///
/// # Performance note: self-query vs held-out
///
/// Batch single-tree builds a query tree and walks it in tandem with the
/// reference tree. The speedup comes from "query splits" (Case 2): when a
/// query node has children at a given scale, the algorithm processes all
/// siblings together — the parent's reference cover sets are inherited by the
/// virtual self-child and copied/filtered for real children. One pruning
/// decision at a reference node applies to ALL descendant query points in
/// that subtree simultaneously.
///
/// In all-NN (self-query), query tree == reference tree, so the query tree
/// has full depth (O(c² log n) levels) with many internal nodes. Every scale
/// level triggers query splits, and bulk pruning eliminates O(N) queries per
/// node check. This is why batch dominates single-tree and dual-tree here.
///
/// In held-out queries, we DO build a query tree from the held-out points
/// (see `batch_single_batch_query_instrumented` in packed/tree.rs), but it
/// is typically shallow: e.g. 128 query points produce a tree with ~2-3
/// levels. Few internal query nodes means few query splits, so most query
/// points reach the base case individually — reducing to essentially the
/// same cost as iterating single-tree queries. The copy overhead for the
/// rare splits that do occur adds insult to injury.
///
/// In short: batch single-tree's speedup scales with query tree depth, not
/// query count. Self-query gives maximum depth; small held-out sets give
/// near-zero depth.
pub fn batch_single_tree_knn_self<'a, T, D>(
    root: &'a Node<T>,
    k: usize,
    metric: &D,
    base: f64,
) -> Vec<(*const T, Vec<(&'a T, f64)>)>
where
    T: Clone,
    D: Distance<T>,
{
    // For self-query, we use k+1 to account for self-match, then filter it out.
    let mut raw_results = batch_single_tree_knn(root, root, k + 1, metric, base);

    // Remove self-matches (by pointer identity) and trim to k
    for (query_ptr, neighbors) in &mut raw_results {
        neighbors.retain(|(point_ref, _)| *point_ref as *const T != *query_ptr);
        neighbors.truncate(k);
    }

    raw_results
}

// ---------------------------------------------------------------------------
// Unpacked k=1 fast path
// ---------------------------------------------------------------------------

/// k=1 fast path for unpacked batch single-tree.
/// Threads K1State by &mut through recursion, bypassing HashMap entirely.
fn batch_single_tree_knn_k1<'a, T, D>(
    query_root: &'a Node<T>,
    ref_root: &'a Node<T>,
    metric: &D,
) -> Vec<(*const T, Vec<(&'a T, f64)>)>
where
    T: Clone,
    D: Distance<T>,
{
    let max_scale = ref_root.level;
    let min_scale = find_min_level(ref_root);

    let n_scales = (max_scale - min_scale + 1).max(1) as usize;
    let mut cover_sets: Vec<Vec<CoverSetEntry<'a, T>>> =
        (0..n_scales).map(|_| Vec::new()).collect();

    let root_dist = metric.distance(&query_root.point, &ref_root.point);
    let scale_idx = (ref_root.level - min_scale) as usize;
    cover_sets[scale_idx].push(CoverSetEntry {
        node: ref_root,
        dist: root_dist,
    });

    let mut zero_set: Vec<CoverSetEntry<'a, T>> = Vec::new();
    if ref_root.children.is_empty() {
        zero_set.push(CoverSetEntry {
            node: ref_root,
            dist: root_dist,
        });
    }

    let mut state = K1State::new(root_dist);
    let mut leaf_results: Vec<(*const T, &'a T, f64)> = Vec::new();
    let mut pool: Vec<Vec<Vec<CoverSetEntry<'a, T>>>> = Vec::new();

    internal_batch_nn_k1(
        query_root,
        &mut cover_sets,
        &mut zero_set,
        max_scale,
        min_scale,
        min_scale,
        metric,
        &mut state,
        &mut leaf_results,
        &mut pool,
    );

    // Collect results: leaf_results contains (query_ptr, best_point, best_dist)
    // for each query node that reached the base case.
    leaf_results
        .into_iter()
        .map(|(ptr, point, dist)| (ptr, vec![(point, dist)]))
        .collect()
}

/// k=1 recursive core for unpacked variant.
fn internal_batch_nn_k1<'a, T, D>(
    query: &'a Node<T>,
    cover_sets: &mut Vec<Vec<CoverSetEntry<'a, T>>>,
    zero_set: &mut Vec<CoverSetEntry<'a, T>>,
    current_scale: i32,
    min_scale: i32,
    min_scale_offset: i32,
    metric: &D,
    state: &mut K1State<'a, T>,
    leaf_results: &mut Vec<(*const T, &'a T, f64)>,
    pool: &mut Vec<Vec<Vec<CoverSetEntry<'a, T>>>>,
) where
    T: Clone,
    D: Distance<T>,
{
    if current_scale < min_scale {
        brute_nearest_k1(query, zero_set, state);
        if !query.is_duplicate {
            if let Some(bp) = state.best_point {
                leaf_results.push((&query.point as *const T, bp, state.best_dist));
            }
        }
        return;
    }

    let query_splits = !query.children.is_empty() && query.level - 1 == current_scale;

    if query_splits {
        // Score self against zero_set
        for entry in zero_set.iter() {
            if !entry.node.is_duplicate {
                state.update(&entry.node.point, entry.dist);
            }
        }

        // Process each real query child
        let n_scales = cover_sets.len();

        for child in &query.children {
            let child_d_parent = if child.d_parent > 0.0 {
                child.d_parent
            } else {
                metric.distance(&query.point, &child.point)
            };

            // Child's initial bound: parent's current best + triangle inequality
            let mut child_state = K1State {
                best_dist: state.best_dist + child_d_parent,
                best_point: None,
                skip_zero: state.skip_zero,
            };
            let new_ub = state.best_dist + child_d_parent;

            // Get scratch from pool or allocate
            let mut scratch_cover = pool.pop().unwrap_or_else(|| {
                (0..n_scales).map(|_| Vec::new()).collect()
            });
            // Ensure correct length and clear
            scratch_cover.resize_with(n_scales, Vec::new);
            for v in &mut scratch_cover { v.clear(); }
            let mut scratch_zero: Vec<CoverSetEntry<'a, T>> = Vec::new();

            copy_cover_sets_into_k1(
                &child.point,
                new_ub,
                cover_sets,
                child_d_parent,
                metric,
                &mut child_state,
                &mut scratch_cover,
            );

            copy_zero_set_into_k1(
                &child.point,
                new_ub,
                zero_set,
                child_d_parent,
                metric,
                &mut child_state,
                &mut scratch_zero,
            );

            internal_batch_nn_k1(
                child,
                &mut scratch_cover,
                &mut scratch_zero,
                current_scale,
                min_scale,
                min_scale_offset,
                metric,
                &mut child_state,
                leaf_results,
                pool,
            );

            // Return scratch to pool
            for v in &mut scratch_cover { v.clear(); }
            pool.push(scratch_cover);
        }

        // Self-child inherits parent's state directly
        internal_batch_nn_k1(
            query,
            cover_sets,
            zero_set,
            current_scale - 1,
            min_scale,
            min_scale_offset,
            metric,
            state,
            leaf_results,
            pool,
        );
    } else {
        descend_k1(
            query,
            current_scale,
            min_scale_offset,
            cover_sets,
            zero_set,
            metric,
            state,
        );

        internal_batch_nn_k1(
            query,
            cover_sets,
            zero_set,
            current_scale - 1,
            min_scale,
            min_scale_offset,
            metric,
            state,
            leaf_results,
            pool,
        );
    }
}

/// k=1 descend for unpacked variant.
fn descend_k1<'a, T, D>(
    query: &'a Node<T>,
    current_scale: i32,
    min_scale_offset: i32,
    cover_sets: &mut Vec<Vec<CoverSetEntry<'a, T>>>,
    zero_set: &mut Vec<CoverSetEntry<'a, T>>,
    metric: &D,
    state: &mut K1State<'a, T>,
) where
    T: Clone,
    D: Distance<T>,
{
    let scale_idx = (current_scale - min_scale_offset) as usize;
    if scale_idx >= cover_sets.len() {
        return;
    }

    let mut entries = std::mem::take(&mut cover_sets[scale_idx]);

    if entries.len() > 1 {
        let mid = entries.len() / 2;
        entries.as_mut_slice()
            .select_nth_unstable_by(mid, |a, b| a.dist.total_cmp(&b.dist));
        entries[..mid].sort_by(|a, b| a.dist.total_cmp(&b.dist));
    }

    for entry in &entries {
        // Score the parent node itself
        if !entry.node.is_duplicate {
            state.update(&entry.node.point, entry.dist);
        }

        // Expand children
        for child in &entry.node.children {
            let kth = state.best_dist;

            if child.d_parent > 0.0 {
                let lower_bound = (entry.dist - child.d_parent).max(0.0);
                if (lower_bound - child.maxdist).max(0.0) > kth {
                    continue;
                }
            }

            let child_dist = if child.is_duplicate && child.d_parent == 0.0 {
                entry.dist
            } else {
                metric.distance_with_bound(
                    &query.point,
                    &child.point,
                    kth + child.maxdist,
                )
            };

            let min_possible = (child_dist - child.maxdist).max(0.0);
            if min_possible > state.best_dist {
                continue;
            }

            if child.children.is_empty() {
                zero_set.push(CoverSetEntry {
                    node: child,
                    dist: child_dist,
                });
            } else {
                let child_scale_idx = (child.level - min_scale_offset) as usize;
                if child_scale_idx < cover_sets.len() {
                    cover_sets[child_scale_idx].push(CoverSetEntry {
                        node: child,
                        dist: child_dist,
                    });
                }
            }
        }
    }
}

/// k=1 copy_cover_sets for unpacked variant.
fn copy_cover_sets_into_k1<'a, T, D>(
    new_query_point: &T,
    new_ub: f64,
    parent_cover_sets: &[Vec<CoverSetEntry<'a, T>>],
    child_d_parent: f64,
    metric: &D,
    state: &mut K1State<'a, T>,
    output: &mut Vec<Vec<CoverSetEntry<'a, T>>>,
) where
    T: Clone,
    D: Distance<T>,
{
    for (i, parent_entries) in parent_cover_sets.iter().enumerate() {
        if parent_entries.is_empty() { continue; }
        for entry in parent_entries {
            let lower_bound = (entry.dist - child_d_parent).max(0.0);
            if lower_bound - entry.node.maxdist > new_ub {
                continue;
            }

            let new_dist = metric.distance_with_bound(
                new_query_point,
                &entry.node.point,
                new_ub + entry.node.maxdist,
            );

            let min_possible = (new_dist - entry.node.maxdist).max(0.0);
            if min_possible > state.best_dist.min(new_ub) {
                continue;
            }

            output[i].push(CoverSetEntry {
                node: entry.node,
                dist: new_dist,
            });
        }
    }
}

/// k=1 copy_zero_set for unpacked variant.
fn copy_zero_set_into_k1<'a, T, D>(
    new_query_point: &T,
    new_ub: f64,
    parent_zero_set: &[CoverSetEntry<'a, T>],
    child_d_parent: f64,
    metric: &D,
    state: &mut K1State<'a, T>,
    output: &mut Vec<CoverSetEntry<'a, T>>,
) where
    T: Clone,
    D: Distance<T>,
{
    for entry in parent_zero_set {
        let lower_bound = (entry.dist - child_d_parent).max(0.0);
        if lower_bound > new_ub {
            continue;
        }

        let new_dist = metric.distance(new_query_point, &entry.node.point);

        if !entry.node.is_duplicate {
            state.update(&entry.node.point, new_dist);
        }

        if new_dist <= state.best_dist.min(new_ub) {
            output.push(CoverSetEntry {
                node: entry.node,
                dist: new_dist,
            });
        }
    }
}

/// k=1 base case for unpacked variant.
#[inline]
fn brute_nearest_k1<'a, T: Clone>(
    query: &'a Node<T>,
    zero_set: &[CoverSetEntry<'a, T>],
    state: &mut K1State<'a, T>,
) {
    let _ = query; // query identity used by caller for result collection
    for entry in zero_set {
        if !entry.node.is_duplicate {
            state.update(&entry.node.point, entry.dist);
        }
    }
}

// ---------------------------------------------------------------------------
// Packed variant
// ---------------------------------------------------------------------------

/// Entry in a packed cover set: a reference node index with precomputed distance.
struct PackedCoverSetEntry {
    idx: usize,
    dist: f64,
}

/// Result collector for packed batch single-tree.
/// Uses HashMap for O(1) amortized lookup instead of O(n) linear scan.
struct PackedBatchResults<'a, T: Clone> {
    states: Vec<KnnState<'a, T>>,
    index: HashMap<usize, usize>, // ptr-as-usize → states index
    k: usize,
}

impl<'a, T: Clone> PackedBatchResults<'a, T> {
    fn new(k: usize) -> Self {
        PackedBatchResults {
            states: Vec::new(),
            index: HashMap::new(),
            k,
        }
    }

    #[inline]
    fn get_state(&mut self, query_ptr: *const T) -> &mut KnnState<'a, T> {
        let key = query_ptr as usize;
        let k = self.k;
        let states = &mut self.states;
        let idx = *self.index.entry(key).or_insert_with(|| {
            let i = states.len();
            states.push(KnnState::new(k));
            i
        });
        &mut self.states[idx]
    }
}

/// Batch single-tree k-NN using packed reference tree + unpacked query tree.
pub fn batch_single_tree_knn_packed<'a, T, D>(
    query_root: &'a Node<T>,
    packed: &'a crate::packed::PackedCoverTree<T, D>,
    ref_root_idx: usize,
    k: usize,
    metric: &D,
    _base: f64,
) -> Vec<(*const T, Vec<(&'a T, f64)>)>
where
    T: Clone,
    D: Distance<T>,
{
    // k=1 fast path: bypass HashMap entirely
    if k == 1 {
        return batch_single_tree_knn_packed_k1(query_root, packed, ref_root_idx, metric);
    }

    let mut results = PackedBatchResults::new(k);

    let ref_node = packed.node(ref_root_idx);
    let max_scale = ref_node.level;
    let min_scale = packed.min_level();

    let n_scales = (max_scale - min_scale + 1).max(1) as usize;
    let mut cover_sets: Vec<Vec<PackedCoverSetEntry>> =
        (0..n_scales).map(|_| Vec::new()).collect();
    let mut zero_set: Vec<PackedCoverSetEntry> = Vec::new();

    let root_dist = metric.distance(&query_root.point, &ref_node.point);
    let scale_idx = (ref_node.level - min_scale) as usize;
    cover_sets[scale_idx].push(PackedCoverSetEntry {
        idx: ref_root_idx,
        dist: root_dist,
    });

    if packed.children_of(ref_root_idx).is_empty() {
        zero_set.push(PackedCoverSetEntry {
            idx: ref_root_idx,
            dist: root_dist,
        });
    }

    let mut upper_bound = root_dist;

    internal_batch_nn_packed(
        query_root,
        &mut cover_sets,
        &mut zero_set,
        max_scale, // start at root level and descend
        min_scale,
        min_scale,
        &mut upper_bound,
        k,
        metric,
        packed,
        &mut results,
    );

    results
        .index
        .into_iter()
        .map(|(ptr_key, state_idx)| {
            let state = std::mem::replace(
                &mut results.states[state_idx],
                KnnState::new(results.k),
            );
            (ptr_key as *const T, state.into_sorted_vec())
        })
        .collect()
}

/// Self-query variant: batch single-tree k=1 NN using packed reference tree.
///
/// For self-query (query tree == reference tree), uses k=1 with zero-distance
/// exclusion instead of k=2 with post-filtering. This allows the k=1 fast path
/// to be used, with self-exclusion handled during traversal rather than via k+1.
pub fn batch_single_tree_knn_packed_self<'a, T, D>(
    query_root: &'a Node<T>,
    packed: &'a crate::packed::PackedCoverTree<T, D>,
    ref_root_idx: usize,
    k: usize,
    metric: &D,
    _base: f64,
) -> Vec<(*const T, Vec<(&'a T, f64)>)>
where
    T: Clone,
    D: Distance<T>,
{
    if k == 1 {
        return batch_single_tree_knn_packed_self_k1(query_root, packed, ref_root_idx, metric);
    }
    // For k>1 self-query, fall back to k+1 with post-filtering
    let mut raw = batch_single_tree_knn_packed(query_root, packed, ref_root_idx, k + 1, metric, _base);
    for (_query_ptr, neighbors) in &mut raw {
        neighbors.retain(|(_point_ref, dist)| *dist > 0.0);
        neighbors.truncate(k);
    }
    raw
}

/// Recursive core for packed batch single-tree.
///
/// `upper_bound` is threaded by `&mut` — tightened on every improving insert.
fn internal_batch_nn_packed<'a, T, D>(
    query: &'a Node<T>,
    cover_sets: &mut Vec<Vec<PackedCoverSetEntry>>,
    zero_set: &mut Vec<PackedCoverSetEntry>,
    current_scale: i32,
    min_scale: i32,
    min_scale_offset: i32,
    upper_bound: &mut f64,
    _k: usize,
    metric: &D,
    packed: &'a crate::packed::PackedCoverTree<T, D>,
    results: &mut PackedBatchResults<'a, T>,
) where
    T: Clone,
    D: Distance<T>,
{
    if current_scale < min_scale {
        brute_nearest_packed(query, zero_set, packed, upper_bound, results);
        return;
    }

    let query_splits = !query.children.is_empty() && query.level - 1 == current_scale;

    if query_splits {
        // Case 2: Split — query has children at this scale

        // Score self against zero_set
        {
            let state = results.get_state(&query.point as *const T);
            for entry in zero_set.iter() {
                let node = packed.node(entry.idx);
                if !node.is_duplicate {
                    if state.insert(&node.point, entry.dist) {
                        *upper_bound = state.kth_distance();
                    }
                }
            }
        }

        // Process each real query child with reused scratch buffers
        let n_scales = cover_sets.len();
        let mut scratch_cover: Vec<Vec<PackedCoverSetEntry>> =
            (0..n_scales).map(|_| Vec::new()).collect();
        let mut scratch_zero: Vec<PackedCoverSetEntry> = Vec::new();

        for child in &query.children {
            let child_d_parent = if child.d_parent > 0.0 {
                child.d_parent
            } else {
                metric.distance(&query.point, &child.point)
            };

            // Child's initial bound: parent's current bound + triangle inequality
            let mut new_ub = *upper_bound + child_d_parent;

            // Clear scratch (retains capacity!)
            for v in &mut scratch_cover { v.clear(); }
            scratch_zero.clear();

            copy_cover_sets_packed_into(
                &child.point,
                &mut new_ub,
                cover_sets,
                child_d_parent,
                metric,
                packed,
                results,
                &child.point as *const T,
                &mut scratch_cover,
            );

            copy_zero_set_packed_into(
                &child.point,
                &mut new_ub,
                zero_set,
                child_d_parent,
                metric,
                packed,
                results,
                &child.point as *const T,
                &mut scratch_zero,
            );

            internal_batch_nn_packed(
                child,
                &mut scratch_cover,
                &mut scratch_zero,
                current_scale,
                min_scale,
                min_scale_offset,
                &mut new_ub,
                _k,
                metric,
                packed,
                results,
            );
        }

        // Self-child inherits directly, continue descending
        internal_batch_nn_packed(
            query,
            cover_sets,
            zero_set,
            current_scale - 1,
            min_scale,
            min_scale_offset,
            upper_bound,
            _k,
            metric,
            packed,
            results,
        );
    } else {
        // Case 3: Descend reference
        descend_packed(
            query,
            current_scale,
            min_scale_offset,
            cover_sets,
            zero_set,
            upper_bound,
            metric,
            packed,
            results,
        );

        internal_batch_nn_packed(
            query,
            cover_sets,
            zero_set,
            current_scale - 1,
            min_scale,
            min_scale_offset,
            upper_bound,
            _k,
            metric,
            packed,
            results,
        );
    }
}

/// Descend reference nodes in packed tree.
///
/// `upper_bound` is updated immediately on every improving insert — cascading tightening.
fn descend_packed<'a, T, D>(
    query: &'a Node<T>,
    current_scale: i32,
    min_scale_offset: i32,
    cover_sets: &mut Vec<Vec<PackedCoverSetEntry>>,
    zero_set: &mut Vec<PackedCoverSetEntry>,
    upper_bound: &mut f64,
    metric: &D,
    packed: &'a crate::packed::PackedCoverTree<T, D>,
    results: &mut PackedBatchResults<'a, T>,
) where
    T: Clone,
    D: Distance<T>,
{
    let scale_idx = (current_scale - min_scale_offset) as usize;
    if scale_idx >= cover_sets.len() {
        return;
    }

    let mut entries = std::mem::take(&mut cover_sets[scale_idx]);

    // Halfsort: sort closer half at every descend call
    if entries.len() > 1 {
        let mid = entries.len() / 2;
        entries.as_mut_slice()
            .select_nth_unstable_by(mid, |a, b| a.dist.total_cmp(&b.dist));
        entries[..mid].sort_by(|a, b| a.dist.total_cmp(&b.dist));
    }

    for entry in &entries {
        let node = packed.node(entry.idx);
        let query_ptr = &query.point as *const T;

        // Score the parent
        if !node.is_duplicate {
            let state = results.get_state(query_ptr);
            if state.insert(&node.point, entry.dist) {
                *upper_bound = state.kth_distance();
            }
        }

        // Use threaded upper_bound for pruning — single f64 read, no HashMap lookup
        let kth = *upper_bound;

        // Expand children
        for &child_idx in packed.children_of(entry.idx) {
            let child_node = packed.node(child_idx);

            // Shell test: triangle inequality pre-filter using parent distance
            if child_node.d_parent > 0.0 {
                let lower_bound = (entry.dist - child_node.d_parent).max(0.0);
                if (lower_bound - child_node.maxdist).max(0.0) > kth {
                    continue;
                }
            }

            let child_dist = if child_node.is_duplicate && child_node.d_parent == 0.0 {
                // Self-child optimization: reuse parent distance
                entry.dist
            } else {
                metric.distance_with_bound(
                    &query.point,
                    &child_node.point,
                    kth + child_node.maxdist,
                )
            };

            let min_possible = (child_dist - child_node.maxdist).max(0.0);
            if min_possible > *upper_bound {
                continue;
            }

            if child_node.child_index_count == 0 {
                zero_set.push(PackedCoverSetEntry {
                    idx: child_idx,
                    dist: child_dist,
                });
            } else {
                let child_scale_idx = (child_node.level - min_scale_offset) as usize;
                if child_scale_idx < cover_sets.len() {
                    cover_sets[child_scale_idx].push(PackedCoverSetEntry {
                        idx: child_idx,
                        dist: child_dist,
                    });
                }
            }
        }
    }
}

/// Copy and filter cover_sets from packed tree for a new query child.
///
/// `new_ub` is threaded by `&mut` — tightened when scoring improves kth-distance.
fn copy_cover_sets_packed_into<'a, T, D>(
    new_query_point: &T,
    new_ub: &mut f64,
    parent_cover_sets: &[Vec<PackedCoverSetEntry>],
    child_d_parent: f64,
    metric: &D,
    packed: &'a crate::packed::PackedCoverTree<T, D>,
    results: &mut PackedBatchResults<'a, T>,
    query_ptr: *const T,
    output: &mut Vec<Vec<PackedCoverSetEntry>>,
) where
    T: Clone,
    D: Distance<T>,
{
    for (i, parent_entries) in parent_cover_sets.iter().enumerate() {
        if parent_entries.is_empty() { continue; } // Skip empty scales
        for entry in parent_entries {
            let node = packed.node(entry.idx);

            // Shell test: triangle inequality pre-filter using parent distance
            let lower_bound = (entry.dist - child_d_parent).max(0.0);
            if lower_bound - node.maxdist > *new_ub {
                continue;
            }

            let new_dist = metric.distance_with_bound(
                new_query_point,
                &node.point,
                *new_ub + node.maxdist,
            );

            let min_possible = (new_dist - node.maxdist).max(0.0);
            let state = results.get_state(query_ptr);
            if min_possible > state.kth_distance().min(*new_ub) {
                continue;
            }

            output[i].push(PackedCoverSetEntry {
                idx: entry.idx,
                dist: new_dist,
            });
        }
    }
}

/// Copy and filter zero_set from packed tree for a new query child.
///
/// `new_ub` is threaded by `&mut` — tightened when scoring improves kth-distance.
fn copy_zero_set_packed_into<'a, T, D>(
    new_query_point: &T,
    new_ub: &mut f64,
    parent_zero_set: &[PackedCoverSetEntry],
    child_d_parent: f64,
    metric: &D,
    packed: &'a crate::packed::PackedCoverTree<T, D>,
    results: &mut PackedBatchResults<'a, T>,
    query_ptr: *const T,
    output: &mut Vec<PackedCoverSetEntry>,
) where
    T: Clone,
    D: Distance<T>,
{
    for entry in parent_zero_set {
        // Shell test: triangle inequality pre-filter
        let lower_bound = (entry.dist - child_d_parent).max(0.0);
        if lower_bound > *new_ub {
            continue;
        }

        let node = packed.node(entry.idx);
        let new_dist = metric.distance(new_query_point, &node.point);

        let state = results.get_state(query_ptr);
        if !node.is_duplicate {
            if state.insert(&node.point, new_dist) {
                *new_ub = state.kth_distance();
            }
        }

        if new_dist <= state.kth_distance().min(*new_ub) {
            output.push(PackedCoverSetEntry {
                idx: entry.idx,
                dist: new_dist,
            });
        }
    }
}

fn brute_nearest_packed<'a, T, D>(
    query: &'a Node<T>,
    zero_set: &[PackedCoverSetEntry],
    packed: &'a crate::packed::PackedCoverTree<T, D>,
    upper_bound: &mut f64,
    results: &mut PackedBatchResults<'a, T>,
) where
    T: Clone,
    D: Distance<T>,
{
    let state = results.get_state(&query.point as *const T);
    for entry in zero_set {
        let node = packed.node(entry.idx);
        if !node.is_duplicate {
            if state.insert(&node.point, entry.dist) {
                *upper_bound = state.kth_distance();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Packed k=1 fast path
// ---------------------------------------------------------------------------

/// k=1 fast path for packed batch single-tree SELF-QUERY.
/// Same as `batch_single_tree_knn_packed_k1` but uses `skip_zero: true`
/// to exclude self-matches (distance 0) instead of using k+1.
fn batch_single_tree_knn_packed_self_k1<'a, T, D>(
    query_root: &'a Node<T>,
    packed: &'a crate::packed::PackedCoverTree<T, D>,
    ref_root_idx: usize,
    metric: &D,
) -> Vec<(*const T, Vec<(&'a T, f64)>)>
where
    T: Clone,
    D: Distance<T>,
{
    let ref_node = packed.node(ref_root_idx);
    let max_scale = ref_node.level;
    let min_scale = packed.min_level();

    let n_scales = (max_scale - min_scale + 1).max(1) as usize;
    let mut cover_sets: Vec<Vec<PackedCoverSetEntry>> =
        (0..n_scales).map(|_| Vec::new()).collect();
    let mut zero_set: Vec<PackedCoverSetEntry> = Vec::new();

    let root_dist = metric.distance(&query_root.point, &ref_node.point);
    let scale_idx = (ref_node.level - min_scale) as usize;
    cover_sets[scale_idx].push(PackedCoverSetEntry {
        idx: ref_root_idx,
        dist: root_dist,
    });

    if packed.children_of(ref_root_idx).is_empty() {
        zero_set.push(PackedCoverSetEntry {
            idx: ref_root_idx,
            dist: root_dist,
        });
    }

    // Use self-query state: skip_zero=true to exclude self-matches at distance 0.
    // Initialize with INFINITY so pruning starts from scratch (root_dist=0 would
    // prune everything since the root IS the query point in a self-query).
    let mut state = K1State::new_self_query(f64::INFINITY);
    let mut leaf_results: Vec<(*const T, &'a T, f64)> = Vec::new();
    let mut pool: Vec<Vec<Vec<PackedCoverSetEntry>>> = Vec::new();

    internal_batch_nn_packed_k1(
        query_root,
        &mut cover_sets,
        &mut zero_set,
        max_scale,
        min_scale,
        min_scale,
        metric,
        packed,
        &mut state,
        &mut leaf_results,
        &mut pool,
    );

    leaf_results
        .into_iter()
        .map(|(ptr, point, dist)| (ptr, vec![(point, dist)]))
        .collect()
}

/// k=1 fast path for packed batch single-tree.
/// Threads K1State by &mut through recursion, bypassing HashMap entirely.
fn batch_single_tree_knn_packed_k1<'a, T, D>(
    query_root: &'a Node<T>,
    packed: &'a crate::packed::PackedCoverTree<T, D>,
    ref_root_idx: usize,
    metric: &D,
) -> Vec<(*const T, Vec<(&'a T, f64)>)>
where
    T: Clone,
    D: Distance<T>,
{
    let ref_node = packed.node(ref_root_idx);
    let max_scale = ref_node.level;
    let min_scale = packed.min_level();

    let n_scales = (max_scale - min_scale + 1).max(1) as usize;
    let mut cover_sets: Vec<Vec<PackedCoverSetEntry>> =
        (0..n_scales).map(|_| Vec::new()).collect();
    let mut zero_set: Vec<PackedCoverSetEntry> = Vec::new();

    let root_dist = metric.distance(&query_root.point, &ref_node.point);
    let scale_idx = (ref_node.level - min_scale) as usize;
    cover_sets[scale_idx].push(PackedCoverSetEntry {
        idx: ref_root_idx,
        dist: root_dist,
    });

    if packed.children_of(ref_root_idx).is_empty() {
        zero_set.push(PackedCoverSetEntry {
            idx: ref_root_idx,
            dist: root_dist,
        });
    }

    let mut state = K1State::new(root_dist);
    let mut leaf_results: Vec<(*const T, &'a T, f64)> = Vec::new();
    let mut pool: Vec<Vec<Vec<PackedCoverSetEntry>>> = Vec::new();

    internal_batch_nn_packed_k1(
        query_root,
        &mut cover_sets,
        &mut zero_set,
        max_scale,
        min_scale,
        min_scale,
        metric,
        packed,
        &mut state,
        &mut leaf_results,
        &mut pool,
    );

    leaf_results
        .into_iter()
        .map(|(ptr, point, dist)| (ptr, vec![(point, dist)]))
        .collect()
}

/// k=1 recursive core for packed variant.
fn internal_batch_nn_packed_k1<'a, T, D>(
    query: &'a Node<T>,
    cover_sets: &mut Vec<Vec<PackedCoverSetEntry>>,
    zero_set: &mut Vec<PackedCoverSetEntry>,
    current_scale: i32,
    min_scale: i32,
    min_scale_offset: i32,
    metric: &D,
    packed: &'a crate::packed::PackedCoverTree<T, D>,
    state: &mut K1State<'a, T>,
    leaf_results: &mut Vec<(*const T, &'a T, f64)>,
    pool: &mut Vec<Vec<Vec<PackedCoverSetEntry>>>,
) where
    T: Clone,
    D: Distance<T>,
{
    if current_scale < min_scale {
        brute_nearest_packed_k1(query, zero_set, packed, state);
        if !query.is_duplicate {
            if let Some(bp) = state.best_point {
                leaf_results.push((&query.point as *const T, bp, state.best_dist));
            }
        }
        return;
    }

    let query_splits = !query.children.is_empty() && query.level - 1 == current_scale;

    if query_splits {
        // Score self against zero_set
        for entry in zero_set.iter() {
            let node = packed.node(entry.idx);
            if !node.is_duplicate {
                state.update(&node.point, entry.dist);
            }
        }

        // Process each real query child
        let n_scales = cover_sets.len();

        for child in &query.children {
            let child_d_parent = if child.d_parent > 0.0 {
                child.d_parent
            } else {
                metric.distance(&query.point, &child.point)
            };

            // Child's initial bound: parent's current best + d(parent, child), which is
            // valid for the child by the triangle inequality.
            let mut child_state = K1State {
                best_dist: state.best_dist + child_d_parent,
                best_point: None,
                skip_zero: state.skip_zero,
            };
            let new_ub = state.best_dist + child_d_parent;

            // Get scratch from pool or allocate
            let mut scratch_cover = pool.pop().unwrap_or_else(|| {
                (0..n_scales).map(|_| Vec::new()).collect()
            });
            scratch_cover.resize_with(n_scales, Vec::new);
            for v in &mut scratch_cover { v.clear(); }
            let mut scratch_zero: Vec<PackedCoverSetEntry> = Vec::new();

            copy_cover_sets_packed_into_k1(
                &child.point,
                new_ub,
                cover_sets,
                child_d_parent,
                metric,
                packed,
                &mut child_state,
                &mut scratch_cover,
            );

            copy_zero_set_packed_into_k1(
                &child.point,
                new_ub,
                zero_set,
                child_d_parent,
                metric,
                packed,
                &mut child_state,
                &mut scratch_zero,
            );

            internal_batch_nn_packed_k1(
                child,
                &mut scratch_cover,
                &mut scratch_zero,
                current_scale,
                min_scale,
                min_scale_offset,
                metric,
                packed,
                &mut child_state,
                leaf_results,
                pool,
            );

            // Return scratch to pool
            for v in &mut scratch_cover { v.clear(); }
            pool.push(scratch_cover);
        }

        // Self-child inherits parent's state directly
        internal_batch_nn_packed_k1(
            query,
            cover_sets,
            zero_set,
            current_scale - 1,
            min_scale,
            min_scale_offset,
            metric,
            packed,
            state,
            leaf_results,
            pool,
        );
    } else {
        descend_packed_k1(
            query,
            current_scale,
            min_scale_offset,
            cover_sets,
            zero_set,
            metric,
            packed,
            state,
        );

        internal_batch_nn_packed_k1(
            query,
            cover_sets,
            zero_set,
            current_scale - 1,
            min_scale,
            min_scale_offset,
            metric,
            packed,
            state,
            leaf_results,
            pool,
        );
    }
}

/// k=1 descend for packed variant. Uses state.best_dist directly instead of
/// HashMap lookup — the core optimization that eliminates per-operation overhead.
fn descend_packed_k1<'a, T, D>(
    query: &'a Node<T>,
    current_scale: i32,
    min_scale_offset: i32,
    cover_sets: &mut Vec<Vec<PackedCoverSetEntry>>,
    zero_set: &mut Vec<PackedCoverSetEntry>,
    metric: &D,
    packed: &'a crate::packed::PackedCoverTree<T, D>,
    state: &mut K1State<'a, T>,
) where
    T: Clone,
    D: Distance<T>,
{
    let scale_idx = (current_scale - min_scale_offset) as usize;
    if scale_idx >= cover_sets.len() {
        return;
    }

    let mut entries = std::mem::take(&mut cover_sets[scale_idx]);

    if entries.len() > 1 {
        let mid = entries.len() / 2;
        entries.as_mut_slice()
            .select_nth_unstable_by(mid, |a, b| a.dist.total_cmp(&b.dist));
        entries[..mid].sort_by(|a, b| a.dist.total_cmp(&b.dist));
    }

    for entry in &entries {
        let node = packed.node(entry.idx);

        // Score the parent — single field read instead of HashMap lookup
        if !node.is_duplicate {
            state.update(&node.point, entry.dist);
        }

        // Expand children
        for &child_idx in packed.children_of(entry.idx) {
            let child_node = packed.node(child_idx);

            // Single field read: ~1 cycle vs HashMap hash+probe ~15-25 cycles
            let kth = state.best_dist;

            // Shell test: triangle inequality pre-filter using parent distance
            if child_node.d_parent > 0.0 {
                let lower_bound = (entry.dist - child_node.d_parent).max(0.0);
                if (lower_bound - child_node.maxdist).max(0.0) > kth {
                    continue;
                }
            }

            let child_dist = if child_node.is_duplicate && child_node.d_parent == 0.0 {
                entry.dist
            } else {
                metric.distance_with_bound(
                    &query.point,
                    &child_node.point,
                    kth + child_node.maxdist,
                )
            };

            let min_possible = (child_dist - child_node.maxdist).max(0.0);
            if min_possible > state.best_dist {
                continue;
            }

            if child_node.child_index_count == 0 {
                zero_set.push(PackedCoverSetEntry {
                    idx: child_idx,
                    dist: child_dist,
                });
            } else {
                let child_scale_idx = (child_node.level - min_scale_offset) as usize;
                if child_scale_idx < cover_sets.len() {
                    cover_sets[child_scale_idx].push(PackedCoverSetEntry {
                        idx: child_idx,
                        dist: child_dist,
                    });
                }
            }
        }
    }
}

/// k=1 copy_cover_sets for packed variant.
fn copy_cover_sets_packed_into_k1<'a, T, D>(
    new_query_point: &T,
    new_ub: f64,
    parent_cover_sets: &[Vec<PackedCoverSetEntry>],
    child_d_parent: f64,
    metric: &D,
    packed: &'a crate::packed::PackedCoverTree<T, D>,
    state: &mut K1State<'a, T>,
    output: &mut Vec<Vec<PackedCoverSetEntry>>,
) where
    T: Clone,
    D: Distance<T>,
{
    for (i, parent_entries) in parent_cover_sets.iter().enumerate() {
        if parent_entries.is_empty() { continue; } // Skip empty scales
        for entry in parent_entries {
            let node = packed.node(entry.idx);

            let lower_bound = (entry.dist - child_d_parent).max(0.0);
            if lower_bound - node.maxdist > new_ub {
                continue;
            }

            let new_dist = metric.distance_with_bound(
                new_query_point,
                &node.point,
                new_ub + node.maxdist,
            );

            let min_possible = (new_dist - node.maxdist).max(0.0);
            if min_possible > state.best_dist.min(new_ub) {
                continue;
            }

            output[i].push(PackedCoverSetEntry {
                idx: entry.idx,
                dist: new_dist,
            });
        }
    }
}

/// k=1 copy_zero_set for packed variant.
fn copy_zero_set_packed_into_k1<'a, T, D>(
    new_query_point: &T,
    new_ub: f64,
    parent_zero_set: &[PackedCoverSetEntry],
    child_d_parent: f64,
    metric: &D,
    packed: &'a crate::packed::PackedCoverTree<T, D>,
    state: &mut K1State<'a, T>,
    output: &mut Vec<PackedCoverSetEntry>,
) where
    T: Clone,
    D: Distance<T>,
{
    for entry in parent_zero_set {
        let lower_bound = (entry.dist - child_d_parent).max(0.0);
        if lower_bound > new_ub {
            continue;
        }

        let node = packed.node(entry.idx);
        let new_dist = metric.distance(new_query_point, &node.point);

        if !node.is_duplicate {
            state.update(&node.point, new_dist);
        }

        if new_dist <= state.best_dist.min(new_ub) {
            output.push(PackedCoverSetEntry {
                idx: entry.idx,
                dist: new_dist,
            });
        }
    }
}

/// k=1 base case for packed variant.
#[inline]
fn brute_nearest_packed_k1<'a, T, D>(
    query: &'a Node<T>,
    zero_set: &[PackedCoverSetEntry],
    packed: &'a crate::packed::PackedCoverTree<T, D>,
    state: &mut K1State<'a, T>,
) where
    T: Clone,
    D: Distance<T>,
{
    let _ = query; // query identity used by caller for result collection
    for entry in zero_set {
        let node = packed.node(entry.idx);
        if !node.is_duplicate {
            state.update(&node.point, entry.dist);
        }
    }
}
