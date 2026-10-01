//! PackedCoverTree - Cache-optimized cover tree with contiguous memory layout
//!
//! This module implements the PackedCoverTree data structure, which stores all nodes
//! in a single contiguous Vec for optimal cache performance.

use crate::Distance;
use crate::simplified::SimplifiedCoverTree;
use super::node::PackedNode;
#[cfg(not(feature = "no-smallvec"))]
use smallvec::SmallVec;

/// A cache-optimized cover tree with a depth-first packed layout
///
/// PackedCoverTree stores all nodes in contiguous memory using depth-first ordering.
/// This provides 15-25% faster queries compared to pointer-based trees due to:
/// - Sequential memory access (CPU prefetcher friendly)
/// - Reduced cache misses (parent + children in same cache lines)
/// - Index arithmetic faster than pointer dereferencing
///
/// # Two-Array Design for Cache Efficiency
///
/// Nodes are stored in `nodes` array (~32 bytes each), keeping them small so
/// more fit in L1/L2 cache. Child relationships stored in separate `child_indices`
/// array (8 bytes per edge). This avoids `Vec<usize>` per node which would bloat
/// nodes to ~96 bytes and reduce cache capacity by 67%.
///
/// # Important: Completely Immutable
///
/// PackedCoverTree is **read-only** after creation. No insertions or modifications
/// are possible. This enables the cache-friendly layout without expensive maintenance.
///
/// # Usage
///
/// ```rust,ignore
/// // Build unpacked tree first
/// let mut tree = SimplifiedCoverTree::new(metric, 1.3);
/// for point in points {
///     tree.insert(point);
/// }
///
/// // Pack for cache-efficient queries
/// let packed = tree.pack();
///
/// // Query (15-25% faster than unpacked)
/// let nearest = packed.find_nearest(&query);
/// ```
///
/// # Performance
///
/// - **Query time**: Same O(c^6 log n) complexity, but 15-25% faster wall-clock time
/// - **Space**: Similar to unpacked tree, only +16% for child index array
/// - **Packing time**: O(n) single traversal
pub struct PackedCoverTree<T: Clone, D: Distance<T>> {
    /// All nodes in contiguous memory (depth-first order)
    ///
    /// Index 0 is not necessarily the root - use root_index field
    nodes: Vec<PackedNode<T>>,

    /// Shared array of child indices
    ///
    /// Each node references a slice of this array via (child_index_start, child_index_count)
    /// This keeps nodes small (~40 bytes) for better cache utilization
    child_indices: Vec<usize>,

    /// Index of root node in nodes Vec
    root_index: usize,

    /// Distance metric
    metric: D,

    /// Base value for covdist/sepdist calculations
    base: f64,

    /// Number of points in the tree
    size: usize,

    /// Minimum level across all nodes (cached to avoid O(N) scan per query)
    min_level: i32,
}

impl<T: Clone, D: Distance<T>> PackedCoverTree<T, D> {
    /// Create a new empty PackedCoverTree
    ///
    /// Note: Typically you don't construct PackedCoverTree directly.
    /// Instead, build a SimplifiedCoverTree and call `.pack()` on it.
    pub(super) fn new(
        nodes: Vec<PackedNode<T>>,
        child_indices: Vec<usize>,
        root_index: usize,
        metric: D,
        base: f64,
        size: usize,
    ) -> Self {
        let min_level = if nodes.is_empty() {
            0
        } else {
            nodes.iter().map(|n| n.level).min().unwrap_or(0)
        };
        Self {
            nodes,
            child_indices,
            root_index,
            metric,
            base,
            size,
            min_level,
        }
    }

    /// Get the number of points in the tree
    pub fn len(&self) -> usize {
        self.size
    }

    /// Check if the tree is empty
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Get a reference to the distance metric
    pub fn metric(&self) -> &D {
        &self.metric
    }

    /// Get the base value used for covdist/sepdist calculations
    pub fn base_value(&self) -> f64 {
        self.base
    }

    /// Get the minimum level across all nodes in the tree.
    ///
    /// This is cached at construction time to avoid an O(N) scan per query.
    /// Used by batch single-tree algorithms to determine the scale range.
    #[inline]
    pub fn min_level(&self) -> i32 {
        self.min_level
    }

    /// Collect tree structure quality metrics from the packed arrays.
    ///
    /// Returns a `TreeStats` summarizing the tree's structure quality.
    /// Returns `None` if the tree is empty.
    pub fn tree_stats(&self) -> Option<crate::node::TreeStats> {
        if self.nodes.is_empty() {
            return None;
        }

        let mut total_nodes: usize = 0;
        let mut leaf_count: usize = 0;
        let mut max_depth: usize = 0;
        let mut max_children: usize = 0;
        let mut total_children: usize = 0;
        let mut internal_count: usize = 0;
        let mut sum_maxdist: f64 = 0.0;
        let mut max_maxdist: f64 = 0.0;
        let mut sum_ratio: f64 = 0.0;
        let mut ratio_count: usize = 0;
        let mut min_level: i32 = self.nodes[self.root_index].level;
        let mut max_level: i32 = min_level;
        let mut children_histogram: Vec<usize> = Vec::new();

        // BFS through packed arrays
        let mut stack: Vec<(usize, usize)> = vec![(self.root_index, 0)];

        while let Some((idx, depth)) = stack.pop() {
            let node = &self.nodes[idx];
            total_nodes += 1;
            if depth > max_depth {
                max_depth = depth;
            }
            if node.level < min_level {
                min_level = node.level;
            }
            if node.level > max_level {
                max_level = node.level;
            }

            let nc = node.child_index_count as usize;
            if nc == 0 {
                leaf_count += 1;
            } else {
                internal_count += 1;
                total_children += nc;
                if nc > max_children {
                    max_children = nc;
                }
            }

            if nc >= children_histogram.len() {
                children_histogram.resize(nc + 1, 0);
            }
            children_histogram[nc] += 1;

            sum_maxdist += node.maxdist;
            if node.maxdist > max_maxdist {
                max_maxdist = node.maxdist;
            }

            if nc > 0 {
                let covdist = self.base.powi(node.level);
                if covdist > 0.0 {
                    sum_ratio += node.maxdist / covdist;
                    ratio_count += 1;
                }
            }

            let child_range = node.child_index_range();
            for &child_idx in &self.child_indices[child_range] {
                stack.push((child_idx, depth + 1));
            }
        }

        Some(crate::node::TreeStats {
            total_nodes,
            leaf_count,
            max_depth,
            avg_children: if internal_count > 0 {
                total_children as f64 / internal_count as f64
            } else {
                0.0
            },
            max_children,
            avg_maxdist: if total_nodes > 0 {
                sum_maxdist / total_nodes as f64
            } else {
                0.0
            },
            max_maxdist,
            children_histogram,
            avg_maxdist_covdist_ratio: if ratio_count > 0 {
                sum_ratio / ratio_count as f64
            } else {
                0.0
            },
            level_span: max_level - min_level,
        })
    }

    /// Build a packed tree using incremental insertion followed by packing.
    ///
    /// This produces significantly better tree structure than `from_batch()` because
    /// incremental insertion creates tighter pruning bounds (lower `maxdist` values),
    /// resulting in 4-8x fewer distance computations on held-out queries.
    ///
    /// # Algorithm
    ///
    /// 1. Insert points one at a time into a `SimplifiedCoverTree`
    /// 2. Recompute exact `maxdist` values (bottom-up)
    /// 3. Pack into cache-optimized contiguous layout
    ///
    /// # Arguments
    ///
    /// * `points` - Points to insert
    /// * `metric` - Distance metric (must be `Clone` for the intermediate tree)
    /// * `base` - Base value for covdist/sepdist calculations
    ///
    /// # Panics
    ///
    /// Panics if `base` is not finite or is below [`MIN_BASE`](crate::MIN_BASE).
    pub fn from_incremental(points: Vec<T>, metric: D, base: f64) -> Self
    where
        D: Clone,
    {
        let mut tree = SimplifiedCoverTree::new(metric, base);
        for p in points {
            tree.insert(p);
        }
        tree.recompute_maxdist();
        tree.pack()
    }

    /// Get the root node index in the packed array.
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn root_index(&self) -> usize {
        self.root_index
    }

    /// Number of nodes in the packed array (including structural duplicates).
    #[inline]
    pub(crate) fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get a reference to a packed node by index.
    ///
    /// Used by the packed dual-tree traversal for index-based reference-side access.
    #[inline(always)]
    pub(crate) fn node(&self, idx: usize) -> &PackedNode<T> {
        &self.nodes[idx]
    }

    /// Get the child indices for a given node.
    ///
    /// Returns a contiguous slice of child node indices from the shared
    /// `child_indices` array. Sequential access to this slice enables
    /// CPU prefetching for cache-efficient traversal.
    #[inline(always)]
    pub(crate) fn children_of(&self, idx: usize) -> &[usize] {
        let node = &self.nodes[idx];
        &self.child_indices[node.child_index_range()]
    }

    /// Find the nearest neighbor to a query point
    ///
    /// Returns `None` if tree is empty, otherwise returns reference to nearest point.
    ///
    /// # Performance
    ///
    /// This implementation is 15-25% faster than unpacked trees due to cache locality.
    /// The algorithm is identical to unpacked find_nearest, but memory access patterns
    /// are dramatically improved.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let packed = tree.pack();
    /// let nearest = packed.find_nearest(&query_point);
    /// assert!(nearest.is_some());
    /// ```
    pub fn find_nearest(&self, query: &T) -> Option<&T> {
        if self.nodes.is_empty() {
            return None;
        }

        let root = &self.nodes[self.root_index];
        let mut best = &root.point;
        let mut best_dist = self.metric.distance(query, best);

        self.find_nearest_internal(
            self.root_index,
            query,
            &mut best,
            &mut best_dist,
        );

        Some(best)
    }

    /// Internal recursive helper for find_nearest
    ///
    /// This is where cache optimization shines - children are stored sequentially
    /// so iterating over them hits the same cache lines.
    fn find_nearest_internal<'a>(
        &'a self,
        node_idx: usize,
        query: &T,
        best: &mut &'a T,
        best_dist: &mut f64,
    ) {
        let node = &self.nodes[node_idx];

        // Check if this node is better
        let dist = self.metric.distance(query, &node.point);
        if dist < *best_dist {
            *best = &node.point;
            *best_dist = dist;
        }

        // Collect and sort children by distance before recursion (critical optimization!)
        //
        // This is the key optimization: child indices are stored contiguously
        // in self.child_indices[child_index_start..child_index_start + child_index_count].
        // Sequential access to this array and then to nodes enables CPU prefetching.
        let child_idx_range = node.child_index_range();
        let children_indices = &self.child_indices[child_idx_range];

        // Collect children with distances and prune
        #[cfg(not(feature = "no-smallvec"))]
        let mut child_dists: SmallVec<[(f64, usize); 16]> = SmallVec::new();
        #[cfg(feature = "no-smallvec")]
        let mut child_dists: Vec<(f64, usize)> = Vec::new();
        for &child_idx in children_indices {
            let child = &self.nodes[child_idx];

            // Self-child optimization: reuse parent's distance for self-children
            let child_dist = if child.is_duplicate && child.d_parent == 0.0 {
                dist
            } else {
                // Triangle inequality shell test
                if cfg!(not(feature = "no-triangle-filter")) && child.d_parent > 0.0 {
                    let lower_bound = (dist - child.d_parent).max(0.0);
                    if lower_bound - child.maxdist >= *best_dist {
                        continue;
                    }
                }

                self.metric.distance_with_bound(query, &child.point, *best_dist + child.maxdist)
            };

            if child_dist - child.maxdist < *best_dist {
                child_dists.push((child_dist, child_idx));
            }
        }

        // Halfsort: for large child sets, only sort the closer half fully.
        // With `no-halfsort` feature, always use full sort for A/B benchmarking.
        #[cfg(feature = "no-halfsort")]
        {
            child_dists.sort_by(|a, b| a.0.total_cmp(&b.0));
        }
        #[cfg(not(feature = "no-halfsort"))]
        {
            if child_dists.len() > 12 {
                let mid = child_dists.len() / 2;
                child_dists.as_mut_slice()
                    .select_nth_unstable_by(mid, |a, b| a.0.total_cmp(&b.0));
                child_dists[..mid].sort_by(|a, b| a.0.total_cmp(&b.0));
            } else {
                child_dists.sort_by(|a, b| a.0.total_cmp(&b.0));
            }
        }

        // Visit children in sorted order
        for (child_dist, child_idx) in child_dists {
            // Re-check pruning (best_dist may have improved)
            // Note: We use continue, not break, because children with larger maxdist
            // might still be explorable even if they're farther away
            if child_dist - self.nodes[child_idx].maxdist >= *best_dist {
                continue;
            }

            self.find_nearest_internal(child_idx, query, best, best_dist);
        }
    }

    /// Find k nearest neighbors to a query point
    ///
    /// Returns up to k nearest neighbors as (point, distance) tuples, sorted by distance.
    ///
    /// # Duplicate Handling
    ///
    /// Algorithm-created duplicates (from tree merging) are automatically skipped.
    /// User's intentional duplicate values are correctly included in results.
    ///
    /// # Performance
    ///
    /// This implementation is 15-25% faster than unpacked k-NN due to cache locality.
    /// The algorithm is identical, but memory layout dramatically improves performance.
    ///
    /// # Arguments
    ///
    /// * `query` - The query point
    /// * `k` - Number of nearest neighbors to find
    ///
    /// # Returns
    ///
    /// Vector of up to k (point reference, distance) pairs, sorted ascending by distance.
    /// Returns empty vector if tree is empty or k=0.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let packed = tree.pack();
    /// let neighbors = packed.find_k_nearest(&query, 5);
    /// for (point, dist) in neighbors {
    ///     println!("Point {:?} at distance {}", point, dist);
    /// }
    /// ```
    pub fn find_k_nearest(&self, query: &T, k: usize) -> Vec<(&T, f64)> {
        use crate::knn::KnnState;

        if k == 0 || self.nodes.is_empty() {
            return Vec::new();
        }

        let mut state = KnnState::new(k);
        self.find_k_nearest_internal(self.root_index, query, &mut state);
        state.into_sorted_vec()
    }

    /// Internal recursive k-NN search
    ///
    /// This is where cache optimization shines - sequential child iteration
    /// hits same cache lines, dramatically reducing memory latency.
    ///
    /// Algorithm-created duplicates (is_duplicate=true) are skipped automatically.
    fn find_k_nearest_internal<'a>(
        &'a self,
        node_idx: usize,
        query: &T,
        state: &mut crate::knn::KnnState<'a, T>,
    ) {
        let node = &self.nodes[node_idx];

        // 1. Check this node as candidate (skip algorithm duplicates)
        let node_dist = self.metric.distance(&node.point, query);
        if !node.is_duplicate {
            state.insert(&node.point, node_dist);
        }

        // 2. If leaf, done
        if node.child_index_count == 0 {
            return;
        }

        // 3. Collect and prune children (cache-friendly sequential access!)
        //
        // Access shared child_indices array - sequential access pattern enables
        // CPU prefetching and keeps working set in L1/L2 cache
        let child_idx_range = node.child_index_range();
        let children_indices = &self.child_indices[child_idx_range];

        #[cfg(not(feature = "no-smallvec"))]
        let mut child_dists: SmallVec<[(f64, usize); 16]> = SmallVec::new();
        #[cfg(feature = "no-smallvec")]
        let mut child_dists: Vec<(f64, usize)> = Vec::new();
        for &child_idx in children_indices {
            let child = &self.nodes[child_idx];

            // Self-child optimization: reuse parent's distance for self-children
            let child_dist = if child.is_duplicate && child.d_parent == 0.0 {
                node_dist
            } else {
                // Triangle inequality shell test
                if cfg!(not(feature = "no-triangle-filter")) && child.d_parent > 0.0 {
                    let lower_bound = (node_dist - child.d_parent).max(0.0);
                    if (lower_bound - child.maxdist).max(0.0) > state.kth_distance() {
                        continue;
                    }
                }

                self.metric.distance_with_bound(&child.point, query, state.kth_distance() + child.maxdist)
            };

            let min_possible_dist = (child_dist - child.maxdist).max(0.0);
            if min_possible_dist <= state.kth_distance() {
                child_dists.push((child_dist, child_idx));
            }
        }

        // Halfsort: for large child sets, only sort the closer half fully.
        // With `no-halfsort` feature, always use full sort for A/B benchmarking.
        #[cfg(feature = "no-halfsort")]
        {
            child_dists.sort_by(|a, b| a.0.total_cmp(&b.0));
        }
        #[cfg(not(feature = "no-halfsort"))]
        {
            if child_dists.len() > 12 {
                let mid = child_dists.len() / 2;
                child_dists.as_mut_slice()
                    .select_nth_unstable_by(mid, |a, b| a.0.total_cmp(&b.0));
                child_dists[..mid].sort_by(|a, b| a.0.total_cmp(&b.0));
            } else {
                child_dists.sort_by(|a, b| a.0.total_cmp(&b.0));
            }
        }

        // 5. Recursively search children
        for (child_dist, idx) in child_dists {
            let child = &self.nodes[idx];

            // Re-check pruning (kth_distance may have improved)
            // Note: We use continue, not break, because children with larger maxdist
            // might still be explorable even if they're farther away
            let min_possible_dist = (child_dist - child.maxdist).max(0.0);
            if min_possible_dist > state.kth_distance() {
                continue;
            }

            self.find_k_nearest_internal(idx, query, state);
        }
    }

    /// Find k nearest neighbors for a batch of query points using dual-tree traversal.
    ///
    /// Builds a temporary query tree from the query points, then runs a packed
    /// dual-tree traversal (unpacked query tree + packed reference tree). This is
    /// significantly faster than running `find_k_nearest` for each query independently,
    /// especially for large batches.
    ///
    /// # Arguments
    ///
    /// * `queries` - Slice of query points
    /// * `k` - Number of nearest neighbors to find per query point
    ///
    /// # Returns
    ///
    /// `Vec<Vec<(&T, f64)>>` — one entry per input query (in the same order as `queries`),
    /// each containing up to k neighbors sorted by distance. Points reference the packed
    /// tree's internal storage.
    ///
    /// # Requires
    ///
    /// `D: Clone` — the metric must be cloneable to construct the temporary query tree.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let packed = tree.pack();
    /// let queries = vec![5.0, 15.0, 25.0];
    /// let results = packed.find_k_nearest_batch(&queries, 3);
    /// assert_eq!(results.len(), 3);
    /// ```
    pub fn find_k_nearest_batch(&self, queries: &[T], k: usize) -> Vec<Vec<(&T, f64)>>
    where
        D: Clone,
    {
        use crate::simplified::SimplifiedCoverTree;
        use crate::core::dual_tree::knn_rules::KnnRules;
        use super::dual_tree::PackedDualTreeTraversal;

        if k == 0 || queries.is_empty() || self.nodes.is_empty() {
            return vec![Vec::new(); queries.len()];
        }

        // For small query sets, single-tree queries avoid dual-tree overhead
        const SINGLE_TREE_THRESHOLD: usize = 16;
        if queries.len() <= SINGLE_TREE_THRESHOLD {
            return queries.iter()
                .map(|q| self.find_k_nearest(q, k))
                .collect();
        }

        // Build a temporary query tree, tracking which pointer belongs to which input index.
        let mut query_tree = SimplifiedCoverTree::new(self.metric.clone(), self.base);
        let mut ptr_to_input: Vec<(*const T, Vec<usize>)> = Vec::with_capacity(queries.len());
        for (qi, q) in queries.iter().enumerate() {
            let ptr = query_tree.insert_returning_ptr(q.clone());
            match ptr_to_input.binary_search_by_key(&(ptr as usize), |&(p, _)| p as usize) {
                Ok(pos) => ptr_to_input[pos].1.push(qi),
                Err(pos) => ptr_to_input.insert(pos, (ptr, vec![qi])),
            }
        }
        query_tree.recompute_maxdist();

        let query_root = match query_tree.root_node() {
            Some(r) => r,
            None => return vec![Vec::new(); queries.len()],
        };

        // Run packed DFS dual-tree traversal
        let mut rules = KnnRules::new(k, &self.metric, false);
        rules.state.init_parent_map(query_root);
        PackedDualTreeTraversal::traverse(query_root, self, self.root_index, &mut rules);

        // Collect results
        let (mut results_vec, query_state_indices, query_ptrs) = rules.state.into_results();

        let mut output: Vec<Vec<(&T, f64)>> = vec![Vec::new(); queries.len()];
        for (qi, &ptr) in query_ptrs.iter().enumerate() {
            let idx = query_state_indices[qi];
            let results = std::mem::take(&mut results_vec[idx]);
            if let Ok(pos) = ptr_to_input.binary_search_by_key(&(ptr as usize), |&(p, _)| p as usize) {
                let input_indices = &ptr_to_input[pos].1;
                if input_indices.len() == 1 {
                    output[input_indices[0]] = results;
                } else {
                    for &qi in &input_indices[..input_indices.len() - 1] {
                        output[qi] = results.clone();
                    }
                    output[*input_indices.last().unwrap()] = results;
                }
            }
        }

        for (i, result) in output.iter_mut().enumerate() {
            if result.is_empty() {
                *result = self.find_k_nearest(&queries[i], k);
            }
        }

        output
    }

    /// Instrumented version of `find_k_nearest_batch` that returns timing breakdowns
    /// and traversal statistics alongside the results.
    ///
    /// Separates query tree construction, maxdist recomputation, parent map init,
    /// traversal, and result collection into individually-timed phases. Also reports
    /// distance computation count and bound cache hit/miss rates.
    pub fn find_k_nearest_batch_instrumented(
        &self,
        queries: &[T],
        k: usize,
    ) -> (Vec<Vec<(&T, f64)>>, crate::core::dual_tree::DualTreeStats)
    where
        D: Clone,
    {
        use std::time::Instant;
        use crate::simplified::SimplifiedCoverTree;
        use crate::core::dual_tree::DualTreeStats;
        use crate::core::dual_tree::knn_rules::KnnRules;
        use crate::distance::{reset_distance_count, get_distance_count};
        use super::dual_tree::PackedDualTreeTraversal;

        let mut stats = DualTreeStats::default();

        if k == 0 || queries.is_empty() || self.nodes.is_empty() {
            return (vec![Vec::new(); queries.len()], stats);
        }

        // Phase 1: Build query tree
        let t0 = Instant::now();
        let mut query_tree = SimplifiedCoverTree::new(self.metric.clone(), self.base);
        let mut ptr_to_input: Vec<(*const T, Vec<usize>)> = Vec::with_capacity(queries.len());
        for (qi, q) in queries.iter().enumerate() {
            let ptr = query_tree.insert_returning_ptr(q.clone());
            match ptr_to_input.binary_search_by_key(&(ptr as usize), |&(p, _)| p as usize) {
                Ok(pos) => ptr_to_input[pos].1.push(qi),
                Err(pos) => ptr_to_input.insert(pos, (ptr, vec![qi])),
            }
        }
        stats.query_tree_build_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Phase 2: Recompute maxdist
        let t1 = Instant::now();
        query_tree.recompute_maxdist();
        stats.query_tree_maxdist_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let query_root = match query_tree.root_node() {
            Some(r) => r,
            None => return (vec![Vec::new(); queries.len()], stats),
        };

        // Phase 3: Init parent map
        let t2 = Instant::now();
        let mut rules = KnnRules::new(k, &self.metric, false);
        rules.state.init_parent_map(query_root);
        stats.init_parent_map_ms = t2.elapsed().as_secs_f64() * 1000.0;

        // Phase 4: Traversal (with distance counter)
        reset_distance_count();
        let t3 = Instant::now();
        PackedDualTreeTraversal::traverse(query_root, self, self.root_index, &mut rules);
        stats.traversal_ms = t3.elapsed().as_secs_f64() * 1000.0;
        stats.distance_computations = get_distance_count();

        let (hits, misses) = rules.bound_cache_stats();
        stats.bound_cache_hits = hits;
        stats.bound_cache_misses = misses;

        // Phase 5: Result collection
        let t4 = Instant::now();
        let (mut results_vec, query_state_indices, query_ptrs) = rules.state.into_results();
        let mut output: Vec<Vec<(&T, f64)>> = vec![Vec::new(); queries.len()];
        for (qi, &ptr) in query_ptrs.iter().enumerate() {
            let idx = query_state_indices[qi];
            let results = std::mem::take(&mut results_vec[idx]);
            if let Ok(pos) = ptr_to_input.binary_search_by_key(&(ptr as usize), |&(p, _)| p as usize) {
                let input_indices = &ptr_to_input[pos].1;
                if input_indices.len() == 1 {
                    output[input_indices[0]] = results;
                } else {
                    for &qi in &input_indices[..input_indices.len() - 1] {
                        output[qi] = results.clone();
                    }
                    output[*input_indices.last().unwrap()] = results;
                }
            }
        }
        for (i, result) in output.iter_mut().enumerate() {
            if result.is_empty() {
                *result = self.find_k_nearest(&queries[i], k);
            }
        }
        stats.result_collection_ms = t4.elapsed().as_secs_f64() * 1000.0;

        (output, stats)
    }

    /// Like `find_k_nearest_batch_instrumented`, but accepts an explicit `BoundMode`.
    ///
    /// The bound mode is passed through to the DFS dual-tree traversal's `KnnRules`.
    pub fn find_k_nearest_batch_with_bound_instrumented(
        &self,
        queries: &[T],
        k: usize,
        bound_mode: crate::core::dual_tree::BoundMode,
    ) -> (Vec<Vec<(&T, f64)>>, crate::core::dual_tree::DualTreeStats)
    where
        D: Clone,
    {
        use std::time::Instant;
        use crate::simplified::SimplifiedCoverTree;
        use crate::core::dual_tree::DualTreeStats;
        use crate::core::dual_tree::knn_rules::KnnRules;
        use crate::distance::{reset_distance_count, get_distance_count};
        use super::dual_tree::PackedDualTreeTraversal;

        let mut stats = DualTreeStats::default();

        if k == 0 || queries.is_empty() || self.nodes.is_empty() {
            return (vec![Vec::new(); queries.len()], stats);
        }

        // Phase 1: Build query tree
        let t0 = Instant::now();
        let mut query_tree = SimplifiedCoverTree::new(self.metric.clone(), self.base);
        let mut ptr_to_input: Vec<(*const T, Vec<usize>)> = Vec::with_capacity(queries.len());
        for (qi, q) in queries.iter().enumerate() {
            let ptr = query_tree.insert_returning_ptr(q.clone());
            match ptr_to_input.binary_search_by_key(&(ptr as usize), |&(p, _)| p as usize) {
                Ok(pos) => ptr_to_input[pos].1.push(qi),
                Err(pos) => ptr_to_input.insert(pos, (ptr, vec![qi])),
            }
        }
        stats.query_tree_build_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Phase 2: Recompute maxdist
        let t1 = Instant::now();
        query_tree.recompute_maxdist();
        stats.query_tree_maxdist_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let query_root = match query_tree.root_node() {
            Some(r) => r,
            None => return (vec![Vec::new(); queries.len()], stats),
        };

        // Phase 3: Init parent map
        let t2 = Instant::now();
        let mut rules = KnnRules::new_with_bound(k, &self.metric, false, bound_mode);
        rules.state.init_parent_map(query_root);
        stats.init_parent_map_ms = t2.elapsed().as_secs_f64() * 1000.0;

        // Phase 4: Traversal (with distance counter)
        reset_distance_count();
        let t3 = Instant::now();
        PackedDualTreeTraversal::traverse(query_root, self, self.root_index, &mut rules);
        stats.traversal_ms = t3.elapsed().as_secs_f64() * 1000.0;
        stats.distance_computations = get_distance_count();

        let (hits, misses) = rules.bound_cache_stats();
        stats.bound_cache_hits = hits;
        stats.bound_cache_misses = misses;

        // Phase 5: Result collection
        let t4 = Instant::now();
        let (mut results_vec, query_state_indices, query_ptrs) = rules.state.into_results();
        let mut output: Vec<Vec<(&T, f64)>> = vec![Vec::new(); queries.len()];
        for (qi, &ptr) in query_ptrs.iter().enumerate() {
            let idx = query_state_indices[qi];
            let results = std::mem::take(&mut results_vec[idx]);
            if let Ok(pos) = ptr_to_input.binary_search_by_key(&(ptr as usize), |&(p, _)| p as usize) {
                let input_indices = &ptr_to_input[pos].1;
                if input_indices.len() == 1 {
                    output[input_indices[0]] = results;
                } else {
                    for &qi in &input_indices[..input_indices.len() - 1] {
                        output[qi] = results.clone();
                    }
                    output[*input_indices.last().unwrap()] = results;
                }
            }
        }
        for (i, result) in output.iter_mut().enumerate() {
            if result.is_empty() {
                *result = self.find_k_nearest(&queries[i], k);
            }
        }
        stats.result_collection_ms = t4.elapsed().as_secs_f64() * 1000.0;

        (output, stats)
    }

    /// Dual-tree k-NN using DFS-both-sides traversal with unpacked trees.
    ///
    /// Builds an unpacked query tree from the input queries and unpacks the
    /// packed reference tree, then runs `DualTreeTraversal::traverse()` between
    /// them. This uses unpacked node trees on both sides (unlike
    /// `find_k_nearest_batch_instrumented` which keeps the reference packed).
    ///
    /// Returns only timing stats (no result references) because the unpacked
    /// reference tree is a local copy and cannot be borrowed into the output.
    /// This is intended for benchmarking only.
    pub fn find_k_nearest_batch_dfs_instrumented(
        &self,
        queries: &[T],
        k: usize,
    ) -> (f64, crate::core::dual_tree::DualTreeStats)
    where
        D: Clone,
    {
        use std::time::Instant;
        use crate::simplified::SimplifiedCoverTree;
        use crate::core::dual_tree::{DualTreeStats, traversal::DualTreeTraversal};
        use crate::core::dual_tree::knn_rules::KnnRules;
        use crate::distance::{reset_distance_count, get_distance_count};

        let t_total = Instant::now();
        let mut stats = DualTreeStats::default();

        if k == 0 || queries.is_empty() || self.nodes.is_empty() {
            return (0.0, stats);
        }

        // Phase 1: Build unpacked query tree
        let t0 = Instant::now();
        let mut query_tree = SimplifiedCoverTree::new(self.metric.clone(), self.base);
        for q in queries.iter() {
            query_tree.insert(q.clone());
        }
        stats.query_tree_build_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Phase 2: Recompute maxdist on query tree
        let t1 = Instant::now();
        query_tree.recompute_maxdist();
        stats.query_tree_maxdist_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let query_root = match query_tree.root_node() {
            Some(r) => r,
            None => return (t_total.elapsed().as_secs_f64() * 1000.0, stats),
        };

        // Phase 3: Unpack reference tree + init parent map
        let t2 = Instant::now();
        let ref_root = match self.unpack_to_node_tree() {
            Some(r) => r,
            None => return (t_total.elapsed().as_secs_f64() * 1000.0, stats),
        };
        // same_set = false: query and reference are different point sets
        let mut rules = KnnRules::new(k, &self.metric, false);
        rules.state.init_parent_map(query_root);
        stats.init_parent_map_ms = t2.elapsed().as_secs_f64() * 1000.0;

        // Phase 4: DFS-both-sides traversal
        reset_distance_count();
        let t3 = Instant::now();
        DualTreeTraversal::traverse(query_root, &ref_root, &mut rules);
        stats.traversal_ms = t3.elapsed().as_secs_f64() * 1000.0;
        stats.distance_computations = get_distance_count();

        let (hits, misses) = rules.bound_cache_stats();
        stats.bound_cache_hits = hits;
        stats.bound_cache_misses = misses;

        // Phase 5: Result collection (drop results since we can't return refs)
        let t4 = Instant::now();
        let _ = rules.state.into_results();
        stats.result_collection_ms = t4.elapsed().as_secs_f64() * 1000.0;

        let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
        (total_ms, stats)
    }

    /// Dual-tree k-NN with a user-supplied query tree.
    ///
    /// Uses the provided unpacked query tree and this packed tree (as reference)
    /// for dual-tree k-NN search. Returns results for each non-duplicate point
    /// in the query tree in DFS order.
    ///
    /// # Arguments
    ///
    /// * `query_tree` - The unpacked query cover tree
    /// * `k` - Number of nearest neighbors to find per query point
    ///
    /// # Returns
    ///
    /// `Vec<Vec<(&T, f64)>>` — one entry per non-duplicate point in the query tree
    /// (DFS order), each containing up to k neighbors sorted by distance.
    ///
    /// # Panics
    ///
    /// Panics if the query tree and reference tree have different base values.
    pub fn find_k_nearest_dual(
        &self,
        query_tree: &crate::simplified::SimplifiedCoverTree<T, D>,
        k: usize,
    ) -> Vec<Vec<(&T, f64)>> {
        use crate::core::dual_tree::knn_rules::KnnRules;
        use super::dual_tree::PackedDualTreeTraversal;

        assert_eq!(
            self.base, query_tree.base_value(),
            "Cannot run dual-tree k-NN with different base values: {} != {}",
            self.base, query_tree.base_value()
        );

        let query_root = match query_tree.root_node() {
            Some(r) => r,
            None => return Vec::new(),
        };

        if k == 0 || self.nodes.is_empty() {
            return Vec::new();
        }

        let mut rules = KnnRules::new(k, &self.metric, false);
        rules.state.init_parent_map(query_root);
        PackedDualTreeTraversal::traverse(query_root, self, self.root_index, &mut rules);

        let (mut results_vec, query_state_indices, _query_ptrs) = rules.state.into_results();

        query_state_indices
            .iter()
            .map(|&idx| std::mem::take(&mut results_vec[idx]))
            .collect()
    }

    /// Same-set k-NN: find k nearest neighbors for each point in the tree itself.
    ///
    /// Uses fully-packed dual-tree traversal where both query and reference sides
    /// index directly into this packed tree's contiguous arrays. No unpacked query
    /// tree is built — zero construction overhead.
    ///
    /// Self-matches are NOT excluded (the query side is the same tree data but
    /// indexed independently).
    ///
    /// # Arguments
    ///
    /// * `k` - Number of nearest neighbors to find per point
    ///
    /// # Returns
    ///
    /// `Vec<Vec<(&T, f64)>>` — one entry per non-duplicate point in the tree
    /// (in DFS order), each containing up to k neighbors sorted by distance.
    pub fn find_k_nearest_self(&self, k: usize) -> Vec<Vec<(&T, f64)>> {
        use crate::core::dual_tree::knn_rules::KnnRules;
        use super::dual_tree_packed::FullyPackedDualTreeTraversal;

        if k == 0 || self.nodes.is_empty() {
            return Vec::new();
        }

        // Full Curtin B1/B2 bounds: a query node's bound must hold for every query
        // point below it, not just its own point, even when query = reference.
        // same_set=true so each point skips itself as a neighbor candidate
        let mut rules = KnnRules::new(k, &self.metric, true);
        rules.state.init_parent_map_packed(self, self.root_index);
        FullyPackedDualTreeTraversal::traverse(self, 0, self.root_index, &mut rules);

        let (mut results_vec, query_state_indices, _query_ptrs) = rules.state.into_results();

        query_state_indices
            .iter()
            .map(|&idx| std::mem::take(&mut results_vec[idx]))
            .collect()
    }

    /// Instrumented version of `find_k_nearest_self` that returns timing breakdowns
    /// and traversal statistics alongside the results.
    ///
    /// Uses fully-packed dual-tree traversal — `query_tree_build_ms` should be ~0
    /// since no query tree is constructed.
    pub fn find_k_nearest_self_instrumented(
        &self,
        k: usize,
    ) -> (Vec<Vec<(&T, f64)>>, crate::core::dual_tree::DualTreeStats) {
        use std::time::Instant;
        use crate::core::dual_tree::knn_rules::KnnRules;
        use crate::core::dual_tree::DualTreeStats;
        use crate::distance::{reset_distance_count, get_distance_count};
        use super::dual_tree_packed::FullyPackedDualTreeTraversal;

        let mut stats = DualTreeStats::default();

        if k == 0 || self.nodes.is_empty() {
            return (Vec::new(), stats);
        }

        // Phase 1: No query tree build needed — fully packed self-query
        stats.query_tree_build_ms = 0.0;
        stats.query_tree_maxdist_ms = 0.0;

        // Phase 2: Init parent map (walks packed array by index)
        let t2 = Instant::now();
        // Full Curtin B1/B2 bounds: a query node's bound must hold for every query
        // point below it, not just its own point, even when query = reference.
        // same_set=true so each point skips itself as a neighbor candidate
        let mut rules = KnnRules::new(k, &self.metric, true);
        rules.state.init_parent_map_packed(self, self.root_index);
        stats.init_parent_map_ms = t2.elapsed().as_secs_f64() * 1000.0;

        // Phase 3: Traversal (with distance counter)
        reset_distance_count();
        let t3 = Instant::now();
        FullyPackedDualTreeTraversal::traverse(self, 0, self.root_index, &mut rules);
        stats.traversal_ms = t3.elapsed().as_secs_f64() * 1000.0;
        stats.distance_computations = get_distance_count();

        let (hits, misses) = rules.bound_cache_stats();
        stats.bound_cache_hits = hits;
        stats.bound_cache_misses = misses;

        // Phase 4: Result collection
        let t4 = Instant::now();
        let (mut results_vec, query_state_indices, _query_ptrs) = rules.state.into_results();
        let results: Vec<Vec<(&T, f64)>> = query_state_indices
            .iter()
            .map(|&idx| std::mem::take(&mut results_vec[idx]))
            .collect();
        stats.result_collection_ms = t4.elapsed().as_secs_f64() * 1000.0;

        (results, stats)
    }

    // ---------------------------------------------------------------------------
    // Unpacking (O(n) DFS structure copy for batch-single queries)
    // ---------------------------------------------------------------------------

    /// Convert the packed tree back to an unpacked Node<T> tree in O(n) time.
    ///
    /// This is used by batch-single queries which need an unpacked query tree.
    /// Much faster than rebuilding via insert-one-by-one (which is O(n² log n)).
    fn unpack_to_node_tree(&self) -> Option<Box<crate::node::Node<T>>> {
        if self.nodes.is_empty() {
            return None;
        }
        Some(self.unpack_subtree(self.root_index))
    }

    /// Recursively unpack a subtree rooted at `idx` into a Node<T> tree.
    fn unpack_subtree(&self, idx: usize) -> Box<crate::node::Node<T>> {
        let packed = &self.nodes[idx];
        let mut node = Box::new(crate::node::Node::new(
            packed.point.clone(),
            packed.level,
            packed.is_duplicate,
        ));
        node.maxdist = packed.maxdist;
        node.d_parent = packed.d_parent;
        node.children = self.children_of(idx)
            .iter()
            .map(|&ci| self.unpack_subtree(ci))
            .collect();
        node
    }

    // ---------------------------------------------------------------------------
    // Batch single-tree queries
    // ---------------------------------------------------------------------------

    /// Batch single-tree all-nearest-neighbors: for every point in the tree, its k
    /// nearest other points.
    ///
    /// Walks the tree as a query tree against itself, performing single-tree
    /// reference descent at each scale level; each query child receives an
    /// independently filtered copy of its parent's reference candidates.
    ///
    /// # Returns
    ///
    /// One `(point, neighbors)` pair per point, in no particular order. Neighbors are
    /// sorted by distance, closest first, and never include the point itself
    /// (other points at distance zero are included).
    pub fn find_k_nearest_batch_single_self(&self, k: usize) -> Vec<(&T, Vec<(&T, f64)>)> {
        crate::core::batch_single::batch_single_tree_knn_packed_self(self, k)
            .into_iter()
            .map(|(idx, row)| (&self.node(idx).point, row))
            .collect()
    }

    /// Batch single-tree k-NN for held-out query points.
    ///
    /// Builds a cover tree over `queries` and walks it against this tree, performing
    /// single-tree reference descent at each scale level.
    ///
    /// # Returns
    ///
    /// One entry per input query (in the same order as `queries`), each containing up
    /// to k neighbors sorted by distance, closest first.
    pub fn find_k_nearest_batch_single(&self, queries: &[T], k: usize) -> Vec<Vec<(&T, f64)>>
    where
        D: Clone,
    {
        let mut output: Vec<Vec<(&T, f64)>> = vec![Vec::new(); queries.len()];
        if k == 0 || queries.is_empty() || self.nodes.is_empty() {
            return output;
        }
        let (query_tree, ptr_to_input) = self.build_query_tree(queries);
        let Some(query_root) = query_tree.root_node() else {
            return output;
        };
        let rows = crate::core::batch_single::batch_single_tree_knn_packed(query_root, self, k);
        for (ptr, row) in rows {
            if let Ok(pos) = ptr_to_input.binary_search_by_key(&(ptr as usize), |&(p, _)| p as usize) {
                for &qi in &ptr_to_input[pos].1 {
                    output[qi] = row.clone();
                }
            }
        }
        output
    }

    /// Builds a cover tree over `queries`, returning it with a map from each tree
    /// point's address to the input indices that produced it (sorted by address).
    #[allow(clippy::type_complexity)]
    fn build_query_tree(
        &self,
        queries: &[T],
    ) -> (crate::simplified::SimplifiedCoverTree<T, D>, Vec<(*const T, Vec<usize>)>)
    where
        D: Clone,
    {
        let mut query_tree = crate::simplified::SimplifiedCoverTree::new(self.metric.clone(), self.base);
        let mut ptr_to_input: Vec<(*const T, Vec<usize>)> = Vec::with_capacity(queries.len());
        for (qi, q) in queries.iter().enumerate() {
            let ptr = query_tree.insert_returning_ptr(q.clone());
            match ptr_to_input.binary_search_by_key(&(ptr as usize), |&(p, _)| p as usize) {
                Ok(pos) => ptr_to_input[pos].1.push(qi),
                Err(pos) => ptr_to_input.insert(pos, (ptr, vec![qi])),
            }
        }
        query_tree.recompute_all();
        (query_tree, ptr_to_input)
    }

    /// Batch single-tree self-query (instrumented): runs
    /// [`find_k_nearest_batch_single_self`](Self::find_k_nearest_batch_single_self)
    /// and returns (elapsed_ms, distance_computations). Distance computations are
    /// only counted with the `instrument` feature.
    pub fn batch_single_self_query_instrumented(&self, k: usize) -> (f64, u64) {
        use std::time::Instant;
        use crate::distance::{reset_distance_count, get_distance_count};

        if k == 0 || self.nodes.is_empty() {
            return (0.0, 0);
        }
        reset_distance_count();
        let t0 = Instant::now();
        let results = crate::core::batch_single::batch_single_tree_knn_packed_self(self, k);
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let dist_count = get_distance_count();
        drop(results);
        (elapsed_ms, dist_count)
    }

    /// Batch single-tree held-out query (instrumented): builds a query tree from the
    /// given query points (not timed), then runs the batch single-tree search against
    /// this tree. Returns (elapsed_ms, distance_computations). Distance computations
    /// are only counted with the `instrument` feature.
    pub fn batch_single_batch_query_instrumented(&self, queries: &[T], k: usize) -> (f64, u64)
    where
        D: Clone,
    {
        use std::time::Instant;
        use crate::distance::{reset_distance_count, get_distance_count};

        if k == 0 || queries.is_empty() || self.nodes.is_empty() {
            return (0.0, 0);
        }
        let (query_tree, _) = self.build_query_tree(queries);
        let Some(query_root) = query_tree.root_node() else {
            return (0.0, 0);
        };
        reset_distance_count();
        let t0 = Instant::now();
        let results = crate::core::batch_single::batch_single_tree_knn_packed(query_root, self, k);
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let dist_count = get_distance_count();
        drop(results);
        (elapsed_ms, dist_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::node::PackedNode;

    #[derive(Clone)]
    struct SimpleDistance;

    impl Distance<f64> for SimpleDistance {
        fn distance(&self, p: &f64, q: &f64) -> f64 {
            (p - q).abs()
        }
    }

    #[test]
    fn test_packed_empty_tree() {
        let tree: PackedCoverTree<f64, SimpleDistance> = PackedCoverTree::new(
            vec![],
            vec![],
            0,
            SimpleDistance,
            1.3,
            0,
        );

        assert_eq!(tree.len(), 0);
        assert!(tree.is_empty());
        assert!(tree.find_nearest(&5.0).is_none());
    }

    #[test]
    fn test_packed_single_node() {
        // Single node tree
        let nodes = vec![PackedNode {
            point: 10.0,
            level: 0,

            maxdist: 0.0,
            d_parent: 0.0,
            child_index_start: 0,
            child_index_count: 0,
            is_duplicate: false,
        }];

        let tree = PackedCoverTree::new(nodes, vec![], 0, SimpleDistance, 1.3, 1);

        assert_eq!(tree.len(), 1);
        assert!(!tree.is_empty());

        let nearest = tree.find_nearest(&12.0);
        assert_eq!(nearest, Some(&10.0));
    }

    #[test]
    fn test_packed_tree_with_children() {
        // Tree structure:
        //     root(10.0, level=1)
        //     ├── child1(5.0, level=0)
        //     └── child2(15.0, level=0)
        //
        // Depth-first layout: [root, child1, child2]
        // Child indices: root's children are at indices [1, 2] stored in child_indices[0..2]
        let nodes = vec![
            PackedNode {
                point: 10.0,
                level: 1,

                maxdist: 5.0,
                d_parent: 0.0,
            child_index_start: 0,  // Indices into child_indices array
                child_index_count: 2,
                is_duplicate: false,
            },
            PackedNode {
                point: 5.0,
                level: 0,

                maxdist: 0.0,
                d_parent: 0.0,
            child_index_start: 0,
                child_index_count: 0,
                is_duplicate: false,
            },
            PackedNode {
                point: 15.0,
                level: 0,

                maxdist: 0.0,
                d_parent: 0.0,
            child_index_start: 0,
                child_index_count: 0,
                is_duplicate: false,
            },
        ];

        // child_indices[0..2] = [1, 2] -> root's children
        let child_indices = vec![1, 2];

        let tree = PackedCoverTree::new(nodes, child_indices, 0, SimpleDistance, 1.3, 3);

        assert_eq!(tree.len(), 3);

        // Query closest to 5.0
        let nearest = tree.find_nearest(&6.0);
        assert_eq!(nearest, Some(&5.0));

        // Query closest to 15.0
        let nearest = tree.find_nearest(&14.0);
        assert_eq!(nearest, Some(&15.0));

        // Query closest to 10.0
        let nearest = tree.find_nearest(&10.0);
        assert_eq!(nearest, Some(&10.0));
    }

    #[test]
    fn test_packed_tree_query_pruning() {
        // Tree with maxdist set to enable pruning test
        //
        //     root(50.0, level=5, maxdist=10.0)
        //     └── child(100.0, level=4, maxdist=0.0)
        //
        // Query at 0.0 should find root and prune child
        // (child is at distance 100, child.maxdist=0, so can't have anything closer than 100)
        let nodes = vec![
            PackedNode {
                point: 50.0,
                level: 5,

                maxdist: 50.0,  // Can reach up to distance 50 from root
                d_parent: 0.0,
            child_index_start: 0,
                child_index_count: 1,
                is_duplicate: false,
            },
            PackedNode {
                point: 100.0,
                level: 4,

                maxdist: 0.0,
                d_parent: 0.0,
            child_index_start: 0,
                child_index_count: 0,
                is_duplicate: false,
            },
        ];

        let child_indices = vec![1];  // root's child is at index 1
        let tree = PackedCoverTree::new(nodes, child_indices, 0, SimpleDistance, 1.3, 2);

        // Query at 0.0: root(50.0) is closest
        let nearest = tree.find_nearest(&0.0);
        assert_eq!(nearest, Some(&50.0));

        // Query at 90.0: child(100.0) is closest (distance 10 vs 40 to root)
        let nearest = tree.find_nearest(&90.0);
        assert_eq!(nearest, Some(&100.0));
    }

    #[test]
    fn test_packed_knn_empty_tree() {
        let tree: PackedCoverTree<f64, SimpleDistance> = PackedCoverTree::new(
            vec![],
            vec![],  // empty child_indices
            0,
            SimpleDistance,
            1.3,
            0,
        );

        let results = tree.find_k_nearest(&5.0, 3);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_packed_knn_k_zero() {
        let nodes = vec![PackedNode {
            point: 10.0,
            level: 0,

            maxdist: 0.0,
            d_parent: 0.0,
            child_index_start: 0,
            child_index_count: 0,
            is_duplicate: false,
        }];

        let child_indices = vec![];  // no children
        let tree = PackedCoverTree::new(nodes, child_indices, 0, SimpleDistance, 1.3, 1);
        let results = tree.find_k_nearest(&5.0, 0);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_packed_knn_k_one_matches_find_nearest() {
        // k=1 should return same result as find_nearest()
        let nodes = vec![
            PackedNode {
                point: 10.0,
                level: 1,

                maxdist: 10.0,
                d_parent: 0.0,
            child_index_start: 0,
                child_index_count: 2,
                is_duplicate: false,
            },
            PackedNode {
                point: 5.0,
                level: 0,

                maxdist: 0.0,
                d_parent: 0.0,
            child_index_start: 0,
                child_index_count: 0,
                is_duplicate: false,
            },
            PackedNode {
                point: 15.0,
                level: 0,

                maxdist: 0.0,
                d_parent: 0.0,
            child_index_start: 0,
                child_index_count: 0,
                is_duplicate: false,
            },
        ];

        let child_indices = vec![1, 2];  // root has children at indices 1 and 2
        let tree = PackedCoverTree::new(nodes, child_indices, 0, SimpleDistance, 1.3, 3);

        let nearest = tree.find_nearest(&12.0);
        let knn = tree.find_k_nearest(&12.0, 1);

        assert_eq!(knn.len(), 1);
        assert_eq!(knn[0].0, nearest.unwrap());
        assert_eq!(*knn[0].0, 10.0);
    }

    #[test]
    fn test_packed_knn_basic() {
        // Build small packed tree: [10, 20, 30, 40, 50]
        // Tree structure (simplified):
        //     30 (root)
        //     ├── 10
        //     ├── 20
        //     ├── 40
        //     └── 50
        let nodes = vec![
            PackedNode {
                point: 30.0,
                level: 2,

                maxdist: 20.0,
                d_parent: 0.0,
            child_index_start: 0,
                child_index_count: 4,
                is_duplicate: false,
            },
            PackedNode {
                point: 10.0,
                level: 1,

                maxdist: 0.0,
                d_parent: 0.0,
            child_index_start: 0,
                child_index_count: 0,
                is_duplicate: false,
            },
            PackedNode {
                point: 20.0,
                level: 1,

                maxdist: 0.0,
                d_parent: 0.0,
            child_index_start: 0,
                child_index_count: 0,
                is_duplicate: false,
            },
            PackedNode {
                point: 40.0,
                level: 1,

                maxdist: 0.0,
                d_parent: 0.0,
            child_index_start: 0,
                child_index_count: 0,
                is_duplicate: false,
            },
            PackedNode {
                point: 50.0,
                level: 1,

                maxdist: 0.0,
                d_parent: 0.0,
            child_index_start: 0,
                child_index_count: 0,
                is_duplicate: false,
            },
        ];

        let child_indices = vec![1, 2, 3, 4];  // root has children at indices 1-4
        let tree = PackedCoverTree::new(nodes, child_indices, 0, SimpleDistance, 1.3, 5);

        // Query: 25, k=3 → expect two at distance 5 (points 20, 30) and one at distance 15 (point 10)
        let results = tree.find_k_nearest(&25.0, 3);
        assert_eq!(results.len(), 3);

        // Should be sorted by distance; tie-breaking order between 20.0 and 30.0
        // (both at distance 5) is implementation-defined.
        assert_eq!(results[0].1, 5.0);
        assert_eq!(results[1].1, 5.0);
        // The two distance-5 points should be {20.0, 30.0} in some order
        let tie_points: Vec<f64> = results[0..2].iter().map(|(p, _)| **p).collect();
        assert!(tie_points.contains(&20.0), "Expected 20.0 in tie group, got {:?}", tie_points);
        assert!(tie_points.contains(&30.0), "Expected 30.0 in tie group, got {:?}", tie_points);
        assert_eq!(*results[2].0, 10.0); // distance 15
        assert_eq!(results[2].1, 15.0);
    }

    #[test]
    fn test_packed_knn_k_greater_than_n() {
        // k > tree.len() should return all points
        let nodes = vec![
            PackedNode {
                point: 10.0,
                level: 1,

                maxdist: 10.0,
                d_parent: 0.0,
            child_index_start: 0,
                child_index_count: 2,
                is_duplicate: false,
            },
            PackedNode {
                point: 5.0,
                level: 0,

                maxdist: 0.0,
                d_parent: 0.0,
            child_index_start: 0,
                child_index_count: 0,
                is_duplicate: false,
            },
            PackedNode {
                point: 15.0,
                level: 0,

                maxdist: 0.0,
                d_parent: 0.0,
            child_index_start: 0,
                child_index_count: 0,
                is_duplicate: false,
            },
        ];

        let child_indices = vec![1, 2];  // root has children at indices 1 and 2
        let tree = PackedCoverTree::new(nodes, child_indices, 0, SimpleDistance, 1.3, 3);

        let results = tree.find_k_nearest(&12.0, 100);
        assert_eq!(results.len(), 3); // Returns all 3 points

        // Verify they're sorted by distance from 12.0
        assert_eq!(*results[0].0, 10.0); // distance 2
        assert_eq!(*results[1].0, 15.0); // distance 3
        assert_eq!(*results[2].0, 5.0);  // distance 7
    }

    #[test]
    fn test_packed_knn_sorted_results() {
        // Verify results are always sorted (closest first)
        let nodes = vec![
            PackedNode {
                point: 100.0,
                level: 2,

                maxdist: 90.0,
                d_parent: 0.0,
            child_index_start: 0,
                child_index_count: 3,
                is_duplicate: false,
            },
            PackedNode {
                point: 10.0,
                level: 1,

                maxdist: 0.0,
                d_parent: 0.0,
            child_index_start: 0,
                child_index_count: 0,
                is_duplicate: false,
            },
            PackedNode {
                point: 50.0,
                level: 1,

                maxdist: 0.0,
                d_parent: 0.0,
            child_index_start: 0,
                child_index_count: 0,
                is_duplicate: false,
            },
            PackedNode {
                point: 30.0,
                level: 1,

                maxdist: 0.0,
                d_parent: 0.0,
            child_index_start: 0,
                child_index_count: 0,
                is_duplicate: false,
            },
        ];

        let child_indices = vec![1, 2, 3];  // root has children at indices 1-3
        let tree = PackedCoverTree::new(nodes, child_indices, 0, SimpleDistance, 1.3, 4);

        let results = tree.find_k_nearest(&25.0, 4);
        assert_eq!(results.len(), 4);

        // Verify sorted order
        assert_eq!(*results[0].0, 30.0);  // distance 5
        assert_eq!(*results[1].0, 10.0);  // distance 15
        assert_eq!(*results[2].0, 50.0);  // distance 25
        assert_eq!(*results[3].0, 100.0); // distance 75

        // Verify distances are ascending
        assert!(results[0].1 <= results[1].1);
        assert!(results[1].1 <= results[2].1);
        assert!(results[2].1 <= results[3].1);
    }

    #[test]
    fn test_packed_knn_skips_duplicates() {
        // Build tree with is_duplicate=true nodes
        // Verify k-NN skips them (returns only non-duplicate points)
        let nodes = vec![
            PackedNode {
                point: 10.0,
                level: 1,

                maxdist: 10.0,
                d_parent: 0.0,
            child_index_start: 0,
                child_index_count: 3,
                is_duplicate: false,
            },
            PackedNode {
                point: 5.0,
                level: 0,

                maxdist: 0.0,
                d_parent: 0.0,
            child_index_start: 0,
                child_index_count: 0,
                is_duplicate: true,  // Algorithm duplicate - should be skipped
            },
            PackedNode {
                point: 15.0,
                level: 0,

                maxdist: 0.0,
                d_parent: 0.0,
            child_index_start: 0,
                child_index_count: 0,
                is_duplicate: false,
            },
            PackedNode {
                point: 20.0,
                level: 0,

                maxdist: 0.0,
                d_parent: 0.0,
            child_index_start: 0,
                child_index_count: 0,
                is_duplicate: true,  // Algorithm duplicate - should be skipped
            },
        ];

        let child_indices = vec![1, 2, 3];  // root has children at indices 1-3
        let tree = PackedCoverTree::new(nodes, child_indices, 0, SimpleDistance, 1.3, 4);

        // Request k=3, but only 2 non-duplicates exist
        let results = tree.find_k_nearest(&12.0, 3);
        assert_eq!(results.len(), 2); // Only non-duplicates returned

        // Verify we got the non-duplicate points
        assert_eq!(*results[0].0, 10.0);
        assert_eq!(*results[1].0, 15.0);

        // Verify duplicates were skipped (5.0 and 20.0 not in results)
        for (point, _) in &results {
            assert_ne!(**point, 5.0);
            assert_ne!(**point, 20.0);
        }
    }

    #[test]
    fn test_packed_knn_pruning() {
        // Build tree with clear pruning opportunities
        // Verify correct k-NN even with pruning
        let nodes = vec![
            PackedNode {
                point: 50.0,
                level: 3,

                maxdist: 45.0,
                d_parent: 0.0,
            child_index_start: 0,
                child_index_count: 2,
                is_duplicate: false,
            },
            PackedNode {
                point: 10.0,
                level: 2,

                maxdist: 5.0,
                d_parent: 0.0,
            child_index_start: 2,
                child_index_count: 1,
                is_duplicate: false,
            },
            PackedNode {
                point: 100.0,
                level: 2,

                maxdist: 0.0,
                d_parent: 0.0,
            child_index_start: 0,
                child_index_count: 0,
                is_duplicate: false,
            },
            PackedNode {
                point: 5.0,
                level: 1,

                maxdist: 0.0,
                d_parent: 0.0,
            child_index_start: 0,
                child_index_count: 0,
                is_duplicate: false,
            },
        ];

        let child_indices = vec![1, 2, 3];  // root has children at 1,2; node 1 has child at 3
        let tree = PackedCoverTree::new(nodes, child_indices, 0, SimpleDistance, 1.3, 4);

        // Query near 10.0, k=2
        // Should find 10.0 and 5.0 (prune 100.0 subtree as too far)
        let results = tree.find_k_nearest(&8.0, 2);
        assert_eq!(results.len(), 2);

        assert_eq!(*results[0].0, 10.0); // distance 2
        assert_eq!(*results[1].0, 5.0);  // distance 3
    }
}

