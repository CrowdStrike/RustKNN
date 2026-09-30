//! Dual-Tree k-NN State Management
//!
//! This module provides per-query-point candidate tracking for batch k-NN queries
//! using dual-tree traversal. Each query point maintains its own `KnnState` (sorted
//! Vec of k best candidates), indexed by integer index for O(1) access.
//!
//! # Bound Caching
//!
//! Computing B(Nq) = min(B1, B2) requires walking all descendants of a query node.
//! Without caching, this is O(subtree_size) per call and dominates traversal cost.
//!
//! We cache (first_bound, second_bound) per query node in a Vec indexed by node
//! integer index. When a base_case improves a query point's kth-distance,
//! we invalidate the cache by walking the parent chain upward (O(depth) Vec writes
//! instead of O(depth) HashMap removes).
//!
//! # B2 (Second Bound) Formulation
//!
//! Following Curtin et al., B2 is computed recursively:
//!   B2(Nq) = min(
//!     min{ Dp[k] + maxdist(Nq) for p in Pq },
//!     min{ B2(Nc) + 2*(maxdist(Nq) - maxdist(Nc)) for Nc in children(Nq) }
//!   )
//! The key insight: using incremental maxdist differences per child (not the full
//! maxdist(Nq)) produces tighter bounds for intermediate nodes.
//!
//! # Vec-Based State (O(1) lookups)
//!
//! In a simplified cover tree, each node holds exactly one point (1:1 correspondence).
//! During `init_parent_map`, every node is assigned an integer index 0..n-1 that serves
//! as both the point index (for `states[idx]`) and the node index (for `bound_cache[idx]`).
//! All hot-path lookups use Vec indexing instead of HashMap probing.

use smallvec::SmallVec;

use crate::knn::KnnState;
use crate::node::Node;
use crate::packed::tree::PackedCoverTree;
use crate::distance::Distance;

/// Controls how pruning bounds are computed during dual-tree traversal.
///
/// The full Curtin et al. B1/B2 bound computation walks entire subtrees on cache miss
/// and invalidates ancestor chains on every improvement. For self-query (same tree as
/// both query and reference), this overhead dominates — a simpler per-point bound
/// suffices because the `parent_bound` parameter already propagates the tightest
/// ancestor bound downward through recursion.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BoundMode {
    /// Full B1/B2 recursive bound with cache invalidation (Curtin et al.).
    /// Use for general dual-tree queries where query ≠ reference.
    CurtinRecursive,
    /// Per-point kth-distance only, O(1). No cache allocation, no invalidation.
    /// Use ONLY for self-query where query=reference tree. NOT safe for batch/external
    /// queries — missing the λ(Nq) margin needed to account for descendant query
    /// points with worse kth-distances (see Curtin et al. Section 5, B2 bound).
    SimpleBound,
    /// Beygelzimer bound: kth_distance(center) + maxdist(Nq). Sound for external
    /// queries with lower overhead than CurtinRecursive — no bound cache needed,
    /// no cache invalidation, no subtree walks. Less tight than full B1/B2
    /// (B2 only, without B1), so may result in more distance computations but
    /// less overhead per bound computation. Good for small query trees.
    Beygelzimer,
}

/// Cached bound values for a query node.
#[derive(Clone, Copy)]
struct CachedBound {
    /// B1: max kth-distance over all descendant query points (most pessimistic).
    first_bound: f64,
    /// B2: computed recursively using incremental maxdist differences.
    second_bound: f64,
    /// Epoch at which this bound was computed. Stale if != DualKnnState.epoch.
    epoch: u64,
}

/// Tracks k-NN candidates for all query points simultaneously.
///
/// Query nodes/points are pre-indexed during `init_parent_map` for O(1) state access.
/// Bound computation results are cached per query node and invalidated lazily.
pub(crate) struct DualKnnState<'a, T: Clone> {
    /// One KnnState per query point, indexed by integer index.
    states: Vec<KnnState<'a, T>>,
    /// Maximum number of neighbors to find per query point.
    k: usize,
    /// How pruning bounds are computed (full recursive vs simple per-point).
    bound_mode: BoundMode,
    /// Monotonically increasing counter, incremented on every successful base-case
    /// insert. Cached bounds with `epoch < self.epoch` are stale and must be
    /// recomputed. Replaces the O(depth) `invalidate_ancestors()` walk.
    epoch: u64,
    /// Cached bounds per query node, indexed by node integer index.
    /// A cached entry is valid only if `cached.epoch == self.epoch`.
    /// Only allocated in `CurtinRecursive` mode.
    bound_cache: Vec<Option<CachedBound>>,
    /// Parent node index for each node. `None` for the root node.
    /// Used to walk the ancestor chain during cache invalidation.
    parent_node_idx: Vec<Option<usize>>,
    /// Children indices for each node, indexed by node integer index.
    /// Built during `init_parent_map` for O(1) child index lookups.
    children_idx: Vec<SmallVec<[usize; 8]>>,
    /// Cached maxdist for each query node, indexed by node integer index.
    /// Populated during `init_parent_map` so `compute_bounds_recursive` can
    /// operate purely on indices without chasing `Box<Node<T>>` pointers.
    maxdist_cache: Vec<f64>,
    /// Query point pointers collected in DFS order during `init_parent_map`.
    /// Only non-duplicate nodes are included. This eliminates the need for
    /// a separate `collect_query_points_dfs` walk after traversal.
    query_ptrs: Vec<*const T>,
    /// State indices parallel to `query_ptrs`. `query_state_indices[i]` is the
    /// integer index into `states` for the query point at `query_ptrs[i]`.
    /// Eliminates the need to return `ptr_to_idx` HashMap from `into_results`.
    query_state_indices: Vec<usize>,
    /// Maps state index → packed node index. Populated by `init_parent_map_packed`
    /// for fully-packed dual-tree traversal where both query and reference are the
    /// same packed tree. Empty when using unpacked query trees.
    packed_idx: Vec<usize>,

    // --- Instrumentation counters ---
    /// Number of `bound_with_idx` calls that hit the cache (O(1) Vec read).
    pub(crate) bound_cache_hits: u64,
    /// Number of `bound_with_idx` calls that missed the cache and triggered
    /// `compute_bounds_recursive`.
    pub(crate) bound_cache_misses: u64,
}

impl<'a, T: Clone> DualKnnState<'a, T> {
    /// Create a new dual k-NN state for tracking up to k neighbors per query point.
    pub fn new(k: usize, bound_mode: BoundMode) -> Self {
        DualKnnState {
            states: Vec::new(),
            k,
            bound_mode,
            epoch: 0,
            bound_cache: Vec::new(),
            parent_node_idx: Vec::new(),
            children_idx: Vec::new(),
            maxdist_cache: Vec::new(),
            query_ptrs: Vec::new(),
            query_state_indices: Vec::new(),
            packed_idx: Vec::new(),
            bound_cache_hits: 0,
            bound_cache_misses: 0,
        }
    }

    /// Count the total number of nodes in a tree rooted at `node`.
    fn count_nodes(node: &Node<T>) -> usize {
        1 + node.children.iter().map(|c| Self::count_nodes(c)).sum::<usize>()
    }

    /// Build the parent-pointer map and point index from the query tree.
    ///
    /// Must be called once before traversal begins. Assigns integer indices to
    /// all query nodes/points and records parent relationships for cache invalidation.
    /// Also collects query point pointers in DFS order (non-duplicates only) and
    /// caches maxdist values for index-only bound computation.
    ///
    /// Returns the root node's integer index (always 0).
    pub fn init_parent_map(&mut self, query_root: &Node<T>) -> usize {
        // Pre-count nodes for exact allocation (avoids Vec reallocation)
        let node_count = Self::count_nodes(query_root);
        self.states.reserve(node_count);
        self.children_idx.reserve(node_count);
        self.query_ptrs.reserve(node_count);
        self.query_state_indices.reserve(node_count);
        if self.bound_mode == BoundMode::CurtinRecursive {
            self.bound_cache.reserve(node_count);
            self.parent_node_idx.reserve(node_count);
            self.maxdist_cache.reserve(node_count);
        } else if self.bound_mode == BoundMode::Beygelzimer {
            // Beygelzimer needs maxdist_cache but NOT bound_cache or parent_node_idx
            self.maxdist_cache.reserve(node_count);
        }
        self.build_parent_map_recursive(query_root, None);
        0 // root is always the first node indexed
    }

    /// Build the parent-pointer map from a packed cover tree's node array.
    ///
    /// Like `init_parent_map`, but walks the packed tree's contiguous arrays
    /// by index instead of following `Box<Node<T>>` pointers. This enables
    /// fully-packed dual-tree self-query where both query and reference sides
    /// are the same packed tree — no unpacked query tree needed.
    ///
    /// Also populates `packed_idx` so the traversal can map state indices back
    /// to packed node indices.
    ///
    /// Returns the root node's state index (always 0).
    pub fn init_parent_map_packed<D: Distance<T>>(
        &mut self,
        packed: &PackedCoverTree<T, D>,
        root_packed_idx: usize,
    ) -> usize {
        // DFS walk over packed tree, assigning state indices in DFS order.
        // The packed tree is already in DFS order, but we still need to walk
        // the parent-child relationships via children_of().
        let mut stack: Vec<(usize, Option<usize>)> = Vec::new(); // (packed_idx, parent_state_idx)
        stack.push((root_packed_idx, None));

        // Pre-allocate: count total nodes reachable from root.
        // For a full packed tree, this is just nodes.len(), but we count to be safe.
        let estimated = packed.len();
        self.states.reserve(estimated);
        self.children_idx.reserve(estimated);
        self.packed_idx.reserve(estimated);
        self.query_ptrs.reserve(estimated);
        self.query_state_indices.reserve(estimated);
        if self.bound_mode == BoundMode::CurtinRecursive {
            self.bound_cache.reserve(estimated);
            self.parent_node_idx.reserve(estimated);
            self.maxdist_cache.reserve(estimated);
        } else if self.bound_mode == BoundMode::Beygelzimer {
            self.maxdist_cache.reserve(estimated);
        }

        // Iterative DFS (avoids recursion stack overflow for deep trees)
        // Process nodes in DFS pre-order: visit node, then push children (reversed).
        while let Some((p_idx, parent_state)) = stack.pop() {
            let node = packed.node(p_idx);
            let state_idx = self.states.len();

            self.states.push(KnnState::new(self.k));
            self.packed_idx.push(p_idx);
            if self.bound_mode == BoundMode::CurtinRecursive {
                self.bound_cache.push(None);
                self.parent_node_idx.push(parent_state);
                self.maxdist_cache.push(node.maxdist);
            } else if self.bound_mode == BoundMode::Beygelzimer {
                self.maxdist_cache.push(node.maxdist);
            }
            self.children_idx.push(SmallVec::new()); // placeholder, filled below

            // Collect non-duplicate query point pointers in DFS order
            if !node.is_duplicate {
                self.query_ptrs.push(&node.point as *const T);
                self.query_state_indices.push(state_idx);
            }

            // Push children in reverse order so first child is processed first (DFS order)
            let children = packed.children_of(p_idx);
            for &child_packed_idx in children.iter().rev() {
                stack.push((child_packed_idx, Some(state_idx)));
            }
        }

        // Second pass: fix up children_idx using the packed_idx mapping.
        // Build packed_idx -> state_idx reverse map.
        let max_packed = self.packed_idx.iter().copied().max().unwrap_or(0) + 1;
        let mut packed_to_state = vec![usize::MAX; max_packed];
        for (si, &pi) in self.packed_idx.iter().enumerate() {
            packed_to_state[pi] = si;
        }

        // Now fill children_idx for each state node
        for si in 0..self.packed_idx.len() {
            let pi = self.packed_idx[si];
            let children = packed.children_of(pi);
            let mut child_indices = SmallVec::<[usize; 8]>::new();
            for &child_packed_idx in children {
                let child_si = packed_to_state[child_packed_idx];
                debug_assert_ne!(child_si, usize::MAX, "child packed idx {} not found", child_packed_idx);
                child_indices.push(child_si);
            }
            self.children_idx[si] = child_indices;
        }

        0 // root is always the first node indexed
    }

    /// Recursive helper to build the point/node index, parent map, children indices,
    /// maxdist cache, and query_ptrs.
    fn build_parent_map_recursive(&mut self, node: &Node<T>, parent_idx: Option<usize>) {
        let point_ptr = &node.point as *const T;

        // Assign the same index for both node and point (1:1 in simplified cover trees)
        let idx = self.states.len();
        self.states.push(KnnState::new(self.k));
        if self.bound_mode == BoundMode::CurtinRecursive {
            self.bound_cache.push(None);
            self.parent_node_idx.push(parent_idx);
            self.maxdist_cache.push(node.maxdist);
        } else if self.bound_mode == BoundMode::Beygelzimer {
            self.maxdist_cache.push(node.maxdist);
        }
        self.children_idx.push(SmallVec::new()); // placeholder, filled below

        // Collect non-duplicate query point pointers in DFS order
        if !node.is_duplicate {
            self.query_ptrs.push(point_ptr);
            self.query_state_indices.push(idx);
        }

        let mut child_indices = SmallVec::<[usize; 8]>::new();
        for child in &node.children {
            let child_idx = self.states.len(); // next index to be assigned
            child_indices.push(child_idx);
            self.build_parent_map_recursive(child, Some(idx));
        }
        self.children_idx[idx] = child_indices;
    }

    /// Process a base case using a pre-resolved query index. Skips HashMap lookup.
    #[inline]
    pub fn base_case_by_idx(&mut self, idx: usize, ref_point: &'a T, distance: f64) {
        if self.states[idx].insert(ref_point, distance) {
            if self.bound_mode == BoundMode::CurtinRecursive {
                // Increment epoch to lazily invalidate all cached bounds.
                // On next access, stale bounds (epoch < self.epoch) are recomputed.
                // This replaces the O(depth) ancestor walk per improvement.
                self.epoch += 1;
            }
        }
    }

    /// Get the kth-distance for a query point by pre-resolved index. O(1) Vec indexing.
    #[inline]
    pub fn kth_distance_by_idx(&self, idx: usize) -> f64 {
        self.states[idx].kth_distance()
    }

    /// Compute the pruning bound B(Nq) using a pre-resolved node index.
    ///
    /// In `SimpleBound` mode: returns `kth_distance(idx)` directly — O(1), no cache.
    /// In `Beygelzimer` mode: returns `kth_distance(idx) + maxdist(idx)` — O(1), no cache.
    /// In `CurtinRecursive` mode: full B1/B2 with caching and subtree walks.
    #[inline]
    pub fn bound_with_idx(&mut self, idx: usize) -> f64 {
        if self.bound_mode == BoundMode::SimpleBound {
            return self.states[idx].kth_distance();
        }

        if self.bound_mode == BoundMode::Beygelzimer {
            // B2 = kth_distance(center) + λ(Nq) where λ(Nq) = maxdist(Nq)
            // Sound for external queries: any descendant qd is at most maxdist from
            // center, so by triangle inequality: kth_distance(qd) <= kth_distance(center) + maxdist
            return self.states[idx].kth_distance() + self.maxdist_cache[idx];
        }

        if let Some(cached) = self.bound_cache[idx] {
            if cached.epoch == self.epoch {
                self.bound_cache_hits += 1;
                return cached.first_bound.min(cached.second_bound);
            }
        }

        self.bound_cache_misses += 1;
        let (first_bound, second_bound) = self.compute_bounds_recursive(idx);

        self.bound_cache[idx] = Some(CachedBound {
            first_bound,
            second_bound,
            epoch: self.epoch,
        });

        first_bound.min(second_bound)
    }

    /// Get the number of children for a query node. O(1) Vec lookup.
    /// Returns an owned value, so no borrow on `self` is held after the call.
    #[inline]
    pub fn num_children(&self, idx: usize) -> usize {
        self.children_idx[idx].len()
    }

    /// Get the child index at a given position. O(1) Vec lookup.
    /// Returns an owned value, so no borrow on `self` is held after the call.
    #[inline]
    pub fn child_at(&self, idx: usize, child_pos: usize) -> usize {
        self.children_idx[idx][child_pos]
    }

    /// Get the packed node index corresponding to a state index.
    /// Only valid after `init_parent_map_packed` has been called.
    #[inline(always)]
    pub fn packed_node_idx(&self, state_idx: usize) -> usize {
        self.packed_idx[state_idx]
    }

    /// Recursively compute B1 and B2 over all descendant query points.
    ///
    /// Operates purely on indices — no `&Node<T>` parameter needed. All node data
    /// is accessed via `maxdist_cache[idx]` and `children_idx[idx]`, avoiding
    /// pointer chasing through `Box<Node<T>>` on the heap.
    ///
    /// B1 = max{ kth_distance for all descendant query points }. This is the most
    /// conservative (loosest) bound — if dmin > B1, no descendant needs any neighbor.
    ///
    /// B2 = max{ kth_distance(q) + d(q, center_Nq) } over all descendants q. This is
    /// Curtin et al.'s tighter bound that accounts for the distance from each descendant
    /// to the node center. B2 >= B1 always (since d(q, center) >= 0).
    fn compute_bounds_recursive(&mut self, idx: usize) -> (f64, f64) {
        if let Some(cached) = self.bound_cache[idx] {
            if cached.epoch == self.epoch {
                return (cached.first_bound, cached.second_bound);
            }
        }

        let own_kth = self.states[idx].kth_distance();
        let node_maxdist = self.maxdist_cache[idx];

        let mut first_bound = own_kth;
        let mut second_bound = own_kth + node_maxdist;

        // Iterate children by positional index to avoid cloning the SmallVec.
        // Each self.children_idx[idx][ci] is two Vec indexes — no allocation.
        // The borrow is released between iterations, so the mutable recursive
        // call is valid.
        let num_children = self.children_idx[idx].len();
        for ci in 0..num_children {
            let child_idx = self.children_idx[idx][ci];
            let child_maxdist = self.maxdist_cache[child_idx];
            let (child_first, child_b2) = self.compute_bounds_recursive(child_idx);

            if child_first > first_bound {
                first_bound = child_first;
            }

            // Per Curtin et al.: B2(Nq) = max over descendants q of { D_q[k] + d(q, center_Nq) }
            // Propagated from child: B2(Nc) + 2*(maxdist(Nq) - maxdist(Nc))
            // Take MAX (most pessimistic) because any descendant could still need neighbors.
            let propagated_b2 = child_b2 + 2.0 * (node_maxdist - child_maxdist);
            if propagated_b2 > second_bound {
                second_bound = propagated_b2;
            }
        }

        self.bound_cache[idx] = Some(CachedBound {
            first_bound,
            second_bound,
            epoch: self.epoch,
        });

        (first_bound, second_bound)
    }

    /// Collect final results for all query points.
    ///
    /// Returns a tuple of:
    /// - `Vec<Vec<(&T, f64)>>` — results indexed by integer index (0..n-1)
    /// - `Vec<usize>` — state indices parallel to query_ptrs (query_state_indices[i]
    ///   is the index into the results Vec for query_ptrs[i])
    /// - `Vec<*const T>` — query point pointers in DFS order (non-duplicates only)
    ///
    /// Callers iterate `query_state_indices` to map each query pointer to its results
    /// without any HashMap lookups.
    #[allow(clippy::type_complexity)]
    pub fn into_results(self) -> (Vec<Vec<(&'a T, f64)>>, Vec<usize>, Vec<*const T>) {
        let results: Vec<Vec<(&'a T, f64)>> = self.states
            .into_iter()
            .map(|state| state.into_sorted_vec())
            .collect();
        (results, self.query_state_indices, self.query_ptrs)
    }

    /// Return bound cache instrumentation counters: (hits, misses).
    pub fn bound_cache_stats(&self) -> (u64, u64) {
        (self.bound_cache_hits, self.bound_cache_misses)
    }
}
