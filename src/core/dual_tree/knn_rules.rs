//! k-NN Rules for Dual-Tree Traversal
//!
//! Implements BaseCase and Bound calculations for dual-tree k-NN search.
//! These rules determine how to update candidate lists during traversal.
//! Pruning (Score) is handled inline by the traversal itself.
//!
//! Based on: Curtin et al. "Tree-Independent Dual-Tree Algorithms" (ICML 2013)
//! with cover-tree-specific adaptations from mlpack.

use crate::distance::Distance;
use super::state::{BoundMode, DualKnnState};

/// Encapsulates the BaseCase, Score, and Bound logic for dual-tree k-NN.
///
/// # Type Parameters
///
/// * `'a` - Lifetime of references into the reference tree
/// * `T` - Point type
/// * `D` - Distance metric type
pub(crate) struct KnnRules<'a, T: Clone, D> {
    /// Per-query-point candidate tracking.
    pub state: DualKnnState<'a, T>,
    /// The distance metric. Accessible to the traversal module for computing
    /// distances during the sort phase (without the side effects of base_case).
    pub(crate) metric: &'a D,
    /// Whether query and reference trees are the same (self-k-NN).
    /// When true, a point is not considered its own neighbor.
    same_set: bool,
}

impl<'a, T: Clone, D: Distance<T>> KnnRules<'a, T, D> {
    /// Create new k-NN rules with full Curtin recursive bounds.
    ///
    /// # Arguments
    ///
    /// * `k` - Number of nearest neighbors to find per query point
    /// * `metric` - The distance metric
    /// * `same_set` - Whether query and reference trees are the same object
    pub fn new(k: usize, metric: &'a D, same_set: bool) -> Self {
        KnnRules {
            state: DualKnnState::new(k, BoundMode::CurtinRecursive),
            metric,
            same_set,
        }
    }

    /// Create new k-NN rules with Beygelzimer bound (kth + maxdist).
    ///
    /// Sound for external queries with lower overhead than CurtinRecursive —
    /// no bound cache, no cache invalidation, no subtree walks. Less tight
    /// than full B1/B2 but O(1) computation per bound. Good for small query trees.
    #[allow(dead_code)]
    pub fn new_beygelzimer_bound(k: usize, metric: &'a D, same_set: bool) -> Self {
        KnnRules {
            state: DualKnnState::new(k, BoundMode::Beygelzimer),
            metric,
            same_set,
        }
    }

    /// Create new k-NN rules with an explicit bound mode.
    pub fn new_with_bound(k: usize, metric: &'a D, same_set: bool, bound_mode: BoundMode) -> Self {
        KnnRules {
            state: DualKnnState::new(k, bound_mode),
            metric,
            same_set,
        }
    }

    /// Get kth-distance for a query point by pre-resolved index. O(1).
    #[inline]
    pub fn kth_distance_by_idx(&self, idx: usize) -> f64 {
        self.state.kth_distance_by_idx(idx)
    }

    /// Run base case using a pre-resolved query index. Skips HashMap lookup.
    /// Handles self-match and duplicate-skip logic.
    #[inline]
    pub fn base_case_by_idx(&mut self, query_point: &T, query_idx: usize, ref_point: &'a T, ref_is_duplicate: bool, dist: f64) -> f64 {
        // Skip inserting self-matches as candidates
        if self.same_set && std::ptr::eq(query_point, ref_point) {
            return dist;
        }

        // Skip algorithm-created duplicate reference nodes
        if ref_is_duplicate {
            return dist;
        }

        self.state.base_case_by_idx(query_idx, ref_point, dist);
        dist
    }

    /// Return bound cache instrumentation counters: (hits, misses).
    pub fn bound_cache_stats(&self) -> (u64, u64) {
        self.state.bound_cache_stats()
    }
}
