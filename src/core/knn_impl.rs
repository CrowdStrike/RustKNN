//! k-Nearest Neighbor Search Implementation
//!
//! This module implements single-tree k-NN search based on the paper's Algorithm 1,
//! extended to find k neighbors instead of just 1. The implementation is shared
//! between both SimplifiedCoverTree and NACoverTree variants.

#[cfg(not(feature = "no-smallvec"))]
use smallvec::SmallVec;

use crate::node::Node;
use crate::Distance;
use crate::knn::KnnState;

/// Implementation of k-NN search for cover trees
pub(crate) struct KnnImpl;

impl KnnImpl {
    /// Find k nearest neighbors of a query point using single-tree search.
    ///
    /// This is an extension of Algorithm 1 (find_nearest) to k neighbors.
    ///
    /// # Algorithm
    ///
    /// 1. Compute distance to current node's point
    /// 2. Insert point as candidate (if not a duplicate)
    /// 3. If leaf, done
    /// 4. Collect and sort children by distance (with pruning)
    /// 5. Recursively search children in order of increasing distance
    ///
    /// # Pruning
    ///
    /// A subtree rooted at child c can be pruned if:
    /// ```text
    /// d(query, c.point) - c.maxdist > kth_distance
    /// ```
    ///
    /// This means even the closest point in the subtree is farther than our kth-best candidate.
    ///
    /// # Deduplication
    ///
    /// Points marked with `is_duplicate = true` are skipped automatically during traversal.
    /// This handles algorithm-created duplicates from merging/level-alignment.
    ///
    /// # Arguments
    ///
    /// * `node` - Current node being visited
    /// * `query` - Query point
    /// * `state` - k-NN state tracking best candidates
    /// * `metric` - Distance metric
    ///
    /// # Type Parameters
    ///
    /// * `T` - Point type (must be Clone)
    /// * `D` - Distance metric type
    pub fn find_k_nearest_internal<'a, T, D>(
        node: &'a Node<T>,
        query: &T,
        state: &mut KnnState<'a, T>,
        metric: &D,
    ) where
        T: Clone,
        D: Distance<T>,
    {
        // 1. Compute distance to this node's point
        let node_dist = metric.distance(&node.point, query);

        // 2. Insert this point as candidate (skip if algorithm duplicate)
        if !node.is_duplicate {
            state.insert(&node.point, node_dist);
        }

        // 3. If leaf, done
        if node.children.is_empty() {
            return;
        }

        // 4. Collect and sort children by distance (with pruning)
        #[cfg(not(feature = "no-smallvec"))]
        let mut child_dists: SmallVec<[(f64, usize); 16]> = SmallVec::new();
        #[cfg(feature = "no-smallvec")]
        let mut child_dists: Vec<(f64, usize)> = Vec::new();
        for (i, child) in node.children.iter().enumerate() {
            // Self-child optimization: if child is a self-child (same center as parent),
            // reuse parent's distance instead of computing a new one.
            let child_dist = if child.is_duplicate && child.d_parent == 0.0 {
                node_dist
            } else {
                // Triangle inequality shell test: use |d(parent, query) - d(parent, child)|
                // as a free lower bound on d(child, query)
                if cfg!(not(feature = "no-triangle-filter")) && child.d_parent > 0.0 {
                    let lower_bound = (node_dist - child.d_parent).max(0.0);
                    if (lower_bound - child.maxdist).max(0.0) > state.kth_distance() {
                        continue;
                    }
                }

                metric.distance_with_bound(&child.point, query, state.kth_distance() + child.maxdist)
            };

            // Pruning: skip if subtree can't contain better neighbors
            let min_possible_dist = (child_dist - child.maxdist).max(0.0);

            if min_possible_dist <= state.kth_distance() {
                child_dists.push((child_dist, i));
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

        // 6. Recursively search children
        for (child_dist, idx) in child_dists {
            let child = &node.children[idx];

            // Recheck pruning (kth_distance may have improved since step 4)
            // Note: We use continue, not break, because children with larger maxdist
            // might still be explorable even if they're farther away
            let min_possible_dist = (child_dist - child.maxdist).max(0.0);
            if min_possible_dist > state.kth_distance() {
                continue;
            }

            Self::find_k_nearest_internal(child, query, state, metric);
        }
    }
}
