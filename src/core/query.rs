//! Single Nearest Neighbor Query Algorithm
//!
//! This module implements Algorithm 1 from the paper - single nearest neighbor search
//! with pruning optimization. The implementation is shared between both SimplifiedCoverTree
//! and NACoverTree variants since the query algorithm is identical for both.
//!
//! # Algorithm Overview
//!
//! Starting from the root, recursively search the tree:
//! 1. Check if current node is closer than best
//! 2. If leaf, return current best
//! 3. Collect and prune children based on: `dist(query, child) - child.maxdist <= best_dist`
//! 4. Sort remaining children by distance (visit closer children first)
//! 5. Recursively search children in order
//!
//! # Pruning
//!
//! A subtree rooted at child c can be pruned if:
//! ```text
//! d(query, c.point) - c.maxdist > best_dist
//! ```
//! This means even the closest point in the subtree is farther than our current best.

#[cfg(not(feature = "no-smallvec"))]
use smallvec::SmallVec;

use crate::node::Node;
use crate::Distance;

/// Find the nearest neighbor to a query point using recursive tree search.
///
/// This implements Algorithm 1 from the paper with pruning optimization.
///
/// # Arguments
///
/// * `node` - Current node being visited
/// * `query` - Query point we're searching for neighbors of
/// * `best_point` - Best candidate found so far
/// * `best_dist` - Distance of best candidate
/// * `metric` - Distance metric to use
///
/// # Returns
///
/// A tuple of (nearest_point, distance) representing the closest point found
/// in this subtree.
///
/// # Type Parameters
///
/// * `T` - Point type (must be Clone)
/// * `D` - Distance metric type
pub(crate) fn find_nearest_internal<'a, T, D>(
    node: &'a Node<T>,
    query: &T,
    best_point: &'a T,
    best_dist: f64,
    metric: &D
) -> (&'a T, f64)
where
    T: Clone,
    D: Distance<T>,
{
    // Check if current node is closer than best
    let node_dist = metric.distance(&node.point, query);
    let (mut current_best, mut current_dist) = if node_dist < best_dist {
        (&node.point, node_dist)
    } else {
        (best_point, best_dist)
    };

    // If no children, return current best
    if node.children.is_empty() {
        return (current_best, current_dist);
    }

    // OPTIMIZATION: Compute distances and prune first, THEN sort only survivors
    // This avoids sorting children that will be pruned anyway
    #[cfg(not(feature = "no-smallvec"))]
    let mut child_dists: SmallVec<[(f64, usize); 16]> = SmallVec::new();
    #[cfg(feature = "no-smallvec")]
    let mut child_dists: Vec<(f64, usize)> = Vec::new();
    for (i, child) in node.children.iter().enumerate() {
        // Self-child optimization: if child is a self-child (same center as parent),
        // reuse parent's distance instead of computing a new one.
        let dist = if child.is_duplicate && child.d_parent == 0.0 {
            node_dist
        } else {
            // Triangle inequality shell test: use |d(parent, query) - d(parent, child)|
            // as a free lower bound on d(child, query) to skip distance computation
            if cfg!(not(feature = "no-triangle-filter")) && child.d_parent > 0.0 {
                let lower_bound = (node_dist - child.d_parent).max(0.0);
                if lower_bound - child.maxdist > current_dist {
                    continue;
                }
            }

            metric.distance_with_bound(&child.point, query, current_dist + child.maxdist)
        };

        // Apply pruning condition before adding to sort list
        if dist - child.maxdist <= current_dist {
            child_dists.push((dist, i));
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

    // Search children in order of increasing distance
    for (child_dist, child_idx) in child_dists {
        let child = &node.children[child_idx];

        // Pruning condition (recheck in case current_dist improved)
        // Note: We use continue, not break, because children with larger maxdist
        // might still be explorable even if they're farther away
        if child_dist - child.maxdist > current_dist {
            continue;
        }

        // Recursively search this child
        let (candidate, candidate_dist) = find_nearest_internal(
            child,
            query,
            current_best,
            current_dist,
            metric
        );

        // Update best if we found something closer
        if candidate_dist < current_dist {
            current_best = candidate;
            current_dist = candidate_dist;
        }
    }

    (current_best, current_dist)
}
