//! Parallel tree builder using Rayon
//!
//! This module implements the main entry point for parallel cover tree construction.

use crate::simplified::SimplifiedCoverTree;
use crate::nearest_ancestor::NACoverTree;
use crate::Distance;
use rayon::prelude::*;
use super::merge_all::merge_all;

/// Which tree variant to use for per-thread construction.
#[derive(Clone, Copy, Debug)]
pub enum ParallelVariant {
    /// Each thread builds a SimplifiedCoverTree (faster construction, no rebalancing)
    Simplified,
    /// Each thread builds a NACoverTree (better tree structure from rebalancing),
    /// then converts to SimplifiedCoverTree for merging
    NearestAncestor,
}

/// Build a SimplifiedCoverTree in parallel from a vector of points.
///
/// Partitions points across threads, builds per-thread trees using the
/// specified variant, then merges using binary reduction.
///
/// The `variant` parameter controls which tree type is built per-thread:
/// - `Simplified`: faster per-thread construction, no rebalancing
/// - `NearestAncestor`: better per-thread tree structure (rebalancing), then
///    converted to SimplifiedCoverTree for merging
///
/// # No minimum threshold
///
/// Parallel construction runs at any dataset size. For small datasets the
/// overhead is measurable in benchmarks, which is the intended use case.
///
/// # Arguments
///
/// * `metric` - Distance metric (must be Send + Sync + Clone)
/// * `base` - Base value for covdist/sepdist calculations (typically 1.3)
/// * `points` - Vector of points to insert (ownership transferred)
/// * `num_threads` - Optional thread count (None = use all available cores)
/// * `variant` - Which tree type to build per-thread
///
/// # Returns
///
/// A SimplifiedCoverTree containing all points
///
/// # Panics
///
/// Panics if `base` is not finite or is below [`MIN_BASE`](crate::MIN_BASE).
///
/// # Type Requirements
///
/// - `T: Clone + Send + Sync + 'static` - Points must be thread-safe
/// - `D: Distance<T> + Send + Sync + Clone` - Metric must be thread-safe and cloneable
pub fn build_parallel<T, D>(
    metric: D,
    base: f64,
    points: Vec<T>,
    num_threads: Option<usize>,
    variant: ParallelVariant,
) -> SimplifiedCoverTree<T, D>
where
    T: Clone + Send + Sync + 'static,
    D: Distance<T> + Send + Sync + Clone,
{
    // Handle empty dataset
    if points.is_empty() {
        return SimplifiedCoverTree::new(metric, base);
    }

    // Determine thread count
    let num_threads = num_threads.unwrap_or_else(rayon::current_num_threads).max(1);

    // Calculate chunk size
    let chunk_size = (points.len() + num_threads - 1) / num_threads;

    // Build trees in parallel, converting to SimplifiedCoverTree per-thread
    let trees: Vec<SimplifiedCoverTree<T, D>> = points
        .into_par_iter()
        .chunks(chunk_size)
        .map(|chunk| match variant {
            ParallelVariant::Simplified => {
                let mut tree = SimplifiedCoverTree::new(metric.clone(), base);
                for point in chunk {
                    tree.insert(point);
                }
                tree.recompute_maxdist();
                tree
            }
            ParallelVariant::NearestAncestor => {
                let mut tree = NACoverTree::new(metric.clone(), base);
                for point in chunk {
                    tree.insert(point);
                }
                tree.into_simplified()
            }
        })
        .collect();

    // Single chunk: no merge needed
    if trees.len() == 1 {
        return trees.into_iter().next().unwrap();
    }

    // Merge all trees using binary reduction
    let mut result = merge_all(trees);
    // Recompute exact maxdist after merging — merge uses approximate maxdist
    // which can leave stale values that cause queries to miss points
    result.recompute_maxdist();
    result
}

/// Build a SimplifiedCoverTree in parallel (convenience wrapper).
///
/// Equivalent to `build_parallel(..., ParallelVariant::Simplified)`.
///
/// # Arguments
///
/// * `metric` - Distance metric (must be Send + Sync + Clone)
/// * `base` - Base value for covdist/sepdist calculations (typically 1.3)
/// * `points` - Vector of points to insert (ownership transferred)
/// * `num_threads` - Optional thread count (None = use all available cores)
///
/// # Returns
///
/// A SimplifiedCoverTree containing all points
///
/// # Panics
///
/// Panics if `base` is not finite or is below [`MIN_BASE`](crate::MIN_BASE).
pub fn build_parallel_simplified<T, D>(
    metric: D,
    base: f64,
    points: Vec<T>,
    num_threads: Option<usize>,
) -> SimplifiedCoverTree<T, D>
where
    T: Clone + Send + Sync + 'static,
    D: Distance<T> + Send + Sync + Clone,
{
    build_parallel(metric, base, points, num_threads, ParallelVariant::Simplified)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct SimpleDistance;

    impl Distance<f64> for SimpleDistance {
        fn distance(&self, p: &f64, q: &f64) -> f64 {
            (p - q).abs()
        }
    }

    #[test]
    fn test_build_parallel_empty() {
        let points: Vec<f64> = Vec::new();
        let tree = build_parallel_simplified(SimpleDistance, 1.3, points, None);
        assert_eq!(tree.len(), 0);
    }

    #[test]
    fn test_build_parallel_small_dataset() {
        let points: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let tree = build_parallel_simplified(SimpleDistance, 1.3, points, None);
        assert_eq!(tree.len(), 100);
    }

    #[test]
    fn test_build_parallel_large_dataset() {
        let points: Vec<f64> = (0..20_000).map(|i| i as f64).collect();
        let tree = build_parallel_simplified(SimpleDistance, 1.3, points, None);
        assert_eq!(tree.len(), 20_000);

        // Spot check queries
        let nearest = tree.find_nearest(&10_000.0);
        assert!(nearest.is_some());
    }

    #[test]
    fn test_build_parallel_explicit_threads() {
        let points: Vec<f64> = (0..15_000).map(|i| i as f64).collect();
        let tree = build_parallel_simplified(SimpleDistance, 1.3, points, Some(4));
        assert_eq!(tree.len(), 15_000);
    }

    #[test]
    fn test_build_parallel_vs_sequential() {
        let points_par: Vec<f64> = (0..10_000).map(|i| i as f64 * 0.1).collect();
        let points_seq = points_par.clone();

        // Build in parallel
        let par_tree = build_parallel_simplified(SimpleDistance, 1.3, points_par, Some(4));

        // Build sequentially
        let mut seq_tree = SimplifiedCoverTree::new(SimpleDistance, 1.3);
        for p in points_seq {
            seq_tree.insert(p);
        }

        // Verify same size
        assert_eq!(par_tree.len(), seq_tree.len());

        // Verify queries give same distances
        for query in [100.5, 500.3, 999.7] {
            let par_nearest = par_tree.find_nearest(&query);
            let seq_nearest = seq_tree.find_nearest(&query);

            assert!(par_nearest.is_some());
            assert!(seq_nearest.is_some());

            let par_dist = SimpleDistance.distance(&query, par_nearest.unwrap());
            let seq_dist = SimpleDistance.distance(&query, seq_nearest.unwrap());

            // Distances should be identical (might find different points at same distance)
            assert!(
                (par_dist - seq_dist).abs() < 1e-10,
                "Query {} gave different distances: parallel={}, sequential={}",
                query,
                par_dist,
                seq_dist
            );
        }
    }

    // Tests for build_parallel with ParallelVariant::NearestAncestor

    #[test]
    fn test_build_parallel_na_empty() {
        let points: Vec<f64> = Vec::new();
        let tree = build_parallel(SimpleDistance, 1.3, points, None, ParallelVariant::NearestAncestor);
        assert_eq!(tree.len(), 0);
    }

    #[test]
    fn test_build_parallel_na_small_dataset() {
        let points: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let tree = build_parallel(SimpleDistance, 1.3, points, None, ParallelVariant::NearestAncestor);
        assert_eq!(tree.len(), 100);

        // Verify queries work
        let nearest = tree.find_nearest(&50.0);
        assert_eq!(nearest, Some(&50.0));
    }

    #[test]
    fn test_build_parallel_na_large_dataset() {
        let points: Vec<f64> = (0..20_000).map(|i| i as f64).collect();
        let tree = build_parallel(SimpleDistance, 1.3, points, None, ParallelVariant::NearestAncestor);
        assert_eq!(tree.len(), 20_000);

        let nearest = tree.find_nearest(&10_000.0);
        assert!(nearest.is_some());
    }

    #[test]
    fn test_build_parallel_na_explicit_threads() {
        let points: Vec<f64> = (0..15_000).map(|i| i as f64).collect();
        let tree = build_parallel(SimpleDistance, 1.3, points, Some(4), ParallelVariant::NearestAncestor);
        assert_eq!(tree.len(), 15_000);
    }

    #[test]
    fn test_build_parallel_variants_same_results() {
        let points_s: Vec<f64> = (0..5_000).map(|i| i as f64 * 0.1).collect();
        let points_na = points_s.clone();

        let tree_s = build_parallel(SimpleDistance, 1.3, points_s, Some(4), ParallelVariant::Simplified);
        let tree_na = build_parallel(SimpleDistance, 1.3, points_na, Some(4), ParallelVariant::NearestAncestor);

        assert_eq!(tree_s.len(), tree_na.len());

        // Both should produce identical nearest-neighbor distances
        for query in [0.5, 100.3, 250.7, 499.9] {
            let s_nearest = tree_s.find_nearest(&query).unwrap();
            let na_nearest = tree_na.find_nearest(&query).unwrap();

            let s_dist = SimpleDistance.distance(&query, s_nearest);
            let na_dist = SimpleDistance.distance(&query, na_nearest);

            assert!(
                (s_dist - na_dist).abs() < 1e-10,
                "Query {} gave different distances: simplified={}, na={}",
                query, s_dist, na_dist
            );
        }
    }
}
