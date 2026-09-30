//! # Naive (brute-force) nearest-neighbor baseline
//!
//! A linear scan over all points, provided as a reference baseline alongside the
//! cover trees and KD-tree. It uses partial distance calculation: each distance
//! exits early once its running sum exceeds the current k-th best.
//!
//! Like the KD-tree, it reuses the shared machinery so the comparison is clean:
//! - point distances go through the [`Distance`] trait, so `distance_with_bound`
//!   (the "partial distance" early-exit) and the `instrument` counter apply, and the
//!   `no-early-exit` feature flag turns it into a full-distance scan;
//! - candidate tracking uses the shared `KnnState`.
//!
//! There is no index and no construction cost — the whole point is to show what
//! "no data structure, just partial-distance pruning" costs, as a reference for when
//! a tree is worth building.

use crate::distance::Distance;
use crate::knn::KnnState;

/// A brute-force k-NN searcher: holds the reference points and the metric.
pub struct NaiveNN<D: Distance<Vec<f64>>> {
    points: Vec<Vec<f64>>,
    metric: D,
}

impl<D: Distance<Vec<f64>>> NaiveNN<D> {
    /// Create a naive searcher over `points`. No construction work is done.
    pub fn new(points: Vec<Vec<f64>>, metric: D) -> Self {
        NaiveNN { points, metric }
    }

    /// Number of reference points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Find the k nearest neighbors to `query` by linear scan, closest-first.
    ///
    /// Uses `distance_with_bound(p, query, kth_distance)` so that once k candidates
    /// are held, each subsequent point can be rejected early as soon as its partial
    /// squared distance exceeds the current k-th best. With the `no-early-exit`
    /// feature this degrades to a full distance per point.
    pub fn find_k_nearest(&self, query: &Vec<f64>, k: usize) -> Vec<(&Vec<f64>, f64)> {
        let mut state: KnnState<Vec<f64>> = KnnState::new(k);
        for p in &self.points {
            let d = self.metric.distance_with_bound(p, query, state.kth_distance());
            state.insert(p, d);
        }
        state.into_sorted_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::EuclideanDistance;

    #[test]
    fn test_naive_matches_direct() {
        let pts = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 2.0],
            vec![3.0, 3.0],
        ];
        let nn = NaiveNN::new(pts, EuclideanDistance);
        let r = nn.find_k_nearest(&vec![0.1, 0.1], 2);
        assert_eq!(r.len(), 2);
        // Nearest is (0,0) then (1,0).
        assert!((r[0].1 - (0.02_f64).sqrt()).abs() < 1e-9);
        assert!(*r[1].0 == vec![1.0, 0.0]);
    }

    #[test]
    fn test_naive_k_exceeds_n() {
        let pts = vec![vec![1.0], vec![2.0]];
        let nn = NaiveNN::new(pts, EuclideanDistance);
        let r = nn.find_k_nearest(&vec![0.0], 5);
        assert_eq!(r.len(), 2); // only 2 points available
    }
}
