//! # KD-Tree (baseline structure for cross-structure benchmarking)
//!
//! A classic median-split KD-tree over `Vec<f64>` points, provided as a *second*
//! spatial index under the same benchmark harness as the cover trees. Its purpose is
//! to test whether the paper's central thesis — that implementation details
//! (distance kernel, early-exit, candidate tracking) drive measured performance as
//! much as the algorithm — holds outside cover trees.
//!
//! To make that test valid, this KD-tree deliberately reuses the *same* machinery as
//! the cover-tree query paths:
//! - point-to-point distances go through the shared [`Distance`] trait, so the
//!   `distance_with_bound` early-exit and the `instrument` distance counter apply
//!   identically, and the same micro-optimization feature flags (`no-early-exit`,
//!   `no-k1-special`, etc.) toggle its behavior;
//! - k-NN candidate tracking uses the shared `KnnState` (k=1 fast path / sorted
//!   array), the same structure the cover trees use.
//!
//! Only the *pruning geometry* is KD-tree-specific: each node stores an axis-aligned
//! bounding box of its points, and a subtree is pruned when the box's lower-bound
//! distance to the query exceeds the current k-th best. The box lower bound is the
//! standard axis-aligned Euclidean bound (exact for L2), computed on coordinates.
//!
//! This is intentionally a *textbook* KD-tree (median split on the widest axis, box
//! pruning) — not a novel contribution. The scientific value is that it shares the
//! cover trees' distance kernel and candidate state, so a like-for-like ablation is
//! possible.

use crate::distance::Distance;
use crate::knn::KnnState;

/// A node in the flat KD-tree array.
///
/// Nodes are stored contiguously in `KdTree::nodes`. Each node owns a contiguous
/// range of the (reordered) point index array `KdTree::indices[lo..hi]`; the point
/// at `point_idx` is the split pivot (median). Children are `left`/`right` node
/// indices (`usize::MAX` == none). `bb_lo`/`bb_hi` are the per-dimension bounding box
/// over all points in this subtree, used for pruning.
struct KdNode {
    point_idx: usize,
    split_axis: usize,
    left: usize,
    right: usize,
    bb_lo: Vec<f64>,
    bb_hi: Vec<f64>,
}

const NONE: usize = usize::MAX;

/// A median-split KD-tree over `Vec<f64>` points with a distance metric `D`.
///
/// Generic over the metric so it shares the exact distance kernel used by the cover
/// trees (see module docs). Build with [`KdTree::new`]; query with
/// [`KdTree::find_k_nearest`] or [`KdTree::find_k_nearest_self`].
pub struct KdTree<D: Distance<Vec<f64>>> {
    points: Vec<Vec<f64>>,
    nodes: Vec<KdNode>,
    root: usize,
    dim: usize,
    metric: D,
}

impl<D: Distance<Vec<f64>>> KdTree<D> {
    /// Build a KD-tree from `points` using `metric`.
    ///
    /// Construction is the standard recursive median split: at each node, split on
    /// the axis of greatest spread, place the median point as the pivot, and recurse
    /// on the lower/upper halves. O(n log n) with a per-node bounding box computed
    /// bottom-up. Distances are NOT used during construction (KD-trees split on
    /// coordinates), so construction does not touch the distance counter.
    pub fn new(points: Vec<Vec<f64>>, metric: D) -> Self {
        let dim = points.first().map(|p| p.len()).unwrap_or(0);
        let n = points.len();
        let mut indices: Vec<usize> = (0..n).collect();
        let mut nodes: Vec<KdNode> = Vec::with_capacity(n);
        let root = if n == 0 {
            NONE
        } else {
            Self::build_recursive(&points, &mut indices, 0, n, dim, &mut nodes)
        };
        KdTree { points, nodes, root, dim, metric }
    }

    /// Number of points in the tree.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Recursively build a subtree over `indices[lo..hi]`. Returns the node index.
    fn build_recursive(
        points: &[Vec<f64>],
        indices: &mut [usize],
        lo: usize,
        hi: usize,
        dim: usize,
        nodes: &mut Vec<KdNode>,
    ) -> usize {
        let _guard = crate::core::utils::StackGuard::enter();
        if lo >= hi {
            return NONE;
        }
        // Bounding box over indices[lo..hi].
        let mut bb_lo = points[indices[lo]].clone();
        let mut bb_hi = bb_lo.clone();
        for &idx in &indices[lo + 1..hi] {
            let p = &points[idx];
            for d in 0..dim {
                if p[d] < bb_lo[d] { bb_lo[d] = p[d]; }
                if p[d] > bb_hi[d] { bb_hi[d] = p[d]; }
            }
        }
        // Split on the axis of greatest spread.
        let mut split_axis = 0;
        let mut best_spread = -1.0;
        for d in 0..dim {
            let spread = bb_hi[d] - bb_lo[d];
            if spread > best_spread {
                best_spread = spread;
                split_axis = d;
            }
        }
        // Median by nth_element on the split axis.
        let mid = lo + (hi - lo) / 2;
        indices[lo..hi].select_nth_unstable_by(mid - lo, |&a, &b| {
            points[a][split_axis]
                .partial_cmp(&points[b][split_axis])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let pivot_idx = indices[mid];

        // Reserve this node's slot before recursing (children fill in after).
        let node_pos = nodes.len();
        nodes.push(KdNode {
            point_idx: pivot_idx,
            split_axis,
            left: NONE,
            right: NONE,
            bb_lo,
            bb_hi,
        });

        let left = Self::build_recursive(points, indices, lo, mid, dim, nodes);
        let right = Self::build_recursive(points, indices, mid + 1, hi, dim, nodes);
        nodes[node_pos].left = left;
        nodes[node_pos].right = right;
        node_pos
    }

    /// Squared axis-aligned box lower bound from `query` to a node's bounding box.
    ///
    /// For each dimension, the closest possible coordinate in the box contributes
    /// `(q - clamp(q, lo, hi))²`. The square root is the minimum possible Euclidean
    /// distance from the query to ANY point in the subtree — a valid pruning bound.
    /// Computed on coordinates (not via the metric), as is standard for KD-trees;
    /// this is exact for L2.
    #[inline]
    fn box_min_dist(&self, query: &[f64], node: &KdNode) -> f64 {
        let mut acc = 0.0_f64;
        for d in 0..self.dim {
            let q = query[d];
            if q < node.bb_lo[d] {
                let diff = node.bb_lo[d] - q;
                acc += diff * diff;
            } else if q > node.bb_hi[d] {
                let diff = q - node.bb_hi[d];
                acc += diff * diff;
            }
        }
        acc.sqrt()
    }

    /// Find the k nearest neighbors to `query`, returned closest-first.
    ///
    /// Uses the shared `KnnState` and the metric's `distance_with_bound`, so the
    /// early-exit and instrumentation behavior matches the cover-tree query paths.
    pub fn find_k_nearest(&self, query: &Vec<f64>, k: usize) -> Vec<(&Vec<f64>, f64)> {
        let mut state: KnnState<Vec<f64>> = KnnState::new(k);
        if self.root != NONE {
            self.search(self.root, query, &mut state);
        }
        state.into_sorted_vec()
    }

    /// Recursive nearest-neighbor descent with box pruning.
    fn search<'a>(&'a self, node_idx: usize, query: &Vec<f64>, state: &mut KnnState<'a, Vec<f64>>) {
        let _guard = crate::core::utils::StackGuard::enter();
        if node_idx == NONE {
            return;
        }
        let node = &self.nodes[node_idx];

        // Prune: if the closest possible point in this box is farther than the
        // current k-th best, the whole subtree can be skipped.
        let bound = state.kth_distance();
        if bound.is_finite() && self.box_min_dist(query, node) > bound {
            return;
        }

        // Score the pivot point through the shared metric (counter + early-exit apply).
        let pivot = &self.points[node.point_idx];
        let d = self.metric.distance_with_bound(pivot, query, state.kth_distance());
        state.insert(pivot, d);

        // Descend near child first (the side of the splitting plane the query is on),
        // so the pruning bound tightens before we consider the far child.
        let axis = node.split_axis;
        let (near, far) = if query[axis] <= pivot[axis] {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };
        self.search(near, query, state);
        self.search(far, query, state);
    }

    /// All-nearest-neighbors self query: for every point in the tree, find its k
    /// nearest neighbors (including itself). Mirrors the cover trees' `all_nn` /
    /// `all_nn_single` benchmark by looping single-point queries.
    pub fn find_k_nearest_self(&self, k: usize) -> Vec<Vec<(&Vec<f64>, f64)>> {
        self.points
            .iter()
            .map(|q| self.find_k_nearest(q, k))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::EuclideanDistance;

    fn brute_force(points: &[Vec<f64>], query: &[f64], k: usize) -> Vec<f64> {
        let _guard = crate::core::utils::StackGuard::enter();
        let mut ds: Vec<f64> = points
            .iter()
            .map(|p| {
                p.iter()
                    .zip(query)
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt()
            })
            .collect();
        ds.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ds.truncate(k);
        ds
    }

    #[test]
    fn test_empty() {
        let t: KdTree<EuclideanDistance> = KdTree::new(vec![], EuclideanDistance);
        assert_eq!(t.len(), 0);
        assert!(t.find_k_nearest(&vec![0.0, 0.0], 1).is_empty());
    }

    #[test]
    fn test_single_point() {
        let pts = vec![vec![1.0, 2.0]];
        let t = KdTree::new(pts, EuclideanDistance);
        let r = t.find_k_nearest(&vec![0.0, 0.0], 1);
        assert_eq!(r.len(), 1);
        assert!((r[0].1 - (5.0_f64).sqrt()).abs() < 1e-9);
    }

    #[test]
    fn test_matches_brute_force_small() {
        // Deterministic pseudo-random points via a simple LCG (no rand dep needed).
        let mut seed = 12345u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (u32::MAX as f64)
        };
        let n = 300;
        let dim = 5;
        let pts: Vec<Vec<f64>> = (0..n).map(|_| (0..dim).map(|_| next()).collect()).collect();
        let tree = KdTree::new(pts.clone(), EuclideanDistance);

        for k in [1usize, 3, 10] {
            for _ in 0..20 {
                let q: Vec<f64> = (0..dim).map(|_| next()).collect();
                let got: Vec<f64> = tree.find_k_nearest(&q, k).iter().map(|(_, d)| *d).collect();
                let want = brute_force(&pts, &q, k);
                assert_eq!(got.len(), want.len(), "k={}", k);
                for (a, b) in got.iter().zip(&want) {
                    assert!((a - b).abs() < 1e-9, "k={}: {} vs {}", k, a, b);
                }
            }
        }
    }

    #[test]
    fn test_self_query_includes_self() {
        let pts = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![5.0, 5.0],
        ];
        let tree = KdTree::new(pts, EuclideanDistance);
        let all = tree.find_k_nearest_self(1);
        // Each point's nearest (k=1) is itself at distance 0.
        for r in all {
            assert_eq!(r.len(), 1);
            assert!((r[0].1).abs() < 1e-9);
        }
    }
}
