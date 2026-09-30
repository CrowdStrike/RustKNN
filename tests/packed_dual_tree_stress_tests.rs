//! Failure-mode stress tests for packed dual-tree k-NN batch queries.
//!
//! These tests exercise edge cases and degenerate inputs that could cause
//! the dual-tree traversal or batch result mapping to produce incorrect
//! or degraded results. Each test compares against single-tree k-NN or
//! brute-force ground truth.

use rustknn::{Distance, SimplifiedCoverTree};

// ============================================================================
// Distance metrics
// ============================================================================

#[derive(Clone)]
struct AbsDistance;
impl Distance<f64> for AbsDistance {
    fn distance(&self, p: &f64, q: &f64) -> f64 {
        (p - q).abs()
    }
}

/// N-dimensional point for high-dimensional tests.
#[derive(Clone, Debug)]
struct PointND {
    coords: Vec<f64>,
}

#[derive(Clone)]
struct EuclideanND;
impl Distance<PointND> for EuclideanND {
    fn distance(&self, p: &PointND, q: &PointND) -> f64 {
        p.coords
            .iter()
            .zip(q.coords.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Brute-force k-NN for ground truth.
fn brute_force_knn<'a, T, D: Distance<T>>(
    query: &T,
    ref_points: &'a [T],
    k: usize,
    metric: &D,
) -> Vec<(&'a T, f64)> {
    let mut dists: Vec<(&T, f64)> = ref_points
        .iter()
        .map(|p| (p, metric.distance(query, p)))
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.truncate(k);
    dists
}

/// Compare two k-NN result sets by distances (epsilon tolerance for floating-point).
fn results_match_by_distance<T>(a: &[(&T, f64)], b: &[(&T, f64)], eps: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for (aa, bb) in a.iter().zip(b.iter()) {
        if (aa.1 - bb.1).abs() > eps {
            return false;
        }
    }
    true
}

/// Build a packed tree from 1D points.
fn build_packed(points: &[f64]) -> rustknn::packed::PackedCoverTree<f64, AbsDistance> {
    let mut tree = SimplifiedCoverTree::new(AbsDistance, 1.3);
    for &p in points {
        tree.insert(p);
    }
    tree.recompute_maxdist();
    tree.pack()
}

/// Simple deterministic pseudo-random number generator (xorshift64).
/// Avoids pulling in `rand` crate just for tests.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Return a f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Return a f64 in [lo, hi).
    fn next_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.next_f64() * (hi - lo)
    }
}

// ============================================================================
// Test 1: Merged tree with duplicates → packed batch k-NN
// ============================================================================

#[test]
fn test_packed_batch_merged_tree_with_duplicates() {
    // Build two trees and merge them. Merge creates is_duplicate nodes.
    // Verify batch k-NN on the merged+packed tree matches single-tree k-NN.
    let mut merged = {
        let mut t1 = SimplifiedCoverTree::new(AbsDistance, 1.3);
        for i in 0..20 { t1.insert(i as f64); }
        let mut t2 = SimplifiedCoverTree::new(AbsDistance, 1.3);
        for i in 10..30 { t2.insert(i as f64); }
        t1.merge(t2)
    };
    merged.recompute_maxdist();
    let packed = merged.pack();

    let queries: Vec<f64> = vec![0.5, 5.0, 15.0, 25.0, 29.5];
    let k = 5;

    let batch_results = packed.find_k_nearest_batch(&queries, k);
    assert_eq!(batch_results.len(), queries.len());

    for (qi, q) in queries.iter().enumerate() {
        let single = packed.find_k_nearest(q, k);
        assert!(
            results_match_by_distance(&batch_results[qi], &single, 1e-10),
            "Merged+packed batch mismatch for query {}: batch_dists={:?}, single_dists={:?}",
            q,
            batch_results[qi].iter().map(|(_, d)| d).collect::<Vec<_>>(),
            single.iter().map(|(_, d)| d).collect::<Vec<_>>(),
        );
    }
}

// ============================================================================
// Test 2: Queries far outside reference tree's covering radius
// ============================================================================

#[test]
fn test_packed_batch_outlier_queries() {
    // Reference points in [0, 100], queries at extreme distances.
    let ref_points: Vec<f64> = (0..50).map(|i| i as f64 * 2.0).collect();
    let packed = build_packed(&ref_points);

    let queries = vec![1_000_000.0, -1_000_000.0, 999_999.0, -500_000.0, 50.0];
    let k = 3;

    let batch_results = packed.find_k_nearest_batch(&queries, k);
    assert_eq!(batch_results.len(), queries.len());

    for (qi, q) in queries.iter().enumerate() {
        assert_eq!(
            batch_results[qi].len(), k,
            "Outlier query {} should still get k={} results", q, k,
        );
        let single = packed.find_k_nearest(q, k);
        assert!(
            results_match_by_distance(&batch_results[qi], &single, 1e-10),
            "Outlier query {} batch vs single mismatch", q,
        );
    }
}

// ============================================================================
// Test 3: All queries identical (degenerate query tree)
// ============================================================================

#[test]
fn test_packed_batch_all_identical_queries() {
    let ref_points: Vec<f64> = (0..30).map(|i| i as f64 * 3.0).collect();
    let packed = build_packed(&ref_points);

    // 50 copies of the same query point
    let queries: Vec<f64> = vec![15.5; 50];
    let k = 5;

    let batch_results = packed.find_k_nearest_batch(&queries, k);
    assert_eq!(batch_results.len(), 50);

    // Every query should produce valid results
    let expected = packed.find_k_nearest(&15.5, k);
    for (qi, result) in batch_results.iter().enumerate() {
        assert!(
            !result.is_empty(),
            "Identical query {} returned empty results", qi,
        );
        assert!(
            results_match_by_distance(result, &expected, 1e-10),
            "Identical query {} has different distances than expected", qi,
        );
    }
}

// ============================================================================
// Test 4: Queries that are also reference points (overlap)
// ============================================================================

#[test]
fn test_packed_batch_queries_overlap_reference() {
    let ref_points: Vec<f64> = (0..20).map(|i| i as f64 * 5.0).collect();
    let packed = build_packed(&ref_points);

    // Queries are exact reference points
    let queries: Vec<f64> = vec![0.0, 25.0, 50.0, 75.0, 95.0];
    let k = 3;

    let batch_results = packed.find_k_nearest_batch(&queries, k);
    assert_eq!(batch_results.len(), queries.len());

    for (qi, q) in queries.iter().enumerate() {
        // First neighbor should be at distance 0.0 (exact match)
        assert!(
            !batch_results[qi].is_empty(),
            "Overlap query {} returned empty results", q,
        );
        assert!(
            batch_results[qi][0].1 < 1e-10,
            "Overlap query {} should have distance-0 first neighbor, got {}",
            q, batch_results[qi][0].1,
        );
        let single = packed.find_k_nearest(q, k);
        assert!(
            results_match_by_distance(&batch_results[qi], &single, 1e-10),
            "Overlap query {} batch vs single mismatch", q,
        );
    }
}

// ============================================================================
// Test 5: High-dimensional data with weak pruning
// ============================================================================

#[test]
fn test_packed_batch_high_dimensional() {
    let dim = 50;
    let mut rng = SimpleRng::new(42);

    // 200 random reference points in 50D
    let ref_points: Vec<PointND> = (0..200)
        .map(|_| PointND {
            coords: (0..dim).map(|_| rng.next_range(0.0, 100.0)).collect(),
        })
        .collect();

    let mut tree = SimplifiedCoverTree::new(EuclideanND, 1.3);
    for p in &ref_points {
        tree.insert(p.clone());
    }
    tree.recompute_maxdist();
    let packed = tree.pack();

    // 30 random query points
    let queries: Vec<PointND> = (0..30)
        .map(|_| PointND {
            coords: (0..dim).map(|_| rng.next_range(0.0, 100.0)).collect(),
        })
        .collect();
    let k = 5;

    let batch_results = packed.find_k_nearest_batch(&queries, k);
    assert_eq!(batch_results.len(), 30);

    // Compare against brute force
    for (qi, q) in queries.iter().enumerate() {
        let bf = brute_force_knn(q, &ref_points, k, &EuclideanND);
        assert_eq!(
            batch_results[qi].len(), bf.len(),
            "High-dim query {} length mismatch: batch={}, bf={}", qi, batch_results[qi].len(), bf.len(),
        );
        // Compare distances
        for (j, ((_, bd), (_, bfd))) in batch_results[qi].iter().zip(bf.iter()).enumerate() {
            assert!(
                (bd - bfd).abs() < 1e-6,
                "High-dim query {} neighbor {} distance mismatch: batch={}, bf={}",
                qi, j, bd, bfd,
            );
        }
    }
}

// ============================================================================
// Test 6: k=1 edge case (single neighbor)
// ============================================================================

#[test]
fn test_packed_batch_k_equals_1() {
    let ref_points: Vec<f64> = (0..500).map(|i| i as f64 * 0.2).collect();
    let packed = build_packed(&ref_points);

    let mut rng = SimpleRng::new(123);
    let queries: Vec<f64> = (0..100).map(|_| rng.next_range(0.0, 100.0)).collect();
    let k = 1;

    let batch_results = packed.find_k_nearest_batch(&queries, k);
    assert_eq!(batch_results.len(), 100);

    for (qi, q) in queries.iter().enumerate() {
        let single = packed.find_k_nearest(q, k);
        assert_eq!(batch_results[qi].len(), 1, "k=1 query {} should return exactly 1 result", qi);
        assert!(
            results_match_by_distance(&batch_results[qi], &single, 1e-10),
            "k=1 query {} batch vs single mismatch: batch_dist={}, single_dist={}",
            qi, batch_results[qi][0].1, single[0].1,
        );
    }
}

// ============================================================================
// Test 7: Reference tree with single point
// ============================================================================

#[test]
fn test_packed_batch_single_ref_point() {
    let packed = build_packed(&[42.0]);

    let queries: Vec<f64> = vec![0.0, 42.0, 100.0, -1000.0, 1000.0];
    let k = 3;

    let batch_results = packed.find_k_nearest_batch(&queries, k);
    assert_eq!(batch_results.len(), 5);

    for (qi, q) in queries.iter().enumerate() {
        assert_eq!(
            batch_results[qi].len(), 1,
            "Single-ref query {} should return 1 result, got {}",
            q, batch_results[qi].len(),
        );
        assert_eq!(*batch_results[qi][0].0, 42.0);
        let expected_dist = (q - 42.0).abs();
        assert!(
            (batch_results[qi][0].1 - expected_dist).abs() < 1e-10,
            "Single-ref query {} distance mismatch: got {}, expected {}",
            q, batch_results[qi][0].1, expected_dist,
        );
    }
}

// ============================================================================
// Test 8: Large-scale random (2000 ref, 200 queries, k=10, 5D)
// ============================================================================

#[test]
fn test_packed_batch_large_scale_random() {
    let dim = 5;
    let mut rng = SimpleRng::new(7777);

    let ref_points: Vec<PointND> = (0..2000)
        .map(|_| PointND {
            coords: (0..dim).map(|_| rng.next_range(0.0, 100.0)).collect(),
        })
        .collect();

    let mut tree = SimplifiedCoverTree::new(EuclideanND, 1.3);
    for p in &ref_points {
        tree.insert(p.clone());
    }
    tree.recompute_maxdist();
    let packed = tree.pack();

    let queries: Vec<PointND> = (0..200)
        .map(|_| PointND {
            coords: (0..dim).map(|_| rng.next_range(0.0, 100.0)).collect(),
        })
        .collect();
    let k = 10;

    let batch_results = packed.find_k_nearest_batch(&queries, k);
    assert_eq!(batch_results.len(), 200);

    // Spot-check every 10th query against brute force
    for qi in (0..200).step_by(10) {
        let bf = brute_force_knn(&queries[qi], &ref_points, k, &EuclideanND);
        assert_eq!(batch_results[qi].len(), bf.len(), "Large-scale query {} length mismatch", qi);
        for (j, ((_, bd), (_, bfd))) in batch_results[qi].iter().zip(bf.iter()).enumerate() {
            assert!(
                (bd - bfd).abs() < 1e-6,
                "Large-scale query {} neighbor {} mismatch: batch={}, bf={}",
                qi, j, bd, bfd,
            );
        }
    }
}

// ============================================================================
// Test 9: find_k_nearest_dual on merged+packed tree
// ============================================================================

#[test]
fn test_packed_dual_merged_tree() {
    let mut t1 = SimplifiedCoverTree::new(AbsDistance, 1.3);
    for i in 0..15 { t1.insert(i as f64); }
    let mut t2 = SimplifiedCoverTree::new(AbsDistance, 1.3);
    for i in 10..25 { t2.insert(i as f64); }

    let mut merged = t1.merge(t2);
    merged.recompute_maxdist();
    let packed = merged.pack();

    // Build a query tree
    let mut query_tree = SimplifiedCoverTree::new(AbsDistance, 1.3);
    for &q in &[2.5, 12.0, 22.0] {
        query_tree.insert(q);
    }
    query_tree.recompute_maxdist();

    let k = 4;
    let results = packed.find_k_nearest_dual(&query_tree, k);

    // Should have results for each non-duplicate query tree point
    assert!(!results.is_empty());

    for (i, result) in results.iter().enumerate() {
        assert!(
            result.len() <= k,
            "Merged dual-tree result {} has {} > {} neighbors", i, result.len(), k,
        );
        assert!(
            !result.is_empty(),
            "Merged dual-tree result {} is empty", i,
        );
        // Verify sorted
        for window in result.windows(2) {
            assert!(
                window[0].1 <= window[1].1,
                "Merged dual-tree result {} not sorted: {} > {}", i, window[0].1, window[1].1,
            );
        }
    }
}

// ============================================================================
// Test 10: Queries at exact covdist boundary
// ============================================================================

#[test]
fn test_packed_batch_queries_at_reference_boundary() {
    // Reference points: 0, 10, 20, 30, 40
    // Queries placed at exact covdist boundaries of the root
    let ref_points: Vec<f64> = (0..5).map(|i| i as f64 * 10.0).collect();
    let packed = build_packed(&ref_points);

    // Queries at boundaries: just inside and just outside the root's covering radius
    let queries: Vec<f64> = vec![
        -0.001, 0.001,     // Near left boundary
        39.999, 40.001,    // Near right boundary
        20.0,              // Center
    ];
    let k = 3;

    let batch_results = packed.find_k_nearest_batch(&queries, k);
    assert_eq!(batch_results.len(), queries.len());

    for (qi, q) in queries.iter().enumerate() {
        let single = packed.find_k_nearest(q, k);
        assert!(
            results_match_by_distance(&batch_results[qi], &single, 1e-10),
            "Boundary query {} batch vs single mismatch", q,
        );
    }
}

// ============================================================================
// Test 11: Tight cluster reference + spread queries
// ============================================================================

#[test]
fn test_packed_batch_clustered_reference_uniform_queries() {
    // Reference: 100 points tightly clustered in [49.0, 51.0]
    let ref_points: Vec<f64> = (0..100).map(|i| 49.0 + (i as f64) * 0.02).collect();
    let packed = build_packed(&ref_points);

    // Queries: spread uniformly across [0, 100]
    let queries: Vec<f64> = (0..20).map(|i| i as f64 * 5.0).collect();
    let k = 5;

    let batch_results = packed.find_k_nearest_batch(&queries, k);
    assert_eq!(batch_results.len(), 20);

    for (qi, q) in queries.iter().enumerate() {
        let bf = brute_force_knn(q, &ref_points, k, &AbsDistance);
        assert!(
            results_match_by_distance(&batch_results[qi], &bf, 1e-10),
            "Clustered-ref query {} batch vs brute-force mismatch: batch_dists={:?}, bf_dists={:?}",
            q,
            batch_results[qi].iter().map(|(_, d)| d).collect::<Vec<_>>(),
            bf.iter().map(|(_, d)| d).collect::<Vec<_>>(),
        );
    }
}

// ============================================================================
// Test 12: Batch = sequential single-tree for 500 queries
// ============================================================================

#[test]
fn test_packed_batch_vs_sequential_single_tree() {
    let ref_points: Vec<f64> = (0..300).map(|i| i as f64 * 0.5).collect();
    let packed = build_packed(&ref_points);

    let mut rng = SimpleRng::new(9999);
    let queries: Vec<f64> = (0..500).map(|_| rng.next_range(-10.0, 160.0)).collect();
    let k = 7;

    let batch_results = packed.find_k_nearest_batch(&queries, k);
    assert_eq!(batch_results.len(), 500);

    for (qi, q) in queries.iter().enumerate() {
        let single = packed.find_k_nearest(q, k);
        assert!(
            results_match_by_distance(&batch_results[qi], &single, 1e-10),
            "Sequential consistency query {} failed: batch_dists={:?}, single_dists={:?}",
            qi,
            batch_results[qi].iter().map(|(_, d)| d).collect::<Vec<_>>(),
            single.iter().map(|(_, d)| d).collect::<Vec<_>>(),
        );
    }
}
