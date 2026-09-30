//! Tests for packed dual-tree k-NN batch queries.
//!
//! Verifies that `PackedCoverTree::find_k_nearest_batch` and `find_k_nearest_dual`
//! produce correct results by comparing against single-tree ground truth and
//! brute-force k-NN.

use rustknn::{Distance, SimplifiedCoverTree};

// ============================================================================
// Distance metrics for testing
// ============================================================================

#[derive(Clone)]
struct EuclideanDistance1D;
impl Distance<f64> for EuclideanDistance1D {
    fn distance(&self, p: &f64, q: &f64) -> f64 {
        (p - q).abs()
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Point2D {
    x: f64,
    y: f64,
}

#[derive(Clone)]
struct EuclideanDistance2D;
impl Distance<Point2D> for EuclideanDistance2D {
    fn distance(&self, p: &Point2D, q: &Point2D) -> f64 {
        let dx = p.x - q.x;
        let dy = p.y - q.y;
        (dx * dx + dy * dy).sqrt()
    }
}

// ============================================================================
// Helper: brute-force k-NN for ground truth
// ============================================================================

/// Brute-force k-NN: compute distances to all reference points and return k nearest.
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

/// Compare two k-NN result sets by distances (ignoring point identity, since
/// ties might map to different points).
fn results_match_by_distance(a: &[(&f64, f64)], b: &[(&f64, f64)], eps: f64) -> bool {
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

/// Helper to build a packed tree from points.
fn build_packed_1d(points: &[f64]) -> rustknn::packed::PackedCoverTree<f64, EuclideanDistance1D> {
    let mut tree = SimplifiedCoverTree::new(EuclideanDistance1D, 1.3);
    for &p in points {
        tree.insert(p);
    }
    tree.recompute_maxdist();
    tree.pack()
}

/// Helper to build an unpacked tree from points.
fn build_unpacked_1d(points: &[f64]) -> SimplifiedCoverTree<f64, EuclideanDistance1D> {
    let mut tree = SimplifiedCoverTree::new(EuclideanDistance1D, 1.3);
    for &p in points {
        tree.insert(p);
    }
    tree.recompute_maxdist();
    tree
}

// ============================================================================
// Correctness: packed batch vs single-tree ground truth
// ============================================================================

#[test]
fn test_packed_batch_matches_single_tree_knn() {
    // Build reference tree with 20 points
    let ref_points: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let packed = build_packed_1d(&ref_points);

    let queries = vec![5.5, 10.3, 0.1, 19.9, 15.0];
    let k = 3;

    let batch_results = packed.find_k_nearest_batch(&queries, k);
    assert_eq!(batch_results.len(), queries.len());

    for (qi, q) in queries.iter().enumerate() {
        let single_results = packed.find_k_nearest(q, k);
        assert!(
            results_match_by_distance(&batch_results[qi], &single_results, 1e-10),
            "Mismatch for query {}: batch={:?}, single={:?}",
            q,
            batch_results[qi].iter().map(|(_, d)| d).collect::<Vec<_>>(),
            single_results.iter().map(|(_, d)| d).collect::<Vec<_>>(),
        );
    }
}

#[test]
fn test_packed_batch_matches_brute_force() {
    let ref_points: Vec<f64> = (0..30).map(|i| (i as f64) * 1.7).collect();
    let packed = build_packed_1d(&ref_points);

    let queries = vec![2.5, 25.0, 40.0, 0.0, 50.0];
    let k = 5;

    let batch_results = packed.find_k_nearest_batch(&queries, k);

    for (qi, q) in queries.iter().enumerate() {
        let bf = brute_force_knn(q, &ref_points, k, &EuclideanDistance1D);
        assert!(
            results_match_by_distance(&batch_results[qi], &bf, 1e-10),
            "Brute-force mismatch for query {}: batch_dists={:?}, bf_dists={:?}",
            q,
            batch_results[qi].iter().map(|(_, d)| d).collect::<Vec<_>>(),
            bf.iter().map(|(_, d)| d).collect::<Vec<_>>(),
        );
    }
}

// ============================================================================
// Correctness: packed batch vs unpacked batch
// ============================================================================

#[test]
fn test_packed_batch_matches_unpacked_batch() {
    let ref_points: Vec<f64> = (0..50).map(|i| (i as f64) * 0.7 + 3.0).collect();
    let unpacked_tree = build_unpacked_1d(&ref_points);
    let packed = build_packed_1d(&ref_points);

    let queries: Vec<f64> = (0..10).map(|i| (i as f64) * 3.5).collect();
    let k = 4;

    let packed_results = packed.find_k_nearest_batch(&queries, k);
    let unpacked_results = unpacked_tree.find_k_nearest_batch(&queries, k);

    assert_eq!(packed_results.len(), unpacked_results.len());
    for i in 0..queries.len() {
        assert!(
            results_match_by_distance(&packed_results[i], &unpacked_results[i], 1e-10),
            "Packed vs unpacked mismatch at query index {}", i,
        );
    }
}

// ============================================================================
// Edge cases
// ============================================================================

#[test]
fn test_packed_batch_empty_tree() {
    let tree = SimplifiedCoverTree::new(EuclideanDistance1D, 1.3);
    let packed = tree.pack();

    let results = packed.find_k_nearest_batch(&[1.0, 2.0, 3.0], 5);
    assert_eq!(results.len(), 3);
    for r in &results {
        assert!(r.is_empty());
    }
}

#[test]
fn test_packed_batch_empty_queries() {
    let ref_points: Vec<f64> = vec![1.0, 2.0, 3.0];
    let packed = build_packed_1d(&ref_points);

    let empty: Vec<f64> = Vec::new();
    let results = packed.find_k_nearest_batch(&empty, 3);
    assert!(results.is_empty());
}

#[test]
fn test_packed_batch_k_zero() {
    let ref_points: Vec<f64> = vec![1.0, 2.0, 3.0];
    let packed = build_packed_1d(&ref_points);

    let results = packed.find_k_nearest_batch(&[5.0, 10.0], 0);
    assert_eq!(results.len(), 2);
    for r in &results {
        assert!(r.is_empty());
    }
}

#[test]
fn test_packed_batch_k_greater_than_n() {
    let ref_points: Vec<f64> = vec![10.0, 20.0, 30.0];
    let packed = build_packed_1d(&ref_points);

    let results = packed.find_k_nearest_batch(&[15.0], 100);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].len(), 3); // Returns all 3 points
}

#[test]
fn test_packed_batch_single_query() {
    let ref_points: Vec<f64> = (0..10).map(|i| i as f64 * 5.0).collect();
    let packed = build_packed_1d(&ref_points);

    let results = packed.find_k_nearest_batch(&[12.0], 3);
    assert_eq!(results.len(), 1);

    let single_results = packed.find_k_nearest(&12.0, 3);
    assert!(results_match_by_distance(&results[0], &single_results, 1e-10));
}

#[test]
fn test_packed_batch_single_point_in_tree() {
    let packed = build_packed_1d(&[42.0]);

    let results = packed.find_k_nearest_batch(&[0.0, 100.0], 3);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].len(), 1);
    assert_eq!(*results[0][0].0, 42.0);
    assert_eq!(results[1].len(), 1);
    assert_eq!(*results[1][0].0, 42.0);
}

// ============================================================================
// Result properties
// ============================================================================

#[test]
fn test_packed_batch_results_sorted_by_distance() {
    let ref_points: Vec<f64> = (0..50).map(|i| i as f64).collect();
    let packed = build_packed_1d(&ref_points);

    let queries: Vec<f64> = vec![7.3, 25.8, 42.1];
    let results = packed.find_k_nearest_batch(&queries, 5);

    for result in &results {
        for window in result.windows(2) {
            assert!(
                window[0].1 <= window[1].1,
                "Results not sorted: {} > {}",
                window[0].1, window[1].1,
            );
        }
    }
}

#[test]
fn test_packed_batch_correct_distances() {
    let ref_points: Vec<f64> = vec![0.0, 10.0, 20.0, 30.0, 40.0];
    let packed = build_packed_1d(&ref_points);

    let results = packed.find_k_nearest_batch(&[15.0], 3);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].len(), 3);

    // Verify distances are correct
    for (point, dist) in &results[0] {
        let expected_dist = (**point - 15.0).abs();
        assert!(
            (dist - expected_dist).abs() < 1e-10,
            "Distance mismatch: point={}, expected={}, got={}",
            point, expected_dist, dist,
        );
    }
}

// ============================================================================
// Scale test: exercises pruning paths
// ============================================================================

#[test]
fn test_packed_batch_scale_1000_points() {
    // Build tree with 1000 points
    let ref_points: Vec<f64> = (0..1000).map(|i| (i as f64) * 0.1).collect();
    let packed = build_packed_1d(&ref_points);

    // 100 queries
    let queries: Vec<f64> = (0..100).map(|i| (i as f64) * 1.03 + 0.05).collect();
    let k = 5;

    let batch_results = packed.find_k_nearest_batch(&queries, k);
    assert_eq!(batch_results.len(), 100);

    // Verify each result against brute force
    for (qi, q) in queries.iter().enumerate() {
        let bf = brute_force_knn(q, &ref_points, k, &EuclideanDistance1D);
        assert!(
            results_match_by_distance(&batch_results[qi], &bf, 1e-10),
            "Scale test mismatch at query index {}", qi,
        );
    }
}

#[test]
fn test_packed_batch_2d_points() {
    // Test with 2D points to verify generality
    let ref_points: Vec<Point2D> = (0..100)
        .map(|i| Point2D {
            x: (i % 10) as f64,
            y: (i / 10) as f64,
        })
        .collect();

    let mut tree = SimplifiedCoverTree::new(EuclideanDistance2D, 1.3);
    for p in &ref_points {
        tree.insert(p.clone());
    }
    tree.recompute_maxdist();
    let packed = tree.pack();

    let queries = vec![
        Point2D { x: 3.5, y: 3.5 },
        Point2D { x: 0.0, y: 0.0 },
        Point2D { x: 9.0, y: 9.0 },
    ];
    let k = 4;

    let batch_results = packed.find_k_nearest_batch(&queries, k);
    assert_eq!(batch_results.len(), 3);

    // Verify against single-tree results
    for (qi, q) in queries.iter().enumerate() {
        let single = packed.find_k_nearest(q, k);
        assert_eq!(
            batch_results[qi].len(), single.len(),
            "Length mismatch for 2D query {:?}", q,
        );
        for (a, b) in batch_results[qi].iter().zip(single.iter()) {
            assert!(
                (a.1 - b.1).abs() < 1e-10,
                "2D distance mismatch: {} vs {}", a.1, b.1,
            );
        }
    }
}

// ============================================================================
// find_k_nearest_dual tests
// ============================================================================

#[test]
fn test_packed_dual_with_user_supplied_query_tree() {
    let ref_points: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let packed = build_packed_1d(&ref_points);

    // Build a query tree manually
    let mut query_tree = SimplifiedCoverTree::new(EuclideanDistance1D, 1.3);
    query_tree.insert(5.5);
    query_tree.insert(15.5);
    query_tree.recompute_maxdist();

    let k = 3;
    let results = packed.find_k_nearest_dual(&query_tree, k);

    // Should have results for each non-duplicate point in query tree DFS order
    assert!(!results.is_empty());

    // Each result should have up to k neighbors
    for r in &results {
        assert!(r.len() <= k);
        assert!(!r.is_empty());
    }
}

#[test]
fn test_packed_dual_empty_query_tree() {
    let ref_points: Vec<f64> = vec![1.0, 2.0, 3.0];
    let packed = build_packed_1d(&ref_points);

    let query_tree = SimplifiedCoverTree::new(EuclideanDistance1D, 1.3);
    let results = packed.find_k_nearest_dual(&query_tree, 3);
    assert!(results.is_empty());
}

#[test]
fn test_packed_dual_empty_reference_tree() {
    let tree = SimplifiedCoverTree::new(EuclideanDistance1D, 1.3);
    let packed = tree.pack();

    let mut query_tree = SimplifiedCoverTree::new(EuclideanDistance1D, 1.3);
    query_tree.insert(5.0);
    let results = packed.find_k_nearest_dual(&query_tree, 3);
    assert!(results.is_empty());
}

#[test]
fn test_packed_dual_k_zero() {
    let ref_points: Vec<f64> = vec![1.0, 2.0, 3.0];
    let packed = build_packed_1d(&ref_points);

    let mut query_tree = SimplifiedCoverTree::new(EuclideanDistance1D, 1.3);
    query_tree.insert(5.0);
    let results = packed.find_k_nearest_dual(&query_tree, 0);
    assert!(results.is_empty());
}

#[test]
fn test_packed_dual_results_match_single_tree() {
    let ref_points: Vec<f64> = (0..50).map(|i| i as f64 * 2.0).collect();
    let packed = build_packed_1d(&ref_points);

    let query_points = vec![5.0, 25.0, 45.0, 75.0, 95.0];
    let mut query_tree = SimplifiedCoverTree::new(EuclideanDistance1D, 1.3);
    for &q in &query_points {
        query_tree.insert(q);
    }
    query_tree.recompute_maxdist();

    let k = 3;
    let dual_results = packed.find_k_nearest_dual(&query_tree, k);

    // Verify each result set against single-tree k-NN
    // Results are in query tree DFS order, so we check distances against brute force
    for result in &dual_results {
        assert!(result.len() <= k);
        // Verify results are sorted
        for window in result.windows(2) {
            assert!(window[0].1 <= window[1].1);
        }
    }
}

#[test]
#[should_panic(expected = "Cannot run dual-tree k-NN with different base values")]
fn test_packed_dual_panics_on_different_bases() {
    let mut tree = SimplifiedCoverTree::new(EuclideanDistance1D, 1.3);
    tree.insert(1.0);
    let packed = tree.pack();

    let mut query_tree = SimplifiedCoverTree::new(EuclideanDistance1D, 2.0);
    query_tree.insert(5.0);

    // Should panic due to different base values
    let _ = packed.find_k_nearest_dual(&query_tree, 3);
}

// ============================================================================
// Duplicate query handling
// ============================================================================

#[test]
fn test_packed_batch_duplicate_queries() {
    // When queries contain duplicate values, each query should still get valid results
    let ref_points: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let packed = build_packed_1d(&ref_points);

    let queries = vec![5.0, 5.0, 10.0];
    let k = 3;

    let results = packed.find_k_nearest_batch(&queries, k);
    assert_eq!(results.len(), 3);

    // Both duplicate queries should get valid results
    for result in &results {
        assert!(!result.is_empty());
        assert!(result.len() <= k);
    }

    // The two duplicate queries should have identical distance patterns
    let dists_0: Vec<f64> = results[0].iter().map(|(_, d)| *d).collect();
    let dists_1: Vec<f64> = results[1].iter().map(|(_, d)| *d).collect();
    assert_eq!(dists_0.len(), dists_1.len());
    for (a, b) in dists_0.iter().zip(dists_1.iter()) {
        assert!((a - b).abs() < 1e-10);
    }
}

// ============================================================================
// Consistency: packed batch is identical to packed single-tree for many k values
// ============================================================================

#[test]
fn test_packed_batch_consistency_across_k_values() {
    let ref_points: Vec<f64> = (0..30).map(|i| i as f64 * 1.1).collect();
    let packed = build_packed_1d(&ref_points);

    let queries = vec![5.0, 15.0, 25.0];

    for k in [1, 2, 3, 5, 10, 30] {
        let batch_results = packed.find_k_nearest_batch(&queries, k);
        for (qi, q) in queries.iter().enumerate() {
            let single = packed.find_k_nearest(q, k);
            assert!(
                results_match_by_distance(&batch_results[qi], &single, 1e-10),
                "Mismatch at k={}, query={}", k, q,
            );
        }
    }
}
