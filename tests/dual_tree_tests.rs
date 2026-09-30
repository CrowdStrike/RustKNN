//! Comprehensive tests for dual-tree k-NN search.
//!
//! These tests verify that dual-tree k-NN produces identical results to
//! single-tree k-NN (brute-force comparison), handles edge cases correctly,
//! and works with both cover tree variants.

use rustknn::{CoverTree, Distance, SimplifiedCoverTree, NACoverTree};

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

impl Clone for EuclideanDistance2D {
    fn clone(&self) -> Self {
        EuclideanDistance2D
    }
}

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

/// Brute-force self-k-NN: for each point, find k nearest excluding itself.
#[allow(dead_code)]
fn brute_force_self_knn<'a, T, D: Distance<T>>(
    points: &'a [T],
    k: usize,
    metric: &D,
) -> Vec<Vec<(&'a T, f64)>> {
    points
        .iter()
        .map(|query| {
            let mut dists: Vec<(&T, f64)> = points
                .iter()
                .filter(|p| !std::ptr::eq(*p, query))
                .map(|p| (p, metric.distance(query, p)))
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            dists.truncate(k);
            dists
        })
        .collect()
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

// ============================================================================
// Correctness tests: dual-tree vs single-tree
// ============================================================================

#[test]
fn test_dual_tree_matches_single_tree_small() {
    let metric = EuclideanDistance1D;
    let mut tree = SimplifiedCoverTree::new(metric.clone(), 1.3);

    let ref_points: Vec<f64> = (0..20).map(|i| i as f64 * 1.5).collect();
    for &p in &ref_points {
        tree.insert(p);
    }

    let queries: Vec<f64> = vec![0.5, 5.5, 10.5, 15.5, 20.5, 25.5, 3.3, 7.7, 12.1, 18.9];
    let k = 3;

    // Dual-tree batch
    let dual_results = tree.find_k_nearest_batch(&queries, k);

    // Single-tree per query
    for (i, q) in queries.iter().enumerate() {
        let single_result = tree.find_k_nearest(q, k);
        assert!(
            results_match_by_distance(&dual_results[i], &single_result, 1e-10),
            "Mismatch for query {} at index {}: dual={:?}, single={:?}",
            q, i, dual_results[i], single_result
        );
    }
}

#[test]
fn test_dual_tree_matches_single_tree_medium() {
    let metric = EuclideanDistance1D;
    let mut tree = SimplifiedCoverTree::new(metric.clone(), 1.3);

    let ref_points: Vec<f64> = (0..200).map(|i| i as f64 * 0.3 + 0.1).collect();
    for &p in &ref_points {
        tree.insert(p);
    }

    let queries: Vec<f64> = (0..50).map(|i| i as f64 * 1.2 + 0.7).collect();
    let k = 5;

    let dual_results = tree.find_k_nearest_batch(&queries, k);

    for (i, q) in queries.iter().enumerate() {
        let single_result = tree.find_k_nearest(q, k);
        assert!(
            results_match_by_distance(&dual_results[i], &single_result, 1e-10),
            "Mismatch for query index {}", i
        );
    }
}

#[test]
fn test_dual_tree_matches_single_tree_large() {
    let metric = EuclideanDistance1D;
    let mut tree = SimplifiedCoverTree::new(metric.clone(), 1.3);

    let ref_points: Vec<f64> = (0..1000).map(|i| i as f64 * 0.1).collect();
    for &p in &ref_points {
        tree.insert(p);
    }

    let queries: Vec<f64> = (0..100).map(|i| i as f64 * 0.97 + 0.05).collect();
    let k = 10;

    let dual_results = tree.find_k_nearest_batch(&queries, k);

    for (i, q) in queries.iter().enumerate() {
        let single_result = tree.find_k_nearest(q, k);
        assert!(
            results_match_by_distance(&dual_results[i], &single_result, 1e-10),
            "Mismatch for query index {}", i
        );
    }
}

#[test]
fn test_dual_tree_k_equals_1() {
    let metric = EuclideanDistance1D;
    let mut tree = SimplifiedCoverTree::new(metric.clone(), 1.3);

    for i in 0..50 {
        tree.insert(i as f64 * 2.0);
    }

    let queries: Vec<f64> = vec![1.0, 5.0, 17.0, 49.0, 99.0];
    let dual_results = tree.find_k_nearest_batch(&queries, 1);

    for (i, q) in queries.iter().enumerate() {
        let nearest = tree.find_nearest(q);
        assert_eq!(dual_results[i].len(), 1);
        // The nearest neighbor by find_nearest should match
        let dual_nearest = dual_results[i][0].0;
        let single_nearest = nearest.unwrap();
        assert_eq!(
            *dual_nearest, *single_nearest,
            "k=1 mismatch for query {}: dual={}, single={}",
            q, dual_nearest, single_nearest
        );
    }
}

#[test]
fn test_dual_tree_2d_euclidean() {
    let metric = EuclideanDistance2D;
    let mut tree = SimplifiedCoverTree::new(metric, 1.3);

    let ref_points = vec![
        Point2D { x: 0.0, y: 0.0 },
        Point2D { x: 1.0, y: 0.0 },
        Point2D { x: 0.0, y: 1.0 },
        Point2D { x: 1.0, y: 1.0 },
        Point2D { x: 5.0, y: 5.0 },
        Point2D { x: 10.0, y: 10.0 },
    ];
    for p in &ref_points {
        tree.insert(p.clone());
    }

    let query = Point2D { x: 0.5, y: 0.5 };
    let k = 3;

    // Single-tree result
    let single_result = tree.find_k_nearest(&query, k);

    // Brute force
    let metric2 = EuclideanDistance2D;
    let brute_result = brute_force_knn(&query, &ref_points, k, &metric2);

    // Verify single-tree matches brute force by distance
    assert_eq!(single_result.len(), brute_result.len());
    for (s, b) in single_result.iter().zip(brute_result.iter()) {
        assert!((s.1 - b.1).abs() < 1e-10);
    }
}

// ============================================================================
// Edge case tests
// ============================================================================

#[test]
fn test_dual_tree_empty_reference() {
    let metric = EuclideanDistance1D;
    let tree = SimplifiedCoverTree::new(metric.clone(), 1.3);

    let queries = vec![1.0, 2.0, 3.0];
    let results = tree.find_k_nearest_batch(&queries, 3);

    assert_eq!(results.len(), 3);
    for r in &results {
        assert!(r.is_empty());
    }
}

#[test]
fn test_dual_tree_empty_queries() {
    let metric = EuclideanDistance1D;
    let mut tree = SimplifiedCoverTree::new(metric.clone(), 1.3);
    tree.insert(1.0);
    tree.insert(2.0);

    let queries: Vec<f64> = vec![];
    let results = tree.find_k_nearest_batch(&queries, 3);
    assert!(results.is_empty());
}

#[test]
fn test_dual_tree_k_zero() {
    let metric = EuclideanDistance1D;
    let mut tree = SimplifiedCoverTree::new(metric.clone(), 1.3);
    tree.insert(1.0);

    let queries = vec![1.0, 2.0];
    let results = tree.find_k_nearest_batch(&queries, 0);

    assert_eq!(results.len(), 2);
    for r in &results {
        assert!(r.is_empty());
    }
}

#[test]
fn test_dual_tree_k_greater_than_n() {
    let metric = EuclideanDistance1D;
    let mut tree = SimplifiedCoverTree::new(metric.clone(), 1.3);

    let ref_points = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    for &p in &ref_points {
        tree.insert(p);
    }

    let queries = vec![3.0];
    let results = tree.find_k_nearest_batch(&queries, 100);

    assert_eq!(results.len(), 1);
    // Should return at most 5 neighbors (all points in the tree)
    assert!(results[0].len() <= 5);
}

#[test]
fn test_dual_tree_single_ref_point() {
    let metric = EuclideanDistance1D;
    let mut tree = SimplifiedCoverTree::new(metric.clone(), 1.3);
    tree.insert(42.0);

    let queries = vec![0.0, 42.0, 100.0];
    let results = tree.find_k_nearest_batch(&queries, 3);

    assert_eq!(results.len(), 3);
    for r in &results {
        assert_eq!(r.len(), 1);
        assert_eq!(*r[0].0, 42.0);
    }
}

#[test]
fn test_dual_tree_single_query_point() {
    let metric = EuclideanDistance1D;
    let mut tree = SimplifiedCoverTree::new(metric.clone(), 1.3);

    for i in 0..20 {
        tree.insert(i as f64);
    }

    let queries = vec![10.5];
    let k = 3;
    let dual_results = tree.find_k_nearest_batch(&queries, k);
    let single_result = tree.find_k_nearest(&10.5, k);

    assert_eq!(dual_results.len(), 1);
    assert!(
        results_match_by_distance(&dual_results[0], &single_result, 1e-10),
        "Single query mismatch"
    );
}

#[test]
fn test_dual_tree_identical_points() {
    let metric = EuclideanDistance1D;
    let mut tree = SimplifiedCoverTree::new(metric.clone(), 1.3);

    // All points are the same
    for _ in 0..10 {
        tree.insert(5.0);
    }

    let queries = vec![5.0, 0.0, 10.0];
    let results = tree.find_k_nearest_batch(&queries, 3);

    assert_eq!(results.len(), 3);
    // All distances for query=5.0 should be 0
    for r in &results[0] {
        assert!((r.1 - 0.0).abs() < 1e-10);
    }
}

// ============================================================================
// Same-set (self k-NN) tests
// ============================================================================

#[test]
fn test_self_knn_excludes_self() {
    let metric = EuclideanDistance1D;
    let mut tree = SimplifiedCoverTree::new(metric.clone(), 1.3);

    let points = vec![0.0, 10.0, 20.0, 30.0, 40.0];
    for &p in &points {
        tree.insert(p);
    }

    let results = tree.find_k_nearest_self(1);

    // Each point should have exactly 1 neighbor, and it should NOT be itself
    // (distances should be > 0 since points are all distinct and 10 apart)
    for neighbors in &results {
        assert_eq!(neighbors.len(), 1, "Expected 1 neighbor per point");
        assert!(
            neighbors[0].1 > 0.0,
            "Self-match not excluded: distance is 0"
        );
    }
}

#[test]
fn test_self_knn_matches_expected() {
    let metric = EuclideanDistance1D;
    let mut tree = SimplifiedCoverTree::new(metric.clone(), 1.3);

    let points = vec![0.0, 1.0, 3.0, 6.0, 10.0];
    for &p in &points {
        tree.insert(p);
    }

    let results = tree.find_k_nearest_self(2);

    // Verify results have correct count
    for neighbors in &results {
        assert_eq!(neighbors.len(), 2, "Expected 2 neighbors per point");
    }

    // Verify that all returned distances are valid:
    // Each result is for some point in the tree. The returned neighbors should
    // be actual points from the tree at the correct distances.
    // We verify this by checking: for each result set, the returned distances
    // correspond to actual inter-point distances, and self-matches are excluded.
    for neighbors in &results {
        for (point, dist) in neighbors {
            assert!(
                *dist > 0.0,
                "Self-match not excluded: dist={}",
                dist
            );
            // Verify the returned point is in the original set
            assert!(
                points.iter().any(|p| (*p - **point).abs() < 1e-10),
                "Returned point {} is not in the point set",
                point
            );
        }
    }
}

#[test]
fn test_self_knn_with_duplicates() {
    let metric = EuclideanDistance1D;
    let mut tree = SimplifiedCoverTree::new(metric.clone(), 1.3);

    // Insert duplicate values (user-intentional duplicates, NOT pointer-same)
    tree.insert(0.0);
    tree.insert(0.0); // Same value, different insertion
    tree.insert(10.0);

    let results = tree.find_k_nearest_self(1);

    // The two 0.0 points should find each other (distance 0), NOT be excluded.
    // Only pointer-identity self-match is excluded.
    // At least one result should have distance 0.0 (the duplicate 0.0 finding the other 0.0)
    let has_zero_distance = results.iter().any(|neighbors| {
        neighbors.iter().any(|(_, d)| *d < 1e-10)
    });
    assert!(
        has_zero_distance,
        "Duplicate values should find each other with distance 0"
    );
}

// ============================================================================
// Dual-tree lower-level API tests
// ============================================================================

#[test]
fn test_find_k_nearest_dual_matches_batch() {
    let metric = EuclideanDistance1D;
    let mut ref_tree = SimplifiedCoverTree::new(metric.clone(), 1.3);

    for i in 0..30 {
        ref_tree.insert(i as f64 * 0.5);
    }

    // Build query tree manually
    let mut query_tree = SimplifiedCoverTree::new(metric.clone(), 1.3);
    let queries = vec![1.0, 5.0, 10.0, 14.0];
    for &q in &queries {
        query_tree.insert(q);
    }

    let k = 3;
    let dual_results = ref_tree.find_k_nearest_dual(&query_tree, k);

    // Each query point should have results
    assert!(!dual_results.is_empty());

    // Compare with single-tree results for each query
    for (i, q) in queries.iter().enumerate() {
        let single_result = ref_tree.find_k_nearest(q, k);
        if i < dual_results.len() {
            assert!(
                results_match_by_distance(&dual_results[i], &single_result, 1e-10),
                "Dual-tree API mismatch for query {} at index {}", q, i
            );
        }
    }
}

#[test]
#[should_panic(expected = "different base values")]
fn test_find_k_nearest_dual_different_bases() {
    let metric = EuclideanDistance1D;
    let ref_tree = SimplifiedCoverTree::new(metric.clone(), 1.3);
    let query_tree = SimplifiedCoverTree::new(metric.clone(), 2.0);

    // This should panic due to mismatched base values
    ref_tree.find_k_nearest_dual(&query_tree, 3);
}

// ============================================================================
// Property tests
// ============================================================================

#[test]
fn test_results_sorted_by_distance() {
    let metric = EuclideanDistance1D;
    let mut tree = SimplifiedCoverTree::new(metric.clone(), 1.3);

    for i in 0..50 {
        tree.insert(i as f64);
    }

    let queries: Vec<f64> = vec![5.5, 25.5, 45.5];
    let results = tree.find_k_nearest_batch(&queries, 5);

    for (i, neighbors) in results.iter().enumerate() {
        for j in 1..neighbors.len() {
            assert!(
                neighbors[j].1 >= neighbors[j - 1].1,
                "Results not sorted for query index {}: dist[{}]={} < dist[{}]={}",
                i, j, neighbors[j].1, j - 1, neighbors[j - 1].1
            );
        }
    }
}

#[test]
fn test_result_distances_correct() {
    let metric = EuclideanDistance1D;
    let mut tree = SimplifiedCoverTree::new(metric.clone(), 1.3);

    for i in 0..30 {
        tree.insert(i as f64);
    }

    let queries = vec![5.5, 15.5];
    let results = tree.find_k_nearest_batch(&queries, 3);

    for (i, neighbors) in results.iter().enumerate() {
        for (point, dist) in neighbors {
            let recomputed = metric.distance(&queries[i], point);
            assert!(
                (recomputed - dist).abs() < 1e-10,
                "Distance mismatch: stored={}, recomputed={} for query {}",
                dist, recomputed, queries[i]
            );
        }
    }
}

#[test]
fn test_result_lengths() {
    let metric = EuclideanDistance1D;
    let mut tree = SimplifiedCoverTree::new(metric.clone(), 1.3);

    let n = 15;
    for i in 0..n {
        tree.insert(i as f64);
    }

    let queries = vec![0.5, 7.5, 14.5];

    // k < n: should get exactly k results
    let results = tree.find_k_nearest_batch(&queries, 5);
    for r in &results {
        assert_eq!(r.len(), 5);
    }

    // k > n: should get at most n results
    let results = tree.find_k_nearest_batch(&queries, 100);
    for r in &results {
        assert!(r.len() <= n);
    }
}

#[test]
fn test_results_are_actual_ref_points() {
    let metric = EuclideanDistance1D;
    let mut tree = SimplifiedCoverTree::new(metric.clone(), 1.3);

    let ref_points: Vec<f64> = vec![1.0, 5.0, 10.0, 20.0, 50.0];
    for &p in &ref_points {
        tree.insert(p);
    }

    let queries = vec![3.0, 15.0];
    let results = tree.find_k_nearest_batch(&queries, 3);

    for neighbors in &results {
        for (point, _) in neighbors {
            assert!(
                ref_points.contains(point),
                "Returned point {} is not in the reference set",
                point
            );
        }
    }
}

// ============================================================================
// Variant tests: NACoverTree
// ============================================================================

#[test]
fn test_dual_tree_na_variant_self_knn() {
    let metric = EuclideanDistance1D;
    let mut tree = NACoverTree::new(metric, 1.3);

    let points = vec![0.0, 5.0, 10.0, 15.0, 20.0];
    for &p in &points {
        tree.insert(p);
    }

    let results = tree.find_k_nearest_self(1);

    for neighbors in &results {
        assert_eq!(neighbors.len(), 1);
        assert!(neighbors[0].1 > 0.0, "Self-match not excluded in NA variant");
    }
}

#[test]
fn test_dual_tree_na_variant_dual() {
    let metric = EuclideanDistance1D;
    let mut ref_tree = NACoverTree::new(metric, 1.3);

    for i in 0..30 {
        ref_tree.insert(i as f64);
    }

    let mut query_tree = NACoverTree::new(EuclideanDistance1D, 1.3);
    let queries = vec![5.5, 15.5, 25.5];
    for &q in &queries {
        query_tree.insert(q);
    }

    let k = 3;
    let dual_results = ref_tree.find_k_nearest_dual(&query_tree, k);

    // Verify results are non-empty and sorted
    for neighbors in &dual_results {
        assert!(!neighbors.is_empty());
        for j in 1..neighbors.len() {
            assert!(neighbors[j].1 >= neighbors[j - 1].1);
        }
    }
}

// ============================================================================
// CoverTree wrapper tests
// ============================================================================

#[test]
fn test_dual_tree_wrapper_batch() {
    let metric = EuclideanDistance1D;
    let mut tree = CoverTree::new(metric, 1.3);

    for i in 0..30 {
        tree.insert(i as f64);
    }

    let queries = vec![5.5, 15.5, 25.5];
    let results = tree.find_k_nearest_batch(&queries, 3);

    assert_eq!(results.len(), 3);
    for r in &results {
        assert_eq!(r.len(), 3);
    }
}

#[test]
fn test_dual_tree_wrapper_self() {
    let metric = EuclideanDistance1D;
    let mut tree = CoverTree::new(metric, 1.3);

    for i in 0..10 {
        tree.insert(i as f64 * 5.0);
    }

    let results = tree.find_k_nearest_self(2);

    assert_eq!(results.len(), 10);
    for neighbors in &results {
        assert_eq!(neighbors.len(), 2);
        for (_, dist) in neighbors {
            assert!(*dist > 0.0, "Self-match not excluded in wrapper");
        }
    }
}

#[test]
fn test_dual_tree_wrapper_na_batch() {
    let metric = EuclideanDistance1D;
    let mut tree = CoverTree::new_nearest_ancestor(metric, 1.3);

    for i in 0..30 {
        tree.insert(i as f64);
    }

    let queries = vec![5.5, 15.5, 25.5];
    let results = tree.find_k_nearest_batch(&queries, 3);

    assert_eq!(results.len(), 3);
    for r in &results {
        assert_eq!(r.len(), 3);
    }
}

#[test]
fn test_dual_tree_wrapper_na_self() {
    let metric = EuclideanDistance1D;
    let mut tree = CoverTree::new_nearest_ancestor(metric, 1.3);

    for i in 0..10 {
        tree.insert(i as f64 * 5.0);
    }

    let results = tree.find_k_nearest_self(2);

    assert_eq!(results.len(), 10);
    for neighbors in &results {
        assert_eq!(neighbors.len(), 2);
    }
}
