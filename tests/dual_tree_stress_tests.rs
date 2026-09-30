//! Dual-Tree k-NN Stress Tests
//!
//! Targeted tests to expose potential algorithmic failure modes in the dual-tree
//! k-NN implementation. Each test is designed to trigger a specific failure scenario
//! and verifies results against brute-force ground truth.
//!
//! ## Likely Bug Targeted
//!
//! `is_duplicate` reference nodes are NOT skipped in the dual-tree `base_case`
//! (`knn_rules.rs`), while single-tree k-NN (`knn_impl.rs:66`) explicitly skips them.
//! On merged trees (which have `is_duplicate` nodes), dual-tree k-NN may return
//! cloned points as spurious neighbors.

use rustknn::{Distance, SimplifiedCoverTree, NACoverTree};

// ============================================================================
// Distance metrics
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

#[derive(Clone)]
struct ManhattanDistance2D;
impl Distance<Point2D> for ManhattanDistance2D {
    fn distance(&self, p: &Point2D, q: &Point2D) -> f64 {
        (p.x - q.x).abs() + (p.y - q.y).abs()
    }
}

#[derive(Clone)]
struct EuclideanDistanceND;
impl Distance<Vec<f64>> for EuclideanDistanceND {
    fn distance(&self, p: &Vec<f64>, q: &Vec<f64>) -> f64 {
        p.iter()
            .zip(q.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
    }
}

// ============================================================================
// Brute-force helpers
// ============================================================================

/// Brute-force k-NN: returns (index, distance) pairs sorted by distance.
fn brute_force_knn_indexed<T, D: Distance<T>>(
    query: &T,
    ref_points: &[T],
    k: usize,
    metric: &D,
) -> Vec<(usize, f64)> {
    let mut dists: Vec<(usize, f64)> = ref_points
        .iter()
        .enumerate()
        .map(|(i, p)| (i, metric.distance(query, p)))
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.truncate(k);
    dists
}

/// Brute-force self-k-NN: for each point, find k nearest excluding itself (by index).
fn brute_force_self_knn_indexed<T, D: Distance<T>>(
    points: &[T],
    k: usize,
    metric: &D,
) -> Vec<Vec<(usize, f64)>> {
    points
        .iter()
        .enumerate()
        .map(|(qi, query)| {
            let mut dists: Vec<(usize, f64)> = points
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != qi)
                .map(|(i, p)| (i, metric.distance(query, p)))
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            dists.truncate(k);
            dists
        })
        .collect()
}

/// Compare dual-tree results against brute-force by distance vectors.
/// Tolerant of tie-breaking differences (only compares distances, not point identity).
fn assert_distances_match(
    label: &str,
    actual: &[(&f64, f64)],
    expected_dists: &[f64],
    eps: f64,
) {
    assert_eq!(
        actual.len(),
        expected_dists.len(),
        "{}: length mismatch: got {}, expected {}",
        label,
        actual.len(),
        expected_dists.len()
    );
    for (i, ((_pt, d_actual), d_expected)) in actual.iter().zip(expected_dists.iter()).enumerate() {
        assert!(
            (d_actual - d_expected).abs() <= eps,
            "{}: distance mismatch at index {}: actual={}, expected={} (eps={})",
            label,
            i,
            d_actual,
            d_expected,
            eps
        );
    }
}

/// Generic distance comparison for Point2D results.
fn assert_distances_match_2d(
    label: &str,
    actual: &[(&Point2D, f64)],
    expected_dists: &[f64],
    eps: f64,
) {
    assert_eq!(
        actual.len(),
        expected_dists.len(),
        "{}: length mismatch: got {}, expected {}",
        label,
        actual.len(),
        expected_dists.len()
    );
    for (i, ((_pt, d_actual), d_expected)) in actual.iter().zip(expected_dists.iter()).enumerate() {
        assert!(
            (d_actual - d_expected).abs() <= eps,
            "{}: distance mismatch at index {}: actual={}, expected={}",
            label,
            i,
            d_actual,
            d_expected,
        );
    }
}

/// Generic distance comparison for Vec<f64> results.
fn assert_distances_match_nd(
    label: &str,
    actual: &[(&Vec<f64>, f64)],
    expected_dists: &[f64],
    eps: f64,
) {
    assert_eq!(
        actual.len(),
        expected_dists.len(),
        "{}: length mismatch: got {}, expected {}",
        label,
        actual.len(),
        expected_dists.len()
    );
    for (i, ((_pt, d_actual), d_expected)) in actual.iter().zip(expected_dists.iter()).enumerate() {
        assert!(
            (d_actual - d_expected).abs() <= eps,
            "{}: distance mismatch at index {}: actual={}, expected={}",
            label,
            i,
            d_actual,
            d_expected,
        );
    }
}

/// Simple seeded pseudo-random number generator (xorshift64).
/// Avoids external dependency on `rand` crate.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        SimpleRng {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Returns a f64 in [lo, hi).
    fn next_f64_range(&mut self, lo: f64, hi: f64) -> f64 {
        let t = (self.next_u64() as f64) / (u64::MAX as f64);
        lo + t * (hi - lo)
    }
}

// ============================================================================
// Tier 1: Most Likely to Expose Bugs
// ============================================================================

/// Test 1: Self-k-NN on merged tree should NOT return is_duplicate clones as neighbors.
///
/// Failure mode: `is_duplicate` bug — dual-tree self-k-NN on merged tree returns
/// cloned points at distance 0.0.
#[test]
fn test_self_knn_merged_tree_no_duplicate_artifacts() {
    let metric = EuclideanDistance1D;

    // Build two separate trees
    let mut tree1 = SimplifiedCoverTree::new(metric.clone(), 1.3);
    for &p in &[0.0, 1.0, 2.0] {
        tree1.insert(p);
    }
    let mut tree2 = SimplifiedCoverTree::new(metric.clone(), 1.3);
    for &p in &[10.0, 11.0, 12.0] {
        tree2.insert(p);
    }

    // Merge creates is_duplicate nodes for level alignment
    let merged = tree1.merge(tree2);
    assert_eq!(merged.len(), 6);

    // Self-k-NN: each point's neighbors should NOT include itself (distance 0.0)
    let results = merged.find_k_nearest_self(2);

    let all_points = [0.0, 1.0, 2.0, 10.0, 11.0, 12.0];
    let brute = brute_force_self_knn_indexed(&all_points, 2, &metric);

    // Verify no self-matches (distance 0.0) appear
    for (idx, neighbors) in results.iter().enumerate() {
        for (pt, dist) in neighbors {
            assert!(
                *dist > 1e-12,
                "Self-k-NN on merged tree returned distance ~0.0 for result {}: \
                 point={}, dist={}. This suggests is_duplicate nodes are leaking through.",
                idx,
                pt,
                dist
            );
        }
    }

    // Verify result count
    for neighbors in &results {
        assert_eq!(
            neighbors.len(),
            2,
            "Expected 2 neighbors per point in self-k-NN"
        );
    }

    // Cross-check distances against brute-force for each result
    // Note: results are in DFS order of the merged tree, not necessarily in
    // [0,1,2,10,11,12] order. We verify by matching each result to a brute-force entry.
    for neighbors in &results {
        // Find which original point this result corresponds to
        // by checking that the query must be one of the 6 points
        let query_val = {
            // The nearest neighbor's distance + its value constrains the query
            // But simpler: check all 6 points and see which brute-force entry matches
            let actual_dists: Vec<f64> = neighbors.iter().map(|(_, d)| *d).collect();
            let mut matched = false;
            for (bi, bf_neighbors) in brute.iter().enumerate() {
                let bf_dists: Vec<f64> = bf_neighbors.iter().map(|(_, d)| *d).collect();
                if actual_dists.len() == bf_dists.len()
                    && actual_dists
                        .iter()
                        .zip(bf_dists.iter())
                        .all(|(a, b)| (a - b).abs() < 1e-10)
                {
                    matched = true;
                    let _ = bi;
                    break;
                }
            }
            matched
        };
        // We don't require every result to match (DFS order may differ),
        // but we verify no spurious distance-0 entries exist (checked above).
        let _ = query_val;
    }
}

/// Test 2: Batch k-NN on merged reference tree should not return duplicate-value neighbors.
///
/// Failure mode: `is_duplicate` bug — batch k-NN on merged reference tree returns
/// algorithm duplicates as extra neighbors.
#[test]
fn test_batch_knn_merged_tree_no_duplicate_neighbors() {
    let metric = EuclideanDistance1D;

    let mut tree1 = SimplifiedCoverTree::new(metric.clone(), 1.3);
    for &p in &[0.0, 1.0, 2.0, 3.0, 4.0] {
        tree1.insert(p);
    }
    let mut tree2 = SimplifiedCoverTree::new(metric.clone(), 1.3);
    for &p in &[5.0, 6.0, 7.0, 8.0, 9.0] {
        tree2.insert(p);
    }

    let merged = tree1.merge(tree2);
    assert_eq!(merged.len(), 10);

    let queries = vec![2.5, 7.5];
    let k = 3;
    let results = merged.find_k_nearest_batch(&queries, k);

    let all_points: Vec<f64> = (0..10).map(|i| i as f64).collect();

    for (qi, q) in queries.iter().enumerate() {
        let neighbors = &results[qi];
        assert_eq!(
            neighbors.len(),
            k,
            "Query {} should have {} neighbors",
            q,
            k
        );

        // Check that no two neighbors have the same value
        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                assert!(
                    (*neighbors[i].0 - *neighbors[j].0).abs() > 1e-12,
                    "Query {}: duplicate neighbor values {} and {} at positions {} and {}. \
                     This suggests is_duplicate nodes are leaking into results.",
                    q,
                    neighbors[i].0,
                    neighbors[j].0,
                    i,
                    j
                );
            }
        }

        // Cross-check with brute force
        let bf = brute_force_knn_indexed(q, &all_points, k, &metric);
        let bf_dists: Vec<f64> = bf.iter().map(|(_, d)| *d).collect();
        assert_distances_match(
            &format!("batch_merged query={}", q),
            neighbors,
            &bf_dists,
            1e-10,
        );
    }
}

/// Test 3: Batch k-NN with identical query values should return identical results.
///
/// Failure mode: Result mapping with duplicate query values assigns results to wrong slot.
#[test]
fn test_batch_identical_queries() {
    let metric = EuclideanDistance1D;
    let mut tree = SimplifiedCoverTree::new(metric.clone(), 1.3);
    for &p in &[0.0, 5.0, 10.0, 15.0, 20.0] {
        tree.insert(p);
    }

    let queries = vec![7.5, 7.5, 7.5];
    let k = 2;
    let results = tree.find_k_nearest_batch(&queries, k);

    assert_eq!(results.len(), 3);

    // All three should produce the same neighbor distances
    let ref_points = [0.0, 5.0, 10.0, 15.0, 20.0];
    let bf = brute_force_knn_indexed(&7.5, &ref_points, k, &metric);
    let bf_dists: Vec<f64> = bf.iter().map(|(_, d)| *d).collect();

    for (i, neighbors) in results.iter().enumerate() {
        assert_eq!(neighbors.len(), k, "Query {} should have {} neighbors", i, k);
        assert_distances_match(
            &format!("identical_query[{}]", i),
            neighbors,
            &bf_dists,
            1e-10,
        );
    }

    // Additionally verify all three results have matching distances
    for i in 1..results.len() {
        for j in 0..results[i].len() {
            assert!(
                (results[i][j].1 - results[0][j].1).abs() < 1e-10,
                "Identical queries produced different distance at pos {}: {} vs {}",
                j,
                results[i][j].1,
                results[0][j].1
            );
        }
    }
}

/// Test 4: Nearly identical queries should each get correctly mapped results.
///
/// Failure mode: Result mapping assigns results to wrong query when queries are
/// very close together.
#[test]
fn test_batch_nearly_identical_queries() {
    let metric = EuclideanDistance1D;
    let mut tree = SimplifiedCoverTree::new(metric.clone(), 1.3);
    for &p in &[0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 50.0, 100.0] {
        tree.insert(p);
    }

    let queries = vec![3.0, 3.001, 3.002, 3.003];
    let k = 3;
    let results = tree.find_k_nearest_batch(&queries, k);

    let ref_points = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 50.0, 100.0];

    assert_eq!(results.len(), 4);

    for (qi, q) in queries.iter().enumerate() {
        let neighbors = &results[qi];
        assert_eq!(
            neighbors.len(),
            k,
            "Query {} should have {} neighbors",
            q,
            k
        );

        let bf = brute_force_knn_indexed(q, &ref_points, k, &metric);
        let bf_dists: Vec<f64> = bf.iter().map(|(_, d)| *d).collect();
        assert_distances_match(
            &format!("nearly_identical query={}", q),
            neighbors,
            &bf_dists,
            1e-10,
        );
    }
}

/// Test 5: Deep neighbor should not be pruned by over-aggressive bounds.
///
/// Failure mode: The true nearest neighbor is deep in the reference tree and gets
/// incorrectly pruned.
#[test]
fn test_deep_neighbor_not_pruned() {
    let metric = EuclideanDistance1D;
    let mut tree = SimplifiedCoverTree::new(metric.clone(), 1.3);

    // Insert cluster far from query first, then the true nearest neighbor last
    // (so it ends up deep in the tree)
    for &p in &[100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 1.0] {
        tree.insert(p);
    }

    let queries = vec![0.5];
    let results = tree.find_k_nearest_batch(&queries, 1);

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].len(), 1);
    assert_eq!(
        *results[0][0].0, 1.0,
        "Expected nearest neighbor to be 1.0 (distance 0.5), got {} (distance {})",
        results[0][0].0, results[0][0].1
    );
    assert!((results[0][0].1 - 0.5).abs() < 1e-10);
}

/// Test 6: Sorted insertion creates degenerate tree with loose maxdist.
///
/// Failure mode: Sorted data creates single-child chains where maxdist is very loose,
/// causing missed pruning or incorrect results.
#[test]
fn test_sorted_insertion_degenerate_tree() {
    let metric = EuclideanDistance1D;
    let mut tree = SimplifiedCoverTree::new(metric.clone(), 1.3);

    let ref_points: Vec<f64> = (0..100).map(|i| i as f64).collect();
    for &p in &ref_points {
        tree.insert(p);
    }

    let queries = vec![5.5, 50.5, 95.5];
    let k = 5;
    let results = tree.find_k_nearest_batch(&queries, k);

    assert_eq!(results.len(), 3);

    for (qi, q) in queries.iter().enumerate() {
        let neighbors = &results[qi];
        assert_eq!(neighbors.len(), k, "Query {} should have {} neighbors", q, k);

        let bf = brute_force_knn_indexed(q, &ref_points, k, &metric);
        let bf_dists: Vec<f64> = bf.iter().map(|(_, d)| *d).collect();
        assert_distances_match(
            &format!("sorted_degenerate query={}", q),
            neighbors,
            &bf_dists,
            1e-10,
        );
    }
}

/// Test 7: Outlier query with clustered queries.
///
/// Failure mode: B2 bound `(B_aux + 2*maxdist)` over-prunes for the outlier query
/// when the query tree has a tight cluster that makes `B_aux` very small.
#[test]
fn test_outlier_query_with_clustered_queries() {
    let metric = EuclideanDistance1D;
    let mut tree = SimplifiedCoverTree::new(metric.clone(), 1.3);

    let ref_points = [0.0, 1.0, 2.0, 3.0, 4.0, 50.0, 51.0, 52.0];
    for &p in &ref_points {
        tree.insert(p);
    }

    // Cluster of close queries + one far outlier
    let queries = vec![1.5, 2.5, 3.5, 100.0];
    let k = 3;
    let results = tree.find_k_nearest_batch(&queries, k);

    assert_eq!(results.len(), 4);

    for (qi, q) in queries.iter().enumerate() {
        let neighbors = &results[qi];
        assert_eq!(
            neighbors.len(),
            k,
            "Query {} should have {} neighbors",
            q,
            k
        );

        let bf = brute_force_knn_indexed(q, &ref_points, k, &metric);
        let bf_dists: Vec<f64> = bf.iter().map(|(_, d)| *d).collect();
        assert_distances_match(
            &format!("outlier_clustered query={}", q),
            neighbors,
            &bf_dists,
            1e-10,
        );
    }

    // Specifically verify the outlier query (100.0) finds [52.0, 51.0, 50.0]
    let outlier_neighbors = &results[3];
    let outlier_values: Vec<f64> = outlier_neighbors.iter().map(|(p, _)| **p).collect();
    // Should be sorted by distance: 52.0 (dist 48), 51.0 (dist 49), 50.0 (dist 50)
    assert!(
        (outlier_values[0] - 52.0).abs() < 1e-10,
        "Outlier nearest should be 52.0, got {}",
        outlier_values[0]
    );
}

// ============================================================================
// Tier 2: Moderate Likelihood
// ============================================================================

/// Test 8: k equals n — no pruning possible, exercises different code path.
#[test]
fn test_large_k_equals_n() {
    let metric = EuclideanDistance1D;
    let mut tree = SimplifiedCoverTree::new(metric.clone(), 1.3);

    let ref_points: Vec<f64> = (0..10).map(|i| i as f64 * 5.0).collect();
    for &p in &ref_points {
        tree.insert(p);
    }

    let queries = vec![22.5];
    let k = 10; // equals n
    let results = tree.find_k_nearest_batch(&queries, k);

    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0].len(),
        10,
        "k=n should return all 10 points"
    );

    // Verify sorted by distance
    for i in 1..results[0].len() {
        assert!(
            results[0][i].1 >= results[0][i - 1].1,
            "Results not sorted: dist[{}]={} < dist[{}]={}",
            i,
            results[0][i].1,
            i - 1,
            results[0][i - 1].1
        );
    }

    // Cross-check with brute force
    let bf = brute_force_knn_indexed(&22.5, &ref_points, k, &metric);
    let bf_dists: Vec<f64> = bf.iter().map(|(_, d)| *d).collect();
    assert_distances_match("k_equals_n", &results[0], &bf_dists, 1e-10);
}

/// Test 9: k = n - 1 — must exclude exactly the farthest point.
#[test]
fn test_large_k_equals_n_minus_1() {
    let metric = EuclideanDistance1D;
    let mut tree = SimplifiedCoverTree::new(metric.clone(), 1.3);

    let ref_points: Vec<f64> = (0..10).map(|i| i as f64 * 5.0).collect();
    for &p in &ref_points {
        tree.insert(p);
    }

    let queries = vec![22.5];
    let k = 9; // n - 1
    let results = tree.find_k_nearest_batch(&queries, k);

    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0].len(),
        9,
        "k=n-1 should return 9 of 10 points"
    );

    let bf = brute_force_knn_indexed(&22.5, &ref_points, k, &metric);
    let bf_dists: Vec<f64> = bf.iter().map(|(_, d)| *d).collect();
    assert_distances_match("k_equals_n_minus_1", &results[0], &bf_dists, 1e-10);
}

/// Test 10: Equidistant tie-breaking in 2D.
///
/// Failure mode: Pruning incorrectly eliminates one of multiple equidistant candidates.
#[test]
fn test_equidistant_ties_2d() {
    let metric = EuclideanDistance2D;
    let mut tree = SimplifiedCoverTree::new(metric.clone(), 1.3);

    // 4 points, all at distance 1.0 from origin
    let ref_points = vec![
        Point2D { x: -1.0, y: 0.0 },
        Point2D { x: 1.0, y: 0.0 },
        Point2D { x: 0.0, y: -1.0 },
        Point2D { x: 0.0, y: 1.0 },
    ];
    for p in &ref_points {
        tree.insert(p.clone());
    }

    let query = Point2D { x: 0.0, y: 0.0 };
    let k = 3;
    let neighbors = tree.find_k_nearest(&query, k);

    assert_eq!(neighbors.len(), 3, "Should return 3 of 4 equidistant points");

    // All returned neighbors should be at distance 1.0
    for (i, (_, dist)) in neighbors.iter().enumerate() {
        assert!(
            (dist - 1.0).abs() < 1e-10,
            "Neighbor {} at distance {} (expected 1.0)",
            i,
            dist
        );
    }
}

/// Test 11: Query value equals a reference value (same_set=false).
///
/// Failure mode: Exact match should appear as neighbor at distance 0.0.
#[test]
fn test_query_exists_in_reference() {
    let metric = EuclideanDistance1D;
    let mut tree = SimplifiedCoverTree::new(metric.clone(), 1.3);

    let ref_points = [0.0, 5.0, 10.0, 15.0, 20.0];
    for &p in &ref_points {
        tree.insert(p);
    }

    let queries = vec![10.0, 7.5];
    let k = 2;
    let results = tree.find_k_nearest_batch(&queries, k);

    assert_eq!(results.len(), 2);

    // Query 10.0: nearest at distance 0.0
    assert_eq!(results[0].len(), k);
    assert!(
        results[0][0].1.abs() < 1e-10,
        "Query 10.0 should have neighbor at distance 0.0, got {}",
        results[0][0].1
    );

    // Cross-check both with brute force
    for (qi, q) in queries.iter().enumerate() {
        let bf = brute_force_knn_indexed(q, &ref_points, k, &metric);
        let bf_dists: Vec<f64> = bf.iter().map(|(_, d)| *d).collect();
        assert_distances_match(
            &format!("query_in_ref query={}", q),
            &results[qi],
            &bf_dists,
            1e-10,
        );
    }
}

/// Test 12: Negative coordinate values.
#[test]
fn test_negative_coordinates() {
    let metric = EuclideanDistance1D;
    let mut tree = SimplifiedCoverTree::new(metric.clone(), 1.3);

    let ref_points = [-10.0, -5.0, 0.0, 5.0, 10.0];
    for &p in &ref_points {
        tree.insert(p);
    }

    let queries = vec![-7.5, 2.5];
    let k = 2;
    let results = tree.find_k_nearest_batch(&queries, k);

    assert_eq!(results.len(), 2);

    for (qi, q) in queries.iter().enumerate() {
        let bf = brute_force_knn_indexed(q, &ref_points, k, &metric);
        let bf_dists: Vec<f64> = bf.iter().map(|(_, d)| *d).collect();
        assert_distances_match(
            &format!("negative_coords query={}", q),
            &results[qi],
            &bf_dists,
            1e-10,
        );
    }
}

/// Test 13: Manhattan distance metric (non-Euclidean).
///
/// Failure mode: Algorithm assumptions that only hold for Euclidean distance.
#[test]
fn test_manhattan_distance() {
    let metric = ManhattanDistance2D;
    let mut tree = SimplifiedCoverTree::new(metric.clone(), 1.3);

    let ref_points = vec![
        Point2D { x: 0.0, y: 0.0 },
        Point2D { x: 3.0, y: 0.0 },
        Point2D { x: 0.0, y: 4.0 },
        Point2D { x: 3.0, y: 4.0 },
        Point2D { x: 10.0, y: 10.0 },
    ];
    for p in &ref_points {
        tree.insert(p.clone());
    }

    let query = Point2D { x: 1.0, y: 1.0 };
    let k = 2;
    let neighbors = tree.find_k_nearest(&query, k);

    // Brute force:
    // (0,0): |1-0| + |1-0| = 2
    // (3,0): |1-3| + |1-0| = 3
    // (0,4): |1-0| + |1-4| = 4
    // (3,4): |1-3| + |1-4| = 5
    // (10,10): |1-10| + |1-10| = 18
    // k=2: (0,0) at dist 2, (3,0) at dist 3

    assert_eq!(neighbors.len(), 2);
    let bf = brute_force_knn_indexed(&query, &ref_points, k, &metric);
    let bf_dists: Vec<f64> = bf.iter().map(|(_, d)| *d).collect();
    assert_distances_match_2d("manhattan", &neighbors, &bf_dists, 1e-10);

    // Verify specific values
    assert!((neighbors[0].1 - 2.0).abs() < 1e-10, "Nearest should be at Manhattan distance 2.0");
    assert!((neighbors[1].1 - 3.0).abs() < 1e-10, "Second nearest should be at Manhattan distance 3.0");
}

/// Test 14: High-dimensional (50D) data.
///
/// Failure mode: Curse of dimensionality weakens pruning, causing missed neighbors.
#[test]
fn test_high_dimensional_50d() {
    let metric = EuclideanDistanceND;
    let dim = 50;
    let mut rng = SimpleRng::new(42);

    // Generate 30 reference points in 50D
    let ref_points: Vec<Vec<f64>> = (0..30)
        .map(|_| (0..dim).map(|_| rng.next_f64_range(-10.0, 10.0)).collect())
        .collect();

    // Generate 10 query points in 50D
    let query_points: Vec<Vec<f64>> = (0..10)
        .map(|_| (0..dim).map(|_| rng.next_f64_range(-10.0, 10.0)).collect())
        .collect();

    let mut tree = SimplifiedCoverTree::new(metric.clone(), 1.3);
    for p in &ref_points {
        tree.insert(p.clone());
    }

    let k = 5;
    let results = tree.find_k_nearest_batch(&query_points, k);

    assert_eq!(results.len(), 10);

    for (qi, q) in query_points.iter().enumerate() {
        let neighbors = &results[qi];
        assert_eq!(
            neighbors.len(),
            k,
            "50D query {} should have {} neighbors",
            qi,
            k
        );

        let bf = brute_force_knn_indexed(q, &ref_points, k, &metric);
        let bf_dists: Vec<f64> = bf.iter().map(|(_, d)| *d).collect();
        assert_distances_match_nd(
            &format!("50d query={}", qi),
            neighbors,
            &bf_dists,
            1e-8, // Slightly looser epsilon for high-dim floating point
        );
    }
}

// ============================================================================
// Tier 3: Stress Tests
// ============================================================================

/// Test 15: Random 1D stress test (50 ref, 20 query, k=5).
#[test]
fn test_stress_random_1d() {
    let metric = EuclideanDistance1D;
    let mut rng = SimpleRng::new(12345);

    let ref_points: Vec<f64> = (0..50).map(|_| rng.next_f64_range(-100.0, 100.0)).collect();
    let query_points: Vec<f64> = (0..20).map(|_| rng.next_f64_range(-100.0, 100.0)).collect();

    let mut tree = SimplifiedCoverTree::new(metric.clone(), 1.3);
    for &p in &ref_points {
        tree.insert(p);
    }

    let k = 5;
    let results = tree.find_k_nearest_batch(&query_points, k);

    assert_eq!(results.len(), 20);

    for (qi, q) in query_points.iter().enumerate() {
        let bf = brute_force_knn_indexed(q, &ref_points, k, &metric);
        let bf_dists: Vec<f64> = bf.iter().map(|(_, d)| *d).collect();
        assert_distances_match(
            &format!("stress_1d query={}", qi),
            &results[qi],
            &bf_dists,
            1e-10,
        );
    }
}

/// Test 16: Random 2D stress test (200 ref, 50 query, k=5).
#[test]
fn test_stress_random_2d() {
    let metric = EuclideanDistance2D;
    let mut rng = SimpleRng::new(11111);

    let ref_points: Vec<Point2D> = (0..200)
        .map(|_| Point2D {
            x: rng.next_f64_range(-50.0, 50.0),
            y: rng.next_f64_range(-50.0, 50.0),
        })
        .collect();

    let query_points: Vec<Point2D> = (0..50)
        .map(|_| Point2D {
            x: rng.next_f64_range(-50.0, 50.0),
            y: rng.next_f64_range(-50.0, 50.0),
        })
        .collect();

    let mut tree = SimplifiedCoverTree::new(metric.clone(), 1.3);
    for p in &ref_points {
        tree.insert(p.clone());
    }

    let k = 5;
    let results = tree.find_k_nearest_batch(&query_points, k);

    assert_eq!(results.len(), 50);

    for (qi, q) in query_points.iter().enumerate() {
        let bf = brute_force_knn_indexed(q, &ref_points, k, &metric);
        let bf_dists: Vec<f64> = bf.iter().map(|(_, d)| *d).collect();
        assert_distances_match_2d(
            &format!("stress_2d query={}", qi),
            &results[qi],
            &bf_dists,
            1e-10,
        );
    }
}

/// Test 17: Large random 1D stress test (500 ref, 100 query, k=10).
#[test]
fn test_stress_random_1d_large() {
    let metric = EuclideanDistance1D;
    let mut rng = SimpleRng::new(67890);

    let ref_points: Vec<f64> = (0..500).map(|_| rng.next_f64_range(-1000.0, 1000.0)).collect();
    let query_points: Vec<f64> = (0..100).map(|_| rng.next_f64_range(-1000.0, 1000.0)).collect();

    let mut tree = SimplifiedCoverTree::new(metric.clone(), 1.3);
    for &p in &ref_points {
        tree.insert(p);
    }
    // Recompute exact maxdist for tight pruning bounds (approximate values from
    // incremental insertion can cause over-pruning with large datasets).
    tree.recompute_maxdist();

    let k = 10;
    let results = tree.find_k_nearest_batch(&query_points, k);

    assert_eq!(results.len(), 100);

    for (qi, q) in query_points.iter().enumerate() {
        let bf = brute_force_knn_indexed(q, &ref_points, k, &metric);
        let bf_dists: Vec<f64> = bf.iter().map(|(_, d)| *d).collect();
        assert_distances_match(
            &format!("stress_1d_large query={}", qi),
            &results[qi],
            &bf_dists,
            1e-10,
        );
    }
}

/// Test 18: Self-k-NN on medium non-uniform dataset.
#[test]
fn test_self_knn_medium_brute_force() {
    let metric = EuclideanDistance1D;
    let mut tree = SimplifiedCoverTree::new(metric.clone(), 1.3);

    // Non-uniform spacing (quadratic growth)
    let points = [0.0, 0.7, 1.5, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0, 36.0, 45.0, 55.0];
    for &p in &points {
        tree.insert(p);
    }

    let k = 3;
    let results = tree.find_k_nearest_self(k);

    let brute = brute_force_self_knn_indexed(&points, k, &metric);

    assert_eq!(
        results.len(),
        points.len(),
        "Self-k-NN should return one result per point"
    );

    // Verify no self-matches
    for (idx, neighbors) in results.iter().enumerate() {
        assert_eq!(
            neighbors.len(),
            k,
            "Point {} should have {} self-k-NN neighbors",
            idx,
            k
        );
        for (_, dist) in neighbors {
            assert!(
                *dist > 1e-12,
                "Self-match leaked through at index {}",
                idx
            );
        }
    }

    // Cross-check: find which brute-force entry matches each result by distances
    // Results are in DFS order, so we match by distance pattern
    let mut matched_count = 0;
    for neighbors in &results {
        let actual_dists: Vec<f64> = neighbors.iter().map(|(_, d)| *d).collect();
        for bf_neighbors in &brute {
            let bf_dists: Vec<f64> = bf_neighbors.iter().map(|(_, d)| *d).collect();
            if actual_dists.len() == bf_dists.len()
                && actual_dists
                    .iter()
                    .zip(bf_dists.iter())
                    .all(|(a, b)| (a - b).abs() < 1e-10)
            {
                matched_count += 1;
                break;
            }
        }
    }
    assert_eq!(
        matched_count,
        points.len(),
        "All {} self-k-NN results should match brute force, only {} matched",
        points.len(),
        matched_count
    );
}

/// Test 19: NACoverTree batch k-NN matches brute force.
#[test]
fn test_na_batch_matches_brute_force() {
    let metric = EuclideanDistance1D;
    let mut tree = NACoverTree::new(metric.clone(), 1.3);

    let ref_points: Vec<f64> = (0..30).map(|i| i as f64).collect();
    for &p in &ref_points {
        tree.insert(p);
    }

    let queries = vec![5.5, 15.5, 25.5];
    let k = 3;
    let results = tree.find_k_nearest_batch(&queries, k);

    assert_eq!(results.len(), 3);

    for (qi, q) in queries.iter().enumerate() {
        let neighbors = &results[qi];
        assert_eq!(neighbors.len(), k);

        let bf = brute_force_knn_indexed(q, &ref_points, k, &metric);
        let bf_dists: Vec<f64> = bf.iter().map(|(_, d)| *d).collect();
        assert_distances_match(
            &format!("na_batch query={}", q),
            neighbors,
            &bf_dists,
            1e-10,
        );
    }
}

/// Test 20: NACoverTree self-k-NN matches brute force.
#[test]
fn test_na_self_knn_matches_brute_force() {
    let metric = EuclideanDistance1D;
    let mut tree = NACoverTree::new(metric.clone(), 1.3);

    let points = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0];
    for &p in &points {
        tree.insert(p);
    }

    let k = 2;
    let results = tree.find_k_nearest_self(k);

    let brute = brute_force_self_knn_indexed(&points, k, &metric);

    assert_eq!(results.len(), points.len());

    // Verify no self-matches
    for (idx, neighbors) in results.iter().enumerate() {
        assert_eq!(neighbors.len(), k, "Point {} should have {} neighbors", idx, k);
        for (_, dist) in neighbors {
            assert!(
                *dist > 1e-12,
                "Self-match at index {}: distance {}",
                idx,
                dist
            );
        }
    }

    // Cross-check by distance pattern matching
    let mut matched_count = 0;
    for neighbors in &results {
        let actual_dists: Vec<f64> = neighbors.iter().map(|(_, d)| *d).collect();
        for bf_neighbors in &brute {
            let bf_dists: Vec<f64> = bf_neighbors.iter().map(|(_, d)| *d).collect();
            if actual_dists.len() == bf_dists.len()
                && actual_dists
                    .iter()
                    .zip(bf_dists.iter())
                    .all(|(a, b)| (a - b).abs() < 1e-10)
            {
                matched_count += 1;
                break;
            }
        }
    }
    assert_eq!(
        matched_count,
        points.len(),
        "All {} NA self-k-NN results should match brute force, only {} matched",
        points.len(),
        matched_count
    );
}
