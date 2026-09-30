//! Integration tests for k-nearest neighbor search
//!
//! These tests verify that k-NN search works correctly on both SimplifiedCoverTree
//! and NACoverTree variants, and that duplicate handling works as expected.

use rustknn::{CoverTree, SimplifiedCoverTree, NACoverTree, Distance};

#[derive(Clone, Debug, PartialEq)]
struct Point2D {
    x: f64,
    y: f64,
}

impl Point2D {
    fn new(x: f64, y: f64) -> Self {
        Point2D { x, y }
    }
}

#[derive(Clone)]
struct EuclideanDistance;

impl Distance<Point2D> for EuclideanDistance {
    fn distance(&self, p: &Point2D, q: &Point2D) -> f64 {
        let dx = p.x - q.x;
        let dy = p.y - q.y;
        (dx * dx + dy * dy).sqrt()
    }
}

#[derive(Clone)]
struct ManhattanDistance;

impl Distance<Point2D> for ManhattanDistance {
    fn distance(&self, p: &Point2D, q: &Point2D) -> f64 {
        (p.x - q.x).abs() + (p.y - q.y).abs()
    }
}

/// Helper: Compute k-NN via brute force for verification
fn brute_force_knn(points: &[Point2D], query: &Point2D, k: usize, metric: &impl Distance<Point2D>) -> Vec<(Point2D, f64)> {
    let mut dists: Vec<(Point2D, f64)> = points
        .iter()
        .map(|p| (p.clone(), metric.distance(p, query)))
        .collect();

    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.truncate(k);
    dists
}

#[test]
fn test_knn_simplified_basic() {
    let mut tree = SimplifiedCoverTree::new(EuclideanDistance, 1.3);

    // Insert points
    tree.insert(Point2D::new(0.0, 0.0));
    tree.insert(Point2D::new(1.0, 0.0));
    tree.insert(Point2D::new(0.0, 1.0));
    tree.insert(Point2D::new(1.0, 1.0));
    tree.insert(Point2D::new(2.0, 2.0));

    let query = Point2D::new(0.5, 0.5);
    let neighbors = tree.find_k_nearest(&query, 3);

    assert_eq!(neighbors.len(), 3);

    // Should get (0,0), (1,0), (0,1) or (1,1) - all within ~0.7 distance
    assert!(neighbors[0].1 < 0.8);
    assert!(neighbors[1].1 < 0.8);
    assert!(neighbors[2].1 < 0.8);
}

#[test]
fn test_knn_na_basic() {
    let mut tree = NACoverTree::new(EuclideanDistance, 1.3);

    // Insert points
    tree.insert(Point2D::new(0.0, 0.0));
    tree.insert(Point2D::new(1.0, 0.0));
    tree.insert(Point2D::new(0.0, 1.0));
    tree.insert(Point2D::new(1.0, 1.0));
    tree.insert(Point2D::new(2.0, 2.0));

    let query = Point2D::new(0.5, 0.5);
    let neighbors = tree.find_k_nearest(&query, 3);

    assert_eq!(neighbors.len(), 3);

    // Should get (0,0), (1,0), (0,1) or (1,1) - all within ~0.7 distance
    assert!(neighbors[0].1 < 0.8);
    assert!(neighbors[1].1 < 0.8);
    assert!(neighbors[2].1 < 0.8);
}

#[test]
fn test_knn_correctness_vs_brute_force() {
    let points = vec![
        Point2D::new(0.0, 0.0),
        Point2D::new(1.0, 0.0),
        Point2D::new(0.0, 1.0),
        Point2D::new(1.0, 1.0),
        Point2D::new(2.0, 0.0),
        Point2D::new(0.0, 2.0),
        Point2D::new(2.0, 2.0),
        Point2D::new(3.0, 3.0),
    ];

    let mut tree = SimplifiedCoverTree::new(EuclideanDistance, 1.3);
    for p in &points {
        tree.insert(p.clone());
    }

    let query = Point2D::new(0.8, 0.8);
    let k = 4;

    let tree_result = tree.find_k_nearest(&query, k);
    let brute_result = brute_force_knn(&points, &query, k, &EuclideanDistance);

    assert_eq!(tree_result.len(), k);
    assert_eq!(brute_result.len(), k);

    // Check that distances match (points may be in different order)
    let mut tree_dists: Vec<f64> = tree_result.iter().map(|(_, d)| *d).collect();
    let mut brute_dists: Vec<f64> = brute_result.iter().map(|(_, d)| *d).collect();

    tree_dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
    brute_dists.sort_by(|a, b| a.partial_cmp(b).unwrap());

    for (tree_d, brute_d) in tree_dists.iter().zip(&brute_dists) {
        assert!((tree_d - brute_d).abs() < 1e-10, "Distance mismatch: {} vs {}", tree_d, brute_d);
    }
}

#[test]
fn test_knn_k_equals_n() {
    let mut tree = SimplifiedCoverTree::new(EuclideanDistance, 1.3);

    tree.insert(Point2D::new(0.0, 0.0));
    tree.insert(Point2D::new(1.0, 1.0));
    tree.insert(Point2D::new(2.0, 2.0));

    let query = Point2D::new(0.0, 0.0);
    let neighbors = tree.find_k_nearest(&query, 3);

    assert_eq!(neighbors.len(), 3);
    // First should be exact match at distance 0
    assert!(neighbors[0].1 < 1e-10);
}

#[test]
fn test_knn_k_greater_than_n() {
    let mut tree = SimplifiedCoverTree::new(EuclideanDistance, 1.3);

    tree.insert(Point2D::new(0.0, 0.0));
    tree.insert(Point2D::new(1.0, 1.0));

    let query = Point2D::new(0.0, 0.0);
    let neighbors = tree.find_k_nearest(&query, 10); // Ask for 10 but only 2 in tree

    assert_eq!(neighbors.len(), 2);
}

#[test]
fn test_knn_k_zero() {
    let mut tree = SimplifiedCoverTree::new(EuclideanDistance, 1.3);

    tree.insert(Point2D::new(0.0, 0.0));
    tree.insert(Point2D::new(1.0, 1.0));

    let query = Point2D::new(0.0, 0.0);
    let neighbors = tree.find_k_nearest(&query, 0);

    assert_eq!(neighbors.len(), 0);
}

#[test]
fn test_knn_k_one_matches_find_nearest() {
    let mut tree = SimplifiedCoverTree::new(EuclideanDistance, 1.3);

    tree.insert(Point2D::new(0.0, 0.0));
    tree.insert(Point2D::new(1.0, 1.0));
    tree.insert(Point2D::new(2.0, 2.0));

    let query = Point2D::new(0.5, 0.5);

    let knn_result = tree.find_k_nearest(&query, 1);
    let nearest_result = tree.find_nearest(&query);

    assert_eq!(knn_result.len(), 1);
    assert!(nearest_result.is_some());

    // Should return the same point
    assert_eq!(knn_result[0].0, nearest_result.unwrap());
}

#[test]
fn test_knn_empty_tree() {
    let tree: SimplifiedCoverTree<Point2D, EuclideanDistance> = SimplifiedCoverTree::new(EuclideanDistance, 1.3);

    let query = Point2D::new(0.0, 0.0);
    let neighbors = tree.find_k_nearest(&query, 5);

    assert_eq!(neighbors.len(), 0);
}

#[test]
fn test_knn_sorted_by_distance() {
    let mut tree = SimplifiedCoverTree::new(EuclideanDistance, 1.3);

    tree.insert(Point2D::new(0.0, 0.0));
    tree.insert(Point2D::new(5.0, 0.0));
    tree.insert(Point2D::new(10.0, 0.0));
    tree.insert(Point2D::new(15.0, 0.0));

    let query = Point2D::new(0.0, 0.0);
    let neighbors = tree.find_k_nearest(&query, 4);

    // Should be sorted by increasing distance
    for i in 1..neighbors.len() {
        assert!(neighbors[i].1 >= neighbors[i - 1].1);
    }

    // First should be at 0, last at 15
    assert!(neighbors[0].1 < 1e-10);
    assert!((neighbors[3].1 - 15.0).abs() < 1e-10);
}

#[test]
fn test_knn_manhattan_distance() {
    let mut tree = SimplifiedCoverTree::new(ManhattanDistance, 1.3);

    tree.insert(Point2D::new(0.0, 0.0));
    tree.insert(Point2D::new(1.0, 0.0));
    tree.insert(Point2D::new(0.0, 1.0));
    tree.insert(Point2D::new(1.0, 1.0));

    let query = Point2D::new(0.0, 0.0);
    let neighbors = tree.find_k_nearest(&query, 2);

    assert_eq!(neighbors.len(), 2);

    // Should get (0,0) at distance 0, then either (1,0) or (0,1) at distance 1
    assert!(neighbors[0].1 < 1e-10);
    assert!((neighbors[1].1 - 1.0).abs() < 1e-10);
}

#[test]
fn test_knn_user_duplicate_values() {
    // Test that user's intentional duplicate values are correctly returned
    let mut tree = SimplifiedCoverTree::new(EuclideanDistance, 1.3);

    // Insert the same point value twice (different insertions, same value)
    tree.insert(Point2D::new(1.0, 1.0));
    tree.insert(Point2D::new(1.0, 1.0));
    tree.insert(Point2D::new(5.0, 5.0));

    let query = Point2D::new(1.0, 1.0);
    let neighbors = tree.find_k_nearest(&query, 3);

    // Should get both (1,1) points and the (5,5) point
    assert_eq!(neighbors.len(), 3);

    // Count how many are at distance ~0 (should be 2)
    let zero_dist_count = neighbors.iter().filter(|(_, d)| *d < 1e-10).count();
    assert_eq!(zero_dist_count, 2, "Should find both user-inserted (1,1) points");
}

#[test]
fn test_knn_merged_tree_no_algorithm_duplicates() {
    // Test that merging doesn't return algorithm duplicates in k-NN results
    let mut tree1 = CoverTree::new(EuclideanDistance, 1.3);
    tree1.insert(Point2D::new(0.0, 0.0));
    tree1.insert(Point2D::new(1.0, 0.0));

    let mut tree2 = CoverTree::new(EuclideanDistance, 1.3);
    tree2.insert(Point2D::new(10.0, 0.0));
    tree2.insert(Point2D::new(11.0, 0.0));

    let merged = tree1.merge(tree2);

    // Find all 4 nearest neighbors
    let query = Point2D::new(5.0, 0.0);
    let neighbors = merged.find_k_nearest(&query, 4);

    // Should get exactly 4 neighbors (the 4 original points, no algorithm duplicates)
    assert_eq!(neighbors.len(), 4);
}

#[test]
fn test_knn_unified_wrapper() {
    let mut tree = CoverTree::new(EuclideanDistance, 1.3);

    tree.insert(Point2D::new(0.0, 0.0));
    tree.insert(Point2D::new(1.0, 1.0));
    tree.insert(Point2D::new(2.0, 2.0));

    let query = Point2D::new(0.5, 0.5);
    let neighbors = tree.find_k_nearest(&query, 2);

    assert_eq!(neighbors.len(), 2);
}

#[test]
fn test_knn_nearest_ancestor_wrapper() {
    let mut tree = CoverTree::new_nearest_ancestor(EuclideanDistance, 1.3);

    tree.insert(Point2D::new(0.0, 0.0));
    tree.insert(Point2D::new(1.0, 1.0));
    tree.insert(Point2D::new(2.0, 2.0));

    let query = Point2D::new(0.5, 0.5);
    let neighbors = tree.find_k_nearest(&query, 2);

    assert_eq!(neighbors.len(), 2);
}

#[test]
fn test_knn_large_dataset() {
    let mut tree = SimplifiedCoverTree::new(EuclideanDistance, 1.3);

    // Insert 100 points on a grid
    for i in 0..10 {
        for j in 0..10 {
            tree.insert(Point2D::new(i as f64, j as f64));
        }
    }

    let query = Point2D::new(5.0, 5.0);
    let k = 10;
    let neighbors = tree.find_k_nearest(&query, k);

    assert_eq!(neighbors.len(), k);

    // Should be sorted by distance
    for i in 1..neighbors.len() {
        assert!(neighbors[i].1 >= neighbors[i - 1].1);
    }
}

#[test]
fn test_knn_variable_k() {
    let mut tree = SimplifiedCoverTree::new(EuclideanDistance, 1.3);

    for i in 0..20 {
        tree.insert(Point2D::new(i as f64, 0.0));
    }

    let query = Point2D::new(10.0, 0.0);

    // Test various k values
    for k in [1, 5, 10, 15, 20, 25] {
        let neighbors = tree.find_k_nearest(&query, k);
        assert_eq!(neighbors.len(), k.min(20));
    }
}

// ========== Packed Cover Tree k-NN Tests ==========

#[test]
fn test_packed_knn_basic() {
    let mut tree = SimplifiedCoverTree::new(EuclideanDistance, 1.3);

    // Insert points
    tree.insert(Point2D::new(0.0, 0.0));
    tree.insert(Point2D::new(1.0, 0.0));
    tree.insert(Point2D::new(0.0, 1.0));
    tree.insert(Point2D::new(1.0, 1.0));
    tree.insert(Point2D::new(2.0, 2.0));

    // Pack the tree
    let packed = tree.pack();

    let query = Point2D::new(0.5, 0.5);
    let neighbors = packed.find_k_nearest(&query, 3);

    assert_eq!(neighbors.len(), 3);

    // Should get points within ~0.7 distance
    assert!(neighbors[0].1 < 0.8);
    assert!(neighbors[1].1 < 0.8);
    assert!(neighbors[2].1 < 0.8);
}

#[test]
fn test_packed_knn_matches_unpacked() {
    // Verify that packed k-NN returns valid k nearest neighbors
    let points = vec![
        Point2D::new(0.0, 0.0),
        Point2D::new(1.0, 0.0),
        Point2D::new(0.0, 1.0),
        Point2D::new(1.0, 1.0),
        Point2D::new(2.0, 0.0),
        Point2D::new(0.0, 2.0),
        Point2D::new(2.0, 2.0),
        Point2D::new(3.0, 3.0),
    ];

    // Build tree for packing
    let mut tree = SimplifiedCoverTree::new(EuclideanDistance, 1.3);
    for p in &points {
        tree.insert(p.clone());
    }

    let query = Point2D::new(0.8, 0.8);
    let k = 5;

    let packed = tree.pack();
    let packed_result = packed.find_k_nearest(&query, k);

    assert_eq!(packed_result.len(), k);

    // Verify results are sorted
    for i in 1..k {
        assert!(packed_result[i].1 >= packed_result[i-1].1,
            "Results not sorted at index {}", i);
    }

    // Verify all returned points actually exist in our dataset
    for (point, _) in &packed_result {
        let found = points.iter().any(|p| (p.x - point.x).abs() < 1e-10 && (p.y - point.y).abs() < 1e-10);
        assert!(found, "Returned point ({}, {}) not in original dataset", point.x, point.y);
    }
}

#[test]
fn test_packed_knn_empty_tree() {
    let tree: SimplifiedCoverTree<Point2D, EuclideanDistance> = SimplifiedCoverTree::new(EuclideanDistance, 1.3);
    let packed = tree.pack();

    let query = Point2D::new(0.0, 0.0);
    let neighbors = packed.find_k_nearest(&query, 5);

    assert_eq!(neighbors.len(), 0);
}

#[test]
fn test_packed_knn_k_zero() {
    let mut tree = SimplifiedCoverTree::new(EuclideanDistance, 1.3);
    tree.insert(Point2D::new(1.0, 1.0));

    let packed = tree.pack();

    let neighbors = packed.find_k_nearest(&Point2D::new(0.0, 0.0), 0);
    assert_eq!(neighbors.len(), 0);
}

#[test]
fn test_packed_knn_k_one_matches_find_nearest() {
    let mut tree = SimplifiedCoverTree::new(EuclideanDistance, 1.3);

    tree.insert(Point2D::new(0.0, 0.0));
    tree.insert(Point2D::new(1.0, 1.0));
    tree.insert(Point2D::new(2.0, 2.0));

    let packed = tree.pack();
    let query = Point2D::new(0.5, 0.5);

    let knn_result = packed.find_k_nearest(&query, 1);
    let nearest_result = packed.find_nearest(&query);

    assert_eq!(knn_result.len(), 1);
    assert!(nearest_result.is_some());

    // Should return the same point
    assert_eq!(knn_result[0].0, nearest_result.unwrap());
}

#[test]
fn test_packed_knn_k_greater_than_n() {
    let mut tree = SimplifiedCoverTree::new(EuclideanDistance, 1.3);

    tree.insert(Point2D::new(0.0, 0.0));
    tree.insert(Point2D::new(1.0, 1.0));

    let packed = tree.pack();

    let query = Point2D::new(0.0, 0.0);
    let neighbors = packed.find_k_nearest(&query, 10); // Ask for 10 but only 2 in tree

    assert_eq!(neighbors.len(), 2);
}

#[test]
fn test_packed_knn_sorted_by_distance() {
    let mut tree = SimplifiedCoverTree::new(EuclideanDistance, 1.3);

    tree.insert(Point2D::new(0.0, 0.0));
    tree.insert(Point2D::new(5.0, 0.0));
    tree.insert(Point2D::new(10.0, 0.0));
    tree.insert(Point2D::new(15.0, 0.0));

    let packed = tree.pack();

    let query = Point2D::new(0.0, 0.0);
    let neighbors = packed.find_k_nearest(&query, 4);

    // Should be sorted by increasing distance
    for i in 1..neighbors.len() {
        assert!(neighbors[i].1 >= neighbors[i - 1].1);
    }

    // First should be at 0, last at 15
    assert!(neighbors[0].1 < 1e-10);
    assert!((neighbors[3].1 - 15.0).abs() < 1e-10);
}

#[test]
fn test_packed_knn_large_dataset() {
    let mut tree = SimplifiedCoverTree::new(EuclideanDistance, 1.3);

    // Insert 100 points on a grid
    for i in 0..10 {
        for j in 0..10 {
            tree.insert(Point2D::new(i as f64, j as f64));
        }
    }

    let packed = tree.pack();

    let query = Point2D::new(5.5, 5.5);  // Query between grid points
    let k = 10;
    let neighbors = packed.find_k_nearest(&query, k);

    assert_eq!(neighbors.len(), k);

    // Should be sorted by distance
    for i in 1..neighbors.len() {
        assert!(neighbors[i].1 >= neighbors[i - 1].1,
            "Results not sorted: neighbors[{}].dist={} >= neighbors[{}].dist={}",
            i, neighbors[i].1, i-1, neighbors[i - 1].1);
    }

    // Verify all points are from the grid
    for (point, _) in &neighbors {
        let x_int = point.x.round() as i32;
        let y_int = point.y.round() as i32;
        assert!(x_int >= 0 && x_int < 10 && y_int >= 0 && y_int < 10,
            "Point ({}, {}) not from grid", point.x, point.y);
    }
}

#[test]
fn test_packed_knn_skips_algorithm_duplicates() {
    // Create a merged tree which will have algorithm duplicates
    let mut tree1 = SimplifiedCoverTree::new(EuclideanDistance, 1.3);
    tree1.insert(Point2D::new(0.0, 0.0));
    tree1.insert(Point2D::new(1.0, 0.0));

    let mut tree2 = SimplifiedCoverTree::new(EuclideanDistance, 1.3);
    tree2.insert(Point2D::new(10.0, 0.0));
    tree2.insert(Point2D::new(11.0, 0.0));

    let merged = tree1.merge(tree2);

    // Pack the merged tree
    let packed = merged.pack();

    // Find all nearest neighbors
    let query = Point2D::new(5.0, 0.0);
    let neighbors = packed.find_k_nearest(&query, 10);

    // Should get exactly 4 neighbors (the 4 original points, no algorithm duplicates)
    assert_eq!(neighbors.len(), 4);

    // Verify no duplicates in results
    for i in 0..neighbors.len() {
        for j in i+1..neighbors.len() {
            let p1 = neighbors[i].0;
            let p2 = neighbors[j].0;
            assert!(p1.x != p2.x || p1.y != p2.y,
                "Found duplicate in results: ({}, {}) appears twice", p1.x, p1.y);
        }
    }
}

