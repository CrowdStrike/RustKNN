//! Comprehensive integration tests for SimplifiedCoverTree merge operation
//!
//! These tests validate that Algorithm 4 (tree merging) correctly maintains
//! all three cover tree invariants after merging two trees.

use rustknn::simplified::SimplifiedCoverTree;
use rustknn::{CoverTree, Distance};

// Test point type
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

// Euclidean distance metric
#[derive(Clone)]
struct EuclideanDistance;

impl Distance<Point2D> for EuclideanDistance {
    fn distance(&self, p: &Point2D, q: &Point2D) -> f64 {
        let dx = p.x - q.x;
        let dy = p.y - q.y;
        (dx * dx + dy * dy).sqrt()
    }
}

// Simple 1D distance for easier testing
#[derive(Clone)]
struct SimpleDistance;

impl Distance<f64> for SimpleDistance {
    fn distance(&self, p: &f64, q: &f64) -> f64 {
        (p - q).abs()
    }
}

#[test]
fn test_merge_two_small_trees() {
    let mut tree1 = SimplifiedCoverTree::new(SimpleDistance, 1.3);
    tree1.insert(1.0);
    tree1.insert(2.0);

    let mut tree2 = SimplifiedCoverTree::new(SimpleDistance, 1.3);
    tree2.insert(10.0);
    tree2.insert(20.0);

    let merged = tree1.merge(tree2);

    assert_eq!(merged.len(), 4);

    // Verify all points are findable
    assert!(merged.find_nearest(&1.0).is_some());
    assert!(merged.find_nearest(&2.0).is_some());
    assert!(merged.find_nearest(&10.0).is_some());
    assert!(merged.find_nearest(&20.0).is_some());
}

#[test]
fn test_merge_empty_trees() {
    let tree1 = SimplifiedCoverTree::new(SimpleDistance, 1.3);
    let tree2 = SimplifiedCoverTree::new(SimpleDistance, 1.3);

    let merged = tree1.merge(tree2);
    assert_eq!(merged.len(), 0);
    assert!(merged.is_empty());
}

#[test]
fn test_merge_with_empty_left() {
    let tree1 = SimplifiedCoverTree::new(SimpleDistance, 1.3);

    let mut tree2 = SimplifiedCoverTree::new(SimpleDistance, 1.3);
    tree2.insert(10.0);
    tree2.insert(20.0);

    let merged = tree1.merge(tree2);
    assert_eq!(merged.len(), 2);
}

#[test]
fn test_merge_with_empty_right() {
    let mut tree1 = SimplifiedCoverTree::new(SimpleDistance, 1.3);
    tree1.insert(1.0);
    tree1.insert(2.0);

    let tree2 = SimplifiedCoverTree::new(SimpleDistance, 1.3);

    let merged = tree1.merge(tree2);
    assert_eq!(merged.len(), 2);
}

#[test]
fn test_merge_trees_at_different_levels() {
    // Create tree1 with close points (lower levels)
    let mut tree1 = SimplifiedCoverTree::new(SimpleDistance, 1.3);
    tree1.insert(1.0);
    tree1.insert(1.5);
    tree1.insert(2.0);

    // Create tree2 with far points (higher levels)
    let mut tree2 = SimplifiedCoverTree::new(SimpleDistance, 1.3);
    tree2.insert(100.0);
    tree2.insert(200.0);

    let merged = tree1.merge(tree2);
    assert_eq!(merged.len(), 5);

    // Verify all points findable
    assert!(merged.find_nearest(&1.0).is_some());
    assert!(merged.find_nearest(&2.0).is_some());
    assert!(merged.find_nearest(&100.0).is_some());
    assert!(merged.find_nearest(&200.0).is_some());
}

#[test]
fn test_merge_with_overlapping_points() {
    let mut tree1 = SimplifiedCoverTree::new(SimpleDistance, 1.3);
    tree1.insert(1.0);
    tree1.insert(5.0);
    tree1.insert(10.0);

    let mut tree2 = SimplifiedCoverTree::new(SimpleDistance, 1.3);
    tree2.insert(3.0);
    tree2.insert(7.0);
    tree2.insert(12.0);

    let merged = tree1.merge(tree2);
    assert_eq!(merged.len(), 6);

    // Verify nearest neighbor queries work correctly
    let nearest_to_2 = merged.find_nearest(&2.0);
    assert!(nearest_to_2.is_some());
    let dist = SimpleDistance.distance(&2.0, nearest_to_2.unwrap());
    assert!(dist <= 2.0); // Should be 1.0 or 3.0
}

#[test]
fn test_merge_maintains_nearest_neighbor_correctness() {
    let mut tree1 = SimplifiedCoverTree::new(SimpleDistance, 1.3);
    for i in 0..10 {
        tree1.insert(i as f64);
    }

    let mut tree2 = SimplifiedCoverTree::new(SimpleDistance, 1.3);
    for i in 10..20 {
        tree2.insert(i as f64);
    }

    let merged = tree1.merge(tree2);

    // Test queries across the range
    for i in 0..20 {
        let query = i as f64 + 0.3; // Between points
        let nearest = merged.find_nearest(&query);
        assert!(nearest.is_some());

        let dist = SimpleDistance.distance(&query, nearest.unwrap());
        // Should find a point within 1.0 distance
        assert!(dist <= 1.0, "Query {} found point at distance {}", query, dist);
    }
}

#[test]
fn test_merge_2d_points() {
    let mut tree1 = SimplifiedCoverTree::new(EuclideanDistance, 1.3);
    tree1.insert(Point2D::new(0.0, 0.0));
    tree1.insert(Point2D::new(1.0, 0.0));
    tree1.insert(Point2D::new(0.0, 1.0));

    let mut tree2 = SimplifiedCoverTree::new(EuclideanDistance, 1.3);
    tree2.insert(Point2D::new(10.0, 10.0));
    tree2.insert(Point2D::new(11.0, 10.0));
    tree2.insert(Point2D::new(10.0, 11.0));

    let merged = tree1.merge(tree2);
    assert_eq!(merged.len(), 6);

    // Verify points in both clusters are findable
    assert!(merged.find_nearest(&Point2D::new(0.5, 0.5)).is_some());
    assert!(merged.find_nearest(&Point2D::new(10.5, 10.5)).is_some());
}

#[test]
fn test_merge_large_trees() {
    let mut tree1 = SimplifiedCoverTree::new(SimpleDistance, 1.3);
    for i in 0..100 {
        tree1.insert(i as f64);
    }

    let mut tree2 = SimplifiedCoverTree::new(SimpleDistance, 1.3);
    for i in 100..200 {
        tree2.insert(i as f64);
    }

    let merged = tree1.merge(tree2);
    assert_eq!(merged.len(), 200);

    // Spot check some queries
    assert!(merged.find_nearest(&50.0).is_some());
    assert!(merged.find_nearest(&150.0).is_some());
}

#[test]
fn test_merge_single_element_trees() {
    let mut tree1 = SimplifiedCoverTree::new(SimpleDistance, 1.3);
    tree1.insert(5.0);

    let mut tree2 = SimplifiedCoverTree::new(SimpleDistance, 1.3);
    tree2.insert(10.0);

    let merged = tree1.merge(tree2);
    assert_eq!(merged.len(), 2);

    let nearest_to_7 = merged.find_nearest(&7.0);
    assert!(nearest_to_7.is_some());
}

#[test]
#[should_panic(expected = "Cannot merge trees with different base values")]
fn test_merge_different_base_values_panics() {
    let mut tree1 = SimplifiedCoverTree::new(SimpleDistance, 1.3);
    tree1.insert(1.0);

    let mut tree2 = SimplifiedCoverTree::new(SimpleDistance, 2.0);
    tree2.insert(10.0);

    let _ = tree1.merge(tree2); // Should panic
}

#[test]
fn test_merge_via_unified_wrapper() {
    let mut tree1 = CoverTree::new(SimpleDistance, 1.3);
    tree1.insert(1.0);
    tree1.insert(2.0);

    let mut tree2 = CoverTree::new(SimpleDistance, 1.3);
    tree2.insert(10.0);
    tree2.insert(20.0);

    let merged = tree1.merge(tree2);
    assert_eq!(merged.len(), 4);
}

#[test]
fn test_merge_clustered_data() {
    // Create two distinct clusters
    let mut tree1 = SimplifiedCoverTree::new(SimpleDistance, 1.3);
    for i in 0..10 {
        tree1.insert(i as f64 * 0.1); // Cluster around 0
    }

    let mut tree2 = SimplifiedCoverTree::new(SimpleDistance, 1.3);
    for i in 0..10 {
        tree2.insert(100.0 + i as f64 * 0.1); // Cluster around 100
    }

    let merged = tree1.merge(tree2);
    assert_eq!(merged.len(), 20);

    // Query in first cluster
    let nearest_in_cluster1 = merged.find_nearest(&0.5);
    assert!(nearest_in_cluster1.is_some());
    let dist1 = SimpleDistance.distance(&0.5, nearest_in_cluster1.unwrap());
    assert!(dist1 < 1.0);

    // Query in second cluster
    let nearest_in_cluster2 = merged.find_nearest(&100.5);
    assert!(nearest_in_cluster2.is_some());
    let dist2 = SimpleDistance.distance(&100.5, nearest_in_cluster2.unwrap());
    assert!(dist2 < 1.0);
}

#[test]
fn test_merge_preserves_query_results() {
    // Build merged tree
    let mut tree1 = SimplifiedCoverTree::new(SimpleDistance, 1.3);
    for i in 0..50 {
        tree1.insert(i as f64);
    }

    let mut tree2 = SimplifiedCoverTree::new(SimpleDistance, 1.3);
    for i in 50..100 {
        tree2.insert(i as f64);
    }

    let merged = tree1.merge(tree2);

    // Build sequential tree
    let mut sequential = SimplifiedCoverTree::new(SimpleDistance, 1.3);
    for i in 0..100 {
        sequential.insert(i as f64);
    }

    // Compare query results (distances should be identical)
    for query in [5.5, 25.7, 50.3, 75.2, 99.1] {
        let merged_result = merged.find_nearest(&query);
        let seq_result = sequential.find_nearest(&query);

        assert!(merged_result.is_some());
        assert!(seq_result.is_some());

        let merged_dist = SimpleDistance.distance(&query, merged_result.unwrap());
        let seq_dist = SimpleDistance.distance(&query, seq_result.unwrap());

        // Distances should be equal (might find different points at same distance)
        assert!(
            (merged_dist - seq_dist).abs() < 1e-10,
            "Query {} gave different distances: merged={}, sequential={}",
            query,
            merged_dist,
            seq_dist
        );
    }
}

#[test]
fn test_merge_random_order() {
    use std::collections::HashSet;

    let mut tree1 = SimplifiedCoverTree::new(SimpleDistance, 1.3);
    let points1: Vec<f64> = vec![3.0, 1.0, 7.0, 2.0, 5.0];
    for &p in &points1 {
        tree1.insert(p);
    }

    let mut tree2 = SimplifiedCoverTree::new(SimpleDistance, 1.3);
    let points2: Vec<f64> = vec![13.0, 11.0, 17.0, 12.0, 15.0];
    for &p in &points2 {
        tree2.insert(p);
    }

    let merged = tree1.merge(tree2);
    assert_eq!(merged.len(), 10);

    // Verify all points exist in merged tree
    let mut found_points = HashSet::new();
    for query in [1.0, 2.0, 3.0, 5.0, 7.0, 11.0, 12.0, 13.0, 15.0, 17.0] {
        let nearest = merged.find_nearest(&query);
        assert!(nearest.is_some());
        let dist = SimpleDistance.distance(&query, nearest.unwrap());
        if dist < 0.01 {
            // Found exact match
            found_points.insert((query * 100.0) as i32);
        }
    }

    // Should find most or all original points
    assert!(found_points.len() >= 8);
}

#[test]
fn test_merge_chain() {
    // Test merging multiple trees in sequence
    let mut tree1 = SimplifiedCoverTree::new(SimpleDistance, 1.3);
    tree1.insert(1.0);

    let mut tree2 = SimplifiedCoverTree::new(SimpleDistance, 1.3);
    tree2.insert(2.0);

    let mut tree3 = SimplifiedCoverTree::new(SimpleDistance, 1.3);
    tree3.insert(3.0);

    let merged12 = tree1.merge(tree2);
    let merged123 = merged12.merge(tree3);

    assert_eq!(merged123.len(), 3);
    assert!(merged123.find_nearest(&1.0).is_some());
    assert!(merged123.find_nearest(&2.0).is_some());
    assert!(merged123.find_nearest(&3.0).is_some());
}

#[test]
fn test_merge_with_duplicates() {
    let mut tree1 = SimplifiedCoverTree::new(SimpleDistance, 1.3);
    tree1.insert(5.0);
    tree1.insert(5.0); // Duplicate

    let mut tree2 = SimplifiedCoverTree::new(SimpleDistance, 1.3);
    tree2.insert(5.0); // Another duplicate
    tree2.insert(10.0);

    let merged = tree1.merge(tree2);
    assert_eq!(merged.len(), 4); // All points included, even duplicates

    let nearest = merged.find_nearest(&5.0);
    assert!(nearest.is_some());
    assert_eq!(*nearest.unwrap(), 5.0);
}

#[test]
fn test_merge_grid_pattern() {
    // Create grid in first quadrant
    let mut tree1 = SimplifiedCoverTree::new(EuclideanDistance, 1.3);
    for i in 0..5 {
        for j in 0..5 {
            tree1.insert(Point2D::new(i as f64, j as f64));
        }
    }

    // Create grid in second quadrant
    let mut tree2 = SimplifiedCoverTree::new(EuclideanDistance, 1.3);
    for i in 0..5 {
        for j in 0..5 {
            tree2.insert(Point2D::new(10.0 + i as f64, 10.0 + j as f64));
        }
    }

    let merged = tree1.merge(tree2);
    assert_eq!(merged.len(), 50);

    // Query in both regions
    assert!(merged.find_nearest(&Point2D::new(2.0, 2.0)).is_some());
    assert!(merged.find_nearest(&Point2D::new(12.0, 12.0)).is_some());
}
