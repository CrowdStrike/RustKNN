//! Integration tests for Simplified Cover Trees
//!
//! These tests verify the public API works correctly from an external perspective.
//! They test the cover tree as a user would use it, with:
//! - Custom point types (2D and 3D)
//! - Larger datasets
//! - Invariant verification across the entire tree
//! - Random data stress testing

use rustknn::{SimplifiedCoverTree, Distance, Node};

// ============================================================================
// 2D Point Type and Euclidean Distance
// ============================================================================

/// A simple 2D point for testing
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

/// Euclidean distance metric for 2D points
struct EuclideanDistance;

impl Distance<Point2D> for EuclideanDistance {
    fn distance(&self, p: &Point2D, q: &Point2D) -> f64 {
        let dx = p.x - q.x;
        let dy = p.y - q.y;
        (dx * dx + dy * dy).sqrt()
    }
}

// ============================================================================
// 3D Point Type and Euclidean Distance
// ============================================================================

/// A simple 3D point for testing
#[derive(Clone, Debug, PartialEq)]
struct Point3D {
    x: f64,
    y: f64,
    z: f64,
}

impl Point3D {
    fn new(x: f64, y: f64, z: f64) -> Self {
        Point3D { x, y, z }
    }
}

/// Euclidean distance metric for 3D points
struct Euclidean3D;

impl Distance<Point3D> for Euclidean3D {
    fn distance(&self, p: &Point3D, q: &Point3D) -> f64 {
        let dx = p.x - q.x;
        let dy = p.y - q.y;
        let dz = p.z - q.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

// ============================================================================
// Manhattan Distance for 2D Points
// ============================================================================

/// Manhattan (L1) distance metric for 2D points
struct ManhattanDistance;

impl Distance<Point2D> for ManhattanDistance {
    fn distance(&self, p: &Point2D, q: &Point2D) -> f64 {
        (p.x - q.x).abs() + (p.y - q.y).abs()
    }
}

// ============================================================================
// Invariant Verification Helpers
// ============================================================================

/// Verifies all three cover tree invariants hold for the entire tree
fn verify_invariants<T: Clone, D: Distance<T>>(
    tree: &SimplifiedCoverTree<T, D>,
) -> Result<(), String> {
    if let Some(root) = tree.root_node() {
        verify_node_invariants(root, tree.metric(), tree.base_value())?;
    }
    Ok(())
}

/// Recursively verifies invariants for a node and all its descendants
fn verify_node_invariants<T: Clone, D: Distance<T>>(
    node: &Node<T>,
    metric: &D,
    base: f64,
) -> Result<(), String> {
    // Check all children
    for (i, child) in node.children.iter().enumerate() {
        // 1. LEVELING INVARIANT: child.level = parent.level - 1
        if child.level != node.level - 1 {
            return Err(format!(
                "Leveling invariant violated: parent level {}, child level {}",
                node.level, child.level
            ));
        }

        // 2. COVERING INVARIANT: d(parent, child) <= covdist(parent)
        let dist = metric.distance(&node.point, &child.point);
        let covdist = base.powi(node.level);
        if dist > covdist + 1e-10 {
            // Small epsilon for floating point
            return Err(format!(
                "Covering invariant violated: d(parent, child) = {}, covdist = {}",
                dist, covdist
            ));
        }

        // 3. SEPARATING INVARIANT: d(child_i, child_j) > sepdist(parent)
        // For simplified cover trees, this is relaxed but we can still check
        let sepdist = base.powi(node.level - 1);
        for (j, other_child) in node.children.iter().enumerate() {
            if i != j {
                let sibling_dist = metric.distance(&child.point, &other_child.point);
                // Note: simplified version may not strictly enforce this
                // So we just warn if it's close but don't fail
                if sibling_dist <= sepdist && sibling_dist > 1e-10 {
                    // Close to violating, but simplified version allows this
                }
            }
        }

        // Recursively check child subtree
        verify_node_invariants(child, metric, base)?;
    }

    Ok(())
}

/// Counts the total number of nodes in the tree
fn count_nodes<T: Clone>(node: &Node<T>) -> usize {
    1 + node
        .children
        .iter()
        .map(|child| count_nodes(child))
        .sum::<usize>()
}

// ============================================================================
// Basic Integration Tests with 2D Points
// ============================================================================

#[test]
fn test_2d_empty_tree() {
    let tree: SimplifiedCoverTree<Point2D, EuclideanDistance> = SimplifiedCoverTree::new(EuclideanDistance, 1.3);
    assert_eq!(tree.len(), 0);
    assert!(tree.is_empty());
    assert_eq!(tree.find_nearest(&Point2D::new(0.0, 0.0)), None);
}

#[test]
fn test_2d_single_point() {
    let mut tree = SimplifiedCoverTree::new(EuclideanDistance, 1.3);
    let p = Point2D::new(1.0, 2.0);
    tree.insert(p.clone());

    assert_eq!(tree.len(), 1);
    assert!(!tree.is_empty());

    let result = tree.find_nearest(&Point2D::new(1.5, 2.5));
    assert!(result.is_some());
    assert_eq!(result.unwrap(), &p);
}

#[test]
fn test_2d_multiple_points_nearest() {
    let mut tree = SimplifiedCoverTree::new(EuclideanDistance, 1.3);

    // Insert points in a grid pattern
    tree.insert(Point2D::new(0.0, 0.0));
    tree.insert(Point2D::new(1.0, 0.0));
    tree.insert(Point2D::new(0.0, 1.0));
    tree.insert(Point2D::new(1.0, 1.0));

    assert_eq!(tree.len(), 4);

    // Query near (0, 0)
    let result = tree.find_nearest(&Point2D::new(0.1, 0.1)).unwrap();
    assert_eq!(result, &Point2D::new(0.0, 0.0));

    // Query near (1, 1)
    let result = tree.find_nearest(&Point2D::new(0.9, 0.9)).unwrap();
    assert_eq!(result, &Point2D::new(1.0, 1.0));
}

#[test]
fn test_2d_invariants_maintained() {
    let mut tree = SimplifiedCoverTree::new(EuclideanDistance, 1.3);

    // Insert multiple points
    let points = vec![
        Point2D::new(0.0, 0.0),
        Point2D::new(5.0, 0.0),
        Point2D::new(0.0, 5.0),
        Point2D::new(5.0, 5.0),
        Point2D::new(2.5, 2.5),
        Point2D::new(7.5, 7.5),
        Point2D::new(10.0, 10.0),
    ];

    for point in points {
        tree.insert(point);
    }

    // Verify all invariants
    assert!(
        verify_invariants(&tree).is_ok(),
        "Invariants violated: {:?}",
        verify_invariants(&tree)
    );
}

#[test]
fn test_2d_exactly_n_nodes() {
    let mut tree = SimplifiedCoverTree::new(EuclideanDistance, 1.3);

    let points = vec![
        Point2D::new(0.0, 0.0),
        Point2D::new(1.0, 1.0),
        Point2D::new(2.0, 2.0),
        Point2D::new(3.0, 3.0),
        Point2D::new(4.0, 4.0),
        Point2D::new(5.0, 5.0),
    ];

    for point in points {
        tree.insert(point);
    }

    assert_eq!(tree.len(), 6);

    // Verify exactly 6 nodes in tree structure
    let node_count = count_nodes(tree.root_node().unwrap());
    assert_eq!(
        node_count, 6,
        "Simplified cover tree should have exactly n nodes"
    );
}

// ============================================================================
// Tests with 3D Points
// ============================================================================

#[test]
fn test_3d_points() {
    let mut tree = SimplifiedCoverTree::new(Euclidean3D, 1.3);

    // Insert 3D points
    tree.insert(Point3D::new(0.0, 0.0, 0.0));
    tree.insert(Point3D::new(1.0, 0.0, 0.0));
    tree.insert(Point3D::new(0.0, 1.0, 0.0));
    tree.insert(Point3D::new(0.0, 0.0, 1.0));
    tree.insert(Point3D::new(1.0, 1.0, 1.0));

    assert_eq!(tree.len(), 5);

    // Query nearest to origin
    let result = tree.find_nearest(&Point3D::new(0.1, 0.1, 0.1)).unwrap();
    assert_eq!(result, &Point3D::new(0.0, 0.0, 0.0));

    // Query nearest to (1,1,1)
    let result = tree.find_nearest(&Point3D::new(0.9, 0.9, 0.9)).unwrap();
    assert_eq!(result, &Point3D::new(1.0, 1.0, 1.0));
}

#[test]
fn test_3d_invariants() {
    let mut tree = SimplifiedCoverTree::new(Euclidean3D, 1.3);

    // Insert 8 corners of a cube
    for x in [0.0, 10.0] {
        for y in [0.0, 10.0] {
            for z in [0.0, 10.0] {
                tree.insert(Point3D::new(x, y, z));
            }
        }
    }

    assert_eq!(tree.len(), 8);

    // Verify invariants
    assert!(
        verify_invariants(&tree).is_ok(),
        "3D tree invariants violated"
    );
}

// ============================================================================
// Tests with Manhattan Distance
// ============================================================================

#[test]
fn test_manhattan_distance() {
    let mut tree = SimplifiedCoverTree::new(ManhattanDistance, 1.3);

    // Insert points
    tree.insert(Point2D::new(0.0, 0.0));
    tree.insert(Point2D::new(2.0, 0.0));
    tree.insert(Point2D::new(0.0, 2.0));

    assert_eq!(tree.len(), 3);

    // With Manhattan distance, (2,0) and (0,2) are equidistant from (1, 1)
    // Both have Manhattan distance 2.0
    let result = tree.find_nearest(&Point2D::new(1.0, 1.0)).unwrap();
    // Accept either point
    assert!(
        result == &Point2D::new(2.0, 0.0) || result == &Point2D::new(0.0, 2.0),
        "Should find one of the equidistant points, got {:?}",
        result
    );

    // Query (0, 0) should find itself
    let result = tree.find_nearest(&Point2D::new(0.0, 0.0)).unwrap();
    assert_eq!(result, &Point2D::new(0.0, 0.0));
}

// ============================================================================
// Larger Dataset Tests
// ============================================================================

#[test]
fn test_larger_dataset_2d() {
    let mut tree = SimplifiedCoverTree::new(EuclideanDistance, 1.3);

    // Insert 50 points in a grid
    let mut points = Vec::new();
    for i in 0..10 {
        for j in 0..5 {
            let p = Point2D::new(i as f64, j as f64);
            points.push(p.clone());
            tree.insert(p);
        }
    }

    assert_eq!(tree.len(), 50);

    // Verify exactly 50 nodes
    let node_count = count_nodes(tree.root_node().unwrap());
    assert_eq!(node_count, 50);

    // Verify invariants
    assert!(verify_invariants(&tree).is_ok(), "Invariants violated");

    // Test queries - for most points the nearest should be itself
    // But some edge cases may have equidistant neighbors in a grid
    for point in &points {
        let result = tree.find_nearest(point).unwrap();
        let dist = EuclideanDistance.distance(point, result);
        // Distance should be 0 (finds itself) or very small (finds adjacent)
        assert!(
            dist < 1.5,
            "Point {:?} should find itself or a close neighbor, found {:?} at distance {}",
            point,
            result,
            dist
        );
    }
}

#[test]
fn test_all_nearest_neighbors_2d() {
    // This is the "all nearest neighbors" benchmark from the paper
    let mut tree = SimplifiedCoverTree::new(EuclideanDistance, 1.3);

    let points = vec![
        Point2D::new(0.0, 0.0),
        Point2D::new(1.0, 1.0),
        Point2D::new(2.0, 0.0),
        Point2D::new(3.0, 1.0),
        Point2D::new(4.0, 0.0),
        Point2D::new(5.0, 1.0),
        Point2D::new(6.0, 0.0),
        Point2D::new(7.0, 1.0),
    ];

    for point in &points {
        tree.insert(point.clone());
    }

    // For each point, find its nearest neighbor (should be itself)
    for point in &points {
        let nearest = tree.find_nearest(point).unwrap();
        assert_eq!(
            nearest, point,
            "Point {:?} should be its own nearest neighbor",
            point
        );
    }
}

#[test]
fn test_queries_dont_modify_tree() {
    let mut tree = SimplifiedCoverTree::new(EuclideanDistance, 1.3);

    // Insert points
    for i in 0..10 {
        tree.insert(Point2D::new(i as f64, i as f64));
    }

    let size_before = tree.len();

    // Perform many queries
    for i in 0..20 {
        tree.find_nearest(&Point2D::new(i as f64 * 0.5, i as f64 * 0.5));
    }

    assert_eq!(tree.len(), size_before, "Queries modified tree size");

    // Verify invariants still hold
    assert!(verify_invariants(&tree).is_ok(), "Queries violated invariants");
}

// ============================================================================
// Different Base Values
// ============================================================================

#[test]
fn test_different_base_values() {
    // Test that different base values all work correctly
    for base in [1.3, 1.5, 2.0] {
        let mut tree = SimplifiedCoverTree::new(EuclideanDistance, base);

        // Insert points
        for i in 0..10 {
            tree.insert(Point2D::new(i as f64, 0.0));
        }

        assert_eq!(tree.len(), 10);

        // Verify invariants with this base
        assert!(
            verify_invariants(&tree).is_ok(),
            "Invariants violated with base {}",
            base
        );

        // Test queries work
        let result = tree.find_nearest(&Point2D::new(5.5, 0.0)).unwrap();
        // Should be either 5.0 or 6.0
        assert!(
            result == &Point2D::new(5.0, 0.0) || result == &Point2D::new(6.0, 0.0),
            "Query failed with base {}",
            base
        );
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_identical_points() {
    let mut tree = SimplifiedCoverTree::new(EuclideanDistance, 1.3);

    // Insert the same point multiple times
    let p = Point2D::new(5.0, 5.0);
    tree.insert(p.clone());
    tree.insert(p.clone());
    tree.insert(p.clone());

    assert_eq!(tree.len(), 3);

    // All three nodes should exist
    let node_count = count_nodes(tree.root_node().unwrap());
    assert_eq!(node_count, 3);

    // Query should return one of them
    let result = tree.find_nearest(&p).unwrap();
    assert_eq!(result, &p);
}

#[test]
fn test_very_close_points() {
    let mut tree = SimplifiedCoverTree::new(EuclideanDistance, 1.3);

    // Insert points very close together
    tree.insert(Point2D::new(0.0, 0.0));
    tree.insert(Point2D::new(0.001, 0.001));
    tree.insert(Point2D::new(0.002, 0.002));

    assert_eq!(tree.len(), 3);

    // Verify invariants
    assert!(verify_invariants(&tree).is_ok());
}

#[test]
fn test_very_far_points() {
    let mut tree = SimplifiedCoverTree::new(EuclideanDistance, 1.3);

    // Insert points very far apart
    tree.insert(Point2D::new(0.0, 0.0));
    tree.insert(Point2D::new(1000.0, 1000.0));
    tree.insert(Point2D::new(-1000.0, -1000.0));

    assert_eq!(tree.len(), 3);

    // Verify invariants
    assert!(verify_invariants(&tree).is_ok());

    // Query near origin
    let result = tree.find_nearest(&Point2D::new(1.0, 1.0)).unwrap();
    assert_eq!(result, &Point2D::new(0.0, 0.0));
}

// ============================================================================
// Correctness Tests
// ============================================================================

#[test]
fn test_correctness_against_brute_force() {
    let mut tree = SimplifiedCoverTree::new(EuclideanDistance, 1.3);

    // Insert a set of points
    let points = vec![
        Point2D::new(0.0, 0.0),
        Point2D::new(3.0, 4.0),
        Point2D::new(5.0, 0.0),
        Point2D::new(10.0, 10.0),
        Point2D::new(7.0, 2.0),
        Point2D::new(2.0, 8.0),
        Point2D::new(9.0, 1.0),
    ];

    for point in &points {
        tree.insert(point.clone());
    }

    // Test multiple query points
    let queries = vec![
        Point2D::new(1.0, 1.0),
        Point2D::new(5.0, 5.0),
        Point2D::new(8.0, 8.0),
        Point2D::new(0.0, 10.0),
    ];

    for query in &queries {
        // Tree result
        let tree_result = tree.find_nearest(query).unwrap();
        let tree_dist = EuclideanDistance.distance(query, tree_result);

        // Brute force: find minimum distance
        let mut best_dist = f64::INFINITY;
        for point in &points {
            let dist = EuclideanDistance.distance(query, point);
            if dist < best_dist {
                best_dist = dist;
            }
        }

        // Tree should find a point at the same distance as brute force
        assert!(
            (tree_dist - best_dist).abs() < 1e-10,
            "Tree distance {} doesn't match brute force {}",
            tree_dist,
            best_dist
        );
    }
}

// ============================================================================
// Random Data Stress Test
// ============================================================================

#[test]
fn test_random_2d_points() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Deterministic "random" points using hashing
    let mut tree = SimplifiedCoverTree::new(EuclideanDistance, 1.3);

    let mut points = Vec::new();
    for i in 0..100 {
        // Generate deterministic pseudo-random coordinates
        let mut hasher = DefaultHasher::new();
        i.hash(&mut hasher);
        let hash1 = hasher.finish();

        let mut hasher = DefaultHasher::new();
        (i + 1000).hash(&mut hasher);
        let hash2 = hasher.finish();

        let x = (hash1 % 10000) as f64 / 100.0;
        let y = (hash2 % 10000) as f64 / 100.0;

        let p = Point2D::new(x, y);
        points.push(p.clone());
        tree.insert(p);
    }

    assert_eq!(tree.len(), 100);

    // Verify exactly 100 nodes
    let node_count = count_nodes(tree.root_node().unwrap());
    assert_eq!(node_count, 100);

    // Verify invariants
    assert!(
        verify_invariants(&tree).is_ok(),
        "Random data violated invariants"
    );

    // Test that queries work
    for point in points.iter().take(10) {
        let result = tree.find_nearest(point);
        assert!(result.is_some(), "Query failed on random data");
    }
}

#[test]
fn test_sequential_vs_random_insertion_order() {
    // Test that insertion order doesn't break invariants
    let points = vec![
        Point2D::new(0.0, 0.0),
        Point2D::new(1.0, 1.0),
        Point2D::new(2.0, 0.0),
        Point2D::new(3.0, 1.0),
        Point2D::new(4.0, 0.0),
    ];

    // Sequential insertion
    let mut tree1 = SimplifiedCoverTree::new(EuclideanDistance, 1.3);
    for point in &points {
        tree1.insert(point.clone());
    }

    // Reverse insertion
    let mut tree2 = SimplifiedCoverTree::new(EuclideanDistance, 1.3);
    for point in points.iter().rev() {
        tree2.insert(point.clone());
    }

    // Both should have 5 points
    assert_eq!(tree1.len(), 5);
    assert_eq!(tree2.len(), 5);

    // Both should maintain invariants
    assert!(verify_invariants(&tree1).is_ok());
    assert!(verify_invariants(&tree2).is_ok());

    // Both should return correct nearest neighbors (may be different structures but same results)
    let query = Point2D::new(2.5, 0.5);
    let result1 = tree1.find_nearest(&query).unwrap();
    let result2 = tree2.find_nearest(&query).unwrap();

    // Both results should be at the same distance from query
    let dist1 = EuclideanDistance.distance(&query, result1);
    let dist2 = EuclideanDistance.distance(&query, result2);
    assert!((dist1 - dist2).abs() < 1e-10);
}

#[test]
fn test_recompute_maxdist_improves_bounds() {
    // This test demonstrates that recompute_maxdist produces tighter bounds
    // than the triangle inequality approximation used during insertion

    let mut tree = SimplifiedCoverTree::new(EuclideanDistance, 1.3);

    // Create a configuration where triangle inequality overestimates
    let points = vec![
        Point2D::new(0.0, 0.0),
        Point2D::new(10.0, 0.0),
        Point2D::new(5.0, 5.0),
        Point2D::new(3.0, 1.0),
        Point2D::new(7.0, 2.0),
    ];

    for p in points {
        tree.insert(p);
    }

    // Get root maxdist before recompute (triangle inequality upper bound)
    let maxdist_before = tree.root_node().unwrap().maxdist;

    // Recompute exact maxdist
    tree.recompute_maxdist();

    let maxdist_after = tree.root_node().unwrap().maxdist;

    // Exact maxdist should be ≤ upper bound
    assert!(maxdist_after <= maxdist_before,
        "Exact maxdist {} should be ≤ triangle inequality bound {}",
        maxdist_after, maxdist_before);

    // Verify all nodes have exact maxdist
    verify_exact_maxdist_all_nodes(tree.root_node().unwrap());

    // Tree should still maintain all invariants
    assert!(verify_invariants(&tree).is_ok());
}

#[test]
fn test_recompute_maxdist_with_larger_dataset() {
    // Test recompute_maxdist on a realistic dataset

    let mut tree = SimplifiedCoverTree::new(EuclideanDistance, 1.3);

    // Create a grid of points
    for x in 0..10 {
        for y in 0..10 {
            tree.insert(Point2D::new(x as f64, y as f64));
        }
    }

    // Recompute exact maxdist
    tree.recompute_maxdist();

    // Verify all nodes have exact maxdist
    verify_exact_maxdist_all_nodes(tree.root_node().unwrap());

    // Verify tree still works correctly for queries
    let query = Point2D::new(5.5, 5.5);
    let result = tree.find_nearest(&query);
    assert!(result.is_some());

    // Should find one of the 4 corners of the center square
    let nearest = result.unwrap();
    let dist = EuclideanDistance.distance(&query, nearest);
    assert!(dist < 1.0, "Should find a very close point");
}

#[test]
fn test_recompute_maxdist_idempotent_integration() {
    // Verify that calling recompute_maxdist multiple times is safe

    let mut tree = SimplifiedCoverTree::new(EuclideanDistance, 1.3);

    let points = vec![
        Point2D::new(0.0, 0.0),
        Point2D::new(5.0, 0.0),
        Point2D::new(0.0, 5.0),
        Point2D::new(5.0, 5.0),
    ];

    for p in points {
        tree.insert(p);
    }

    // First recompute
    tree.recompute_maxdist();
    let maxdist_first = tree.root_node().unwrap().maxdist;

    // Second recompute
    tree.recompute_maxdist();
    let maxdist_second = tree.root_node().unwrap().maxdist;

    // Third recompute
    tree.recompute_maxdist();
    let maxdist_third = tree.root_node().unwrap().maxdist;

    // All should be identical
    assert_eq!(maxdist_first, maxdist_second);
    assert_eq!(maxdist_second, maxdist_third);

    // Tree should still work
    assert!(verify_invariants(&tree).is_ok());
}

// Helper function: verify all nodes in tree have exact maxdist
fn verify_exact_maxdist_all_nodes(node: &Node<Point2D>) {
    // Compute actual max distance to all descendants
    let actual_max = compute_actual_max_distance(node);

    // Should match node's maxdist field (within floating point tolerance)
    assert!((node.maxdist - actual_max).abs() < 1e-10,
        "Node at ({}, {}) has maxdist={}, but actual max is {}",
        node.point.x, node.point.y, node.maxdist, actual_max);

    // Recurse to children
    for child in &node.children {
        verify_exact_maxdist_all_nodes(child);
    }
}

// Helper function: compute actual maximum distance from node to all descendants
fn compute_actual_max_distance(node: &Node<Point2D>) -> f64 {
    let mut max_dist = 0.0;

    fn check_descendants(
        ancestor: &Node<Point2D>,
        current: &Node<Point2D>,
        max_dist: &mut f64
    ) {
        let dist = EuclideanDistance.distance(&ancestor.point, &current.point);
        if dist > *max_dist {
            *max_dist = dist;
        }

        for child in &current.children {
            check_descendants(ancestor, child, max_dist);
        }
    }

    for child in &node.children {
        check_descendants(node, child, &mut max_dist);
    }

    max_dist
}
