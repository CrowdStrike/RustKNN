//! Comprehensive validation tests comparing packed, unpacked, and brute force implementations
//!
//! These tests verify that both packed and unpacked cover trees produce identical results
//! to a ground-truth brute force implementation for datasets up to 100 points.

use rustknn::simplified::SimplifiedCoverTree;
use rustknn::distance::Distance;
use std::collections::HashSet;

// ============================================================================
// Test Point Types and Distance Metrics
// ============================================================================

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

#[derive(Clone, Debug)]
struct PointND {
    coords: Vec<f64>,
}

impl PartialEq for PointND {
    fn eq(&self, other: &Self) -> bool {
        self.coords.len() == other.coords.len()
            && self.coords.iter().zip(&other.coords).all(|(a, b)| (a - b).abs() < 1e-10)
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

struct EuclideanDistanceND;

impl Distance<PointND> for EuclideanDistanceND {
    fn distance(&self, p: &PointND, q: &PointND) -> f64 {
        p.coords
            .iter()
            .zip(&q.coords)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

// ============================================================================
// Ground Truth: Brute Force Implementation
// ============================================================================

fn brute_force_nearest<T: Clone, D: Distance<T>>(
    points: &[T],
    query: &T,
    metric: &D,
) -> Option<(T, f64)> {
    points
        .iter()
        .map(|p| (p.clone(), metric.distance(p, query)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
}

fn brute_force_knn<T: Clone, D: Distance<T>>(
    points: &[T],
    query: &T,
    metric: &D,
    k: usize,
) -> Vec<(T, f64)> {
    let mut distances: Vec<(T, f64)> = points
        .iter()
        .map(|p| (p.clone(), metric.distance(p, query)))
        .collect();

    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    distances.truncate(k);
    distances
}

// ============================================================================
// Nearest Neighbor Tests (Packed vs Unpacked vs Brute Force)
// ============================================================================

#[test]
fn test_nn_all_methods_2d_grid_10() {
    let metric = EuclideanDistance2D;

    // Build dataset: 10x10 grid
    let mut points = Vec::new();
    for i in 0..10 {
        for j in 0..10 {
            points.push(Point2D::new(i as f64, j as f64));
        }
    }

    // Build trees
    let mut tree_unpacked = SimplifiedCoverTree::new(EuclideanDistance2D, 1.3);
    let mut tree_for_packing = SimplifiedCoverTree::new(EuclideanDistance2D, 1.3);

    for point in &points {
        tree_unpacked.insert(point.clone());
        tree_for_packing.insert(point.clone());
    }

    let tree_packed = tree_for_packing.pack();

    // Test 20 queries
    for qx in 0..4 {
        for qy in 0..5 {
            let query = Point2D::new(qx as f64 + 0.5, qy as f64 + 0.5);

            let brute_result = brute_force_nearest(&points, &query, &metric);
            let unpacked_result = tree_unpacked.find_nearest(&query);
            let packed_result = tree_packed.find_nearest(&query);

            // All three should return same result
            assert!(brute_result.is_some(), "Brute force should find result");
            assert!(unpacked_result.is_some(), "Unpacked should find result");
            assert!(packed_result.is_some(), "Packed should find result");

            let (_brute_point, brute_dist) = brute_result.unwrap();
            let unpacked = unpacked_result.unwrap();
            let packed = packed_result.unwrap();

            // Verify distances match (handles ties - multiple points at same distance)
            let unpacked_dist = metric.distance(unpacked, &query);
            let packed_dist = metric.distance(packed, &query);

            assert!((unpacked_dist - brute_dist).abs() < 1e-10,
                "Unpacked distance {} != brute {} for query ({}, {})",
                unpacked_dist, brute_dist, qx, qy);
            assert!((packed_dist - brute_dist).abs() < 1e-10,
                "Packed distance {} != brute {} for query ({}, {})",
                packed_dist, brute_dist, qx, qy);
        }
    }
}

#[test]
fn test_nn_all_methods_random_2d_50() {
    let metric = EuclideanDistance2D;

    // Generate 50 random points
    let mut rng = 42u64;
    let mut points = Vec::new();

    for _ in 0..50 {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let x = ((rng >> 16) % 10000) as f64 / 100.0;
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let y = ((rng >> 16) % 10000) as f64 / 100.0;
        points.push(Point2D::new(x, y));
    }

    // Build trees
    let mut tree_unpacked = SimplifiedCoverTree::new(EuclideanDistance2D, 1.3);
    let mut tree_for_packing = SimplifiedCoverTree::new(EuclideanDistance2D, 1.3);

    for point in &points {
        tree_unpacked.insert(point.clone());
        tree_for_packing.insert(point.clone());
    }

    let tree_packed = tree_for_packing.pack();

    // Test 25 random queries
    for _ in 0..25 {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let qx = ((rng >> 16) % 10000) as f64 / 100.0;
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let qy = ((rng >> 16) % 10000) as f64 / 100.0;
        let query = Point2D::new(qx, qy);

        let brute_result = brute_force_nearest(&points, &query, &metric);
        let unpacked_result = tree_unpacked.find_nearest(&query);
        let packed_result = tree_packed.find_nearest(&query);

        assert!(brute_result.is_some() && unpacked_result.is_some() && packed_result.is_some());

        let (_brute_point, brute_dist) = brute_result.unwrap();
        let unpacked = unpacked_result.unwrap();
        let packed = packed_result.unwrap();

        let unpacked_dist = metric.distance(unpacked, &query);
        let packed_dist = metric.distance(packed, &query);

        assert!((unpacked_dist - brute_dist).abs() < 1e-10,
            "Unpacked distance {} != brute {}", unpacked_dist, brute_dist);
        assert!((packed_dist - brute_dist).abs() < 1e-10,
            "Packed distance {} != brute {}", packed_dist, brute_dist);
    }
}

#[test]
fn test_nn_all_methods_random_2d_100() {
    let metric = EuclideanDistance2D;

    // Generate 100 random points
    let mut rng = 12345u64;
    let mut points = Vec::new();

    for _ in 0..100 {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let x = ((rng >> 16) % 10000) as f64 / 100.0;
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let y = ((rng >> 16) % 10000) as f64 / 100.0;
        points.push(Point2D::new(x, y));
    }

    // Build trees
    let mut tree_unpacked = SimplifiedCoverTree::new(EuclideanDistance2D, 1.3);
    let mut tree_for_packing = SimplifiedCoverTree::new(EuclideanDistance2D, 1.3);

    for point in &points {
        tree_unpacked.insert(point.clone());
        tree_for_packing.insert(point.clone());
    }

    let tree_packed = tree_for_packing.pack();

    // Test 50 random queries
    for _ in 0..50 {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let qx = ((rng >> 16) % 10000) as f64 / 100.0;
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let qy = ((rng >> 16) % 10000) as f64 / 100.0;
        let query = Point2D::new(qx, qy);

        let brute_result = brute_force_nearest(&points, &query, &metric);
        let unpacked_result = tree_unpacked.find_nearest(&query);
        let packed_result = tree_packed.find_nearest(&query);

        assert!(brute_result.is_some() && unpacked_result.is_some() && packed_result.is_some());

        let (_brute_point, brute_dist) = brute_result.unwrap();
        let unpacked = unpacked_result.unwrap();
        let packed = packed_result.unwrap();

        let unpacked_dist = metric.distance(unpacked, &query);
        let packed_dist = metric.distance(packed, &query);

        if (unpacked_dist - brute_dist).abs() >= 1e-10 {
            println!("Unpacked mismatch: query ({}, {}), brute dist {}, unpacked dist {}",
                qx, qy, brute_dist, unpacked_dist);
        }
        if (packed_dist - brute_dist).abs() >= 1e-10 {
            println!("Packed mismatch: query ({}, {}), brute dist {}, packed dist {}",
                qx, qy, brute_dist, packed_dist);
        }

        assert!((unpacked_dist - brute_dist).abs() < 1e-10);
        assert!((packed_dist - brute_dist).abs() < 1e-10);
    }
}

#[test]
fn test_nn_all_methods_high_dim_50() {
    let metric = EuclideanDistanceND;
    let dim = 50;

    // Generate 50 points in 50D
    let mut rng = 999u64;
    let mut points = Vec::new();

    for _ in 0..50 {
        let mut coords = Vec::new();
        for _ in 0..dim {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            coords.push(((rng >> 16) % 10000) as f64 / 100.0);
        }
        points.push(PointND { coords });
    }

    // Build trees
    let mut tree_unpacked = SimplifiedCoverTree::new(EuclideanDistanceND, 1.3);
    let mut tree_for_packing = SimplifiedCoverTree::new(EuclideanDistanceND, 1.3);

    for point in &points {
        tree_unpacked.insert(point.clone());
        tree_for_packing.insert(point.clone());
    }

    let tree_packed = tree_for_packing.pack();

    // Test 20 queries
    for _ in 0..20 {
        let mut coords = Vec::new();
        for _ in 0..dim {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            coords.push(((rng >> 16) % 10000) as f64 / 100.0);
        }
        let query = PointND { coords };

        let brute_result = brute_force_nearest(&points, &query, &metric);
        let unpacked_result = tree_unpacked.find_nearest(&query);
        let packed_result = tree_packed.find_nearest(&query);

        assert!(brute_result.is_some() && unpacked_result.is_some() && packed_result.is_some());

        let (_brute_point, brute_dist) = brute_result.unwrap();
        let unpacked = unpacked_result.unwrap();
        let packed = packed_result.unwrap();

        let unpacked_dist = metric.distance(unpacked, &query);
        let packed_dist = metric.distance(packed, &query);

        assert!((unpacked_dist - brute_dist).abs() < 1e-8);
        assert!((packed_dist - brute_dist).abs() < 1e-8);
    }
}

// ============================================================================
// k-NN Tests (Packed vs Unpacked vs Brute Force)
// ============================================================================

#[test]
fn test_knn_all_methods_2d_grid_10() {
    let metric = EuclideanDistance2D;

    // Build dataset: 10x10 grid
    let mut points = Vec::new();
    for i in 0..10 {
        for j in 0..10 {
            points.push(Point2D::new(i as f64, j as f64));
        }
    }

    // Build trees
    let mut tree_unpacked = SimplifiedCoverTree::new(EuclideanDistance2D, 1.3);
    let mut tree_for_packing = SimplifiedCoverTree::new(EuclideanDistance2D, 1.3);

    for point in &points {
        tree_unpacked.insert(point.clone());
        tree_for_packing.insert(point.clone());
    }

    let tree_packed = tree_for_packing.pack();

    // Test various k values
    for k in [1, 5, 10, 20] {
        let query = Point2D::new(5.5, 5.5);

        let brute = brute_force_knn(&points, &query, &metric, k);
        let unpacked = tree_unpacked.find_k_nearest(&query, k);
        let packed = tree_packed.find_k_nearest(&query, k);

        assert_eq!(brute.len(), k);
        assert_eq!(unpacked.len(), k);
        assert_eq!(packed.len(), k);

        // Verify distances match (order might differ for ties)
        for i in 0..k {
            assert!((unpacked[i].1 - brute[i].1).abs() < 1e-10,
                "Unpacked k={} distance mismatch at index {}: {} vs {}",
                k, i, unpacked[i].1, brute[i].1);
            assert!((packed[i].1 - brute[i].1).abs() < 1e-10,
                "Packed k={} distance mismatch at index {}: {} vs {}",
                k, i, packed[i].1, brute[i].1);
        }
    }
}

#[test]
fn test_knn_all_methods_random_2d_50() {
    let metric = EuclideanDistance2D;

    // Generate 50 random points
    let mut rng = 777u64;
    let mut points = Vec::new();

    for _ in 0..50 {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let x = ((rng >> 16) % 10000) as f64 / 100.0;
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let y = ((rng >> 16) % 10000) as f64 / 100.0;
        points.push(Point2D::new(x, y));
    }

    // Build trees
    let mut tree_unpacked = SimplifiedCoverTree::new(EuclideanDistance2D, 1.3);
    let mut tree_for_packing = SimplifiedCoverTree::new(EuclideanDistance2D, 1.3);

    for point in &points {
        tree_unpacked.insert(point.clone());
        tree_for_packing.insert(point.clone());
    }

    let tree_packed = tree_for_packing.pack();

    // Test k=1, 5, 10 with 20 queries
    for k in [1, 5, 10] {
        for _ in 0..20 {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let qx = ((rng >> 16) % 10000) as f64 / 100.0;
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let qy = ((rng >> 16) % 10000) as f64 / 100.0;
            let query = Point2D::new(qx, qy);

            let brute = brute_force_knn(&points, &query, &metric, k);
            let unpacked = tree_unpacked.find_k_nearest(&query, k);
            let packed = tree_packed.find_k_nearest(&query, k);

            assert_eq!(brute.len(), k);
            assert_eq!(unpacked.len(), k);
            assert_eq!(packed.len(), k);

            // For k=1, points should match exactly
            if k == 1 {
                assert_eq!(unpacked[0].0, &brute[0].0);
                assert_eq!(packed[0].0, &brute[0].0);
            }

            // Verify k-th distances match (handles ties)
            let brute_kth = brute.last().unwrap().1;
            let unpacked_kth = unpacked.last().unwrap().1;
            let packed_kth = packed.last().unwrap().1;

            assert!((unpacked_kth - brute_kth).abs() < 1e-10,
                "Unpacked k-th distance mismatch for k={}: {} vs {}", k, unpacked_kth, brute_kth);
            assert!((packed_kth - brute_kth).abs() < 1e-10,
                "Packed k-th distance mismatch for k={}: {} vs {}", k, packed_kth, brute_kth);
        }
    }
}

#[test]
fn test_knn_all_methods_random_2d_100() {
    let metric = EuclideanDistance2D;

    // Generate 100 random points
    let mut rng = 54321u64;
    let mut points = Vec::new();

    for _ in 0..100 {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let x = ((rng >> 16) % 10000) as f64 / 100.0;
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let y = ((rng >> 16) % 10000) as f64 / 100.0;
        points.push(Point2D::new(x, y));
    }

    // Build trees
    let mut tree_unpacked = SimplifiedCoverTree::new(EuclideanDistance2D, 1.3);
    let mut tree_for_packing = SimplifiedCoverTree::new(EuclideanDistance2D, 1.3);

    for point in &points {
        tree_unpacked.insert(point.clone());
        tree_for_packing.insert(point.clone());
    }

    let tree_packed = tree_for_packing.pack();

    // Test k=1, 10, 20 with 30 queries
    for k in [1, 10, 20] {
        for _ in 0..30 {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let qx = ((rng >> 16) % 10000) as f64 / 100.0;
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let qy = ((rng >> 16) % 10000) as f64 / 100.0;
            let query = Point2D::new(qx, qy);

            let brute = brute_force_knn(&points, &query, &metric, k);
            let unpacked = tree_unpacked.find_k_nearest(&query, k);
            let packed = tree_packed.find_k_nearest(&query, k);

            assert_eq!(brute.len(), k);
            assert_eq!(unpacked.len(), k);
            assert_eq!(packed.len(), k);

            // Verify all results are valid
            for i in 0..k {
                // Verify distances are computed correctly
                let unpacked_actual = metric.distance(unpacked[i].0, &query);
                let packed_actual = metric.distance(packed[i].0, &query);

                assert!((unpacked_actual - unpacked[i].1).abs() < 1e-10);
                assert!((packed_actual - packed[i].1).abs() < 1e-10);
            }

            // Verify no duplicates
            let unpacked_set: HashSet<_> = unpacked.iter().map(|p| (p.0.x as i64, p.0.y as i64)).collect();
            let packed_set: HashSet<_> = packed.iter().map(|p| (p.0.x as i64, p.0.y as i64)).collect();

            assert_eq!(unpacked_set.len(), k, "Unpacked has duplicates");
            assert_eq!(packed_set.len(), k, "Packed has duplicates");
        }
    }
}

#[test]
fn test_knn_all_methods_high_dim_100() {
    let metric = EuclideanDistanceND;
    let dim = 20; // Use 20D for reasonable test time

    // Generate 100 points in 20D
    let mut rng = 11111u64;
    let mut points = Vec::new();

    for _ in 0..100 {
        let mut coords = Vec::new();
        for _ in 0..dim {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            coords.push(((rng >> 16) % 10000) as f64 / 100.0);
        }
        points.push(PointND { coords });
    }

    // Build trees
    let mut tree_unpacked = SimplifiedCoverTree::new(EuclideanDistanceND, 1.3);
    let mut tree_for_packing = SimplifiedCoverTree::new(EuclideanDistanceND, 1.3);

    for point in &points {
        tree_unpacked.insert(point.clone());
        tree_for_packing.insert(point.clone());
    }

    let tree_packed = tree_for_packing.pack();

    // Test k=1, 10, 20 with 20 queries
    for k in [1, 10, 20] {
        for _ in 0..20 {
            let mut coords = Vec::new();
            for _ in 0..dim {
                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                coords.push(((rng >> 16) % 10000) as f64 / 100.0);
            }
            let query = PointND { coords };

            let brute = brute_force_knn(&points, &query, &metric, k);
            let unpacked = tree_unpacked.find_k_nearest(&query, k);
            let packed = tree_packed.find_k_nearest(&query, k);

            assert_eq!(brute.len(), k);
            assert_eq!(unpacked.len(), k);
            assert_eq!(packed.len(), k);

            // Verify k-th distances match
            let brute_kth = brute.last().unwrap().1;
            let unpacked_kth = unpacked.last().unwrap().1;
            let packed_kth = packed.last().unwrap().1;

            assert!((unpacked_kth - brute_kth).abs() < 1e-6);
            assert!((packed_kth - brute_kth).abs() < 1e-6);
        }
    }
}
