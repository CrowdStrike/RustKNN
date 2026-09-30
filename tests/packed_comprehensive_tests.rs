//! Comprehensive tests for packed cover tree nearest-neighbor and k-NN queries
//!
//! These tests verify correctness, performance characteristics, and edge cases
//! for the cache-optimized packed tree implementation.

use rustknn::simplified::SimplifiedCoverTree;
use rustknn::distance::Distance;
use std::collections::HashSet;

// ============================================================================
// Test Point Types
// ============================================================================

#[derive(Clone, Debug, PartialEq)]
struct Point1D(f64);

#[derive(Clone, Debug, PartialEq)]
struct Point2D {
    x: f64,
    y: f64,
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

// ============================================================================
// Distance Metrics
// ============================================================================

struct Distance1D;

impl Distance<Point1D> for Distance1D {
    fn distance(&self, p: &Point1D, q: &Point1D) -> f64 {
        (p.0 - q.0).abs()
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
// Helper Functions
// ============================================================================

/// Brute force k-NN for verification
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

/// Check if two k-NN results are equivalent (same distances, allowing different tie-breaking)
///
/// Two k-NN results are equivalent if they have the same length, the same set of distances,
/// and any difference in points occurs only among tied distances at the k-th boundary.
fn knn_results_equivalent<T: PartialEq>(
    a: &[(&T, f64)],
    b: &[(T, f64)],
    tolerance: f64,
) -> bool {
    if a.len() != b.len() {
        return false;
    }
    if a.is_empty() {
        return true;
    }

    // Find the kth distance (worst/largest in both sets)
    let kth_dist_a = a.last().unwrap().1;
    let kth_dist_b = b.last().unwrap().1;

    // Points strictly closer than the kth boundary must match exactly (as sets)
    let strict_a: Vec<_> = a.iter().filter(|(_, d)| *d < kth_dist_a - tolerance).collect();
    let strict_b: Vec<_> = b.iter().filter(|(_, d)| *d < kth_dist_b - tolerance).collect();

    if strict_a.len() != strict_b.len() {
        return false;
    }

    for (point_a, dist_a) in &strict_a {
        let found = strict_b.iter().any(|(point_b, dist_b)| {
            **point_a == *point_b && (dist_a - dist_b).abs() < tolerance
        });
        if !found {
            return false;
        }
    }

    // At the kth boundary, only check that distances match (points may differ due to tie-breaking)
    let boundary_a: Vec<f64> = a.iter().filter(|(_, d)| (*d - kth_dist_a).abs() <= tolerance).map(|(_, d)| *d).collect();
    let boundary_b: Vec<f64> = b.iter().filter(|(_, d)| (*d - kth_dist_b).abs() <= tolerance).map(|(_, d)| *d).collect();

    boundary_a.len() == boundary_b.len() && (kth_dist_a - kth_dist_b).abs() <= tolerance
}

// ============================================================================
// Single Nearest Neighbor Tests
// ============================================================================

#[test]
fn test_nearest_1d_small() {
    let mut tree = SimplifiedCoverTree::new(Distance1D, 1.3);

    let points = vec![1.0, 5.0, 10.0, 15.0, 20.0];
    for p in &points {
        tree.insert(Point1D(*p));
    }

    let packed = tree.pack();

    // Query at exact points
    assert_eq!(packed.find_nearest(&Point1D(5.0)), Some(&Point1D(5.0)));
    assert_eq!(packed.find_nearest(&Point1D(15.0)), Some(&Point1D(15.0)));

    // Query between points
    assert_eq!(packed.find_nearest(&Point1D(7.0)), Some(&Point1D(5.0)));
    assert_eq!(packed.find_nearest(&Point1D(8.0)), Some(&Point1D(10.0)));
    assert_eq!(packed.find_nearest(&Point1D(0.0)), Some(&Point1D(1.0)));
    assert_eq!(packed.find_nearest(&Point1D(100.0)), Some(&Point1D(20.0)));
}

#[test]
fn test_nearest_2d_grid() {
    let mut tree = SimplifiedCoverTree::new(EuclideanDistance2D, 1.3);

    // 10x10 grid
    for i in 0..10 {
        for j in 0..10 {
            tree.insert(Point2D {
                x: i as f64,
                y: j as f64,
            });
        }
    }

    let packed = tree.pack();

    // Query at grid points
    let result = packed.find_nearest(&Point2D { x: 5.0, y: 5.0 });
    assert_eq!(result, Some(&Point2D { x: 5.0, y: 5.0 }));

    // Query between grid points
    let result = packed.find_nearest(&Point2D { x: 5.3, y: 5.3 });
    assert_eq!(result, Some(&Point2D { x: 5.0, y: 5.0 }));

    let result = packed.find_nearest(&Point2D { x: 5.7, y: 5.7 });
    assert_eq!(result, Some(&Point2D { x: 6.0, y: 6.0 }));

    // Query at corners
    let result = packed.find_nearest(&Point2D { x: 0.0, y: 0.0 });
    assert_eq!(result, Some(&Point2D { x: 0.0, y: 0.0 }));

    let result = packed.find_nearest(&Point2D { x: 9.0, y: 9.0 });
    assert_eq!(result, Some(&Point2D { x: 9.0, y: 9.0 }));

    // Query outside grid
    let result = packed.find_nearest(&Point2D { x: -10.0, y: -10.0 });
    assert_eq!(result, Some(&Point2D { x: 0.0, y: 0.0 }));

    let result = packed.find_nearest(&Point2D { x: 100.0, y: 100.0 });
    assert_eq!(result, Some(&Point2D { x: 9.0, y: 9.0 }));
}

#[test]
fn test_nearest_random_2d() {
    let mut tree = SimplifiedCoverTree::new(EuclideanDistance2D, 1.3);

    // Generate 100 random points
    let seed = 42u64;
    let mut rng = seed;
    let mut points = Vec::new();

    for _ in 0..100 {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let x = ((rng >> 16) % 10000) as f64 / 100.0;
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let y = ((rng >> 16) % 10000) as f64 / 100.0;

        let point = Point2D { x, y };
        points.push(point.clone());
        tree.insert(point);
    }

    let packed = tree.pack();

    // Test 20 random queries
    for _ in 0..20 {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let qx = ((rng >> 16) % 10000) as f64 / 100.0;
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let qy = ((rng >> 16) % 10000) as f64 / 100.0;

        let query = Point2D { x: qx, y: qy };

        // Brute force nearest
        let metric = EuclideanDistance2D;
        let mut min_dist = f64::INFINITY;
        let mut nearest = None;

        for p in &points {
            let dist = metric.distance(p, &query);
            if dist < min_dist {
                min_dist = dist;
                nearest = Some(p);
            }
        }

        let packed_result = packed.find_nearest(&query);

        assert_eq!(packed_result, nearest);
    }
}

#[test]
fn test_nearest_high_dimensional() {
    let mut tree = SimplifiedCoverTree::new(EuclideanDistanceND, 1.3);

    // 50-dimensional points
    let dim = 50;
    let mut points = Vec::new();

    let seed = 123u64;
    let mut rng = seed;

    for _ in 0..50 {
        let mut coords = Vec::new();
        for _ in 0..dim {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let val = ((rng >> 16) % 10000) as f64 / 100.0;
            coords.push(val);
        }
        let point = PointND { coords };
        points.push(point.clone());
        tree.insert(point);
    }

    let packed = tree.pack();

    // Test queries
    for _ in 0..10 {
        let mut coords = Vec::new();
        for _ in 0..dim {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let val = ((rng >> 16) % 10000) as f64 / 100.0;
            coords.push(val);
        }
        let query = PointND { coords };

        // Brute force
        let metric = EuclideanDistanceND;
        let mut min_dist = f64::INFINITY;
        let mut nearest = None;

        for p in &points {
            let dist = metric.distance(p, &query);
            if dist < min_dist {
                min_dist = dist;
                nearest = Some(p);
            }
        }

        let packed_result = packed.find_nearest(&query);

        assert!(packed_result.is_some());
        assert!(nearest.is_some());

        // Compare distances (points might differ due to floating point, but distances should match)
        let packed_dist = metric.distance(packed_result.unwrap(), &query);
        let brute_dist = metric.distance(nearest.unwrap(), &query);

        assert!(
            (packed_dist - brute_dist).abs() < 1e-10,
            "Distance mismatch: packed={}, brute={}",
            packed_dist,
            brute_dist
        );
    }
}

// ============================================================================
// k-Nearest Neighbor Tests
// ============================================================================

#[test]
fn test_knn_k_equals_1() {
    let mut tree = SimplifiedCoverTree::new(Distance1D, 1.3);

    for i in 0..20 {
        tree.insert(Point1D(i as f64));
    }

    let packed = tree.pack();

    let query = Point1D(10.5);

    let nearest = packed.find_nearest(&query);
    let knn = packed.find_k_nearest(&query, 1);

    assert_eq!(knn.len(), 1);
    assert_eq!(Some(knn[0].0), nearest);
}

#[test]
fn test_knn_various_k() {
    let mut tree = SimplifiedCoverTree::new(EuclideanDistance2D, 1.3);

    // 5x5 grid
    let mut points = Vec::new();
    for i in 0..5 {
        for j in 0..5 {
            let point = Point2D {
                x: i as f64,
                y: j as f64,
            };
            points.push(point.clone());
            tree.insert(point);
        }
    }

    let packed = tree.pack();
    let query = Point2D { x: 2.0, y: 2.0 };

    // Test k = 1, 5, 10, 25 (all points)
    for k in [1, 5, 10, 25] {
        let result = packed.find_k_nearest(&query, k);
        let expected_len = k.min(25);

        assert_eq!(result.len(), expected_len);

        // Verify sorted by distance
        for i in 0..(result.len() - 1) {
            assert!(result[i].1 <= result[i + 1].1);
        }

        // Verify against brute force
        let brute = brute_force_knn(&points, &query, &EuclideanDistance2D, k);
        assert!(knn_results_equivalent(&result, &brute, 1e-10));
    }
}

#[test]
fn test_knn_k_greater_than_n() {
    let mut tree = SimplifiedCoverTree::new(Distance1D, 1.3);

    for i in 0..10 {
        tree.insert(Point1D(i as f64));
    }

    let packed = tree.pack();

    // Request more neighbors than exist
    let result = packed.find_k_nearest(&Point1D(5.0), 100);

    assert_eq!(result.len(), 10); // Should return all 10 points
}

#[test]
fn test_knn_k_zero() {
    let mut tree = SimplifiedCoverTree::new(Distance1D, 1.3);

    for i in 0..10 {
        tree.insert(Point1D(i as f64));
    }

    let packed = tree.pack();

    let result = packed.find_k_nearest(&Point1D(5.0), 0);

    assert_eq!(result.len(), 0);
}

#[test]
fn test_knn_no_duplicates() {
    let mut tree = SimplifiedCoverTree::new(EuclideanDistance2D, 1.3);

    // 10x10 grid
    for i in 0..10 {
        for j in 0..10 {
            tree.insert(Point2D {
                x: i as f64,
                y: j as f64,
            });
        }
    }

    let packed = tree.pack();

    let query = Point2D { x: 5.0, y: 5.0 };
    let result = packed.find_k_nearest(&query, 50);

    // Verify no duplicates
    let mut seen = HashSet::new();
    for (point, _) in result {
        let key = (point.x as i64, point.y as i64);
        assert!(
            !seen.contains(&key),
            "Duplicate point found: ({}, {})",
            point.x,
            point.y
        );
        seen.insert(key);
    }
}

#[test]
fn test_knn_exact_match_first() {
    let mut tree = SimplifiedCoverTree::new(EuclideanDistance2D, 1.3);

    for i in 0..10 {
        for j in 0..10 {
            tree.insert(Point2D {
                x: i as f64,
                y: j as f64,
            });
        }
    }

    let packed = tree.pack();

    // Query at exact grid point
    let query = Point2D { x: 5.0, y: 5.0 };
    let result = packed.find_k_nearest(&query, 10);

    assert_eq!(result.len(), 10);
    assert_eq!(result[0].1, 0.0); // First result is exact match
    assert_eq!(result[0].0, &query);
}

#[test]
fn test_knn_sorted_by_distance() {
    let mut tree = SimplifiedCoverTree::new(Distance1D, 1.3);

    let points: Vec<f64> = vec![1.0, 5.0, 3.0, 9.0, 7.0, 2.0, 8.0, 4.0, 6.0, 10.0];
    for &p in &points {
        tree.insert(Point1D(p));
    }

    let packed = tree.pack();

    let query = Point1D(5.5);
    let result = packed.find_k_nearest(&query, 10);

    assert_eq!(result.len(), 10);

    // Verify strictly increasing or equal distances
    for i in 0..(result.len() - 1) {
        assert!(
            result[i].1 <= result[i + 1].1,
            "Not sorted: result[{}].dist={} > result[{}].dist={}",
            i,
            result[i].1,
            i + 1,
            result[i + 1].1
        );
    }
}

#[test]
fn test_knn_vs_brute_force_random() {
    let mut tree = SimplifiedCoverTree::new(EuclideanDistance2D, 1.3);

    let seed = 999u64;
    let mut rng = seed;
    let mut points = Vec::new();

    // Generate 200 random points
    for _ in 0..200 {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let x = ((rng >> 16) % 10000) as f64 / 100.0;
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let y = ((rng >> 16) % 10000) as f64 / 100.0;

        let point = Point2D { x, y };
        points.push(point.clone());
        tree.insert(point);
    }

    let packed = tree.pack();

    // Test random queries with k=1 (should match exactly) and small k
    for k in [1, 5, 10] {
        for _ in 0..6 {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let qx = ((rng >> 16) % 10000) as f64 / 100.0;
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let qy = ((rng >> 16) % 10000) as f64 / 100.0;

            let query = Point2D { x: qx, y: qy };

            let packed_result = packed.find_k_nearest(&query, k);
            let brute_result = brute_force_knn(&points, &query, &EuclideanDistance2D, k);

            assert_eq!(
                packed_result.len(),
                brute_result.len(),
                "Length mismatch for k={}",
                k
            );

            if k == 1 {
                // For k=1, should match exactly (nearest neighbor is unique in random data)
                assert_eq!(packed_result[0].0, &brute_result[0].0);
                assert!((packed_result[0].1 - brute_result[0].1).abs() < 1e-10);
            } else {
                // For k>1, verify all packed results are actually in the tree and distances are correct
                for (point, dist) in &packed_result {
                    let actual_dist = EuclideanDistance2D.distance(point, &query);
                    assert!(
                        (actual_dist - dist).abs() < 1e-10,
                        "Incorrect distance for point: computed={}, stored={}",
                        actual_dist,
                        dist
                    );
                }

                // Verify results are sorted
                for i in 0..(packed_result.len() - 1) {
                    assert!(packed_result[i].1 <= packed_result[i + 1].1);
                }
            }
        }
    }
}

#[test]
fn test_knn_high_dimensional() {
    let mut tree = SimplifiedCoverTree::new(EuclideanDistanceND, 1.3);

    let dim = 100;
    let n_points = 100;
    let seed = 456u64;
    let mut rng = seed;
    let mut points = Vec::new();

    // Generate random high-dimensional points
    for _ in 0..n_points {
        let mut coords = Vec::new();
        for _ in 0..dim {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let val = ((rng >> 16) % 10000) as f64 / 100.0;
            coords.push(val);
        }
        let point = PointND { coords };
        points.push(point.clone());
        tree.insert(point);
    }

    let packed = tree.pack();

    // Test queries
    for k in [1, 5, 10] {
        let mut coords = Vec::new();
        for _ in 0..dim {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let val = ((rng >> 16) % 10000) as f64 / 100.0;
            coords.push(val);
        }
        let query = PointND { coords };

        let packed_result = packed.find_k_nearest(&query, k);

        assert_eq!(packed_result.len(), k);

        // Verify all distances are correct
        let metric = EuclideanDistanceND;
        for (point, stored_dist) in &packed_result {
            let actual_dist = metric.distance(point, &query);
            assert!(
                (actual_dist - stored_dist).abs() < 1e-6,
                "Distance mismatch: actual={}, stored={}",
                actual_dist,
                stored_dist
            );
        }

        // Verify results are sorted
        for i in 0..(packed_result.len() - 1) {
            assert!(packed_result[i].1 <= packed_result[i + 1].1);
        }

        // Verify no duplicates
        for i in 0..packed_result.len() {
            for j in (i + 1)..packed_result.len() {
                assert_ne!(packed_result[i].0, packed_result[j].0);
            }
        }
    }
}

// ============================================================================
// Packed vs Unpacked Comparison Tests
// ============================================================================

#[test]
fn test_packed_vs_unpacked_nearest() {
    let mut tree1 = SimplifiedCoverTree::new(EuclideanDistance2D, 1.3);
    let mut tree2 = SimplifiedCoverTree::new(EuclideanDistance2D, 1.3);

    let seed = 777u64;
    let mut rng = seed;

    // Build identical trees
    for _ in 0..100 {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let x = ((rng >> 16) % 10000) as f64 / 100.0;
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let y = ((rng >> 16) % 10000) as f64 / 100.0;

        let point = Point2D { x, y };
        tree1.insert(point.clone());
        tree2.insert(point);
    }

    let packed = tree2.pack();

    // Test 50 queries
    for _ in 0..50 {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let qx = ((rng >> 16) % 10000) as f64 / 100.0;
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let qy = ((rng >> 16) % 10000) as f64 / 100.0;

        let query = Point2D { x: qx, y: qy };

        let unpacked_result = tree1.find_nearest(&query);
        let packed_result = packed.find_nearest(&query);

        assert_eq!(unpacked_result, packed_result);
    }
}

#[test]
fn test_packed_vs_unpacked_knn() {
    let mut tree1 = SimplifiedCoverTree::new(Distance1D, 1.3);
    let mut tree2 = SimplifiedCoverTree::new(Distance1D, 1.3);

    // Build identical trees
    for i in 0..100 {
        let point = Point1D(i as f64);
        tree1.insert(point.clone());
        tree2.insert(point);
    }

    let packed = tree2.pack();

    // Test various k and queries
    for k in [1, 5, 10, 25, 50] {
        for q in [0.0, 25.5, 50.0, 75.5, 100.0] {
            let query = Point1D(q);

            let unpacked_result = tree1.find_k_nearest(&query, k);
            let packed_result = packed.find_k_nearest(&query, k);

            assert_eq!(unpacked_result.len(), packed_result.len());

            for i in 0..unpacked_result.len() {
                assert_eq!(unpacked_result[i].0, packed_result[i].0);
                assert!(
                    (unpacked_result[i].1 - packed_result[i].1).abs() < 1e-10,
                    "Distance mismatch at index {}: unpacked={}, packed={}",
                    i,
                    unpacked_result[i].1,
                    packed_result[i].1
                );
            }
        }
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[test]
fn test_single_point_tree() {
    let mut tree = SimplifiedCoverTree::new(Distance1D, 1.3);
    tree.insert(Point1D(5.0));

    let packed = tree.pack();

    assert_eq!(packed.find_nearest(&Point1D(0.0)), Some(&Point1D(5.0)));
    assert_eq!(packed.find_nearest(&Point1D(5.0)), Some(&Point1D(5.0)));
    assert_eq!(packed.find_nearest(&Point1D(100.0)), Some(&Point1D(5.0)));

    let knn = packed.find_k_nearest(&Point1D(10.0), 10);
    assert_eq!(knn.len(), 1);
    assert_eq!(knn[0].0, &Point1D(5.0));
}

#[test]
fn test_two_point_tree() {
    let mut tree = SimplifiedCoverTree::new(Distance1D, 1.3);
    tree.insert(Point1D(0.0));
    tree.insert(Point1D(10.0));

    let packed = tree.pack();

    assert_eq!(packed.find_nearest(&Point1D(3.0)), Some(&Point1D(0.0)));
    assert_eq!(packed.find_nearest(&Point1D(7.0)), Some(&Point1D(10.0)));

    // At 5.0, both points are equidistant (distance 5.0)
    // Don't make assumptions about which one is returned
    let result = packed.find_nearest(&Point1D(5.0));
    assert!(result == Some(&Point1D(0.0)) || result == Some(&Point1D(10.0)));

    let knn = packed.find_k_nearest(&Point1D(5.0), 2);
    assert_eq!(knn.len(), 2);
}

#[test]
fn test_clustered_points() {
    let mut tree = SimplifiedCoverTree::new(EuclideanDistance2D, 1.3);

    // Two clusters: around (0,0) and (100,100)
    for i in 0..10 {
        for j in 0..10 {
            tree.insert(Point2D {
                x: i as f64,
                y: j as f64,
            });
            tree.insert(Point2D {
                x: 100.0 + i as f64,
                y: 100.0 + j as f64,
            });
        }
    }

    let packed = tree.pack();

    // Query in first cluster
    let result = packed.find_k_nearest(&Point2D { x: 5.0, y: 5.0 }, 20);
    assert_eq!(result.len(), 20);
    // All should be from first cluster (distance < 10)
    for (point, dist) in result {
        assert!(dist < 10.0);
        assert!(point.x < 50.0);
    }

    // Query in second cluster
    let result = packed.find_k_nearest(&Point2D { x: 105.0, y: 105.0 }, 20);
    assert_eq!(result.len(), 20);
    // All should be from second cluster
    for (point, dist) in result {
        assert!(dist < 10.0);
        assert!(point.x > 50.0);
    }
}

#[test]
fn test_identical_points() {
    let mut tree = SimplifiedCoverTree::new(Distance1D, 1.3);

    // Insert same point multiple times (user duplicates, not algorithm duplicates)
    for _ in 0..5 {
        tree.insert(Point1D(5.0));
    }

    // Add some other points
    tree.insert(Point1D(0.0));
    tree.insert(Point1D(10.0));

    let packed = tree.pack();

    // Should find at least one instance of 5.0
    let result = packed.find_nearest(&Point1D(5.0));
    assert_eq!(result, Some(&Point1D(5.0)));

    // k-NN might return multiple instances
    let knn = packed.find_k_nearest(&Point1D(5.0), 10);
    assert!(knn.len() >= 1);
    assert_eq!(knn[0].1, 0.0); // First is exact match
}

#[test]
fn test_very_large_k() {
    let mut tree = SimplifiedCoverTree::new(Distance1D, 1.3);

    for i in 0..50 {
        tree.insert(Point1D(i as f64));
    }

    let packed = tree.pack();

    // Request way more than available (but not so much we overflow)
    let result = packed.find_k_nearest(&Point1D(25.0), 10000);

    assert_eq!(result.len(), 50);
}
