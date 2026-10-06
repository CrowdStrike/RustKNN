// Packed-tree k-NN on a regular grid, where many distances tie. Checks that the
// traversal visits each node once, returns no duplicate points, and finds the true
// nearest neighbors.

use rustknn::simplified::SimplifiedCoverTree;
use rustknn::distance::Distance;

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

struct EuclideanDistance;

impl Distance<Point2D> for EuclideanDistance {
    fn distance(&self, p: &Point2D, q: &Point2D) -> f64 {
        let dx = p.x - q.x;
        let dy = p.y - q.y;
        (dx * dx + dy * dy).sqrt()
    }
}

#[test]
fn test_packed_5x5_grid() {
    // Build 5x5 grid
    let mut tree = SimplifiedCoverTree::new(EuclideanDistance, 1.3);

    for i in 0..5 {
        for j in 0..5 {
            tree.insert(Point2D::new(i as f64, j as f64));
        }
    }

    // Pack the tree
    let packed = tree.pack();

    // Query at (2,2) - should find exact match at distance 0
    let query = Point2D::new(2.0, 2.0);
    let neighbors = packed.find_k_nearest(&query, 10);

    // CRITICAL: First result must be exact match at distance 0
    assert_eq!(neighbors.len(), 10);
    assert_eq!(neighbors[0].1, 0.0, "First neighbor must be exact match at distance 0");
    assert_eq!(neighbors[0].0.x, 2.0);
    assert_eq!(neighbors[0].0.y, 2.0);

    // Verify no duplicates in results
    for i in 0..neighbors.len() {
        for j in (i + 1)..neighbors.len() {
            assert_ne!(
                neighbors[i].0, neighbors[j].0,
                "Found duplicate point in k-NN results at indices {} and {}",
                i, j
            );
        }
    }

    // Verify distances are reasonable (all should be ≤ 2.0 for 10 nearest in 5x5 grid)
    for (i, (point, dist)) in neighbors.iter().enumerate() {
        assert!(
            *dist <= 2.0,
            "Neighbor {} at distance {} is too far (point: {:?})",
            i, dist, point
        );
    }

    // Verify results are sorted by distance
    for i in 0..(neighbors.len() - 1) {
        assert!(
            neighbors[i].1 <= neighbors[i + 1].1,
            "Results not sorted: neighbors[{}].dist={} > neighbors[{}].dist={}",
            i, neighbors[i].1, i + 1, neighbors[i + 1].1
        );
    }
}

#[test]
fn test_packed_matches_unpacked_knn() {
    // Verify packed and unpacked k-NN return identical results

    // Test multiple queries
    for query_x in 0..4 {
        for query_y in 0..4 {
            // Build two identical trees (since pack() consumes and we need both packed and unpacked)
            let mut tree1 = SimplifiedCoverTree::new(EuclideanDistance, 1.3);
            let mut tree2 = SimplifiedCoverTree::new(EuclideanDistance, 1.3);

            // Insert 4x4 grid into both
            for i in 0..4 {
                for j in 0..4 {
                    let point = Point2D::new(i as f64, j as f64);
                    tree1.insert(point.clone());
                    tree2.insert(point);
                }
            }

            let query = Point2D::new(query_x as f64 + 0.5, query_y as f64 + 0.5);

            let unpacked_result = tree1.find_k_nearest(&query, 5);
            let packed = tree2.pack();
            let packed_result = packed.find_k_nearest(&query, 5);

            assert_eq!(
                unpacked_result.len(), packed_result.len(),
                "Result count mismatch for query ({}, {})",
                query_x, query_y
            );

            // Verify same distances (point identity may differ for equidistant results)
            for i in 0..unpacked_result.len() {
                assert!(
                    (unpacked_result[i].1 - packed_result[i].1).abs() < 1e-10,
                    "Distance mismatch at index {} for query ({}, {}): unpacked={}, packed={}",
                    i, query_x, query_y, unpacked_result[i].1, packed_result[i].1
                );
            }
        }
    }
}

#[test]
fn test_packed_large_dataset_no_duplicates() {
    // Test with larger dataset to ensure no duplicates or visiting issues
    let mut tree = SimplifiedCoverTree::new(EuclideanDistance, 1.3);

    // Insert 10x10 grid
    for i in 0..10 {
        for j in 0..10 {
            tree.insert(Point2D::new(i as f64, j as f64));
        }
    }

    let packed = tree.pack();

    // Query at center
    let query = Point2D::new(5.0, 5.0);
    let neighbors = packed.find_k_nearest(&query, 20);

    assert_eq!(neighbors.len(), 20);

    // Exact match at distance 0
    assert_eq!(neighbors[0].1, 0.0);
    assert_eq!(neighbors[0].0, &Point2D::new(5.0, 5.0));

    // No duplicates
    for i in 0..neighbors.len() {
        for j in (i + 1)..neighbors.len() {
            assert_ne!(neighbors[i].0, neighbors[j].0);
        }
    }

    // Results sorted
    for i in 0..(neighbors.len() - 1) {
        assert!(neighbors[i].1 <= neighbors[i + 1].1);
    }
}
