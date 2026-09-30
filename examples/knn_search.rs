//! Example demonstrating k-Nearest Neighbor search with Cover Trees
//!
//! This example shows how to use the `find_k_nearest()` method to find
//! multiple nearest neighbors of a query point.

use rustknn::{CoverTree, Distance};

#[derive(Clone, Debug)]
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

fn main() {
    println!("=== Cover Tree k-Nearest Neighbor Search Demo ===\n");

    // Create a cover tree
    let mut tree = CoverTree::new(EuclideanDistance, 1.3);

    // Insert points on a grid
    println!("Building tree with points on a 5x5 grid...");
    for i in 0..5 {
        for j in 0..5 {
            tree.insert(Point2D::new(i as f64, j as f64));
        }
    }
    println!("Tree contains {} points\n", tree.len());

    // Query point
    let query = Point2D::new(2.5, 2.5);
    println!("Query point: ({}, {})\n", query.x, query.y);

    // Find k nearest neighbors for various k values
    for k in [1, 3, 5, 10] {
        println!("Finding {} nearest neighbors:", k);
        let neighbors = tree.find_k_nearest(&query, k);

        for (i, (point, dist)) in neighbors.iter().enumerate() {
            println!("  #{}: ({}, {}) at distance {:.4}", i + 1, point.x, point.y, dist);
        }
        println!();
    }

    // Demonstrate with a different query point
    println!("=== Different Query Point ===\n");
    let query2 = Point2D::new(0.0, 0.0);
    println!("Query point: ({}, {})\n", query2.x, query2.y);

    let neighbors = tree.find_k_nearest(&query2, 5);
    println!("5 nearest neighbors:");
    for (i, (point, dist)) in neighbors.iter().enumerate() {
        println!("  #{}: ({}, {}) at distance {:.4}", i + 1, point.x, point.y, dist);
    }
    println!();

    // Demonstrate that user's duplicate values are correctly returned
    println!("=== Handling Duplicate Values ===\n");
    let mut tree2 = CoverTree::new(EuclideanDistance, 1.3);
    tree2.insert(Point2D::new(1.0, 1.0));
    tree2.insert(Point2D::new(1.0, 1.0));  // Same value inserted twice
    tree2.insert(Point2D::new(2.0, 2.0));
    tree2.insert(Point2D::new(3.0, 3.0));

    let query3 = Point2D::new(1.0, 1.0);
    println!("Query point: ({}, {})", query3.x, query3.y);
    println!("Note: We inserted (1.0, 1.0) twice\n");

    let neighbors = tree2.find_k_nearest(&query3, 3);
    println!("3 nearest neighbors:");
    for (i, (point, dist)) in neighbors.iter().enumerate() {
        println!("  #{}: ({}, {}) at distance {:.4}", i + 1, point.x, point.y, dist);
    }
    println!("\nNotice: Both (1.0, 1.0) points are returned!");

    println!("\n=== Summary ===");
    println!("✓ k-NN search finds k closest points efficiently");
    println!("✓ Results are sorted by distance (closest first)");
    println!("✓ User's duplicate values are correctly returned");
    println!("✓ Time complexity: O(c^6 log n) per query");
}
