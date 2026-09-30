//! Example: Using Cover Trees with 2D Euclidean points
//!
//! Run with: `cargo run --example euclidean_2d`
//!
//! This example demonstrates:
//! - Defining a custom point type
//! - Implementing the Distance trait for Euclidean distance
//! - Creating a cover tree
//! - Inserting points
//! - Finding nearest neighbors
//! - Verifying correctness

use rustknn::{SimplifiedCoverTree, Distance};

/// A simple 2D point
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
///
/// This implements the Distance trait, which allows the cover tree
/// to work with our custom point type.
struct EuclideanDistance;

impl Distance<Point2D> for EuclideanDistance {
    fn distance(&self, p: &Point2D, q: &Point2D) -> f64 {
        let dx = p.x - q.x;
        let dy = p.y - q.y;
        (dx * dx + dy * dy).sqrt()
    }
}

fn main() {
    println!("=== Cover Tree Example: 2D Euclidean Points ===\n");

    // Create a new cover tree with Euclidean distance and base 1.3
    // Base 1.3 is the paper's recommended value for optimal performance
    let metric = EuclideanDistance;
    let mut tree = SimplifiedCoverTree::new(metric, 1.3);

    println!("Created empty cover tree with base 1.3");
    println!("Tree size: {}\n", tree.len());

    // Define some cities with their coordinates (simplified)
    let cities = vec![
        ("San Francisco", Point2D::new(0.0, 0.0)),
        ("Los Angeles", Point2D::new(1.0, 4.0)),
        ("San Diego", Point2D::new(1.5, 5.0)),
        ("Sacramento", Point2D::new(0.5, 2.0)),
        ("Las Vegas", Point2D::new(3.0, 4.0)),
        ("Phoenix", Point2D::new(5.0, 5.0)),
        ("Portland", Point2D::new(0.0, 7.0)),
        ("Seattle", Point2D::new(0.0, 9.0)),
        ("Denver", Point2D::new(8.0, 4.0)),
        ("Salt Lake City", Point2D::new(7.0, 3.0)),
    ];

    // Insert all cities into the tree
    println!("Inserting {} cities...", cities.len());
    for (name, point) in &cities {
        tree.insert(point.clone());
        println!("  - Inserted: {}", name);
    }

    println!("\nTree size after insertions: {}", tree.len());
    println!("(Simplified cover tree has exactly n nodes for n insertions)\n");

    // Demonstrate nearest neighbor queries
    println!("=== Nearest Neighbor Queries ===\n");

    // Query 1: Point near San Francisco
    let query1 = Point2D::new(0.2, 0.3);
    println!("Query 1: Finding nearest city to ({}, {})", query1.x, query1.y);
    if let Some(nearest) = tree.find_nearest(&query1) {
        let city_name = cities
            .iter()
            .find(|(_, p)| p == nearest)
            .map(|(name, _)| name)
            .unwrap();
        let dist = EuclideanDistance.distance(&query1, nearest);
        println!("  → Nearest: {} at ({}, {})", city_name, nearest.x, nearest.y);
        println!("  → Distance: {:.2}\n", dist);
    }

    // Query 2: Point near Las Vegas
    let query2 = Point2D::new(3.5, 4.2);
    println!("Query 2: Finding nearest city to ({}, {})", query2.x, query2.y);
    if let Some(nearest) = tree.find_nearest(&query2) {
        let city_name = cities
            .iter()
            .find(|(_, p)| p == nearest)
            .map(|(name, _)| name)
            .unwrap();
        let dist = EuclideanDistance.distance(&query2, nearest);
        println!("  → Nearest: {} at ({}, {})", city_name, nearest.x, nearest.y);
        println!("  → Distance: {:.2}\n", dist);
    }

    // Query 3: Point near Seattle
    let query3 = Point2D::new(0.1, 8.8);
    println!("Query 3: Finding nearest city to ({}, {})", query3.x, query3.y);
    if let Some(nearest) = tree.find_nearest(&query3) {
        let city_name = cities
            .iter()
            .find(|(_, p)| p == nearest)
            .map(|(name, _)| name)
            .unwrap();
        let dist = EuclideanDistance.distance(&query3, nearest);
        println!("  → Nearest: {} at ({}, {})", city_name, nearest.x, nearest.y);
        println!("  → Distance: {:.2}\n", dist);
    }

    // Query 4: Exactly at a city location
    let query4 = Point2D::new(5.0, 5.0); // Phoenix
    println!(
        "Query 4: Finding nearest city to ({}, {}) (exact match)",
        query4.x, query4.y
    );
    if let Some(nearest) = tree.find_nearest(&query4) {
        let city_name = cities
            .iter()
            .find(|(_, p)| p == nearest)
            .map(|(name, _)| name)
            .unwrap();
        let dist = EuclideanDistance.distance(&query4, nearest);
        println!("  → Nearest: {} at ({}, {})", city_name, nearest.x, nearest.y);
        println!("  → Distance: {:.2} (exact match!)\n", dist);
    }

    // Verification: each city should be its own nearest neighbor
    println!("=== Verification: All Nearest Neighbors Test ===\n");
    println!("Verifying that each city is its own nearest neighbor...");
    let mut all_correct = true;
    for (name, point) in &cities {
        let nearest = tree.find_nearest(point).unwrap();
        if nearest != point {
            println!("  ✗ {}: Expected itself, got different point", name);
            all_correct = false;
        } else {
            println!("  ✓ {}", name);
        }
    }

    if all_correct {
        println!("\n✓ All cities correctly found themselves as nearest neighbors!");
    } else {
        println!("\n✗ Some cities failed the verification");
    }

    // Performance characteristics
    println!("\n=== Performance Characteristics ===\n");
    println!("Query time complexity: O(c^6 log n)");
    println!("  where c is the doubling constant (intrinsic dimensionality)");
    println!("  and n is the number of points ({} in this example)", cities.len());
    println!("\nInsertion time complexity: O(c^6 log n)");
    println!("Space complexity: O(n) - exactly one node per point");
    println!("\nThe cover tree efficiently prunes the search space using:");
    println!("  1. Hierarchical structure with exponentially growing levels");
    println!("  2. maxdist caching for quick pruning decisions");
    println!("  3. Sorting children by distance for early termination");

    println!("\n=== Example Complete ===");
}
