//! Performance comparison between Simplified and Nearest Ancestor cover trees.
//!
//! This example benchmarks insertion time and maxdist recomputation time for both
//! tree variants on random 512-dimensional vectors.
//!
//! Run with: cargo run --release --example performance_comparison

use rustknn::{CoverTree, Distance};
use rand::Rng;
use std::time::Instant;

/// A 512-dimensional point (vector).
#[derive(Clone, Debug)]
struct Vector512 {
    data: [f64; 512],
}

impl Vector512 {
    /// Creates a random 512-dimensional vector with values in [0, 1).
    fn random() -> Self {
        let mut rng = rand::thread_rng();
        let mut data = [0.0; 512];
        for i in 0..512 {
            data[i] = rng.gen::<f64>();
        }
        Vector512 { data }
    }
}

/// Euclidean distance metric for 512-dimensional vectors.
struct EuclideanDistance;

impl Distance<Vector512> for EuclideanDistance {
    fn distance(&self, p: &Vector512, q: &Vector512) -> f64 {
        let mut sum = 0.0;
        for i in 0..512 {
            let diff = p.data[i] - q.data[i];
            sum += diff * diff;
        }
        sum.sqrt()
    }
}

/// Runs performance test for a given number of points.
fn benchmark_size(n: usize) {
    println!("\n========================================");
    println!("Benchmarking with {} points", n);
    println!("========================================");

    // Generate random points
    println!("Generating {} random 512-dimensional vectors...", n);
    let start = Instant::now();
    let points: Vec<Vector512> = (0..n).map(|_| Vector512::random()).collect();
    let gen_time = start.elapsed();
    println!("Generated in {:.2?}", gen_time);

    // Test Simplified variant
    println!("\n--- Simplified Cover Tree ---");
    let mut simplified_tree = CoverTree::new(EuclideanDistance, 1.3);

    let start = Instant::now();
    for point in points.iter() {
        simplified_tree.insert(point.clone());
    }
    let insert_time = start.elapsed();
    println!("Insertion time: {:.2?}", insert_time);
    println!("Tree size: {} nodes", simplified_tree.len());

    let start = Instant::now();
    simplified_tree.recompute_maxdist();
    let recompute_time = start.elapsed();
    println!("Recompute maxdist time: {:.2?}", recompute_time);
    println!("Total time (insert + recompute): {:.2?}", insert_time + recompute_time);

    // Test Nearest Ancestor variant
    println!("\n--- Nearest Ancestor Cover Tree ---");
    let mut na_tree = CoverTree::new_nearest_ancestor(EuclideanDistance, 1.3);

    let start = Instant::now();
    for point in points.iter() {
        na_tree.insert(point.clone());
    }
    let na_insert_time = start.elapsed();
    println!("Insertion time (with rebalancing): {:.2?}", na_insert_time);
    println!("Tree size: {} nodes", na_tree.len());

    // NA variant maintains exact maxdist automatically, so no recompute needed
    println!("Recompute maxdist time: N/A (maintains exact maxdist automatically)");
    println!("Total time: {:.2?}", na_insert_time);

    // Comparison
    println!("\n--- Comparison ---");
    let slowdown = na_insert_time.as_secs_f64() / insert_time.as_secs_f64();
    println!("NA insertion slowdown: {:.2}x", slowdown);

    let simplified_total = insert_time + recompute_time;
    let speedup = simplified_total.as_secs_f64() / na_insert_time.as_secs_f64();
    if speedup > 1.0 {
        println!("NA total construction: {:.2}x FASTER than Simplified+recompute", speedup);
    } else {
        println!("NA total construction: {:.2}x slower than Simplified+recompute", 1.0 / speedup);
    }
}

fn main() {
    println!("===========================================");
    println!("Cover Tree Performance Comparison");
    println!("===========================================");
    println!("Point type: 512-dimensional vectors");
    println!("Distance metric: Euclidean distance");
    println!("Base value: 1.3");
    println!();
    println!("NOTE: Run with --release for accurate timings!");
    println!("  cargo run --release --example performance_comparison");

    // Uniform random 512-d data has no low-dimensional structure, which is the
    // worst case for cover trees; sizes are kept small so the example finishes
    // in well under a minute.
    benchmark_size(2_000);
    benchmark_size(5_000);

    println!("\n===========================================");
    println!("Benchmark complete!");
    println!("===========================================");
}
