//! # RustKNN
//!
//! Cover trees for exact k-nearest-neighbor search in arbitrary metric spaces.
//!
//! This crate is the implementation accompanying *"KNN Implementation Details Can
//! Dramatically Change Performance: An Example from Cover Trees"* (Khanna & Raff,
//! NeurIPS 2026). It builds on the cover tree of Beygelzimer, Kakade & Langford
//! (ICML 2006) and the simplified / nearest-ancestor variants of Izbicki & Shelton
//! (ICML 2015).
//!
//! ## What are Cover Trees?
//!
//! Cover trees are tree data structures that enable fast nearest neighbor queries
//! over arbitrary metric spaces (not just Euclidean). They maintain three key invariants:
//!
//! 1. **Leveling**: Each node has a level, children are at `level - 1`
//! 2. **Covering**: Children are within covering distance: `d(parent, child) ≤ base^level`
//! 3. **Separating**: Children are separated from each other: `d(child1, child2) > base^(level-1)`
//!
//! Unlike kd-trees which only work well in low-dimensional Euclidean spaces, cover trees
//! work with any distance metric and have performance guarantees that depend on the
//! intrinsic dimensionality (doubling constant) rather than the ambient dimension.
//!
//! ## Overview
//!
//! This library provides three cover tree construction variants (Simplified, Nearest
//! Ancestor, and Packed) and three batch query modes (dual-tree, single-tree, and batch
//! single-tree), plus tree merging and parallel construction. A median-split KD-tree
//! ([`kdtree`]) and a brute-force scan ([`naive`]) are included as reference baselines.
//! See the feature list below for details.
//!
//! ## Quick Start
//!
//! ```
//! use rustknn::{CoverTree, Distance};
//!
//! // Define a simple point type
//! #[derive(Clone, Debug, PartialEq)]
//! struct Point2D { x: f64, y: f64 }
//!
//! // Implement the Distance trait for Euclidean distance
//! struct EuclideanDistance;
//!
//! impl Distance<Point2D> for EuclideanDistance {
//!     fn distance(&self, p: &Point2D, q: &Point2D) -> f64 {
//!         let dx = p.x - q.x;
//!         let dy = p.y - q.y;
//!         (dx * dx + dy * dy).sqrt()
//!     }
//! }
//!
//! // Create a cover tree with base 1.3 (paper's recommended value)
//! let metric = EuclideanDistance;
//! let mut tree = CoverTree::new(metric, 1.3);
//!
//! // Insert points
//! tree.insert(Point2D { x: 0.0, y: 0.0 });
//! tree.insert(Point2D { x: 1.0, y: 1.0 });
//! tree.insert(Point2D { x: 2.0, y: 2.0 });
//!
//! // Find nearest neighbor
//! let query = Point2D { x: 1.5, y: 1.5 };
//! let nearest = tree.find_nearest(&query);
//!
//! assert!(nearest.is_some());
//! println!("Nearest neighbor: {:?}", nearest);
//! ```
//!
//! ## Features
//!
//! ### Core Features
//!
//! **Simplified Cover Trees**:
//! - **Generic over point types**: Works with any type implementing `Clone`
//! - **Generic over metrics**: Works with any distance function via the `Distance` trait
//! - **Configurable base value**: Default 1.3; tunable per dataset
//! - **Efficient insertion**: O(c^6 log n) time complexity
//! - **Efficient queries**: O(c^6 log n) time complexity with pruning
//! - **Zero-copy queries**: Returns references without unnecessary allocations
//! - **Space efficient**: Exactly n nodes for n points (no duplicates)
//!
//! **Nearest Ancestor Cover Trees**:
//! - **Rebalancing**: Maintains nearest ancestor invariant for improved query performance
//! - **Better tree structure**: More balanced tree via rebalancing during insertion
//! - **Automatic restructuring**: Points may move to better ancestors when beneficial
//! - **10-30% fewer distance computations**: Per paper's benchmarks (query performance)
//! - **Same O(c^6 log n) complexity**: With improved constants
//!
//! **Choosing a Variant**:
//! - Use `CoverTree::new()` for **Simplified** (faster construction)
//! - Use `CoverTree::new_nearest_ancestor()` for **Nearest Ancestor** (better queries)
//!
//! ### Additional Features
//!
//! - **k-NN queries**: `find_k_nearest()` and `find_k_nearest_batch()` (dual-tree)
//! - **Self k-NN**: `find_k_nearest_self()` for all-nearest-neighbors queries
//! - **Batch single-tree queries**: `find_k_nearest_batch_single_self()` on
//!   `SimplifiedCoverTree` and `PackedCoverTree`, and
//!   `PackedCoverTree::find_k_nearest_batch_single()` for held-out queries
//! - **Tree merging**: `merge()` combines two cover trees (Algorithm 4)
//! - **Parallel construction**: Multi-threaded build via Rayon with binary reduction merge
//! - **Packed trees**: Cache-optimized depth-first layout via `pack()`
//!
//! ### Not Yet Implemented
//!
//! - **Range queries**: Find all points within a given radius
//!
//! ## Performance Characteristics
//!
//! - **Query time**: O(c^6 log n) where c is the doubling constant
//! - **Construction time**: O(c^6 n log n)
//! - **Space**: O(n) - exactly one node per data point
//!
//! The doubling constant c characterizes the intrinsic dimensionality of your data.
//! For well-clustered data, c is small and operations are very fast. For uniformly
//! distributed high-dimensional data, c can be large.
//!
//! ## Examples
//!
//! ### Custom Point Types
//!
//! You can use cover trees with any point type:
//!
//! ```
//! use rustknn::{CoverTree, Distance};
//!
//! // 3D points
//! #[derive(Clone)]
//! struct Point3D { x: f64, y: f64, z: f64 }
//!
//! struct Euclidean3D;
//! impl Distance<Point3D> for Euclidean3D {
//!     fn distance(&self, p: &Point3D, q: &Point3D) -> f64 {
//!         let dx = p.x - q.x;
//!         let dy = p.y - q.y;
//!         let dz = p.z - q.z;
//!         (dx * dx + dy * dy + dz * dz).sqrt()
//!     }
//! }
//!
//! let mut tree = CoverTree::new(Euclidean3D, 1.3);
//! tree.insert(Point3D { x: 0.0, y: 0.0, z: 0.0 });
//! ```
//!
//! ### Different Distance Metrics
//!
//! Cover trees work with any distance metric:
//!
//! ```
//! use rustknn::{CoverTree, Distance};
//!
//! #[derive(Clone)]
//! struct Point2D { x: f64, y: f64 }
//!
//! // Manhattan (L1) distance
//! struct ManhattanDistance;
//! impl Distance<Point2D> for ManhattanDistance {
//!     fn distance(&self, p: &Point2D, q: &Point2D) -> f64 {
//!         (p.x - q.x).abs() + (p.y - q.y).abs()
//!     }
//! }
//!
//! let mut tree = CoverTree::new(ManhattanDistance, 1.3);
//! tree.insert(Point2D { x: 0.0, y: 0.0 });
//! ```
//!
//! ### Working with Integers
//!
//! ```
//! use rustknn::{CoverTree, Distance};
//!
//! struct IntDistance;
//! impl Distance<i32> for IntDistance {
//!     fn distance(&self, p: &i32, q: &i32) -> f64 {
//!         (p - q).abs() as f64
//!     }
//! }
//!
//! let mut tree = CoverTree::new(IntDistance, 1.3);
//! tree.insert(10);
//! tree.insert(20);
//! tree.insert(30);
//!
//! let nearest = tree.find_nearest(&15);
//! assert!(nearest == Some(&10) || nearest == Some(&20));
//! ```
//!
//! ### Using the Nearest Ancestor Variant
//!
//! The Nearest Ancestor variant adds rebalancing during insertion to maintain
//! the nearest ancestor invariant, resulting in better query performance:
//!
//! ```
//! use rustknn::{CoverTree, Distance};
//!
//! struct SimpleDistance;
//! impl Distance<f64> for SimpleDistance {
//!     fn distance(&self, p: &f64, q: &f64) -> f64 {
//!         (p - q).abs()
//!     }
//! }
//!
//! // Create Nearest Ancestor tree (better query performance)
//! let mut tree = CoverTree::new_nearest_ancestor(SimpleDistance, 1.3);
//!
//! // Insert points (slower due to rebalancing)
//! tree.insert(0.0);
//! tree.insert(100.0);
//! tree.insert(50.0);  // May trigger rebalancing
//! tree.insert(25.0);
//! tree.insert(75.0);
//!
//! // Queries are 10-30% faster (fewer distance computations)
//! let nearest = tree.find_nearest(&60.0);
//! assert!(nearest == Some(&50.0) || nearest == Some(&75.0));
//! ```
//!
//! **Trade-off**: Nearest Ancestor has slower construction (rebalancing overhead)
//! but faster queries. Use it when queries dominate your workload.
//!
//! ## Design Decisions
//!
//! ### Why `base` instead of hardcoded 2.0?
//!
//! The original cover tree paper uses base 2.0, but "Faster Cover Trees" shows that
//! base 1.3 reduces distance computations by 10-71% compared to base 2.0. We make
//! the base configurable so you can benchmark different values for your dataset.
//!
//! ### Why `T: Clone` instead of `T: Copy`?
//!
//! Using `Clone` instead of `Copy` is more flexible - it works with any type that
//! can be cloned, not just primitive types. This allows using cover trees with
//! complex point types (e.g., points with string labels).
//!
//! ### Why cache `maxdist` in nodes?
//!
//! Caching the maximum distance to descendants in each node enables O(1) pruning
//! decisions during queries. This is critical for achieving the O(c^6 log n)
//! query time complexity. Without caching, we'd need to traverse entire subtrees
//! to compute bounds.
//!
//! ### Why exactly n nodes?
//!
//! The simplified cover tree variant maintains exactly one node per
//! data point - no duplicate nodes at multiple levels. This simplifies the
//! implementation while still providing good query performance. The Nearest
//! Ancestor variant allows points to appear at multiple levels during
//! rebalancing, which can improve query performance at the cost of higher memory usage.
//!
//! ## Running the Examples
//!
//! ```bash
//! cargo run --example euclidean_2d
//! cargo run --example knn_search
//! cargo run --release --example performance_comparison
//! ```
//!
//! ## Running Tests
//!
//! ```bash
//! # Run all tests (unit + integration + doc tests)
//! cargo test
//!
//! # Run a single integration test file
//! cargo test --test simplified_tests
//!
//! # Run with output
//! cargo test -- --nocapture
//! ```
//!
//! ## References
//!
//! - Beygelzimer, A., Kakade, S., & Langford, J. (2006). *Cover trees for nearest
//!   neighbor*. In Proceedings of the 23rd International Conference on Machine Learning (ICML).
//! - Izbicki, M., & Shelton, C. R. (2015). *Faster Cover Trees*. In Proceedings of
//!   the 32nd International Conference on Machine Learning (ICML).
//!
//! ## License
//!
//! Licensed under the MIT license.

// Module declarations
pub mod distance;
pub mod node;
pub mod simplified;
pub mod tree;
pub mod nearest_ancestor;
pub mod parallel;
pub mod packed;
pub mod kdtree;
pub mod naive;

// Shared core algorithms (internal, except stats)
pub(crate) mod core;

// Re-export instrumentation stats and bound modes
pub use core::dual_tree::DualTreeStats;
pub use core::dual_tree::BoundMode;

// k-NN support module (internal)
pub(crate) mod knn;

// Re-export main types for convenient access
pub use distance::{Distance, EuclideanDistance, ManhattanDistance, reset_distance_count, get_distance_count};
pub use core::utils::MIN_BASE;
pub use node::Node;
pub use node::TreeStats;
pub use tree::{CoverTree, TreeVariant};
pub use simplified::SimplifiedCoverTree;
pub use nearest_ancestor::NACoverTree;

// Compile and run the README's Rust examples as doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
