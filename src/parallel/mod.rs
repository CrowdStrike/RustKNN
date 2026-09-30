//! Parallel Cover Tree Construction
//!
//! This module implements parallel construction of SimplifiedCoverTrees using Rayon.
//! The approach follows the paper's Algorithm 4 (tree merging) combined with parallel
//! tree building.
//!
//! # Strategy
//!
//! 1. Partition dataset into chunks (one per thread)
//! 2. Build trees in parallel using either SimplifiedCoverTree or NACoverTree
//! 3. Merge results using binary tree reduction
//!
//! # Variants
//!
//! Use [`ParallelVariant`] to control which tree type is built per-thread:
//! - `Simplified`: Faster per-thread construction, no rebalancing
//! - `NearestAncestor`: Better per-thread tree structure (rebalancing),
//!   converted to SimplifiedCoverTree for merging
//!
//! # Performance
//!
//! - Near-linear speedup for large datasets
//! - Scales with number of CPU cores
//!
//! # Examples
//!
//! ```rust,ignore
//! use rustknn::parallel::{build_parallel, ParallelVariant};
//! use rustknn::Distance;
//!
//! struct EuclideanDistance;
//! impl Distance<f64> for EuclideanDistance {
//!     fn distance(&self, p: &f64, q: &f64) -> f64 {
//!         (p - q).abs()
//!     }
//! }
//!
//! let points: Vec<f64> = (0..100_000).map(|i| i as f64).collect();
//!
//! // Build with Simplified per-thread trees (default)
//! let tree = build_parallel(
//!     EuclideanDistance, 1.3, points.clone(), None,
//!     ParallelVariant::Simplified,
//! );
//!
//! // Build with NearestAncestor per-thread trees
//! let tree = build_parallel(
//!     EuclideanDistance, 1.3, points, None,
//!     ParallelVariant::NearestAncestor,
//! );
//! ```

mod builder;
mod merge_all;

pub use builder::build_parallel;
pub use builder::build_parallel_simplified;
pub use builder::ParallelVariant;
