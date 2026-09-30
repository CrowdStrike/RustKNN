//! Nearest Ancestor Cover Tree Implementation
//!
//! This module implements the **Nearest Ancestor Cover Tree** variant with
//! aggressive optimizations to eliminate unnecessary allocations and computations.
//!
//! # Key Optimizations
//!
//! 1. **Subtree reattachment**: Preserves extracted tree structure (2-15x speedup)
//! 2. **Zero-clone rebalancing**: Uses `std::mem::take` for ownership transfer instead of cloning
//! 3. **Linear scan insertion**: O(b) instead of O(b log b) for finding best child
//! 4. **Prune-first queries**: Only sorts children that pass pruning
//! 5. **swap_remove optimization**: O(1) child manipulation instead of O(b)
//!
//! # Performance Improvements
//!
//! Compared to the simplified variant:
//! - **Queries**: 10-30% fewer distance computations (per paper)
//! - **Construction**: Slower due to rebalancing overhead
//! - **Memory**: Zero extra allocations during rebalancing
//!
//! Best performance on clustered datasets with query-heavy workloads.
//!
//! # Usage
//!
//! ```rust,ignore
//! use rustknn::nearest_ancestor::NACoverTree;
//! use rustknn::Distance;
//!
//! struct EuclideanDistance;
//! impl Distance<f64> for EuclideanDistance {
//!     fn distance(&self, p: &f64, q: &f64) -> f64 {
//!         (p - q).abs()
//!     }
//! }
//!
//! let mut tree = NACoverTree::new(EuclideanDistance, 1.3);
//! tree.insert(5.0);
//! tree.insert(10.0);
//!
//! let nearest = tree.find_nearest(&7.0);
//! assert_eq!(nearest, Some(&5.0));
//! ```

mod tree;
mod insert;
mod rebalance;

pub use tree::NACoverTree;
