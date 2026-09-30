//! Simplified Cover Tree Implementation
//!
//! This module implements the **Simplified Cover Tree** variant from the paper
//! "Faster Cover Trees" (ICML 2015). The simplified variant maintains exactly
//! n nodes for n points and does not perform rebalancing.
//!
//! # Key Characteristics
//!
//! - **Exactly n nodes**: One node per point (no redundant nodes)
//! - **No rebalancing**: Points stay where they're first inserted
//! - **Faster construction**: No restructuring overhead compared to Nearest Ancestor variant
//! - **Three invariants**: Maintains leveling, covering, and separating invariants
//!
//! # Performance
//!
//! - **Construction**: O(c^6 log n) per point insertion
//! - **Query**: O(c^6 log n) per nearest neighbor query
//! - **Space**: O(n) nodes
//!
//! Where c is the doubling constant of the dataset.
//!
//! # When to Use
//!
//! Use the simplified variant when:
//! - Construction speed is more important than query speed
//! - Dataset is small to medium size (< 100K points)
//! - Query performance is "good enough" for your use case
//!
//! For better query performance at the cost of slower construction, use the
//! Nearest Ancestor variant in `crate::nearest_ancestor`.
//!
//! # Example
//!
//! ```rust,ignore
//! use rustknn::simplified::SimplifiedCoverTree;
//! use rustknn::Distance;
//!
//! struct EuclideanDistance;
//! impl Distance<f64> for EuclideanDistance {
//!     fn distance(&self, p: &f64, q: &f64) -> f64 {
//!         (p - q).abs()
//!     }
//! }
//!
//! let mut tree = SimplifiedCoverTree::new(EuclideanDistance, 1.3);
//! tree.insert(5.0);
//! tree.insert(10.0);
//! tree.insert(15.0);
//!
//! let nearest = tree.find_nearest(&7.0);
//! assert_eq!(nearest, Some(&5.0));
//! ```

mod tree;
mod insert;
pub(crate) mod merge;  // Make merge accessible to NACoverTree

pub use tree::SimplifiedCoverTree;
