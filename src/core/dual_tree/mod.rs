//! Dual-Tree k-NN Search
//!
//! This module implements cover-tree-specific dual-tree k-NN search, which uses
//! two cover trees (a query tree and a reference tree) to accelerate batch k-NN
//! queries by pruning entire subtree combinations.
//!
//! # Algorithm
//!
//! The dual-tree traversal exploits the cover tree's level/scale structure:
//! - Query tree is traversed depth-first
//! - Reference tree is expanded breadth-first by scale level
//! - At each step, (query_node, ref_node) pairs are scored and pruned
//!
//! This dramatically reduces distance computations compared to running single-tree
//! k-NN for each query point independently.
//!
//! # Modules
//!
//! - [`state`] - Per-query-point candidate tracking (`DualKnnState`)
//! - [`knn_rules`] - BaseCase, Score, and Bound calculations
//! - [`traversal`] - Cover-tree-specific dual-tree traversal algorithm

pub(crate) mod state;
pub(crate) mod knn_rules;
pub(crate) mod traversal;

pub use state::BoundMode;

/// Instrumentation stats from a dual-tree traversal.
///
/// Returned by `*_instrumented` methods to quantify traversal overhead.
#[derive(Clone, Debug, Default)]
pub struct DualTreeStats {
    /// Total distance computations during traversal (from thread-local counter).
    pub distance_computations: u64,
    /// Number of `bound_with_idx` calls that hit the cache.
    pub bound_cache_hits: u64,
    /// Number of `bound_with_idx` calls that missed the cache.
    pub bound_cache_misses: u64,
    /// Time (ms) spent building the query tree (for batch methods).
    pub query_tree_build_ms: f64,
    /// Time (ms) spent on recompute_maxdist of the query tree.
    pub query_tree_maxdist_ms: f64,
    /// Time (ms) spent on init_parent_map.
    pub init_parent_map_ms: f64,
    /// Time (ms) spent on the actual dual-tree traversal.
    pub traversal_ms: f64,
    /// Time (ms) spent collecting and reordering results.
    pub result_collection_ms: f64,
}

