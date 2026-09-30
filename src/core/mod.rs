//! Shared Core Algorithms for Cover Trees
//!
//! This module contains algorithms shared between the Simplified and
//! Nearest Ancestor cover tree variants. By consolidating identical
//! implementations here, we eliminate code duplication while maintaining
//! clear separation between variant-specific logic.
//!
//! # Modules
//!
//! - [`knn_impl`] - k-nearest neighbor search algorithm
//! - [`query`] - Single nearest neighbor search algorithm
//! - [`utils`] - Utility functions for tree operations
//! - [`dual_tree`] - Dual-tree k-NN search (batch queries using two cover trees)
//!
//! # Design Philosophy
//!
//! Code is placed in this module if and only if:
//! 1. The implementation is identical between both variants
//! 2. The algorithm operates on the shared Node structure
//! 3. No variant-specific logic is required
//!
//! Variant-specific algorithms (insertion, rebalancing, merging) remain
//! in their respective modules to maintain clarity and separation of concerns.

pub(crate) mod knn_impl;
pub(crate) mod query;
pub(crate) mod utils;
pub(crate) mod dual_tree;
pub(crate) mod batch_single;
