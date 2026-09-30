//! Utility Functions for Cover Tree Operations
//!
//! This module contains utility functions shared between both cover tree variants.

use crate::node::Node;

/// Recursively adjusts the level of a node and all its descendants.
///
/// This is used during level raising when a new point is too far from the current root,
/// or when aligning tree levels during merge operations.
///
/// # Arguments
///
/// * `node` - The node whose level (and descendants' levels) to adjust
/// * `adjustment` - The amount to add to each level (can be negative)
///
/// # Example
///
/// ```ignore
/// // Raise entire subtree by 2 levels
/// utils::adjust_levels(&mut node, 2);
///
/// // Lower entire subtree by 1 level
/// utils::adjust_levels(&mut node, -1);
/// ```
pub(crate) fn adjust_levels<T: Clone>(node: &mut Node<T>, adjustment: i32) {
    node.level += adjustment;
    for child in &mut node.children {
        adjust_levels(child, adjustment);
    }
}
