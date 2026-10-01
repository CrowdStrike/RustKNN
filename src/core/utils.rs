//! Utility Functions for Cover Tree Operations
//!
//! This module contains utility functions shared between both cover tree variants.

use crate::node::Node;

/// Smallest accepted cover tree base.
///
/// Every constructor that takes a `base` panics if it is below this value or not finite.
/// Levels are computed as `ceil(ln(distance) / ln(base))`, so a base near 1 produces
/// extreme levels. With `base >= MIN_BASE`, every level derived from a finite `f64`
/// distance fits comfortably in an `i32` (|level| < 8,000).
///
/// Applications that accept a base from untrusted input can check it against this
/// constant before constructing a tree.
pub const MIN_BASE: f64 = 1.1;

/// Panics unless `base` is finite and at least [`MIN_BASE`].
pub(crate) fn validate_base(base: f64) {
    assert!(
        base.is_finite() && base >= MIN_BASE,
        "cover tree base must be a finite number >= {} (got {})",
        MIN_BASE,
        base
    );
}

/// Smallest level whose covering distance `base^level` is at least `dist`.
///
/// # Panics
///
/// Panics if `dist` is not finite and positive, which means the distance metric
/// returned an invalid value (NaN, infinity, or a non-positive distance where a
/// strictly positive one is required).
pub(crate) fn level_for_distance(dist: f64, base: f64) -> i32 {
    assert!(
        dist.is_finite() && dist > 0.0,
        "distance metric returned an invalid distance: {}",
        dist
    );
    // With base >= MIN_BASE and a finite positive dist, |level| < 8,000, so the
    // float-to-int cast never saturates.
    (dist.ln() / base.ln()).ceil() as i32
}

/// Most exact copies of a point that one node holds as direct children.
///
/// Exact duplicates cannot be separated by distance, so without a limit every copy
/// would either chain one level deeper (deep recursion) or hang off a single node
/// (huge fan-out, which makes dual-tree traversal quadratic in time and memory).
/// Copies beyond this many spill into the child copy with the fewest children,
/// keeping groups of duplicates balanced: depth grows as log8 of the copy count.
pub(crate) const DUPLICATE_FANOUT: usize = 8;

/// Where to put a new exact copy of `node.point`: `None` to attach it as a new leaf
/// child of `node`, or `Some(i)` to insert it under child `i`, itself a copy.
///
/// Callers must move the chosen child to the end of `node.children` so that ties
/// between equally full copies are broken round-robin; otherwise the first copy's
/// branch would absorb every insert and grow linearly deep.
pub(crate) fn place_duplicate<T: Clone, D: crate::Distance<T>>(
    node: &Node<T>,
    point: &T,
    metric: &D,
) -> Option<usize> {
    let mut copies = 0;
    let mut emptiest: Option<(usize, usize)> = None; // (child index, its child count)
    for (i, child) in node.children.iter().enumerate() {
        if metric.distance(&child.point, point) == 0.0 {
            copies += 1;
            if emptiest.map_or(true, |(_, n)| child.children.len() < n) {
                emptiest = Some((i, child.children.len()));
            }
        }
    }
    if copies < DUPLICATE_FANOUT { None } else { emptiest.map(|(i, _)| i) }
}

/// Shift needed to place an old root directly below a new root at `new_level`.
///
/// # Panics
///
/// Panics if the shift does not fit in an `i32`. This cannot happen with a base of at
/// least [`MIN_BASE`]; the check guards against silent wraparound in release builds.
pub(crate) fn root_level_adjustment(new_level: i32, old_level: i32) -> i32 {
    new_level
        .checked_sub(1)
        .and_then(|l| l.checked_sub(old_level))
        .expect("cover tree level adjustment overflowed i32")
}

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
    node.level = node
        .level
        .checked_add(adjustment)
        .expect("cover tree level overflowed i32");
    for child in &mut node.children {
        adjust_levels(child, adjustment);
    }
}
