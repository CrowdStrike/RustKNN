//! Insertion logic for Simplified Cover Trees (Algorithm 2)
//!
//! This module implements the simplified cover tree insertion algorithm from the paper
//! "Faster Cover Trees" (ICML 2015). The simplified variant maintains exactly n nodes
//! for n points and does not perform rebalancing.
//!
//! # Algorithm Overview
//!
//! Algorithm 2 (Simplified Insertion) works as follows:
//! 1. Find the first child that can accommodate the point (within its covering distance)
//! 2. If found, recursively insert into that child
//! 3. If not found, add the point as a new child at level (parent.level - 1)
//!
//! # Key Differences from Nearest Ancestor
//!
//! - **No rebalancing**: Points stay where they're first inserted
//! - **Simpler logic**: Just find first suitable child or add as new child
//! - **Faster construction**: No restructuring overhead
//! - **Slightly slower queries**: May have suboptimal tree structure

use crate::distance::Distance;
use crate::node::Node;

/// Internal implementation of simplified cover tree insertion.
///
/// This struct provides the recursive insertion logic for the simplified variant.
/// It's used by SimplifiedCoverTree but kept separate for modularity.
pub(super) struct InsertImpl;

impl InsertImpl {
    /// Recursive insertion that returns a pointer to the inserted point's storage.
    ///
    /// This is used by `SimplifiedCoverTree::insert_returning_ptr` to track which
    /// tree node owns each inserted point, enabling O(m) result mapping in batch
    /// k-NN instead of O(m²) greedy matching.
    ///
    /// # Algorithm Steps
    ///
    /// 1. **Find candidate**: Check each child to see if d(child, point) <= covdist(child)
    /// 2. **Recurse if found**: If a candidate exists, recurse into it
    /// 3. **Add as child if not**: If no candidate, add point as new child at parent.level - 1
    ///
    /// # Invariants Maintained
    ///
    /// - **Leveling**: New children are at parent.level - 1
    /// - **Covering**: New children satisfy d(parent, child) <= covdist(parent)
    /// - **Separating**: Maintained because we only add new children when no existing child
    ///   can cover the point (implicit separation)
    ///
    /// # Time Complexity
    ///
    /// O(c^6 log n) where c is the doubling constant, per the paper's analysis.
    pub(super) fn insert_recursive_returning_ptr<T: Clone, D: Distance<T>>(
        parent: &mut Node<T>,
        point: T,
        metric: &D,
        base: f64,
    ) -> *const T {
        let _guard = crate::core::utils::StackGuard::enter();
        // Calculate distance from parent to point (we'll need it)
        let dist = metric.distance(&parent.point, &point);

        // Exact duplicates of this node's point go into a balanced group of copies
        // below it (see `DUPLICATE_FANOUT`). Ordinary descent would put every further
        // copy one level deeper, and a long run of copies would make the tree deep
        // enough for recursive operations to overflow the stack.
        if dist == 0.0 {
            if let Some(i) = crate::core::utils::place_duplicate(parent, &point, metric) {
                let ptr = Self::insert_recursive_returning_ptr(&mut parent.children[i], point, metric, base);
                // Rotate the chosen copy to the back so ties go round-robin and the
                // group fills evenly. Boxed nodes don't move, so `ptr` stays valid.
                let chosen = parent.children.remove(i);
                parent.children.push(chosen);
                return ptr;
            }
            let mut duplicate = Node::new(point, parent.level - 1, false);
            duplicate.d_parent = 0.0;
            parent.children.push(Box::new(duplicate));
            return &parent.children.last().unwrap().point;
        }

        // Find first child that can accommodate this point
        // A child can accommodate if d(child, point) <= covdist(child)
        let mut found: Option<(usize, f64)> = None;
        for i in 0..parent.children.len() {
            let child = &parent.children[i];
            let child_dist = metric.distance(&child.point, &point);
            if child_dist <= child.covdist(base) {
                found = Some((i, child_dist));
                break;
            }
        }

        if let Some((chosen_idx, _)) = found {
            let ptr = Self::insert_recursive_returning_ptr(&mut parent.children[chosen_idx], point, metric, base);
            // The new point is now a descendant of `parent`; its distance to the parent
            // was computed above, so maxdist stays a valid bound on all descendants.
            if dist > parent.maxdist {
                parent.maxdist = dist;
            }
            return ptr;
        }

        // No child can accommodate this point - add it as a new child
        // New child is at parent.level - 1
        let child_level = parent.level - 1;
        let mut new_child = Node::new(point, child_level, false);
        // Cache the distance from parent to this child for dual-tree traversal
        new_child.d_parent = dist;

        // Update parent's maxdist before adding child
        parent.update_maxdist_approx(dist, new_child.maxdist);

        // Add as child
        parent.children.push(Box::new(new_child));

        // Get pointer to the newly inserted point
        let ptr: *const T = &parent.children.last().unwrap().point;

        // Debug assertion: verify leveling invariant
        debug_assert_eq!(parent.children.last().unwrap().level, parent.level - 1,
            "Leveling invariant violated: child level should be parent.level - 1");

        // Debug assertion: verify covering invariant
        debug_assert!(dist <= parent.covdist(base),
            "Covering invariant violated: child too far from parent");

        ptr
    }
}
