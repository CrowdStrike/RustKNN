//! Optimized Insertion with Subtree Reattachment Rebalancing
//!
//! This module uses the subtree reattachment rebalancing strategy for 2-15x speedup.

use crate::distance::Distance;
use crate::node::Node;
use super::rebalance::RebalanceImpl;

/// Implementation of optimized insertion with subtree reattachment.
pub(super) struct InsertImpl;

impl InsertImpl {
    /// Main insertion function with subtree reattachment rebalancing.
    pub(super) fn insert_internal<T: Clone, D: Distance<T>>(
        tree: Box<Node<T>>,
        point: T,
        metric: &D,
        base: f64
    ) -> Box<Node<T>> {
        let dist_to_root = metric.distance(&tree.point, &point);

        if dist_to_root > tree.covdist(base) {
            Self::create_raised_root(tree, point, dist_to_root, base)
        } else {
            Self::insert_recursive(tree, point, dist_to_root, metric, base)
        }
    }

    /// Creates a new root at a higher level to accommodate a far point.
    fn create_raised_root<T: Clone>(
        mut old_tree: Box<Node<T>>,
        point: T,
        dist_to_root: f64,
        base: f64
    ) -> Box<Node<T>> {
        let new_level = crate::core::utils::level_for_distance(dist_to_root, base);

        let adjustment = crate::core::utils::root_level_adjustment(new_level, old_tree.level);
        if adjustment != 0 {
            crate::core::utils::adjust_levels(&mut old_tree, adjustment);
        }

        let mut new_root = Node::new(point, new_level, false);
        new_root.update_maxdist_approx(dist_to_root, old_tree.maxdist);
        // Cache d_parent for old_tree (now a child of new_root)
        old_tree.d_parent = dist_to_root;
        new_root.children.push(old_tree);

        Box::new(new_root)
    }

    /// Recursive insertion with subtree reattachment rebalancing.
    ///
    /// `dist_to_parent` is `d(parent.point, point)`, already computed by the caller.
    fn insert_recursive<T: Clone, D: Distance<T>>(
        mut parent: Box<Node<T>>,
        point: T,
        dist_to_parent: f64,
        metric: &D,
        base: f64
    ) -> Box<Node<T>> {
        let _guard = crate::core::utils::StackGuard::enter();
        // Exact duplicates of this node's point go into a balanced group of copies
        // below it (see `DUPLICATE_FANOUT`). Ordinary descent would put every further
        // copy one level deeper, and a long run of copies would make the tree deep
        // enough for recursive operations to overflow the stack.
        if dist_to_parent == 0.0 {
            if let Some(i) = crate::core::utils::place_duplicate(&parent, &point, metric) {
                // Rotate the chosen copy to the back so ties go round-robin.
                let child = parent.children.remove(i);
                let child = Self::insert_recursive(child, point, 0.0, metric, base);
                parent.children.push(child);
                return parent;
            }
            let level = parent.level - 1;
            parent.children.push(Box::new(Node::new(point, level, false)));
            return parent;
        }

        // Linear scan to find closest child that can accommodate
        let mut best_child_idx = None;
        let mut best_child_dist = f64::INFINITY;

        for (i, child) in parent.children.iter().enumerate() {
            let dist = metric.distance(&child.point, &point);

            if dist <= child.covdist(base) && dist < best_child_dist {
                best_child_dist = dist;
                best_child_idx = Some(i);
            }
        }

        if let Some(idx) = best_child_idx {
            // Found a child that can accommodate - recurse
            let child = parent.children.swap_remove(idx);
            let child_level = child.level;
            let mut result = Self::insert_recursive(child, point, best_child_dist, metric, base);
            if result.level != child_level {
                // Rebalancing re-inserted points into the child's subtree and raised it
                // under a new root point; refresh the cached distance to this parent.
                // Traversals treat d_parent == 0 as "same point as the parent".
                result.d_parent = metric.distance(&parent.point, &result.point);
            }

            // The new point is now a descendant of `parent`. Rebalancing below only
            // moves points within the child's subtree, so covering the new point keeps
            // maxdist a valid bound on all descendants.
            if dist_to_parent > parent.maxdist {
                parent.maxdist = dist_to_parent;
            }

            parent.children.push(result);
            return parent;
        }

        // No child can accommodate - rebalance with subtree reattachment
        RebalanceImpl::rebalance(parent, point, metric, base)
    }
}
