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
            Self::insert_recursive(tree, point, metric, base)
        }
    }

    /// Creates a new root at a higher level to accommodate a far point.
    fn create_raised_root<T: Clone>(
        mut old_tree: Box<Node<T>>,
        point: T,
        dist_to_root: f64,
        base: f64
    ) -> Box<Node<T>> {
        let new_level = (dist_to_root.ln() / base.ln()).ceil() as i32;

        let adjustment = (new_level - 1) - old_tree.level;
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
    fn insert_recursive<T: Clone, D: Distance<T>>(
        mut parent: Box<Node<T>>,
        point: T,
        metric: &D,
        base: f64
    ) -> Box<Node<T>> {
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
            let result = Self::insert_recursive(child, point, metric, base);

            let result_maxdist = result.maxdist;
            parent.update_maxdist_approx(best_child_dist, result_maxdist);

            parent.children.push(result);
            return parent;
        }

        // No child can accommodate - rebalance with subtree reattachment
        RebalanceImpl::rebalance(parent, point, metric, base)
    }
}
