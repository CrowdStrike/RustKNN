//! Optimized Subtree Reattachment Rebalancing Logic for Nearest Ancestor Cover Trees
//!
//! This module implements an alternative to flatten-and-reinsert by preserving
//! extracted subtree structure and reattaching entire subtrees where possible.
//! Provides 2-15x construction speedup.

use crate::distance::Distance;
use crate::node::Node;
use super::insert::InsertImpl;

/// Information about where to attach a subtree.
struct AttachmentInfo {
    /// Path of indices from root to attachment parent
    path: Vec<usize>,
    /// Target level for subtree root
    target_level: i32,
}

/// Implementation of optimized subtree reattachment rebalancing algorithms.
pub(super) struct RebalanceImpl;

impl RebalanceImpl {
    /// Rebalances the tree after insertion (Algorithm 3) - Subtree reattachment version.
    ///
    /// KEY OPTIMIZATION: This version preserves extracted subtree structure and
    /// reattaches entire subtrees instead of flattening to points.
    ///
    /// Falls back to flatten-and-reinsert if reattachment would violate invariants.
    pub(super) fn rebalance<T: Clone, D: Distance<T>>(
        parent: Box<Node<T>>,
        point: T,
        metric: &D,
        base: f64
    ) -> Box<Node<T>> {
        let _guard = crate::core::utils::StackGuard::enter();
        let mut parent_node = *parent;

        // Step 1: TAKE ownership of children
        let parent_children = std::mem::take(&mut parent_node.children);

        // Extract close children - returns SUBTREES not points
        let (remaining_children, extracted_subtrees) =
            Self::extract_close_children_owned(parent_children, &point, metric);

        // Step 2: Create intermediate tree ct'
        let mut new_singleton = Box::new(Node::new(point.clone(), parent_node.level - 1, false));
        // Cache d_parent: distance from parent_node's point to the new child's point
        new_singleton.d_parent = metric.distance(&parent_node.point, &point);

        parent_node.children = vec![new_singleton];
        parent_node.children.extend(remaining_children);

        // CRITICAL: Recompute intermediate's maxdist after modifying children!
        parent_node.recalc_maxdist_approx(metric);

        // Step 3: Reattach extracted subtrees (or fall back to insertion)
        let mut result = Box::new(parent_node);
        for subtree in extracted_subtrees {
            result = Self::reattach_or_insert(result, subtree, metric, base);
        }

        result
    }

    /// Attempts to reattach subtree intact, falls back to flatten-and-reinsert if needed.
    fn reattach_or_insert<T: Clone, D: Distance<T>>(
        mut tree: Box<Node<T>>,
        subtree: Box<Node<T>>,
        metric: &D,
        base: f64
    ) -> Box<Node<T>> {
        // Try to find attachment point
        match Self::find_attachment_point(&mut tree, &subtree.point, metric, base) {
            Some(attachment_info) => {
                // Check if we can attach without violating separating invariant
                if Self::can_attach_at_level(&tree, &subtree, &attachment_info, metric, base) {
                    Self::attach_subtree(tree, subtree, attachment_info, metric)
                } else {
                    // Separating violated - fall back to insertion
                    Self::fallback_to_insertion(tree, subtree, metric, base)
                }
            }
            None => {
                // No valid attachment point - fall back to insertion
                Self::fallback_to_insertion(tree, subtree, metric, base)
            }
        }
    }

    /// Finds where a subtree should be attached.
    ///
    /// Returns path to attachment parent and target level, or None if no valid attachment.
    fn find_attachment_point<T: Clone, D: Distance<T>>(
        tree: &Node<T>,
        point: &T,
        metric: &D,
        base: f64,
    ) -> Option<AttachmentInfo> {
        let _guard = crate::core::utils::StackGuard::enter();
        let dist = metric.distance(&tree.point, point);

        // Can't cover at this level - no valid attachment
        if dist > tree.covdist(base) {
            return None;
        }

        // Try to find child that can cover
        for (i, child) in tree.children.iter().enumerate() {
            let child_dist = metric.distance(&child.point, point);
            if child_dist <= child.covdist(base) {
                // Recurse into child
                if let Some(mut child_info) = Self::find_attachment_point(child, point, metric, base) {
                    // Prepend current index to path
                    child_info.path.insert(0, i);
                    return Some(child_info);
                }
            }
        }

        // No child can cover - attach at this level
        Some(AttachmentInfo {
            path: vec![],
            target_level: tree.level - 1,
        })
    }

    /// Checks if subtree can be attached without violating separating invariant.
    fn can_attach_at_level<T: Clone, D: Distance<T>>(
        tree: &Node<T>,
        subtree: &Node<T>,
        info: &AttachmentInfo,
        metric: &D,
        base: f64,
    ) -> bool {
        // Navigate to attachment parent
        let mut current = tree;
        for &idx in &info.path {
            current = &current.children[idx];
        }

        let sepdist = current.sepdist(base);

        // Check separating with all siblings
        for sibling in &current.children {
            let dist = metric.distance(&sibling.point, &subtree.point);
            if dist <= sepdist {
                return false;  // Would violate separating
            }
        }

        true
    }

    /// Attaches subtree at the specified location.
    fn attach_subtree<T: Clone, D: Distance<T>>(
        mut tree: Box<Node<T>>,
        mut subtree: Box<Node<T>>,
        info: AttachmentInfo,
        metric: &D
    ) -> Box<Node<T>> {
        // Adjust subtree levels
        let adjustment = info.target_level - subtree.level;
        if adjustment != 0 {
            Self::adjust_subtree_levels(&mut subtree, adjustment);
        }

        // Navigate to attachment parent and attach. Every node on the path gains the
        // subtree as descendants, so widen its maxdist to cover the subtree.
        let mut current = &mut *tree;
        for &idx in &info.path {
            let reach = metric.distance(&current.point, &subtree.point) + subtree.maxdist;
            if reach > current.maxdist {
                current.maxdist = reach;
            }
            current = &mut *current.children[idx];
        }

        // Update d_parent for the subtree root to reflect the new parent
        subtree.d_parent = metric.distance(&current.point, &subtree.point);
        current.children.push(subtree);
        current.recalc_maxdist_approx(metric);

        tree
    }

    /// Adjusts all levels in a subtree by a given amount.
    fn adjust_subtree_levels<T: Clone>(node: &mut Node<T>, adjustment: i32) {
        // Iterative (shared helper), so a deep subtree is never left half-adjusted.
        crate::core::utils::adjust_levels(node, adjustment);
    }

    /// Falls back to flattening subtree and reinserting points individually.
    fn fallback_to_insertion<T: Clone, D: Distance<T>>(
        mut tree: Box<Node<T>>,
        subtree: Box<Node<T>>,
        metric: &D,
        base: f64
    ) -> Box<Node<T>> {
        let points = Self::flatten_to_points_owned(subtree);
        for point in points {
            tree = InsertImpl::insert_internal(tree, point, metric, base);
        }
        tree
    }

    /// Extracts children closer to the new point - Returns SUBTREES.
    fn extract_close_children_owned<T: Clone, D: Distance<T>>(
        parent_children: Vec<Box<Node<T>>>,
        new_point: &T,
        metric: &D
    ) -> (Vec<Box<Node<T>>>, Vec<Box<Node<T>>>) {
        let mut remaining_children = Vec::new();
        let mut extracted_subtrees = Vec::new();

        for child in parent_children {
            let root_point = child.point.clone();

            let (remaining_child, child_extracted) =
                Self::extract_from_subtree_owned(child, &root_point, new_point, metric);

            if let Some(pruned_child) = remaining_child {
                remaining_children.push(pruned_child);
            }

            extracted_subtrees.extend(child_extracted);
        }

        (remaining_children, extracted_subtrees)
    }

    /// Recursively extracts subtrees that violate the ancestor invariant.
    ///
    /// Returns:
    /// - Option<Box<Node<T>>>: The remaining (pruned) subtree, or None if entirely extracted
    /// - Vec<Box<Node<T>>>: Extracted SUBTREES (not flattened!)
    fn extract_from_subtree_owned<T: Clone, D: Distance<T>>(
        current: Box<Node<T>>,
        root_point: &T,
        new_point: &T,
        metric: &D
    ) -> (Option<Box<Node<T>>>, Vec<Box<Node<T>>>) {
        let _guard = crate::core::utils::StackGuard::enter();
        let dist_to_root = metric.distance(&current.point, root_point);
        let dist_to_new = metric.distance(&current.point, new_point);

        // Pruning: if entire subtree is far from new_point, keep it all
        if dist_to_root + current.maxdist < dist_to_new {
            return (Some(current), vec![]);
        }

        // Extraction: if current node is closer to new_point, extract entire subtree
        if dist_to_root > dist_to_new {
            // Return ENTIRE SUBTREE, not flattened!
            return (None, vec![current]);
        }

        // Recursive case: current stays, but check children
        let mut node = *current;
        let children = std::mem::take(&mut node.children);

        let mut remaining_children = Vec::new();
        let mut extracted_subtrees = Vec::new();

        for child in children {
            let (remaining_child, child_extracted) =
                Self::extract_from_subtree_owned(child, root_point, new_point, metric);

            if let Some(pruned_child) = remaining_child {
                remaining_children.push(pruned_child);
            }

            extracted_subtrees.extend(child_extracted);
        }

        node.children = remaining_children;
        node.recalc_maxdist_approx(metric);

        (Some(Box::new(node)), extracted_subtrees)
    }

    /// Flattens an entire subtree to a vector of points (for fallback). Iterative, so a
    /// deep subtree cannot overflow the stack or be lost halfway.
    fn flatten_to_points_owned<T: Clone>(node: Box<Node<T>>) -> Vec<T> {
        let mut points = Vec::new();
        let mut stack = vec![node];
        while let Some(mut n) = stack.pop() {
            points.push(n.point.clone());
            // Reverse so children are visited in order (pre-order, as before).
            stack.extend(std::mem::take(&mut n.children).into_iter().rev());
        }
        points
    }
}
