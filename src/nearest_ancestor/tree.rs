//! Nearest Ancestor Cover Tree with Subtree Reattachment
//!
//! This implementation uses subtree reattachment during rebalancing for optimal performance.

use crate::distance::Distance;
use crate::node::Node;
use super::insert::InsertImpl;

/// A Nearest Ancestor Cover Tree using subtree reattachment.
///
/// This implementation preserves extracted subtree structure during rebalancing
/// and reattaches entire subtrees instead of flattening to points, providing
/// significant construction speedup while maintaining query performance.
#[derive(Debug)]
pub struct NACoverTree<T: Clone, D: Distance<T>> {
    pub(super) root: Option<Box<Node<T>>>,
    pub(super) metric: D,
    pub(super) base: f64,
    size: usize,
}

impl<T: Clone, D: Distance<T>> NACoverTree<T, D> {
    /// Creates a new empty nearest ancestor cover tree.
    ///
    /// # Panics
    ///
    /// Panics if `base` is not finite or is below [`MIN_BASE`](crate::MIN_BASE).
    pub fn new(metric: D, base: f64) -> Self {
        crate::core::utils::validate_base(base);
        NACoverTree {
            root: None,
            metric,
            base,
            size: 0,
        }
    }

    /// Returns the number of points in the tree.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Returns true if the tree contains no points.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Returns a reference to the root node, if it exists.
    pub fn root_node(&self) -> Option<&Node<T>> {
        self.root.as_deref()
    }

    /// Returns a reference to the distance metric.
    pub fn metric(&self) -> &D {
        &self.metric
    }

    /// Returns the base value used by this tree.
    pub fn base_value(&self) -> f64 {
        self.base
    }

    /// Inserts a point into the cover tree using subtree reattachment.
    pub fn insert(&mut self, point: T) {
        self.root = match self.root.take() {
            None => {
                Some(Box::new(Node::new(point, 0, false)))
            }
            Some(root) => {
                Some(InsertImpl::insert_internal(root, point, &self.metric, self.base))
            }
        };
        self.size += 1;
    }

    /// Recomputes exact maxdist for all nodes in the tree.
    ///
    /// The NA tree's incremental maxdist updates during rebalancing use the
    /// triangle inequality approximation, which can accumulate slack over many
    /// insertions. Calling this after construction replaces all approximate
    /// maxdist values with exact ones, improving query pruning.
    pub fn recompute_maxdist(&mut self) {
        if let Some(ref mut root) = self.root {
            Self::recompute_maxdist_recursive(root, &self.metric);
        }
    }

    /// Finds the nearest neighbor to a query point.
    pub fn find_nearest(&self, query: &T) -> Option<&T> {
        use crate::core::query;

        self.root.as_ref()?;

        let root = self.root.as_ref().unwrap();

        let (best_point, _best_dist) = query::find_nearest_internal(
            root,
            query,
            &root.point,
            self.metric.distance(&root.point, query),
            &self.metric
        );

        Some(best_point)
    }

    /// Find k nearest neighbors of a query point.
    ///
    /// Returns up to k nearest neighbors as `(point, distance)` tuples, sorted by distance (ascending).
    ///
    /// # Arguments
    ///
    /// * `query` - The query point to search for
    /// * `k` - Number of nearest neighbors to find
    ///
    /// # Returns
    ///
    /// * `Vec<(&T, f64)>` - Vector of up to k nearest neighbors as (point reference, distance) pairs
    ///   - Sorted by distance (closest first)
    ///   - Length ≤ min(k, tree.len())
    ///   - Returns empty vector if k=0 or tree is empty
    ///
    /// # Deduplication
    ///
    /// Algorithm-created duplicate nodes (from tree merging/level alignment) are automatically
    /// skipped. User's intentional duplicate values are correctly returned.
    ///
    /// # Time Complexity
    ///
    /// O(c^6 log n) where c is the doubling constant and n is the number of points.
    /// Same asymptotic complexity as single nearest neighbor search.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustknn::NACoverTree;
    /// use rustknn::Distance;
    ///
    /// struct EuclideanDistance;
    /// impl Distance<f64> for EuclideanDistance {
    ///     fn distance(&self, p: &f64, q: &f64) -> f64 {
    ///         (p - q).abs()
    ///     }
    /// }
    ///
    /// let mut tree = NACoverTree::new(EuclideanDistance, 1.3);
    /// tree.insert(1.0);
    /// tree.insert(5.0);
    /// tree.insert(10.0);
    /// tree.insert(15.0);
    ///
    /// // Find 3 nearest neighbors of 6.0
    /// let neighbors = tree.find_k_nearest(&6.0, 3);
    /// assert_eq!(neighbors.len(), 3);
    ///
    /// // Should be: 5.0 (dist=1), 1.0 (dist=5), 10.0 (dist=4) -> sorted: 5.0, 10.0, 1.0
    /// assert_eq!(*neighbors[0].0, 5.0);
    /// assert!((neighbors[0].1 - 1.0).abs() < 1e-10);
    /// ```
    pub fn find_k_nearest(&self, query: &T, k: usize) -> Vec<(&T, f64)> {
        use crate::knn::KnnState;
        use crate::core::knn_impl::KnnImpl;

        if k == 0 {
            return Vec::new();
        }

        let Some(root) = self.root.as_deref() else {
            return Vec::new();
        };

        let mut state = KnnState::new(k);
        KnnImpl::find_k_nearest_internal(root, query, &mut state, &self.metric);
        state.into_sorted_vec()
    }

    /// Batch k-NN: builds a query tree internally and uses dual-tree traversal.
    ///
    /// For each query point, finds up to k nearest neighbors in this tree. Uses
    /// dual-tree traversal to prune entire subtree combinations, dramatically
    /// reducing distance computations compared to running single-tree k-NN for
    /// each query independently.
    ///
    /// # Arguments
    ///
    /// * `queries` - Slice of query points
    /// * `k` - Number of nearest neighbors to find per query point
    ///
    /// # Returns
    ///
    /// `Vec<Vec<(&T, f64)>>` — outer Vec has one entry per query point (same order
    /// as input), inner Vec contains up to k neighbors sorted by distance (ascending).
    pub fn find_k_nearest_batch(&self, queries: &[T], k: usize) -> Vec<Vec<(&T, f64)>>
    where
        D: Clone,
    {
        use std::collections::HashMap;
        use crate::core::dual_tree::knn_rules::KnnRules;
        use crate::core::dual_tree::traversal::DualTreeTraversal;
        use crate::simplified::SimplifiedCoverTree;

        if k == 0 || queries.is_empty() || self.root.is_none() {
            return vec![Vec::new(); queries.len()];
        }

        // Build a temporary query tree, tracking which pointer belongs to which input index.
        // insert_returning_ptr gives us O(1) per insert to record the mapping.
        let mut query_tree = SimplifiedCoverTree::new(self.metric.clone(), self.base);
        let mut ptr_to_input: HashMap<*const T, Vec<usize>> = HashMap::new();
        for (qi, q) in queries.iter().enumerate() {
            let ptr = query_tree.insert_returning_ptr(q.clone());
            ptr_to_input.entry(ptr).or_default().push(qi);
        }
        // Recompute exact maxdist on the query tree so dual-tree pruning bounds
        // are tight. Without this, approximate maxdist from incremental insertion
        // can cause the traversal to over-prune, missing valid neighbors.
        query_tree.recompute_maxdist();

        let query_root = match query_tree.root_node() {
            Some(r) => r,
            None => return vec![Vec::new(); queries.len()],
        };
        let ref_root = self.root.as_deref().unwrap();

        // Run dual-tree traversal
        let mut rules = KnnRules::new(k, &self.metric, false);
        rules.state.init_parent_map(query_root);
        DualTreeTraversal::traverse(query_root, ref_root, &mut rules);

        // Collect results as Vec indexed by integer index
        let (mut results_vec, query_state_indices, query_ptrs) = rules.state.into_results();

        // Map results to input order using the ptr_to_input index (O(m) total).
        let mut output: Vec<Vec<(&T, f64)>> = vec![Vec::new(); queries.len()];
        for (qi, &ptr) in query_ptrs.iter().enumerate() {
            let idx = query_state_indices[qi];
            let results = std::mem::take(&mut results_vec[idx]);
            if let Some(input_indices) = ptr_to_input.get(&ptr) {
                if input_indices.len() == 1 {
                    output[input_indices[0]] = results; // move, no clone
                } else {
                    for &qi in &input_indices[..input_indices.len() - 1] {
                        output[qi] = results.clone();
                    }
                    output[*input_indices.last().unwrap()] = results; // move last
                }
            }
        }

        // Fill unmatched queries (could happen if query tree deduplicated them)
        // by running single-tree k-NN as fallback
        for (i, result) in output.iter_mut().enumerate() {
            if result.is_empty() {
                *result = self.find_k_nearest(&queries[i], k);
            }
        }

        output
    }

    /// Dual-tree k-NN with a user-supplied query tree.
    ///
    /// Uses the provided query tree and this tree (as reference) for dual-tree
    /// k-NN search. Returns results for each non-duplicate point in the query tree
    /// in DFS order.
    ///
    /// # Arguments
    ///
    /// * `query_tree` - The query cover tree (NACoverTree)
    /// * `k` - Number of nearest neighbors to find per query point
    ///
    /// # Returns
    ///
    /// `Vec<Vec<(&T, f64)>>` — one entry per non-duplicate point in the query tree
    /// (DFS order), each containing up to k neighbors sorted by distance.
    ///
    /// # Panics
    ///
    /// Panics if the query tree and reference tree have different base values.
    pub fn find_k_nearest_dual(
        &self,
        query_tree: &NACoverTree<T, D>,
        k: usize,
    ) -> Vec<Vec<(&T, f64)>> {
        use crate::core::dual_tree::knn_rules::KnnRules;
        use crate::core::dual_tree::traversal::DualTreeTraversal;

        assert_eq!(
            self.base, query_tree.base,
            "Cannot run dual-tree k-NN with different base values: {} != {}",
            self.base, query_tree.base
        );

        if k == 0 || self.root.is_none() || query_tree.root.is_none() {
            return Vec::new();
        }

        let query_root = query_tree.root.as_deref().unwrap();
        let ref_root = self.root.as_deref().unwrap();

        let mut rules = KnnRules::new(k, &self.metric, false);
        rules.state.init_parent_map(query_root);
        DualTreeTraversal::traverse(query_root, ref_root, &mut rules);

        let (mut results_vec, query_state_indices, _query_ptrs) = rules.state.into_results();

        query_state_indices
            .iter()
            .map(|&idx| std::mem::take(&mut results_vec[idx]))
            .collect()
    }

    /// Same-set k-NN: find k nearest neighbors for each point in the tree itself.
    ///
    /// Self-matches are excluded (a point is not its own neighbor). Uses dual-tree
    /// traversal with the same tree as both query and reference.
    ///
    /// # Arguments
    ///
    /// * `k` - Number of nearest neighbors to find per point
    ///
    /// # Returns
    ///
    /// `Vec<Vec<(&T, f64)>>` — one entry per non-duplicate point in the tree
    /// (DFS order), each containing up to k neighbors sorted by distance.
    pub fn find_k_nearest_self(&self, k: usize) -> Vec<Vec<(&T, f64)>> {
        use crate::core::dual_tree::knn_rules::KnnRules;
        use crate::core::dual_tree::traversal::DualTreeTraversal;

        if k == 0 || self.root.is_none() {
            return Vec::new();
        }

        let root = self.root.as_deref().unwrap();

        // Full Curtin B1/B2 bounds: a query node's bound must hold for every query
        // point below it, not just its own point, even when query = reference.
        let mut rules = KnnRules::new(k, &self.metric, true);
        rules.state.init_parent_map(root);
        DualTreeTraversal::traverse(root, root, &mut rules);

        let (mut results_vec, query_state_indices, _query_ptrs) = rules.state.into_results();

        query_state_indices
            .iter()
            .map(|&idx| std::mem::take(&mut results_vec[idx]))
            .collect()
    }

    /// Convert this NACoverTree into a SimplifiedCoverTree.
    ///
    /// Transfers the internal node structure directly, then recomputes maxdist
    /// values (O(n)). This is valid because NACoverTree's invariants are a
    /// superset of SimplifiedCoverTree's — every valid NACoverTree is also a
    /// valid SimplifiedCoverTree.
    pub fn into_simplified(self) -> crate::simplified::SimplifiedCoverTree<T, D> {
        let mut tree = crate::simplified::SimplifiedCoverTree::from_parts(
            self.root, self.metric, self.base, self.size,
        );
        tree.recompute_maxdist();
        tree
    }

    /// Merge two cover trees, consuming both and returning a SimplifiedCoverTree
    ///
    /// **IMPORTANT**: Merging two NACoverTrees does NOT produce a tree that maintains
    /// the nearest ancestor invariant. The merge algorithm only maintains the three
    /// basic cover tree invariants (leveling, covering, separating). Therefore, this
    /// method returns a SimplifiedCoverTree, not a NACoverTree.
    ///
    /// If you need a NACoverTree after merging, you must rebuild by inserting all
    /// points into a new NACoverTree, which will trigger proper rebalancing.
    ///
    /// # Arguments
    ///
    /// * `other` - The tree to merge with this one
    /// * `randomize` - Whether to randomize child order during merge (default: false)
    ///
    /// # Returns
    ///
    /// A SimplifiedCoverTree containing all points from both trees, with exact maxdist values
    ///
    /// # Panics
    ///
    /// Panics if the base values of the two trees differ
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use rustknn::nearest_ancestor::NACoverTree;
    /// use rustknn::simplified::SimplifiedCoverTree;
    /// use rustknn::Distance;
    ///
    /// struct SimpleDistance;
    /// impl Distance<f64> for SimpleDistance {
    ///     fn distance(&self, p: &f64, q: &f64) -> f64 {
    ///         (p - q).abs()
    ///     }
    /// }
    ///
    /// let mut tree1 = NACoverTree::new(SimpleDistance, 1.3);
    /// tree1.insert(1.0);
    /// tree1.insert(2.0);
    ///
    /// let mut tree2 = NACoverTree::new(SimpleDistance, 1.3);
    /// tree2.insert(10.0);
    /// tree2.insert(20.0);
    ///
    /// let merged: SimplifiedCoverTree<_, _> = tree1.merge(tree2);
    /// assert_eq!(merged.len(), 4);
    /// ```
    pub fn merge(self, other: Self) -> crate::simplified::SimplifiedCoverTree<T, D> {
        use crate::simplified::merge::MergeImpl;

        assert_eq!(
            self.base, other.base,
            "Cannot merge trees with different base values: {} != {}",
            self.base, other.base
        );

        // Handle empty tree cases
        if self.root.is_none() {
            return crate::simplified::SimplifiedCoverTree::from_parts(
                other.root,
                other.metric,
                other.base,
                other.size,
            );
        }
        if other.root.is_none() {
            return crate::simplified::SimplifiedCoverTree::from_parts(
                self.root,
                self.metric,
                self.base,
                self.size,
            );
        }

        // Merge the roots using SimplifiedCoverTree's merge algorithm
        let merged_root = MergeImpl::merge(
            self.root.unwrap(),
            other.root.unwrap(),
            &self.metric,
            self.base,
        );

        // Create the merged SimplifiedCoverTree
        let mut merged_tree = crate::simplified::SimplifiedCoverTree::from_parts(
            Some(merged_root),
            self.metric,
            self.base,
            self.size + other.size,
        );

        // CRITICAL: Call recompute_maxdist to get exact maxdist values
        // The merge process uses triangle inequality (approximation), so we need
        // to compute exact maxdist values for optimal query performance
        merged_tree.recompute_maxdist();

        merged_tree
    }

    /// Recursive helper for exact maxdist computation.
    fn recompute_maxdist_recursive(node: &mut Node<T>, metric: &D) -> f64 {
        if node.children.is_empty() {
            node.maxdist = 0.0;
            return 0.0;
        }

        // Post-order: recursively compute children's maxdist first
        for child in &mut node.children {
            Self::recompute_maxdist_recursive(child, metric);
        }

        // Compute exact maxdist for this node
        let mut max_dist = 0.0;

        for child in &node.children {
            let dist_to_child = metric.distance(&node.point, &child.point);

            if dist_to_child > max_dist {
                max_dist = dist_to_child;
            }

            // Branch-and-bound pruning
            if dist_to_child + child.maxdist <= max_dist {
                continue;
            }

            let max_via_child = Self::compute_max_descendant_dist(
                node,
                child,
                metric,
                max_dist
            );

            if max_via_child > max_dist {
                max_dist = max_via_child;
            }
        }

        node.maxdist = max_dist;
        max_dist
    }

    /// Computes the maximum distance from ancestor to any point in subtree_root's subtree.
    fn compute_max_descendant_dist(
        ancestor: &Node<T>,
        subtree_root: &Node<T>,
        metric: &D,
        current_max: f64
    ) -> f64 {
        let mut max_dist = metric.distance(&ancestor.point, &subtree_root.point);

        for child in &subtree_root.children {
            let dist_to_child = metric.distance(&ancestor.point, &child.point);

            if dist_to_child > max_dist {
                max_dist = dist_to_child;
            }

            if dist_to_child + child.maxdist <= current_max.max(max_dist) {
                continue;
            }

            let max_via_child = Self::compute_max_descendant_dist(
                ancestor,
                child,
                metric,
                current_max.max(max_dist)
            );

            if max_via_child > max_dist {
                max_dist = max_via_child;
            }
        }

        max_dist
    }
}
