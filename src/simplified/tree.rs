//! Simplified Cover Tree Implementation
//!
//! This module implements the Simplified variant of cover trees,
//! which maintains exactly n nodes for n points with no rebalancing.
//!
//! # Key Features
//!
//! - **Simplified Structure**: Exactly n nodes for n points
//! - **No Rebalancing**: Faster construction than Nearest Ancestor variant
//! - **Three Invariants**: Maintains leveling, covering, and separating invariants
//! - **Configurable Base**: Supports different base values for performance tuning
//!
//! # Usage
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
//!
//! let nearest = tree.find_nearest(&7.0);
//! assert_eq!(nearest, Some(&5.0));
//! ```

use crate::distance::Distance;
use crate::node::Node;
use super::insert::InsertImpl;

/// A simplified cover tree for efficient nearest neighbor search.
///
/// This variant maintains exactly n nodes for n points
/// without rebalancing. Construction is faster than the Nearest Ancestor variant,
/// but queries may perform slightly more distance computations.
///
/// # Type Parameters
///
/// * `T` - The type of points stored. Must implement `Clone`.
/// * `D` - The distance metric. Must implement `Distance<T>`.
#[derive(Debug)]
pub struct SimplifiedCoverTree<T: Clone, D: Distance<T>> {
    /// The root node of the tree, or None if empty.
    root: Option<Box<Node<T>>>,

    /// The distance metric used for all distance calculations.
    metric: D,

    /// The base value used for covdist and sepdist calculations.
    base: f64,

    /// The number of points stored in the tree.
    size: usize,
}

impl<T: Clone, D: Distance<T>> SimplifiedCoverTree<T, D> {
    /// Creates a new empty simplified cover tree.
    ///
    /// # Arguments
    ///
    /// * `metric` - The distance function to use
    /// * `base` - The base value for distance calculations (typically 1.3)
    ///
    /// # Returns
    ///
    /// A new empty `SimplifiedCoverTree`.
    ///
    /// # Panics
    ///
    /// Panics if `base` is not finite or is below [`MIN_BASE`](crate::MIN_BASE).
    ///
    /// # Examples
    ///
    /// ```rust,ignore
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
    /// let tree = SimplifiedCoverTree::new(SimpleDistance, 1.3);
    /// assert_eq!(tree.len(), 0);
    /// ```
    pub fn new(metric: D, base: f64) -> Self {
        crate::core::utils::validate_base(base);
        SimplifiedCoverTree {
            root: None,
            metric,
            base,
            size: 0,
        }
    }

    /// Internal constructor for creating SimplifiedCoverTree from parts.
    ///
    /// This is used by NACoverTree.merge() which needs to construct a SimplifiedCoverTree
    /// from the merged components. Not part of the public API.
    pub(crate) fn from_parts(
        root: Option<Box<Node<T>>>,
        metric: D,
        base: f64,
        size: usize,
    ) -> Self {
        SimplifiedCoverTree {
            root,
            metric,
            base,
            size,
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
    ///
    /// This is primarily for testing and validation purposes.
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

    /// Collect tree structure quality metrics.
    ///
    /// Returns `None` if the tree is empty, otherwise returns a `TreeStats`
    /// summarizing the tree's structure quality for diagnostic purposes.
    pub fn tree_stats(&self) -> Option<crate::node::TreeStats> {
        self.root.as_deref().map(|root| root.collect_stats(self.base))
    }

    /// Inserts a point into the cover tree.
    ///
    /// This implements Algorithm 2 from the paper (simplified cover tree insertion).
    ///
    /// # Arguments
    ///
    /// * `point` - The point to insert (ownership transferred)
    ///
    /// # Time Complexity
    ///
    /// O(c^6 log n) where c is the doubling constant and n is the number of points.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let mut tree = SimplifiedCoverTree::new(SimpleDistance, 1.3);
    /// tree.insert(10.0);
    /// tree.insert(20.0);
    /// assert_eq!(tree.len(), 2);
    /// ```
    pub fn insert(&mut self, point: T) {
        self.insert_impl(point);
    }

    /// Insert a point and return a raw pointer to its storage location in the tree.
    ///
    /// This is used by batch k-NN to build an O(1) mapping from input query index
    /// to query-tree node pointer, replacing the O(m²) greedy distance-matching.
    ///
    /// # Safety
    ///
    /// The returned pointer is valid as long as the tree is not modified (no further
    /// inserts, merges, or rebalancing). It points into a `Box<Node<T>>` owned by
    /// the tree.
    pub(crate) fn insert_returning_ptr(&mut self, point: T) -> *const T {
        self.insert_impl(point)
    }

    /// Core insert logic shared by `insert` and `insert_returning_ptr`.
    /// Returns a raw pointer to the inserted point's storage.
    fn insert_impl(&mut self, point: T) -> *const T {
        // Handle empty tree case
        if self.root.is_none() {
            self.root = Some(Box::new(Node::new(point, 0, false)));
            self.size = 1;
            let ptr: *const T = &self.root.as_ref().unwrap().point;
            return ptr;
        }

        // Check if point is within covering distance of root
        let dist_to_root = {
            let root = self.root.as_ref().unwrap();
            self.metric.distance(&root.point, &point)
        };

        let needs_raise = {
            let root = self.root.as_ref().unwrap();
            dist_to_root > root.covdist(self.base)
        };

        let ptr = if needs_raise {
            // Point is too far - need to raise tree level (reuse cached distance)
            self.raise_tree_level_returning_ptr(point, dist_to_root)
        } else {
            // Point is close enough - insert recursively
            let root = self.root.as_mut().unwrap();
            InsertImpl::insert_recursive_returning_ptr(
                root,
                point,
                &self.metric,
                self.base,
            )
        };

        self.size += 1;
        ptr
    }

    /// Finds the nearest neighbor to a query point.
    ///
    /// This implements Algorithm 1 from the paper (nearest neighbor query with pruning).
    ///
    /// # Arguments
    ///
    /// * `query` - The query point (borrowed)
    ///
    /// # Returns
    ///
    /// * `Some(&T)` - A reference to the nearest point in the tree
    /// * `None` - If the tree is empty
    ///
    /// # Time Complexity
    ///
    /// O(c^6 log n) where c is the doubling constant and n is the number of points.
    pub fn find_nearest(&self, query: &T) -> Option<&T> {
        use crate::core::query;

        // Return None if tree is empty
        self.root.as_ref()?;

        // Start search at root
        let root = self.root.as_ref().unwrap();

        // Initialize best with the root point
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
    /// use rustknn::SimplifiedCoverTree;
    /// use rustknn::Distance;
    ///
    /// struct EuclideanDistance;
    /// impl Distance<f64> for EuclideanDistance {
    ///     fn distance(&self, p: &f64, q: &f64) -> f64 {
    ///         (p - q).abs()
    ///     }
    /// }
    ///
    /// let mut tree = SimplifiedCoverTree::new(EuclideanDistance, 1.3);
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

    /// Raises the tree level and returns a pointer to the newly inserted point.
    ///
    /// The new point becomes the root, so the pointer is to `root.point`.
    fn raise_tree_level_returning_ptr(&mut self, point: T, dist_to_root: f64) -> *const T {
        // Calculate new level needed to accommodate the distance
        let new_level = crate::core::utils::level_for_distance(dist_to_root, self.base);

        // Take ownership of the old root
        let mut old_root = self.root.take().unwrap();

        // Adjust old root and all its descendants' levels
        let level_adjustment = crate::core::utils::root_level_adjustment(new_level, old_root.level);
        crate::core::utils::adjust_levels(&mut old_root, level_adjustment);

        // Create new root with the new point
        let mut new_root = Node::new(point, new_level, false);

        // Add old root as direct child
        let dist = self.metric.distance(&new_root.point, &old_root.point);
        new_root.update_maxdist_approx(dist, old_root.maxdist);
        old_root.d_parent = dist;
        new_root.children.push(old_root);

        // Install new root
        self.root = Some(Box::new(new_root));

        // Return pointer to the new root's point (the inserted point)
        &self.root.as_ref().unwrap().point as *const T
    }

    /// Recomputes d_parent, maxdist, and sorts children in a single traversal.
    ///
    /// Fuses `recompute_d_parent`, `recompute_maxdist`, and `sort_children_by_distance`
    /// into one bottom-up DFS that shares the `distance(parent, child)` computation
    /// across all three operations. Eliminates redundant distance computations.
    pub fn recompute_all(&mut self) {
        if let Some(ref mut root) = self.root {
            root.d_parent = 0.0;
            Self::recompute_all_recursive(root, &self.metric);
        }
    }

    /// Bottom-up DFS that recomputes d_parent, maxdist, and sorts children.
    ///
    /// For each node:
    /// 1. Compute distance(parent, child) once per edge → set child.d_parent
    /// 2. Recurse into children (post-order)
    /// 3. Compute exact maxdist using branch-and-bound
    /// 4. Sort children by d_parent (ascending) for optimal pruning order
    fn recompute_all_recursive(node: &mut Node<T>, metric: &D) -> f64 {
        let _guard = crate::core::utils::StackGuard::enter();
        if node.children.is_empty() {
            node.maxdist = 0.0;
            return 0.0;
        }

        // Step 1: Compute d_parent for all children (shared distance computation)
        for child in &mut node.children {
            child.d_parent = metric.distance(&node.point, &child.point);
        }

        // Step 2: Recurse into children (post-order: children's maxdist computed first)
        for child in &mut node.children {
            Self::recompute_all_recursive(child, metric);
        }

        // Step 3: Compute exact maxdist for this node
        // We already have d_parent = distance(node, child) cached on each child
        let mut max_dist: f64 = 0.0;
        for child in &node.children {
            let dist_to_child = child.d_parent;

            if dist_to_child > max_dist {
                max_dist = dist_to_child;
            }

            // Branch-and-bound pruning: skip subtree if can't exceed current max
            if dist_to_child + child.maxdist <= max_dist {
                continue;
            }

            let max_via_child = Self::compute_max_descendant_dist(
                node,
                child,
                metric,
                max_dist,
            );

            if max_via_child > max_dist {
                max_dist = max_via_child;
            }
        }
        node.maxdist = max_dist;

        // Step 4: Sort children by d_parent (ascending) for query pruning order
        node.children.sort_by(|a, b| a.d_parent.total_cmp(&b.d_parent));

        max_dist
    }

    /// Recomputes exact maxdist for all nodes in the tree.
    ///
    /// Call this after batch insertions to get tighter pruning bounds for queries.
    pub fn recompute_maxdist(&mut self) {
        if let Some(ref mut root) = self.root {
            Self::recompute_maxdist_recursive(root, &self.metric);
        }
    }

    /// Recomputes `d_parent` for every node in the tree.
    ///
    /// After merging, `d_parent` values can become stale because children may end up
    /// under a different parent than they were originally inserted under. Stale `d_parent`
    /// values cause the triangle inequality shell test to incorrectly prune subtrees
    /// during queries, leading to missed nearest neighbors.
    ///
    /// This method walks the tree recursively and sets each child's `d_parent` to
    /// `metric.distance(&parent.point, &child.point)`.
    pub fn recompute_d_parent(&mut self) {
        if let Some(ref mut root) = self.root {
            root.d_parent = 0.0; // root has no parent
            Self::recompute_d_parent_recursive(root, &self.metric);
        }
    }

    /// Recursive helper for d_parent recomputation.
    ///
    /// Also rebuilds `maxdist` bottom-up as `max(child.d_parent + child.maxdist)`, a
    /// valid (if not tight) bound by the triangle inequality, at no extra distance cost.
    /// Merging can move subtrees, which leaves the old maxdist values unsound.
    fn recompute_d_parent_recursive(node: &mut Node<T>, metric: &D) {
        let _guard = crate::core::utils::StackGuard::enter();
        let mut bound = 0.0_f64;
        for child in &mut node.children {
            child.d_parent = metric.distance(&node.point, &child.point);
            Self::recompute_d_parent_recursive(child, metric);
            bound = bound.max(child.d_parent + child.maxdist);
        }
        node.maxdist = bound;
    }

    /// Sort all children lists by d_parent (ascending) for optimal pruning order.
    ///
    /// During scale-organized descent, children are visited in iteration order.
    /// Visiting closer children first (smaller d_parent) tightens the kth-distance
    /// bound sooner, enabling more distant children to be pruned. This eliminates
    /// the need for runtime cover-set sorting.
    ///
    /// Call after `recompute_maxdist()` / `recompute_d_parent()` to ensure d_parent
    /// values are accurate before sorting.
    pub fn sort_children_by_distance(&mut self) {
        if let Some(ref mut root) = self.root {
            Self::sort_children_recursive(root);
        }
    }

    fn sort_children_recursive(node: &mut Node<T>) {
        let _guard = crate::core::utils::StackGuard::enter();
        node.children.sort_by(|a, b| a.d_parent.total_cmp(&b.d_parent));
        for child in &mut node.children {
            Self::sort_children_recursive(child);
        }
    }

    /// Recursive helper for exact maxdist computation.
    fn recompute_maxdist_recursive(node: &mut Node<T>, metric: &D) -> f64 {
        let _guard = crate::core::utils::StackGuard::enter();
        // Base case: leaf nodes have no descendants
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
        let _guard = crate::core::utils::StackGuard::enter();
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

    /// Merge two SimplifiedCoverTrees, consuming both and returning a new merged tree.
    ///
    /// This implements Algorithm 4 from the paper (tree merging). The merge operation
    /// combines two trees while maintaining all three cover tree invariants.
    ///
    /// # Arguments
    ///
    /// * `other` - The tree to merge with this one (must have same base value)
    /// * `randomize` - Whether to randomize child order during merge (default: false)
    ///
    /// # Returns
    ///
    /// A new `SimplifiedCoverTree` containing all points from both trees.
    ///
    /// # Panics
    ///
    /// Panics if the two trees have different base values, or if their levels are so far
    /// apart that aligning them would build an excessively deep chain of nodes (only
    /// possible for data spanning hundreds of orders of magnitude at a base near
    /// [`MIN_BASE`](crate::MIN_BASE)).
    ///
    /// # Time Complexity
    ///
    /// O(n + m) where n and m are the sizes of the two trees.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let mut tree1 = SimplifiedCoverTree::new(metric, 1.3);
    /// tree1.insert(1.0);
    /// tree1.insert(2.0);
    ///
    /// let mut tree2 = SimplifiedCoverTree::new(metric, 1.3);
    /// tree2.insert(10.0);
    /// tree2.insert(20.0);
    ///
    /// let merged = tree1.merge(tree2);
    /// assert_eq!(merged.len(), 4);
    /// ```
    pub fn merge(self, other: Self) -> Self {
        use super::merge::MergeImpl;

        // Verify base values match
        assert_eq!(
            self.base, other.base,
            "Cannot merge trees with different base values: {} != {}",
            self.base, other.base
        );

        // Handle empty tree cases
        if self.root.is_none() {
            return other;
        }
        if other.root.is_none() {
            return self;
        }

        // Merge the roots
        let merged_root = MergeImpl::merge(
            self.root.unwrap(),
            other.root.unwrap(),
            &self.metric,
            self.base,
        );

        let mut merged = SimplifiedCoverTree {
            root: Some(merged_root),
            metric: self.metric,
            base: self.base,
            size: self.size + other.size,
        };

        // Recompute d_parent after merging — children may end up under different
        // parents during merge, making cached d_parent values stale. Stale d_parent
        // causes the triangle inequality shell test to incorrectly prune subtrees.
        merged.recompute_d_parent();

        merged
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
    ///
    /// # Examples
    ///
    /// ```
    /// use rustknn::SimplifiedCoverTree;
    /// use rustknn::Distance;
    ///
    /// #[derive(Clone)]
    /// struct EuclideanDistance;
    /// impl Distance<f64> for EuclideanDistance {
    ///     fn distance(&self, p: &f64, q: &f64) -> f64 {
    ///         (p - q).abs()
    ///     }
    /// }
    ///
    /// let mut tree = SimplifiedCoverTree::new(EuclideanDistance, 1.3);
    /// for i in 0..20 {
    ///     tree.insert(i as f64);
    /// }
    ///
    /// let queries = vec![5.5, 15.5];
    /// let results = tree.find_k_nearest_batch(&queries, 3);
    /// assert_eq!(results.len(), 2);
    /// assert!(results[0].len() <= 3);
    /// assert!(results[1].len() <= 3);
    /// ```
    pub fn find_k_nearest_batch(&self, queries: &[T], k: usize) -> Vec<Vec<(&T, f64)>>
    where
        D: Clone,
    {
        use crate::core::dual_tree::traversal::DualTreeTraversal;
        use crate::core::dual_tree::knn_rules::KnnRules;

        if k == 0 || queries.is_empty() || self.root.is_none() {
            return vec![Vec::new(); queries.len()];
        }

        // For small query sets, single-tree queries avoid dual-tree overhead
        // (query tree build, maxdist recompute, parent map init).
        // Threshold determined empirically: dual-tree overhead dominates for Q <= 16.
        const SINGLE_TREE_THRESHOLD: usize = 16;
        if queries.len() <= SINGLE_TREE_THRESHOLD {
            return queries.iter()
                .map(|q| self.find_k_nearest(q, k))
                .collect();
        }

        // Build a temporary query tree, tracking which pointer belongs to which input index.
        let mut query_tree = SimplifiedCoverTree {
            root: None,
            metric: self.metric.clone(),
            base: self.base,
            size: 0,
        };

        #[cfg(not(feature = "hashmap-batch"))]
        let mut ptr_to_input: Vec<(*const T, Vec<usize>)> = Vec::with_capacity(queries.len());
        #[cfg(feature = "hashmap-batch")]
        let mut ptr_to_input: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::with_capacity(queries.len());

        for (qi, q) in queries.iter().enumerate() {
            let ptr = query_tree.insert_returning_ptr(q.clone());
            #[cfg(not(feature = "hashmap-batch"))]
            {
                match ptr_to_input.binary_search_by_key(&(ptr as usize), |&(p, _)| p as usize) {
                    Ok(pos) => ptr_to_input[pos].1.push(qi),
                    Err(pos) => ptr_to_input.insert(pos, (ptr, vec![qi])),
                }
            }
            #[cfg(feature = "hashmap-batch")]
            {
                ptr_to_input.entry(ptr as usize).or_insert_with(Vec::new).push(qi);
            }
        }
        // Recompute exact maxdist on the query tree so dual-tree pruning bounds
        // are tight. Without this, approximate maxdist from incremental insertion
        // can cause the traversal to over-prune, missing valid neighbors.
        query_tree.recompute_maxdist();

        let query_root = match query_tree.root.as_deref() {
            Some(r) => r,
            None => return vec![Vec::new(); queries.len()],
        };
        let ref_root = self.root.as_deref().unwrap();

        // Run DFS dual-tree traversal
        let mut rules = KnnRules::new(k, &self.metric, false);
        rules.state.init_parent_map(query_root);
        DualTreeTraversal::traverse(query_root, ref_root, &mut rules);

        // Collect results as Vec indexed by integer index
        let (mut results_vec, query_state_indices, query_ptrs) = rules.state.into_results();

        // Map results to input order
        let mut output: Vec<Vec<(&T, f64)>> = vec![Vec::new(); queries.len()];
        for (qi, &ptr) in query_ptrs.iter().enumerate() {
            let idx = query_state_indices[qi];
            let results = std::mem::take(&mut results_vec[idx]);

            #[cfg(not(feature = "hashmap-batch"))]
            let input_indices_opt = ptr_to_input
                .binary_search_by_key(&(ptr as usize), |&(p, _)| p as usize)
                .ok()
                .map(|pos| &ptr_to_input[pos].1);
            #[cfg(feature = "hashmap-batch")]
            let input_indices_opt = ptr_to_input.get(&(ptr as usize));

            if let Some(input_indices) = input_indices_opt {
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

    /// Instrumented version of `find_k_nearest_batch` that returns timing breakdowns
    /// and traversal statistics alongside the results.
    ///
    /// Separates query tree construction, maxdist recomputation, parent map init,
    /// traversal, and result collection into individually-timed phases. Also reports
    /// distance computation count and bound cache hit/miss rates.
    pub fn find_k_nearest_batch_instrumented(
        &self,
        queries: &[T],
        k: usize,
    ) -> (Vec<Vec<(&T, f64)>>, crate::core::dual_tree::DualTreeStats)
    where
        D: Clone,
    {
        use std::time::Instant;
        use crate::core::dual_tree::traversal::DualTreeTraversal;
        use crate::core::dual_tree::DualTreeStats;
        use crate::core::dual_tree::knn_rules::KnnRules;
        use crate::distance::{reset_distance_count, get_distance_count};

        let mut stats = DualTreeStats::default();

        if k == 0 || queries.is_empty() || self.root.is_none() {
            return (vec![Vec::new(); queries.len()], stats);
        }

        // Phase 1: Build query tree
        let t0 = Instant::now();
        let mut query_tree = SimplifiedCoverTree {
            root: None,
            metric: self.metric.clone(),
            base: self.base,
            size: 0,
        };
        let mut ptr_to_input: Vec<(*const T, Vec<usize>)> = Vec::with_capacity(queries.len());
        for (qi, q) in queries.iter().enumerate() {
            let ptr = query_tree.insert_returning_ptr(q.clone());
            match ptr_to_input.binary_search_by_key(&(ptr as usize), |&(p, _)| p as usize) {
                Ok(pos) => ptr_to_input[pos].1.push(qi),
                Err(pos) => ptr_to_input.insert(pos, (ptr, vec![qi])),
            }
        }
        stats.query_tree_build_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Phase 2: Recompute maxdist
        let t1 = Instant::now();
        query_tree.recompute_maxdist();
        stats.query_tree_maxdist_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let query_root = match query_tree.root.as_deref() {
            Some(r) => r,
            None => return (vec![Vec::new(); queries.len()], stats),
        };
        let ref_root = self.root.as_deref().unwrap();

        // Phase 3: Init parent map
        let t2 = Instant::now();
        let mut rules = KnnRules::new(k, &self.metric, false);
        rules.state.init_parent_map(query_root);
        stats.init_parent_map_ms = t2.elapsed().as_secs_f64() * 1000.0;

        // Phase 4: Traversal (with distance counter)
        reset_distance_count();
        let t3 = Instant::now();
        DualTreeTraversal::traverse(query_root, ref_root, &mut rules);
        stats.traversal_ms = t3.elapsed().as_secs_f64() * 1000.0;
        stats.distance_computations = get_distance_count();

        let (hits, misses) = rules.bound_cache_stats();
        stats.bound_cache_hits = hits;
        stats.bound_cache_misses = misses;

        // Phase 5: Result collection
        let t4 = Instant::now();
        let (mut results_vec, query_state_indices, query_ptrs) = rules.state.into_results();
        let mut output: Vec<Vec<(&T, f64)>> = vec![Vec::new(); queries.len()];
        for (qi, &ptr) in query_ptrs.iter().enumerate() {
            let idx = query_state_indices[qi];
            let results = std::mem::take(&mut results_vec[idx]);
            if let Ok(pos) = ptr_to_input.binary_search_by_key(&(ptr as usize), |&(p, _)| p as usize) {
                let input_indices = &ptr_to_input[pos].1;
                if input_indices.len() == 1 {
                    output[input_indices[0]] = results;
                } else {
                    for &qi in &input_indices[..input_indices.len() - 1] {
                        output[qi] = results.clone();
                    }
                    output[*input_indices.last().unwrap()] = results;
                }
            }
        }
        for (i, result) in output.iter_mut().enumerate() {
            if result.is_empty() {
                *result = self.find_k_nearest(&queries[i], k);
            }
        }
        stats.result_collection_ms = t4.elapsed().as_secs_f64() * 1000.0;

        (output, stats)
    }

    /// Like `find_k_nearest_batch_instrumented`, but accepts an explicit `BoundMode`.
    ///
    /// The bound mode is passed through to the DFS dual-tree traversal's `KnnRules`.
    pub fn find_k_nearest_batch_with_bound_instrumented(
        &self,
        queries: &[T],
        k: usize,
        _bound_mode: crate::core::dual_tree::BoundMode,
    ) -> (Vec<Vec<(&T, f64)>>, crate::core::dual_tree::DualTreeStats)
    where
        D: Clone,
    {
        // BoundMode only applies to packed DFS traversal; delegate directly for unpacked.
        self.find_k_nearest_batch_instrumented(queries, k)
    }

    /// Dual-tree k-NN with a user-supplied query tree.
    ///
    /// Uses the provided query tree and this tree (as reference) for dual-tree
    /// k-NN search. Returns results for each non-duplicate point in the query tree
    /// in DFS order.
    ///
    /// # Arguments
    ///
    /// * `query_tree` - The query cover tree
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
        query_tree: &SimplifiedCoverTree<T, D>,
        k: usize,
    ) -> Vec<Vec<(&T, f64)>> {
        use crate::core::dual_tree::traversal::DualTreeTraversal;
        use crate::core::dual_tree::knn_rules::KnnRules;

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
    /// Self-matches are excluded by pointer identity.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustknn::SimplifiedCoverTree;
    /// use rustknn::Distance;
    ///
    /// struct EuclideanDistance;
    /// impl Distance<f64> for EuclideanDistance {
    ///     fn distance(&self, p: &f64, q: &f64) -> f64 {
    ///         (p - q).abs()
    ///     }
    /// }
    ///
    /// let mut tree = SimplifiedCoverTree::new(EuclideanDistance, 1.3);
    /// tree.insert(0.0);
    /// tree.insert(1.0);
    /// tree.insert(3.0);
    ///
    /// let results = tree.find_k_nearest_self(1);
    /// assert_eq!(results.len(), 3);
    /// // Each point's nearest neighbor should not be itself
    /// for neighbors in &results {
    ///     assert_eq!(neighbors.len(), 1);
    /// }
    /// ```
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

    /// Instrumented version of `find_k_nearest_self` that returns timing breakdowns
    /// and traversal statistics alongside the results.
    ///
    /// For self-query, there is no separate query tree construction — the same tree
    /// is used for both query and reference. Timing covers init_parent_map, traversal,
    /// and result collection. Distance counter and bound cache stats are reported.
    pub fn find_k_nearest_self_instrumented(
        &self,
        k: usize,
    ) -> (Vec<Vec<(&T, f64)>>, crate::core::dual_tree::DualTreeStats) {
        use std::time::Instant;
        use crate::core::dual_tree::knn_rules::KnnRules;
        use crate::core::dual_tree::traversal::DualTreeTraversal;
        use crate::core::dual_tree::DualTreeStats;
        use crate::distance::{reset_distance_count, get_distance_count};

        let mut stats = DualTreeStats::default();

        if k == 0 || self.root.is_none() {
            return (Vec::new(), stats);
        }

        let root = self.root.as_deref().unwrap();

        // Init parent map
        let t0 = Instant::now();
        // Full Curtin B1/B2 bounds: a query node's bound must hold for every query
        // point below it, not just its own point, even when query = reference.
        let mut rules = KnnRules::new(k, &self.metric, true);
        rules.state.init_parent_map(root);
        stats.init_parent_map_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Traversal
        reset_distance_count();
        let t1 = Instant::now();
        DualTreeTraversal::traverse(root, root, &mut rules);
        stats.traversal_ms = t1.elapsed().as_secs_f64() * 1000.0;
        stats.distance_computations = get_distance_count();

        let (hits, misses) = rules.bound_cache_stats();
        stats.bound_cache_hits = hits;
        stats.bound_cache_misses = misses;

        // Result collection
        let t2 = Instant::now();
        let (mut results_vec, query_state_indices, _query_ptrs) = rules.state.into_results();
        let results = query_state_indices
            .iter()
            .map(|&idx| std::mem::take(&mut results_vec[idx]))
            .collect();
        stats.result_collection_ms = t2.elapsed().as_secs_f64() * 1000.0;

        (results, stats)
    }

    /// Convert tree to cache-optimized packed layout
    ///
    /// Creates a PackedCoverTree with all nodes stored in contiguous memory using
    /// depth-first ordering. This provides 15-25% faster queries
    /// due to improved cache locality.
    ///
    /// # Performance
    ///
    /// - **Packing time**: O(n) - single depth-first traversal
    /// - **Query speedup**: 15-25% faster than unpacked tree
    /// - **Memory**: Same space as unpacked (one node per point)
    ///
    /// # Important
    ///
    /// PackedCoverTree is **completely immutable** - no insertions possible.
    /// Keep the original unpacked tree if you need to perform more insertions.
    ///
    /// # Example
    ///
    /// ```rust,ignore
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
    /// // Build tree (unpacked, mutable)
    /// let mut tree = SimplifiedCoverTree::new(SimpleDistance, 1.3);
    /// for i in 0..10000 {
    ///     tree.insert(i as f64);
    /// }
    ///
    /// // Pack for cache-efficient queries
    /// let packed = tree.pack();
    ///
    /// // Queries are 15-25% faster
    /// let nearest = packed.find_nearest(&5000.0);
    /// ```
    pub fn pack(self) -> crate::packed::PackedCoverTree<T, D> {
        use crate::packed::pack::PackImpl;
        PackImpl::pack(self.root, self.metric, self.base, self.size)
    }

    // ---------------------------------------------------------------------------
    // Batch single-tree queries
    // ---------------------------------------------------------------------------

    /// Batch single-tree all-nearest-neighbors: for every point in the tree, its k
    /// nearest other points.
    ///
    /// Walks the tree as a query tree against itself, performing single-tree
    /// reference descent at each scale level. Each query child receives an
    /// independently filtered copy of the parent's reference candidates with
    /// recomputed distances.
    ///
    /// # Returns
    ///
    /// One `(pointer to point, neighbors)` pair per point, in no particular order.
    /// Neighbors are sorted by distance, closest first, and never include the point
    /// itself (other points at distance zero are included).
    pub fn find_k_nearest_batch_single_self(
        &self,
        k: usize,
    ) -> Vec<(*const T, Vec<(&T, f64)>)> {
        let root = match self.root.as_deref() {
            Some(r) => r,
            None => return Vec::new(),
        };
        crate::core::batch_single::batch_single_tree_knn_self(root, k, &self.metric)
    }
}
