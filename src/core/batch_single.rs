//! Batch Single-Tree k-NN Algorithm
//!
//! Implements the batch nearest-neighbor query of Beygelzimer, Kakade & Langford,
//! "Cover Trees for Nearest Neighbor" (ICML 2006): walks the query tree hierarchy but
//! performs single-tree reference descent at each scale level. Each query child
//! receives an independently filtered COPY of the parent's reference candidates with
//! recomputed distances, avoiding the O(children_q × children_r) combinatorial
//! explosion that Curtin-style dual-tree traversal creates.
//!
//! # Algorithm Overview
//!
//! For the query node being processed, the search keeps a stack of "cover sets"
//! (reference nodes not yet expanded, grouped by scale) and a "zero set" (reference
//! points that are candidates but need no further expansion). Each entry carries its
//! distance to the query node's point. Scales are processed from coarse to fine:
//!
//! 1. **Split**: once the query node is at least as coarse as the current reference
//!    scale, each query child gets its own copy of the cover sets and zero set, with
//!    distances recomputed to the child's point, and is searched recursively. The
//!    query node then continues as a single point.
//!
//! 2. **Descend**: expand every reference node at the current scale. Its own point
//!    moves to the zero set and its children go to the zero set (leaves) or to the
//!    cover set for their scale (internal nodes).
//!
//! 3. **Base case**: when no unexpanded reference nodes remain, the query point's k
//!    nearest neighbors are the k closest zero-set entries.
//!
//! # Adaptation for Simplified Cover Trees
//!
//! The original algorithm runs on cover trees in which every point is repeated as a
//! "self-child" down to the leaves. Here every point is stored exactly once, which
//! changes three things:
//!
//! - An internal reference node's point is a candidate in its own right, so it moves
//!   to the zero set when the node is expanded instead of being dropped.
//! - Pruning uses each node's `maxdist` (the radius of its subtree) rather than a
//!   radius derived from its level. While a query node still has unsearched children,
//!   bounds are widened by the query subtree's radius so they hold for every point
//!   below it.
//! - Levels only order the work. Trees built by merging or nearest-ancestor
//!   rebalancing can contain children whose level is not below their parent's, so a
//!   node is always scheduled strictly after the scale that discovered it.
//!
//! # Pruning bound
//!
//! Each query node tracks the k smallest distances from its point to distinct
//! candidate reference points seen so far, capped by the bound inherited from its
//! parent (`parent_kth + d(parent, child)`, valid by the triangle inequality). Points
//! excluded from the result (the query point itself in a self-query, and structural
//! duplicate nodes) never count toward the bound.

use crate::Distance;
use crate::node::Node;
use crate::packed::PackedCoverTree;

// ---------------------------------------------------------------------------
// Tree views
// ---------------------------------------------------------------------------

/// Read-only access to a cover tree, abstracting over the pointer-based and packed
/// layouts so one implementation serves both as query tree and reference tree.
pub(crate) trait TreeView<'a, T: 'a> {
    type Id: Copy + PartialEq;
    fn root(&self) -> Option<Self::Id>;
    fn point(&self, id: Self::Id) -> &'a T;
    fn level(&self, id: Self::Id) -> i32;
    fn maxdist(&self, id: Self::Id) -> f64;
    /// `d(parent.point, point)`, or 0.0 if unknown.
    fn d_parent(&self, id: Self::Id) -> f64;
    fn is_duplicate(&self, id: Self::Id) -> bool;
    fn num_children(&self, id: Self::Id) -> usize;
    fn child(&self, id: Self::Id, i: usize) -> Self::Id;
}

/// Pointer-based node handle compared by identity.
pub(crate) struct NodeRef<'a, T: Clone>(pub(crate) &'a Node<T>);

impl<T: Clone> Clone for NodeRef<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: Clone> Copy for NodeRef<'_, T> {}
impl<T: Clone> PartialEq for NodeRef<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.0, other.0)
    }
}

/// View of a pointer-based tree rooted at `root`.
pub(crate) struct NodeTree<'a, T: Clone>(pub(crate) Option<&'a Node<T>>);

impl<'a, T: Clone + 'a> TreeView<'a, T> for NodeTree<'a, T> {
    type Id = NodeRef<'a, T>;
    fn root(&self) -> Option<Self::Id> {
        self.0.map(NodeRef)
    }
    #[inline]
    fn point(&self, id: Self::Id) -> &'a T {
        &id.0.point
    }
    #[inline]
    fn level(&self, id: Self::Id) -> i32 {
        id.0.level
    }
    #[inline]
    fn maxdist(&self, id: Self::Id) -> f64 {
        id.0.maxdist
    }
    #[inline]
    fn d_parent(&self, id: Self::Id) -> f64 {
        id.0.d_parent
    }
    #[inline]
    fn is_duplicate(&self, id: Self::Id) -> bool {
        id.0.is_duplicate
    }
    #[inline]
    fn num_children(&self, id: Self::Id) -> usize {
        id.0.children.len()
    }
    #[inline]
    fn child(&self, id: Self::Id, i: usize) -> Self::Id {
        NodeRef(&id.0.children[i])
    }
}

impl<'a, T: Clone + 'a, D: Distance<T>> TreeView<'a, T> for &'a PackedCoverTree<T, D> {
    type Id = usize;
    fn root(&self) -> Option<usize> {
        if self.node_count() == 0 { None } else { Some(self.root_index()) }
    }
    #[inline]
    fn point(&self, id: usize) -> &'a T {
        &self.node(id).point
    }
    #[inline]
    fn level(&self, id: usize) -> i32 {
        self.node(id).level
    }
    #[inline]
    fn maxdist(&self, id: usize) -> f64 {
        self.node(id).maxdist
    }
    #[inline]
    fn d_parent(&self, id: usize) -> f64 {
        self.node(id).d_parent
    }
    #[inline]
    fn is_duplicate(&self, id: usize) -> bool {
        self.node(id).is_duplicate
    }
    #[inline]
    fn num_children(&self, id: usize) -> usize {
        self.children_of(id).len()
    }
    #[inline]
    fn child(&self, id: usize, i: usize) -> usize {
        self.children_of(id)[i]
    }
}

// ---------------------------------------------------------------------------
// Search state
// ---------------------------------------------------------------------------

/// A reference node (in a cover set) or reference point (in the zero set) with its
/// distance to the current query point.
struct Entry<I> {
    id: I,
    dist: f64,
}

/// Upper bound on the current query point's k-th nearest neighbor distance.
struct Bound {
    k: usize,
    /// Bound inherited from the parent query node.
    cap: f64,
    /// Smallest distances to distinct candidates, ascending, at most `k` of them.
    best: Vec<f64>,
}

impl Bound {
    fn new(k: usize, cap: f64) -> Self {
        Bound { k, cap, best: Vec::new() }
    }

    #[inline]
    fn kth(&self) -> f64 {
        match self.best.last() {
            Some(&worst) if self.best.len() == self.k => worst.min(self.cap),
            _ => self.cap,
        }
    }

    #[inline]
    fn add(&mut self, d: f64) {
        if self.best.len() == self.k {
            match self.best.last() {
                Some(&worst) if d < worst => {
                    self.best.pop();
                }
                _ => return,
            }
        }
        let pos = self.best.partition_point(|&b| b <= d);
        self.best.insert(pos, d);
    }
}

/// Pruning threshold for the current query subtree: the k-th neighbor bound widened
/// by `2 * rho` (the query subtree radius on both sides) and by a relative margin.
///
/// Bounds are built from sums of computed distances. When the triangle inequality is
/// tight (e.g. collinear points), a candidate's directly computed distance can exceed
/// the summed bound by an ulp, and pruning it would drop an exact neighbor. The
/// margin is far larger than any accumulated rounding error and far too small to
/// affect pruning in practice.
#[inline]
fn prune_threshold(kth: f64, rho: f64) -> f64 {
    let t = kth + 2.0 * rho;
    t + t * 1e-9
}

/// Sort the closer half of `entries` by distance and leave the rest unordered.
/// Visiting closer reference nodes first tightens the bound sooner.
fn halfsort<I>(entries: &mut [Entry<I>]) {
    if entries.len() > 1 {
        let mid = entries.len() / 2;
        entries.select_nth_unstable_by(mid, |a, b| a.dist.total_cmp(&b.dist));
        entries[..mid].sort_by(|a, b| a.dist.total_cmp(&b.dist));
    }
}

/// One result row: a query node and its neighbors (point, distance), closest first.
type Row<'a, Q, T> = (Q, Vec<(&'a T, f64)>);

struct Search<'s, 'q, 'a, T, D, Q, R, X>
where
    Q: TreeView<'q, T>,
    R: TreeView<'a, T>,
{
    q: &'s Q,
    r: &'s R,
    metric: &'s D,
    k: usize,
    /// True if a reference point must not be reported for a query node (the query
    /// point itself in a self-query).
    exclude: X,
    /// Largest level in the reference tree; cover set `i` holds scale `top - i`.
    top: i32,
    rows: Vec<Row<'a, Q::Id, T>>,
}

impl<'s, 'q, 'a, T, D, Q, R, X> Search<'s, 'q, 'a, T, D, Q, R, X>
where
    T: 'q + 'a,
    D: Distance<T>,
    Q: TreeView<'q, T>,
    R: TreeView<'a, T>,
    X: Fn(Q::Id, R::Id) -> bool,
{
    #[inline]
    fn counts(&self, qn: Q::Id, rn: R::Id) -> bool {
        !self.r.is_duplicate(rn) && !(self.exclude)(qn, rn)
    }

    /// Search the query subtree rooted at `qn`. `cover` and `zero` hold distances to
    /// `qn`'s point; cover sets before index `cur` are already empty.
    fn search(
        &mut self,
        qn: Q::Id,
        mut cover: Vec<Vec<Entry<R::Id>>>,
        mut zero: Vec<Entry<R::Id>>,
        mut cur: usize,
        mut pending: usize,
        mut bound: Bound,
    ) {
        let (q, r, metric) = (self.q, self.r, self.metric);
        let x = q.point(qn);
        let mut as_point = q.num_children(qn) == 0;

        loop {
            let ref_scale = self.top - cur as i32;
            if !as_point && (pending == 0 || q.level(qn) > ref_scale) {
                self.split(qn, &cover, &zero, cur, &bound);
                as_point = true;
                continue;
            }
            if pending == 0 {
                self.emit(qn, zero);
                return;
            }

            // Query subtree radius: bounds must hold for every query point below qn.
            let rho = if as_point { 0.0 } else { q.maxdist(qn) };

            if cur < cover.len() && !cover[cur].is_empty() {
                let mut entries = std::mem::take(&mut cover[cur]);
                pending -= entries.len();
                halfsort(&mut entries);

                for e in &entries {
                    let slack = prune_threshold(bound.kth(), rho);
                    if e.dist - r.maxdist(e.id) > slack {
                        continue; // whole reference subtree is too far
                    }
                    // The node's own point is a candidate (registered when discovered).
                    if e.dist <= slack {
                        zero.push(Entry { id: e.id, dist: e.dist });
                    }
                    for i in 0..r.num_children(e.id) {
                        let c = r.child(e.id, i);
                        let slack = prune_threshold(bound.kth(), rho);
                        let c_radius = r.maxdist(c);
                        let dp = r.d_parent(c);
                        // Triangle inequality: d(x, c) >= |d(x, parent) - d(parent, c)|.
                        if dp > 0.0 && (e.dist - dp).abs() - c_radius > slack {
                            continue;
                        }
                        let dc = metric.distance_with_bound(x, r.point(c), slack + c_radius);
                        if dc - c_radius > slack {
                            continue;
                        }
                        if self.counts(qn, c) {
                            bound.add(dc);
                        }
                        if r.num_children(c) == 0 {
                            zero.push(Entry { id: c, dist: dc });
                        } else {
                            // Schedule strictly after the current scale, even if the
                            // child's level is not below its parent's.
                            let off = ((self.top - r.level(c)).max(cur as i32 + 1)) as usize;
                            if off >= cover.len() {
                                cover.resize_with(off + 1, Vec::new);
                            }
                            cover[off].push(Entry { id: c, dist: dc });
                            pending += 1;
                        }
                    }
                }
            }
            cur += 1;
        }
    }

    /// Search each child of `qn` with its own copy of the candidates.
    fn split(
        &mut self,
        qn: Q::Id,
        cover: &[Vec<Entry<R::Id>>],
        zero: &[Entry<R::Id>],
        cur: usize,
        bound: &Bound,
    ) {
        let (q, r, metric) = (self.q, self.r, self.metric);
        let x = q.point(qn);
        // A child's k-th neighbor is within `parent_kth + d(parent, child)`: the
        // parent's k candidates, with the child itself swapped for the parent point if
        // needed, are all that close. That relies on the parent point being a valid
        // neighbor of the child, which a structural duplicate is not (and its bound can
        // be 0 because the real copy of its point is at distance 0), so its children
        // start without an inherited bound.
        let parent_kth = if q.is_duplicate(qn) { f64::INFINITY } else { bound.kth() };

        for i in 0..q.num_children(qn) {
            let qc = q.child(qn, i);
            let y = q.point(qc);
            let dq = match q.d_parent(qc) {
                d if d > 0.0 => d,
                _ => metric.distance(x, y),
            };
            let mut cb = Bound::new(self.k, parent_kth + dq);
            let rho = if q.num_children(qc) == 0 { 0.0 } else { q.maxdist(qc) };

            let mut child_cover: Vec<Vec<Entry<R::Id>>> =
                (0..cover.len()).map(|_| Vec::new()).collect();
            let mut pending = 0;
            for (off, list) in cover.iter().enumerate().skip(cur) {
                for e in list {
                    let e_radius = r.maxdist(e.id);
                    let slack = prune_threshold(cb.kth(), rho);
                    // Triangle inequality: d(y, e) >= |d(x, e) - d(x, y)|.
                    if (e.dist - dq).abs() - e_radius > slack {
                        continue;
                    }
                    let d = metric.distance_with_bound(y, r.point(e.id), slack + e_radius);
                    if d - e_radius > slack {
                        continue;
                    }
                    if self.counts(qc, e.id) {
                        cb.add(d);
                    }
                    child_cover[off].push(Entry { id: e.id, dist: d });
                    pending += 1;
                }
            }

            let mut child_zero = Vec::new();
            for e in zero {
                let slack = prune_threshold(cb.kth(), rho);
                if (e.dist - dq).abs() > slack {
                    continue;
                }
                let d = metric.distance_with_bound(y, r.point(e.id), slack);
                if d > slack {
                    continue;
                }
                if self.counts(qc, e.id) {
                    cb.add(d);
                }
                child_zero.push(Entry { id: e.id, dist: d });
            }

            self.search(qc, child_cover, child_zero, cur, pending, cb);
        }
    }

    /// Record the k closest zero-set candidates for `qn`.
    fn emit(&mut self, qn: Q::Id, zero: Vec<Entry<R::Id>>) {
        if self.q.is_duplicate(qn) {
            return; // structural copy of a point stored elsewhere in the query tree
        }
        let mut found: Vec<Entry<R::Id>> =
            zero.into_iter().filter(|e| self.counts(qn, e.id)).collect();
        if found.len() > self.k {
            found.select_nth_unstable_by(self.k, |a, b| a.dist.total_cmp(&b.dist));
            found.truncate(self.k);
        }
        found.sort_by(|a, b| a.dist.total_cmp(&b.dist));
        // Build the row in a buffer sized to the result. Collecting from `found`
        // directly would reuse its allocation, which can be as large as the zero set,
        // and keep it alive for as long as the row.
        let r = self.r;
        let mut row = Vec::with_capacity(found.len());
        row.extend(found.iter().map(|e| (r.point(e.id), e.dist)));
        self.rows.push((qn, row));
    }
}

/// Largest level anywhere in the tree (iterative, so deep trees cannot overflow
/// the stack).
fn max_level<'a, T: 'a, R: TreeView<'a, T>>(r: &R, root: R::Id) -> i32 {
    let mut top = r.level(root);
    let mut stack = vec![root];
    while let Some(n) = stack.pop() {
        top = top.max(r.level(n));
        for i in 0..r.num_children(n) {
            stack.push(r.child(n, i));
        }
    }
    top
}

/// Exact k-NN of every non-duplicate query-tree point against the reference tree.
///
/// Returns one row per non-duplicate query node, in no particular order. A reference
/// point `rn` is never reported for query node `qn` when `exclude(qn, rn)` is true.
fn batch_knn<'q, 'a, T, D, Q, R, X>(
    q: &Q,
    r: &R,
    k: usize,
    metric: &D,
    exclude: X,
) -> Vec<Row<'a, Q::Id, T>>
where
    T: 'q + 'a,
    D: Distance<T>,
    Q: TreeView<'q, T>,
    R: TreeView<'a, T>,
    X: Fn(Q::Id, R::Id) -> bool,
{
    let (Some(qroot), Some(rroot)) = (q.root(), r.root()) else {
        return Vec::new();
    };
    let top = max_level(r, rroot);
    let mut search = Search { q, r, metric, k, exclude, top, rows: Vec::new() };

    let d0 = metric.distance(q.point(qroot), r.point(rroot));
    let mut bound = Bound::new(k, f64::INFINITY);
    if search.counts(qroot, rroot) {
        bound.add(d0);
    }
    let root_entry = Entry { id: rroot, dist: d0 };
    let (cover, zero, pending) = if r.num_children(rroot) == 0 {
        (Vec::new(), vec![root_entry], 0)
    } else {
        let off = (top - r.level(rroot)) as usize;
        let mut cover: Vec<Vec<Entry<R::Id>>> = (0..=off).map(|_| Vec::new()).collect();
        cover[off].push(root_entry);
        (cover, Vec::new(), 1)
    };

    if k > 0 {
        search.search(qroot, cover, zero, 0, pending, bound);
    } else {
        // Nothing to find, but still report one (empty) row per query point.
        let mut stack = vec![qroot];
        while let Some(n) = stack.pop() {
            if !q.is_duplicate(n) {
                search.rows.push((n, Vec::new()));
            }
            for i in 0..q.num_children(n) {
                stack.push(q.child(n, i));
            }
        }
    }
    search.rows
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

/// Self-query on a pointer-based tree: the k nearest neighbors of every
/// non-duplicate point, excluding the point itself. Rows are keyed by a pointer to
/// the query point.
pub fn batch_single_tree_knn_self<'a, T, D>(
    root: &'a Node<T>,
    k: usize,
    metric: &D,
) -> Vec<(*const T, Vec<(&'a T, f64)>)>
where
    T: Clone,
    D: Distance<T>,
{
    let tree = NodeTree(Some(root));
    batch_knn(&tree, &tree, k, metric, |qn: NodeRef<'a, T>, rn: NodeRef<'a, T>| qn == rn)
        .into_iter()
        .map(|(qn, row)| (&qn.0.point as *const T, row))
        .collect()
}

/// Self-query on a packed tree: rows are keyed by packed node index.
pub fn batch_single_tree_knn_packed_self<'a, T, D>(
    packed: &'a PackedCoverTree<T, D>,
    k: usize,
) -> Vec<(usize, Vec<(&'a T, f64)>)>
where
    T: Clone,
    D: Distance<T>,
{
    batch_knn(&packed, &packed, k, packed.metric(), |qn: usize, rn: usize| qn == rn)
}

/// Held-out queries: the k nearest reference points of every non-duplicate point in
/// the pointer-based query tree. Rows are keyed by a pointer to the query point.
pub fn batch_single_tree_knn_packed<'a, 'q, T, D>(
    query_root: &'q Node<T>,
    packed: &'a PackedCoverTree<T, D>,
    k: usize,
) -> Vec<(*const T, Vec<(&'a T, f64)>)>
where
    T: Clone,
    D: Distance<T>,
{
    let queries = NodeTree(Some(query_root));
    batch_knn(&queries, &packed, k, packed.metric(), |_: NodeRef<'q, T>, _: usize| false)
        .into_iter()
        .map(|(qn, row)| (&qn.0.point as *const T, row))
        .collect()
}
