//! Correctness at dimensions where the early-exit distance kernels engage (16+), on every
//! query path that returns neighbors, and robustness on very deep trees: no operation may
//! abort the process; each either completes or fails with a catchable panic.

use rustknn::parallel::build_parallel_simplified;
use rustknn::{
    BoundMode, CoverTree, Distance, EuclideanDistance, ManhattanDistance, NACoverTree, Node,
    SimplifiedCoverTree, MIN_BASE,
};
use std::panic::{catch_unwind, AssertUnwindSafe};

type P = Vec<f64>;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn points(seed: u64, n: usize, dim: usize) -> Vec<P> {
    let mut r = Rng(seed);
    (0..n).map(|_| (0..dim).map(|_| r.next()).collect()).collect()
}

fn key(d: impl IntoIterator<Item = f64>) -> Vec<i64> {
    let mut v: Vec<f64> = d.into_iter().collect();
    v.sort_by(f64::total_cmp);
    v.into_iter().map(|x| (x * 1e9).round() as i64).collect()
}

fn brute<M: Distance<P>>(m: &M, pts: &[P], q: &P, k: usize, skip: Option<usize>) -> Vec<i64> {
    let mut d: Vec<f64> = pts
        .iter()
        .enumerate()
        .filter(|(i, _)| Some(*i) != skip)
        .map(|(_, p)| m.distance(p, q))
        .collect();
    d.sort_by(f64::total_cmp);
    d.truncate(k);
    key(d)
}

fn rows<'a>(rs: impl IntoIterator<Item = &'a Vec<(&'a P, f64)>>) -> Vec<Vec<i64>> {
    rs.into_iter().map(|r| key(r.iter().map(|x| x.1))).collect()
}

fn sorted(mut v: Vec<Vec<i64>>) -> Vec<Vec<i64>> {
    v.sort();
    v
}

/// Every neighbor-returning query on `pts` with metric `m` must match brute force.
fn check_all_modes<M: Distance<P> + Clone>(m: M, pts: &[P], qs: &[P], k: usize, what: &str) {
    let want: Vec<Vec<i64>> = qs.iter().map(|q| brute(&m, pts, q, k, None)).collect();
    let want_self = sorted((0..pts.len()).map(|i| brute(&m, pts, &pts[i], k, Some(i))).collect());

    let mut t = SimplifiedCoverTree::new(m.clone(), 1.3);
    let mut na = NACoverTree::new(m.clone(), 1.3);
    let mut qt = SimplifiedCoverTree::new(m.clone(), 1.3);
    let mut na_qt = NACoverTree::new(m.clone(), 1.3);
    for p in pts {
        t.insert(p.clone());
        na.insert(p.clone());
    }
    for q in qs {
        qt.insert(q.clone());
        na_qt.insert(q.clone());
    }
    let single: Vec<_> = qs.iter().map(|q| t.find_k_nearest(q, k)).collect();
    assert_eq!(rows(&single), want, "{what}: single-tree");
    assert_eq!(rows(&t.find_k_nearest_batch(qs, k)), want, "{what}: dual batch");
    assert_eq!(rows(&t.find_k_nearest_batch_instrumented(qs, k).0), want, "{what}: dual batch instrumented");
    for mode in [BoundMode::CurtinRecursive, BoundMode::Beygelzimer] {
        let got = t.find_k_nearest_batch_with_bound_instrumented(qs, k, mode).0;
        assert_eq!(rows(&got), want, "{what}: dual batch, explicit bound mode");
    }
    assert_eq!(sorted(rows(&t.find_k_nearest_dual(&qt, k))), sorted(want.clone()), "{what}: find_k_nearest_dual");
    assert_eq!(sorted(rows(&t.find_k_nearest_self(k))), want_self, "{what}: dual self");
    assert_eq!(sorted(rows(&t.find_k_nearest_self_instrumented(k).0)), want_self, "{what}: dual self instrumented");
    let bs: Vec<_> = t.find_k_nearest_batch_single_self(k).into_iter().map(|(_, r)| r).collect();
    assert_eq!(sorted(rows(&bs)), want_self, "{what}: batch single self");

    let single: Vec<_> = qs.iter().map(|q| na.find_k_nearest(q, k)).collect();
    assert_eq!(rows(&single), want, "{what}: NA single-tree");
    assert_eq!(rows(&na.find_k_nearest_batch(qs, k)), want, "{what}: NA dual batch");
    assert_eq!(sorted(rows(&na.find_k_nearest_dual(&na_qt, k))), sorted(want.clone()), "{what}: NA find_k_nearest_dual");
    assert_eq!(sorted(rows(&na.find_k_nearest_self(k))), want_self, "{what}: NA dual self");

    let p = t.pack();
    let single: Vec<_> = qs.iter().map(|q| p.find_k_nearest(q, k)).collect();
    assert_eq!(rows(&single), want, "{what}: packed single-tree");
    assert_eq!(rows(&p.find_k_nearest_batch(qs, k)), want, "{what}: packed dual batch");
    assert_eq!(rows(&p.find_k_nearest_batch_instrumented(qs, k).0), want, "{what}: packed dual batch instrumented");
    for mode in [BoundMode::CurtinRecursive, BoundMode::Beygelzimer] {
        let got = p.find_k_nearest_batch_with_bound_instrumented(qs, k, mode).0;
        assert_eq!(rows(&got), want, "{what}: packed dual batch, explicit bound mode");
    }
    assert_eq!(sorted(rows(&p.find_k_nearest_dual(&qt, k))), sorted(want.clone()), "{what}: packed find_k_nearest_dual");
    assert_eq!(sorted(rows(&p.find_k_nearest_self(k))), want_self, "{what}: packed dual self");
    assert_eq!(sorted(rows(&p.find_k_nearest_self_instrumented(k).0)), want_self, "{what}: packed dual self instrumented");
    let bs: Vec<_> = p.find_k_nearest_batch_single_self(k).into_iter().map(|(_, r)| r).collect();
    assert_eq!(sorted(rows(&bs)), want_self, "{what}: packed batch single self");
    assert_eq!(rows(&p.find_k_nearest_batch_single(qs, k)), want, "{what}: packed batch single held-out");
}

/// The reported case: Euclidean, 16 dimensions, n = 1000, k = 10.
#[test]
fn early_exit_dimensions_euclidean() {
    for &(seed, n, dim) in &[(7u64, 1000usize, 16usize), (11, 600, 34), (13, 400, 64), (17, 150, 784)] {
        let pts = points(seed, n, dim);
        let qs = points(seed + 1000, 60, dim);
        for &k in &[1usize, 10] {
            check_all_modes(EuclideanDistance, &pts, &qs, k, &format!("euclidean d={dim} k={k}"));
        }
    }
}

#[test]
fn early_exit_dimensions_manhattan() {
    for &(seed, n, dim) in &[(19u64, 500usize, 32usize), (23, 300, 64)] {
        let pts = points(seed, n, dim);
        let qs = points(seed + 1000, 40, dim);
        for &k in &[1usize, 10] {
            check_all_modes(ManhattanDistance, &pts, &qs, k, &format!("manhattan d={dim} k={k}"));
        }
    }
}

#[test]
fn k_zero_on_remaining_entry_points() {
    let pts = points(29, 40, 3);
    let qs = points(31, 6, 3);
    let mut na = NACoverTree::new(EuclideanDistance, 1.3);
    let mut ct = CoverTree::new_nearest_ancestor(EuclideanDistance, 1.3);
    let mut cs = CoverTree::new(EuclideanDistance, 1.3);
    for p in &pts {
        na.insert(p.clone());
        ct.insert(p.clone());
        cs.insert(p.clone());
    }
    assert!(na.find_k_nearest_batch(&qs, 0).iter().all(Vec::is_empty));
    assert!(na.find_k_nearest_self(0).iter().all(Vec::is_empty));
    for t in [&ct, &cs] {
        assert!(t.find_k_nearest_batch(&qs, 0).iter().all(Vec::is_empty));
        assert!(t.find_k_nearest_self(0).iter().all(Vec::is_empty));
    }
}

// ---------------------------------------------------------------------------
// deep trees
// ---------------------------------------------------------------------------

fn depth(n: &Node<P>) -> usize {
    let mut best = 0;
    let mut stack = vec![(n, 1usize)];
    while let Some((x, d)) = stack.pop() {
        best = best.max(d);
        stack.extend(x.children.iter().map(|c| (&**c, d + 1)));
    }
    best
}

/// Runs `f` on a 2 MiB thread and reports whether it completed (`true`) or panicked
/// (`false`). A stack overflow would abort the whole test process instead. A panic must
/// be one of the library's documented refusals, not an unrelated failure.
fn on_2mib<F: FnOnce() + Send + 'static>(f: F) -> bool {
    let r = std::thread::Builder::new()
        .stack_size(2 << 20)
        .spawn(move || catch_unwind(AssertUnwindSafe(f)))
        .unwrap()
        .join()
        .unwrap();
    match r {
        Ok(()) => true,
        Err(e) => {
            let msg = e
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| e.downcast_ref::<&str>().copied())
                .unwrap_or("");
            assert!(
                msg.contains("recursion stack budget") || msg.contains("cannot merge cover trees"),
                "unexpected panic: {msg}"
            );
            false
        }
    }
}

type Tree = SimplifiedCoverTree<P, EuclideanDistance>;
type Op = fn(Tree, Tree);

fn qs() -> Vec<P> {
    (0..20).map(|i| vec![i as f64 * 0.05]).collect()
}

/// Every query and maintenance operation, each run on its own 2 MiB thread with two fresh
/// trees from `build` (the second is the query tree or merge partner, so dual-tree queries
/// and merges are deep on both sides). Returns whether each completed (`false` = caught
/// panic).
fn exercise(build: fn() -> Tree) -> Vec<(&'static str, bool)> {
    let ops: [(&'static str, Op); 16] = [
        ("find_nearest", |t, _| { t.find_nearest(&vec![0.2]); }),
        ("find_k_nearest", |t, _| { t.find_k_nearest(&vec![0.2], 2); }),
        ("find_k_nearest_batch", |t, _| { t.find_k_nearest_batch(&qs(), 2); }),
        ("find_k_nearest_dual", |t, o| { t.find_k_nearest_dual(&o, 2); }),
        ("find_k_nearest_self", |t, _| { t.find_k_nearest_self(2); }),
        ("instrumented", |t, _| {
            t.find_k_nearest_batch_instrumented(&qs(), 2);
            t.find_k_nearest_self_instrumented(2);
        }),
        ("find_k_nearest_batch_single_self", |t, _| { t.find_k_nearest_batch_single_self(2); }),
        ("recompute", |mut t, _| { t.recompute_maxdist(); t.recompute_all(); }),
        ("d_parent_and_sort", |mut t, _| { t.recompute_d_parent(); t.sort_children_by_distance(); }),
        ("pack_and_query", |t, _| {
            let p = t.pack();
            p.find_nearest(&vec![0.2]);
            p.find_k_nearest(&vec![0.2], 2);
            p.find_k_nearest_self(2);
            p.find_k_nearest_batch_single_self(2);
        }),
        ("packed_batch_queries", |t, o| {
            let p = t.pack();
            p.find_k_nearest_batch(&qs(), 2);
            p.find_k_nearest_dual(&o, 2);
            p.find_k_nearest_batch_single(&qs(), 2);
            p.find_k_nearest_batch_dfs_instrumented(&qs(), 2);
            p.find_k_nearest_self_instrumented(2);
        }),
        ("merge", |t, _| {
            let mut o = SimplifiedCoverTree::new(EuclideanDistance, 1.3);
            o.insert(vec![0.3]);
            t.merge(o).len();
        }),
        ("merge_two_deep_trees", |t, o| { t.merge(o).len(); }),
        ("debug_format", |t, _| { let _ = format!("{:?}", t.root_node()); }),
        ("drop", |t, o| { drop(t); drop(o); }),
        ("tree_stats", |t, _| { t.tree_stats(); }),
    ];
    ops.iter()
        .map(|&(name, op)| (name, on_2mib(move || op(build(), build()))))
        .collect()
}

fn outlier_tree() -> Tree {
    build_parallel_simplified(EuclideanDistance, 1.3, vec![vec![0.0], vec![1.0], vec![0.5], vec![1e154]], Some(2))
}

fn outlier_tree_min_base() -> SimplifiedCoverTree<P, EuclideanDistance> {
    build_parallel_simplified(EuclideanDistance, MIN_BASE, vec![vec![0.0], vec![1.0], vec![0.5], vec![1e154]], Some(2))
}

fn geometric_tree() -> Tree {
    let mut t = SimplifiedCoverTree::new(EuclideanDistance, 1.3);
    let mut x = 1.0;
    t.insert(vec![0.0]);
    for _ in 0..3000 {
        t.insert(vec![x]);
        x /= 1.3;
    }
    t
}

fn ordinary_tree() -> Tree {
    let mut t = SimplifiedCoverTree::new(EuclideanDistance, 1.3);
    for p in points(37, 3000, 1) {
        t.insert(p);
    }
    t
}

/// The reported reproduction: points 0, 1, 0.5 and 1e154 built in parallel give a tree
/// over a thousand levels deep whose self-query used to abort on a 2 MiB thread. Every
/// operation must now complete or panic; reaching the end of the test means none aborted.
#[test]
fn deep_tree_from_parallel_outlier_never_aborts() {
    // Building may itself exceed the recursion budget in debug builds (a catchable
    // panic); when it succeeds, the tree is over a thousand levels deep.
    let built = on_2mib(|| assert!(depth(outlier_tree().root_node().unwrap()) > 1000));
    if built {
        let _ = exercise(outlier_tree);
    }
    // MIN_BASE: about 3,700 levels.
    let _ = on_2mib(|| assert!(depth(outlier_tree_min_base().root_node().unwrap()) > 3000));
    let _ = on_2mib(|| { outlier_tree_min_base().find_k_nearest_self(2); });
    let _ = on_2mib(|| { outlier_tree_min_base().find_k_nearest(&vec![0.2], 2); });
}

/// A geometric sequence inserted one point at a time also builds a very deep tree.
#[test]
fn deep_tree_from_geometric_inserts_never_aborts() {
    // Building may itself exceed the recursion budget in debug builds; it must panic,
    // not abort, and then there is no tree to query.
    if on_2mib(|| { geometric_tree(); }) {
        let _ = exercise(geometric_tree);
    }
}

/// Ordinary trees are well within the recursion budget: nothing may panic.
#[test]
fn ordinary_trees_are_within_the_recursion_budget() {
    assert!(depth(ordinary_tree().root_node().unwrap()) < 100);
    for (name, ok) in exercise(ordinary_tree) {
        assert!(ok, "{name} panicked on an ordinary tree");
    }
}
fn geometric_na_tree() -> NACoverTree<P, EuclideanDistance> {
    let mut t = NACoverTree::new(EuclideanDistance, 1.3);
    let mut x = 1.0;
    t.insert(vec![0.0]);
    for _ in 0..1500 {
        t.insert(vec![x]);
        x /= 1.3;
    }
    t
}

/// The nearest-ancestor tree's own queries and maintenance on a deep tree.
#[test]
fn deep_nearest_ancestor_tree_never_aborts() {
    type NaOp = fn(NACoverTree<P, EuclideanDistance>, NACoverTree<P, EuclideanDistance>);
    // Building may itself exceed the recursion budget in debug builds.
    if !on_2mib(|| assert!(depth(geometric_na_tree().root_node().unwrap()) > 500)) {
        return;
    }
    let ops: [NaOp; 9] = [
        |t, _| { t.find_nearest(&vec![0.2]); },
        |t, _| { t.find_k_nearest(&vec![0.2], 2); },
        |t, _| { t.find_k_nearest_batch(&qs(), 2); },
        |t, o| { t.find_k_nearest_dual(&o, 2); },
        |t, _| { t.find_k_nearest_self(2); },
        |mut t, _| t.recompute_maxdist(),
        |t, _| { t.into_simplified().find_k_nearest_self(2); },
        |t, o| { t.merge(o).find_k_nearest(&vec![0.2], 2); },
        |t, o| { drop(t); drop(o); },
    ];
    for op in ops {
        let _ = on_2mib(move || op(geometric_na_tree(), geometric_na_tree()));
    }
}

/// If a nearest-ancestor insert exceeds the recursion budget, the tree is left empty and
/// consistent, and stays usable.
#[test]
fn nearest_ancestor_insert_panic_leaves_a_consistent_tree() {
    let ok = on_2mib(|| {
        let mut na = NACoverTree::new(EuclideanDistance, 1.3);
        let mut x = 1.0;
        na.insert(vec![0.0]);
        let mut panicked = false;
        for _ in 0..6000 {
            let p = vec![x];
            x /= 1.3;
            if catch_unwind(AssertUnwindSafe(|| na.insert(p))).is_err() {
                panicked = true;
                break;
            }
        }
        if panicked {
            assert_eq!(na.len(), 0);
            assert!(na.root_node().is_none());
            na.insert(vec![1.0]);
            na.insert(vec![2.0]);
            assert_eq!(na.find_nearest(&vec![1.9]), Some(&vec![2.0]));
        }
    });
    assert!(ok);
}
