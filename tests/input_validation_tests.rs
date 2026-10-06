//! Tests for caller-supplied parameters: the tree `base` and the
//! neighbor count `k`. Invalid values must fail with a catchable panic (for `base`)
//! or be handled gracefully (for `k`), never hang or abort the process.

use rustknn::kdtree::KdTree;
use rustknn::naive::NaiveNN;
use rustknn::packed::PackedCoverTree;
use rustknn::parallel::{build_parallel, build_parallel_simplified, ParallelVariant};
use rustknn::{
    CoverTree, Distance, EuclideanDistance, NACoverTree, SimplifiedCoverTree, TreeVariant,
    MIN_BASE,
};
use std::panic::{catch_unwind, AssertUnwindSafe};

#[derive(Clone)]
struct Abs;
impl Distance<f64> for Abs {
    fn distance(&self, p: &f64, q: &f64) -> f64 {
        (p - q).abs()
    }
}

/// Runs `f`, expecting a panic whose message contains `needle`.
fn expect_panic<R>(f: impl FnOnce() -> R, needle: &str) {
    let err = match catch_unwind(AssertUnwindSafe(f)) {
        Ok(_) => panic!("expected a panic containing {:?}", needle),
        Err(e) => e,
    };
    let msg = err
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| err.downcast_ref::<&str>().copied())
        .unwrap_or("");
    assert!(msg.contains(needle), "panic message {:?} does not contain {:?}", msg, needle);
}

/// Runs `f` on a thread with a 2 MiB stack (the default for spawned and Rayon worker
/// threads), so deep recursion surfaces as a test failure rather than passing on the
/// larger main-thread stack.
fn on_small_stack<R: Send + 'static>(f: impl FnOnce() -> R + Send + 'static) -> R {
    std::thread::Builder::new()
        .stack_size(2 << 20)
        .spawn(f)
        .unwrap()
        .join()
        .unwrap_or_else(|e| std::panic::resume_unwind(e))
}

// ---------------------------------------------------------------------------
// base validation
// ---------------------------------------------------------------------------

const INVALID_BASES: [f64; 9] = [
    1.0 - 1e-10,
    1.0,
    1.0 + 1e-5,
    0.5,
    -2.0,
    f64::NAN,
    f64::INFINITY,
    f64::NEG_INFINITY,
    1.0999,
];

const BASE_MSG: &str = "cover tree base must be";

#[test]
fn every_constructor_rejects_invalid_base() {
    for &base in &INVALID_BASES {
        expect_panic(|| SimplifiedCoverTree::<f64, _>::new(Abs, base), BASE_MSG);
        expect_panic(|| NACoverTree::<f64, _>::new(Abs, base), BASE_MSG);
        expect_panic(|| CoverTree::<f64, _>::new(Abs, base), BASE_MSG);
        expect_panic(|| CoverTree::<f64, _>::new_nearest_ancestor(Abs, base), BASE_MSG);
        expect_panic(
            || CoverTree::<f64, _>::with_options(Abs, base, TreeVariant::Simplified),
            BASE_MSG,
        );
        expect_panic(
            || CoverTree::<f64, _>::with_options(Abs, base, TreeVariant::NearestAncestor),
            BASE_MSG,
        );
        expect_panic(|| PackedCoverTree::from_incremental(vec![0.0, 2.0], Abs, base), BASE_MSG);
        expect_panic(|| build_parallel_simplified(Abs, base, vec![0.0, 2.0, 100.0], Some(2)), BASE_MSG);
        expect_panic(
            || build_parallel(Abs, base, vec![0.0, 2.0, 100.0], Some(2), ParallelVariant::NearestAncestor),
            BASE_MSG,
        );
        // Empty input still validates the base.
        expect_panic(|| build_parallel_simplified(Abs, base, Vec::<f64>::new(), Some(2)), BASE_MSG);
    }
}

#[test]
fn min_base_and_default_base_are_accepted() {
    for &base in &[MIN_BASE, 1.3, 2.0] {
        let mut t = SimplifiedCoverTree::new(Abs, base);
        t.insert(0.0);
        t.insert(2.0);
        t.insert(100.0);
        assert_eq!(t.find_k_nearest(&1.9, 1)[0].0, &2.0);
        let mut na = NACoverTree::new(Abs, base);
        na.insert(0.0);
        na.insert(2.0);
        assert_eq!(na.find_nearest(&1.9), Some(&2.0));
    }
}

/// A base just below 1 would overflow the level arithmetic; it must be rejected.
#[test]
fn base_just_below_one_is_rejected() {
    expect_panic(
        || {
            let mut t = SimplifiedCoverTree::new(Abs, 1.0 - 1e-10);
            t.insert(0.0);
            t.insert(2.0);
            t.find_k_nearest_batch_single_self(2).len()
        },
        BASE_MSG,
    );
}

/// A base just above 1 would make merge() bridge a level gap of tens of thousands of
/// levels; it must be rejected.
#[test]
fn merge_with_base_just_above_one_is_rejected() {
    on_small_stack(|| {
        expect_panic(
            || {
                let base = 1.0 + 1e-5;
                let mut a = SimplifiedCoverTree::new(Abs, base);
                a.insert(0.0);
                a.insert(2.0);
                let mut b = SimplifiedCoverTree::new(Abs, base);
                b.insert(100.0);
                a.merge(b).len()
            },
            BASE_MSG,
        )
    });
}

/// The same base, reached through the parallel builder (which merges internally).
#[test]
fn parallel_with_base_just_above_one_is_rejected() {
    on_small_stack(|| {
        expect_panic(
            || build_parallel_simplified(Abs, 1.0 + 1e-5, vec![0.0, 2.0, 100.0], Some(2)).len(),
            BASE_MSG,
        )
    });
}

/// Worst case for an accepted base: data spanning ~600 orders of magnitude at
/// MIN_BASE. Every operation must either succeed or fail with a catchable panic.
#[test]
fn extreme_dynamic_range_at_min_base_never_aborts() {
    on_small_stack(|| {
        let points = vec![0.0, 1e-300, 1e300, 5.0, 7.0];

        let mut a = SimplifiedCoverTree::new(Abs, MIN_BASE);
        for &p in &points[..3] {
            a.insert(p);
        }
        assert_eq!(a.find_k_nearest(&1e-300, 1)[0].0, &1e-300);
        assert_eq!(a.find_k_nearest_batch_single_self(1).len(), 3);
        let packed = SimplifiedCoverTree::new(Abs, MIN_BASE);
        let mut packed = packed;
        for &p in &points[..3] {
            packed.insert(p);
        }
        let packed = packed.pack();
        assert_eq!(packed.find_k_nearest(&1e300, 2).len(), 2);
        assert_eq!(packed.find_k_nearest_self(1).len(), 3);

        // Root levels ~7,248 apart: merge refuses rather than building the chain.
        let mut b = SimplifiedCoverTree::new(Abs, MIN_BASE);
        b.insert(5.0);
        expect_panic(|| a.merge(b).len(), "cannot merge cover trees");

        expect_panic(
            || build_parallel_simplified(Abs, MIN_BASE, points.clone(), Some(2)).len(),
            "cannot merge cover trees",
        );
    });
}

#[test]
fn infinite_distance_from_metric_is_rejected() {
    struct Broken;
    impl Distance<f64> for Broken {
        fn distance(&self, p: &f64, q: &f64) -> f64 {
            if p == q { 0.0 } else { f64::INFINITY }
        }
    }
    expect_panic(
        || {
            let mut t = SimplifiedCoverTree::new(Broken, 1.3);
            t.insert(0.0);
            t.insert(1.0);
        },
        "invalid distance",
    );
    expect_panic(
        || {
            let mut t = NACoverTree::new(Broken, 1.3);
            t.insert(0.0);
            t.insert(1.0);
        },
        "invalid distance",
    );
}

#[test]
fn parallel_build_with_zero_threads_uses_one() {
    let t = build_parallel_simplified(Abs, 1.3, vec![0.0, 2.0, 100.0], Some(0));
    assert_eq!(t.len(), 3);
}

// ---------------------------------------------------------------------------
// oversized k
// ---------------------------------------------------------------------------

const HUGE_KS: [usize; 2] = [1 << 50, usize::MAX];

fn grid() -> Vec<Vec<f64>> {
    (0..16).map(|i| vec![(i % 4) as f64, (i / 4) as f64]).collect()
}

fn queries() -> Vec<Vec<f64>> {
    (0..8).map(|i| vec![i as f64 * 0.4, 0.3]).collect()
}

fn simplified() -> SimplifiedCoverTree<Vec<f64>, EuclideanDistance> {
    let mut t = SimplifiedCoverTree::new(EuclideanDistance, 1.3);
    for p in grid() {
        t.insert(p);
    }
    t
}

fn na() -> NACoverTree<Vec<f64>, EuclideanDistance> {
    let mut t = NACoverTree::new(EuclideanDistance, 1.3);
    for p in grid() {
        t.insert(p);
    }
    t
}

fn cover_tree(variant: TreeVariant) -> CoverTree<Vec<f64>, EuclideanDistance> {
    let mut t = CoverTree::with_options(EuclideanDistance, 1.3, variant);
    for p in grid() {
        t.insert(p);
    }
    t
}

/// Sorted distances of one result list, rounded to 1e-9. Ties may come back in any
/// order, and different tree variants evaluate `d(p, q)` versus `d(q, p)`, which the
/// fused multiply-add kernel can round differently in the last bit.
fn dists<T>(r: &[(&T, f64)]) -> Vec<f64> {
    let mut d: Vec<f64> = r.iter().map(|(_, d)| (d * 1e9).round() / 1e9).collect();
    d.sort_by(f64::total_cmp);
    d
}

/// Distance lists for a set of queries, as a sorted multiset of rows so that
/// variants returning queries in different orders compare equal.
fn rows<T>(rs: &[Vec<(&T, f64)>]) -> Vec<Vec<f64>> {
    let mut v: Vec<Vec<f64>> = rs.iter().map(|r| dists(r)).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

/// Same as `rows`, for the batch single-tree result shape.
fn batch_single_rows(rs: Vec<(*const Vec<f64>, Vec<(&Vec<f64>, f64)>)>) -> Vec<Vec<f64>> {
    let mut v: Vec<Vec<f64>> = rs.iter().map(|(_, r)| dists(r)).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

#[test]
fn huge_k_returns_every_point_for_single_queries() {
    let q = vec![0.5, 0.5];
    let n = grid().len();
    let packed = simplified().pack();
    let kd = KdTree::new(grid(), EuclideanDistance);
    let naive = NaiveNN::new(grid(), EuclideanDistance);
    let expected = dists(&simplified().find_k_nearest(&q, n));
    assert_eq!(expected.len(), n);
    for &k in &HUGE_KS {
        assert_eq!(dists(&simplified().find_k_nearest(&q, k)), expected, "simplified k={k}");
        assert_eq!(dists(&packed.find_k_nearest(&q, k)), expected, "packed k={k}");
        assert_eq!(dists(&na().find_k_nearest(&q, k)), expected, "na k={k}");
        for variant in [TreeVariant::Simplified, TreeVariant::NearestAncestor] {
            assert_eq!(dists(&cover_tree(variant).find_k_nearest(&q, k)), expected, "covertree k={k}");
        }
        assert_eq!(dists(&kd.find_k_nearest(&q, k)), expected, "kdtree k={k}");
        assert_eq!(dists(&naive.find_k_nearest(&q, k)), expected, "naive k={k}");
    }
}

#[test]
fn huge_k_returns_every_point_for_batch_queries() {
    let qs = queries();
    let n = grid().len();
    let packed = simplified().pack();
    let expected = rows(&simplified().find_k_nearest_batch(&qs, n));
    assert!(expected.iter().all(|d| d.len() == n));
    for &k in &HUGE_KS {
        assert_eq!(rows(&simplified().find_k_nearest_batch(&qs, k)), expected, "simplified k={k}");
        assert_eq!(rows(&packed.find_k_nearest_batch(&qs, k)), expected, "packed k={k}");
        assert_eq!(rows(&packed.find_k_nearest_batch_single(&qs, k)), expected, "packed batch single k={k}");
        assert_eq!(rows(&na().find_k_nearest_batch(&qs, k)), expected, "na k={k}");
        for variant in [TreeVariant::Simplified, TreeVariant::NearestAncestor] {
            assert_eq!(rows(&cover_tree(variant).find_k_nearest_batch(&qs, k)), expected, "covertree k={k}");
        }
    }
}

#[test]
fn huge_k_returns_every_other_point_for_self_queries() {
    let n = grid().len();
    let packed = simplified().pack();
    let kd = KdTree::new(grid(), EuclideanDistance);
    let expected = rows(&simplified().find_k_nearest_self(n - 1));
    assert!(expected.iter().all(|d| d.len() == n - 1));
    let expected_kd = rows(&kd.find_k_nearest_self(n));
    for &k in &HUGE_KS {
        assert_eq!(rows(&simplified().find_k_nearest_self(k)), expected, "simplified k={k}");
        assert_eq!(rows(&packed.find_k_nearest_self(k)), expected, "packed k={k}");
        assert_eq!(rows(&na().find_k_nearest_self(k)), expected, "na k={k}");
        for variant in [TreeVariant::Simplified, TreeVariant::NearestAncestor] {
            assert_eq!(rows(&cover_tree(variant).find_k_nearest_self(k)), expected, "covertree k={k}");
        }
        // KdTree::find_k_nearest_self includes each point itself (distance 0).
        assert_eq!(rows(&kd.find_k_nearest_self(k)), expected_kd, "kdtree k={k}");
        assert_eq!(
            batch_single_rows(simplified().find_k_nearest_batch_single_self(k)),
            expected,
            "batch single k={k}"
        );
        let packed_bs: Vec<Vec<f64>> = {
            let mut v: Vec<Vec<f64>> = packed
                .find_k_nearest_batch_single_self(k)
                .iter()
                .map(|(_, r)| dists(r))
                .collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v
        };
        assert_eq!(packed_bs, expected, "packed batch single k={k}");
        // Timing-only packed batch single-tree entry points: must complete.
        packed.batch_single_self_query_instrumented(k);
        packed.batch_single_batch_query_instrumented(&queries(), k);
    }
}

#[test]
fn k_zero_returns_nothing() {
    let q = vec![0.5, 0.5];
    let qs = queries();
    let packed = simplified().pack();
    assert!(simplified().find_k_nearest(&q, 0).is_empty());
    assert!(packed.find_k_nearest(&q, 0).is_empty());
    assert!(na().find_k_nearest(&q, 0).is_empty());
    assert!(KdTree::new(grid(), EuclideanDistance).find_k_nearest(&q, 0).is_empty());
    assert!(NaiveNN::new(grid(), EuclideanDistance).find_k_nearest(&q, 0).is_empty());
    assert!(simplified().find_k_nearest_batch(&qs, 0).iter().all(Vec::is_empty));
    assert!(packed.find_k_nearest_batch(&qs, 0).iter().all(Vec::is_empty));
    assert!(simplified().find_k_nearest_self(0).iter().all(Vec::is_empty));
    assert!(packed.find_k_nearest_self(0).iter().all(Vec::is_empty));
    assert!(simplified().find_k_nearest_batch_single_self(0).iter().all(|(_, r)| r.is_empty()));
    assert!(packed.find_k_nearest_batch_single_self(0).iter().all(|(_, r)| r.is_empty()));
    assert!(packed.find_k_nearest_batch_single(&qs, 0).iter().all(Vec::is_empty));
}
