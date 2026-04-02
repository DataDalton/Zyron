//! Inverted index for full-text search.
//!
//! Structure-of-Arrays (SoA) PostingsList for cache-friendly SIMD scoring.
//! Tiered SIMD BM25: AVX-512 (8-wide) -> AVX2 (4-wide) on x86_64,
//! NEON (2-wide) on aarch64. Merge intersection for phrase search.
//! Dense ScoreAccumulator with threshold-based top-K extraction.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::io::{Read as IoRead, Write as IoWrite};
use std::path::Path;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;
use zyron_common::{Result, ZyronError};

use super::analyzer::{AnalysisBuffer, Analyzer};
use super::query::{FtsQuery, edit_distance, wildcard_match};
use super::scoring::RelevanceScorer;

pub type DocId = u64;
const MAX_DOC_PAGE_ID: u64 = 0xFFFF_FFFF_FFFF;
const DENSE_SCORE_LIMIT: u64 = 4_000_000;

pub fn encode_doc_id(page_id: u64, slot_id: u16) -> Result<DocId> {
    if page_id > MAX_DOC_PAGE_ID {
        return Err(ZyronError::Internal(
            "page_id exceeds 48-bit DocId limit".into(),
        ));
    }
    Ok((page_id << 16) | (slot_id as u64))
}

pub fn decode_doc_id(doc_id: DocId) -> (u64, u16) {
    (doc_id >> 16, (doc_id & 0xFFFF) as u16)
}

#[derive(Debug, Clone)]
pub struct TermInfo {
    pub doc_frequency: u32,
    pub postings_offset: u64,
    pub postings_length: u32,
}

/// View type for public API compatibility.
#[derive(Debug, Clone)]
pub struct Posting {
    pub doc_id: DocId,
    pub term_frequency: u16,
    pub field_length: u32,
    pub positions: Vec<u32>,
}

/// SoA postings list. Parallel arrays sorted by doc_id.
#[derive(Debug, Clone)]
pub struct PostingsList {
    pub doc_ids: Vec<u64>,
    pub term_freqs: Vec<u16>,
    pub field_lengths: Vec<u32>,
    pub position_offsets: Vec<u32>,
    pub positions: Vec<u32>,
}

impl PostingsList {
    fn new() -> Self {
        Self {
            doc_ids: Vec::new(),
            term_freqs: Vec::new(),
            field_lengths: Vec::new(),
            position_offsets: Vec::new(),
            positions: Vec::new(),
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.doc_ids.len()
    }
    #[inline]
    fn is_empty(&self) -> bool {
        self.doc_ids.is_empty()
    }

    fn positions_for(&self, idx: usize) -> &[u32] {
        let start = self.position_offsets[idx] as usize;
        let end = if idx + 1 < self.position_offsets.len() {
            self.position_offsets[idx + 1] as usize
        } else {
            self.positions.len()
        };
        &self.positions[start..end]
    }

    fn push(&mut self, doc_id: DocId, tf: u16, fl: u32, positions: &[u32]) {
        self.doc_ids.push(doc_id);
        self.term_freqs.push(tf);
        self.field_lengths.push(fl);
        self.position_offsets.push(self.positions.len() as u32);
        self.positions.extend_from_slice(positions);
    }

    /// Start a new document entry without collecting positions into an intermediate Vec.
    /// Call push_position() for each position, then move to the next term group.
    #[inline]
    fn begin_doc(&mut self, doc_id: DocId, tf: u16, fl: u32) {
        self.doc_ids.push(doc_id);
        self.term_freqs.push(tf);
        self.field_lengths.push(fl);
        self.position_offsets.push(self.positions.len() as u32);
    }

    #[inline]
    fn push_position(&mut self, pos: u32) {
        self.positions.push(pos);
    }

    fn insert_sorted(&mut self, idx: usize, doc_id: DocId, tf: u16, fl: u32, positions: &[u32]) {
        self.doc_ids.insert(idx, doc_id);
        self.term_freqs.insert(idx, tf);
        self.field_lengths.insert(idx, fl);
        let pos_start = self.positions.len() as u32;
        self.position_offsets.insert(idx, pos_start);
        self.positions.extend_from_slice(positions);
    }

    fn replace(&mut self, idx: usize, tf: u16, fl: u32, positions: &[u32]) {
        self.term_freqs[idx] = tf;
        self.field_lengths[idx] = fl;
        let old_start = self.position_offsets[idx] as usize;
        let old_end = if idx + 1 < self.position_offsets.len() {
            self.position_offsets[idx + 1] as usize
        } else {
            self.positions.len()
        };
        let old_len = old_end - old_start;
        let new_len = positions.len();
        if old_len == new_len {
            self.positions[old_start..old_end].copy_from_slice(positions);
        } else {
            self.positions
                .splice(old_start..old_end, positions.iter().copied());
            let diff = new_len as i64 - old_len as i64;
            for off in &mut self.position_offsets[idx + 1..] {
                *off = (*off as i64 + diff) as u32;
            }
        }
    }

    fn remove(&mut self, idx: usize) {
        let pos_start = self.position_offsets[idx] as usize;
        let pos_end = if idx + 1 < self.position_offsets.len() {
            self.position_offsets[idx + 1] as usize
        } else {
            self.positions.len()
        };
        let pos_len = pos_end - pos_start;
        self.doc_ids.remove(idx);
        self.term_freqs.remove(idx);
        self.field_lengths.remove(idx);
        self.position_offsets.remove(idx);
        if pos_len > 0 {
            self.positions.drain(pos_start..pos_end);
            for off in &mut self.position_offsets[idx..] {
                *off -= pos_len as u32;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ScoreAccumulator
// ---------------------------------------------------------------------------

enum ScoreAccumulator {
    Dense {
        scores: Vec<f64>,
        scored_ids: Vec<u32>,
    },
    Sparse(HashMap<DocId, f64>),
}

impl ScoreAccumulator {
    fn new(max_doc_id: u64) -> Self {
        if max_doc_id < DENSE_SCORE_LIMIT {
            ScoreAccumulator::Dense {
                scores: vec![0.0f64; (max_doc_id + 1) as usize],
                scored_ids: Vec::with_capacity(4096),
            }
        } else {
            ScoreAccumulator::Sparse(HashMap::with_capacity(1024))
        }
    }

    #[inline(always)]
    fn add(&mut self, doc_id: DocId, score: f64) {
        match self {
            ScoreAccumulator::Dense { scores, scored_ids } => {
                let idx = doc_id as usize;
                unsafe {
                    let slot = scores.get_unchecked_mut(idx);
                    if *slot == 0.0 {
                        scored_ids.push(idx as u32);
                    }
                    *slot += score;
                }
            }
            ScoreAccumulator::Sparse(map) => {
                *map.entry(doc_id).or_insert(0.0) += score;
            }
        }
    }

    fn get_mut(&mut self, doc_id: DocId) -> Option<&mut f64> {
        match self {
            ScoreAccumulator::Dense { scores, .. } => {
                let i = doc_id as usize;
                if i < scores.len() && scores[i] != 0.0 {
                    Some(&mut scores[i])
                } else {
                    None
                }
            }
            ScoreAccumulator::Sparse(map) => map.get_mut(&doc_id),
        }
    }

    /// Reset only the entries that were actually scored, avoiding full array zeroing.
    /// For high-density scoring (>25% of array touched), bulk memset is faster
    /// than random-access clearing of individual entries.
    fn clear_scored(&mut self) {
        match self {
            ScoreAccumulator::Dense { scores, scored_ids } => {
                if scored_ids.len() > scores.len() / 4 {
                    scores.fill(0.0);
                } else {
                    for &idx in scored_ids.iter() {
                        unsafe {
                            *scores.get_unchecked_mut(idx as usize) = 0.0;
                        }
                    }
                }
                scored_ids.clear();
            }
            ScoreAccumulator::Sparse(map) => {
                map.clear();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Top-K with threshold-based early rejection
// ---------------------------------------------------------------------------

#[derive(PartialEq)]
struct ScoredDoc(f64, DocId);
impl Eq for ScoredDoc {}
impl PartialOrd for ScoredDoc {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ScoredDoc {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

fn top_k_from_accumulator(acc: ScoreAccumulator, limit: usize) -> Vec<(DocId, f64)> {
    match acc {
        ScoreAccumulator::Dense { scores, scored_ids } => {
            if scored_ids.is_empty() {
                return Vec::new();
            }
            let mut heap: BinaryHeap<Reverse<ScoredDoc>> = BinaryHeap::with_capacity(limit + 1);
            let mut threshold = f64::NEG_INFINITY;
            for idx in scored_ids {
                let score = unsafe { *scores.get_unchecked(idx as usize) };
                if heap.len() >= limit && score <= threshold {
                    continue;
                }
                heap.push(Reverse(ScoredDoc(score, idx as DocId)));
                if heap.len() > limit {
                    heap.pop();
                    if let Some(Reverse(ScoredDoc(min, _))) = heap.peek() {
                        threshold = *min;
                    }
                }
            }
            let mut r: Vec<(DocId, f64)> = heap
                .into_iter()
                .map(|Reverse(ScoredDoc(s, d))| (d, s))
                .collect();
            r.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            r
        }
        ScoreAccumulator::Sparse(map) => {
            if map.is_empty() {
                return Vec::new();
            }
            let mut heap: BinaryHeap<Reverse<ScoredDoc>> = BinaryHeap::with_capacity(limit + 1);
            let mut threshold = f64::NEG_INFINITY;
            for (doc_id, score) in map {
                if heap.len() >= limit && score <= threshold {
                    continue;
                }
                heap.push(Reverse(ScoredDoc(score, doc_id)));
                if heap.len() > limit {
                    heap.pop();
                    if let Some(Reverse(ScoredDoc(min, _))) = heap.peek() {
                        threshold = *min;
                    }
                }
            }
            let mut r: Vec<(DocId, f64)> = heap
                .into_iter()
                .map(|Reverse(ScoredDoc(s, d))| (d, s))
                .collect();
            r.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            r
        }
    }
}

// ---------------------------------------------------------------------------
// Inlined BM25
// ---------------------------------------------------------------------------

#[inline(always)]
fn bm25_idf(df: u32, total_docs: u64) -> f64 {
    if total_docs == 0 {
        return 0.0;
    }
    let n = total_docs as f64;
    let df = (df as f64).min(n);
    ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
}

#[inline(always)]
fn bm25_tf_norm(tf: u16, fl: u32, avg_dl: f64, k1: f64, b: f64) -> f64 {
    if tf == 0 {
        return 0.0;
    }
    let tf = tf as f64;
    let dl = fl as f64;
    let avgdl = if avg_dl > 0.0 { avg_dl } else { 1.0 };
    (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * dl / avgdl))
}

// ---------------------------------------------------------------------------
// SIMD scoring
// ---------------------------------------------------------------------------

type ScoreFn =
    unsafe fn(&[u64], &[u16], &[u32], f64, f64, f64, f64, f64, &mut [f64], &mut Vec<u32>);
static SCORE_FN: OnceLock<ScoreFn> = OnceLock::new();
fn get_score_fn() -> ScoreFn {
    *SCORE_FN.get_or_init(select_score_fn)
}

#[cfg(target_arch = "x86_64")]
fn select_score_fn() -> ScoreFn {
    if is_x86_feature_detected!("avx512f") {
        score_avx512
    } else {
        score_avx2
    }
}

#[cfg(target_arch = "aarch64")]
fn select_score_fn() -> ScoreFn {
    score_neon
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn select_score_fn() -> ScoreFn {
    score_generic
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn score_avx512(
    doc_ids: &[u64],
    tfs: &[u16],
    fls: &[u32],
    idf: f64,
    k1p1: f64,
    k1: f64,
    omb: f64,
    bdav: f64,
    scores: &mut [f64],
    scored_ids: &mut Vec<u32>,
) {
    use std::arch::x86_64::*;
    let n = doc_ids.len();
    let chunks = n / 8;
    let vi = _mm512_set1_pd(idf);
    let vk = _mm512_set1_pd(k1p1);
    let v1 = _mm512_set1_pd(k1);
    let vo = _mm512_set1_pd(omb);
    let vb = _mm512_set1_pd(bdav);
    for c in 0..chunks {
        let b = c * 8;
        let vt = _mm512_set_pd(
            tfs[b + 7] as f64,
            tfs[b + 6] as f64,
            tfs[b + 5] as f64,
            tfs[b + 4] as f64,
            tfs[b + 3] as f64,
            tfs[b + 2] as f64,
            tfs[b + 1] as f64,
            tfs[b] as f64,
        );
        let vd = _mm512_set_pd(
            fls[b + 7] as f64,
            fls[b + 6] as f64,
            fls[b + 5] as f64,
            fls[b + 4] as f64,
            fls[b + 3] as f64,
            fls[b + 2] as f64,
            fls[b + 1] as f64,
            fls[b] as f64,
        );
        let num = _mm512_mul_pd(vt, vk);
        let den = _mm512_add_pd(
            vt,
            _mm512_mul_pd(v1, _mm512_add_pd(vo, _mm512_mul_pd(vb, vd))),
        );
        let sc = _mm512_mul_pd(vi, _mm512_div_pd(num, den));
        let mut out = [0.0f64; 8];
        _mm512_storeu_pd(out.as_mut_ptr(), sc);
        for j in 0..8 {
            let did = *doc_ids.get_unchecked(b + j) as usize;
            let slot = scores.get_unchecked_mut(did);
            if *slot == 0.0 {
                scored_ids.push(did as u32);
            }
            *slot += *out.get_unchecked(j);
        }
    }
    for i in (chunks * 8)..n {
        let sc =
            idf * ((tfs[i] as f64 * k1p1) / (tfs[i] as f64 + k1 * (omb + bdav * fls[i] as f64)));
        let did = doc_ids[i] as usize;
        let slot = scores.get_unchecked_mut(did);
        if *slot == 0.0 {
            scored_ids.push(did as u32);
        }
        *slot += sc;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn score_avx2(
    doc_ids: &[u64],
    tfs: &[u16],
    fls: &[u32],
    idf: f64,
    k1p1: f64,
    k1: f64,
    omb: f64,
    bdav: f64,
    scores: &mut [f64],
    scored_ids: &mut Vec<u32>,
) {
    use std::arch::x86_64::*;
    let n = doc_ids.len();
    let chunks = n / 4;
    let vi = _mm256_set1_pd(idf);
    let vk = _mm256_set1_pd(k1p1);
    let v1 = _mm256_set1_pd(k1);
    let vo = _mm256_set1_pd(omb);
    let vb = _mm256_set1_pd(bdav);
    for c in 0..chunks {
        let b = c * 4;
        let vt = _mm256_set_pd(
            tfs[b + 3] as f64,
            tfs[b + 2] as f64,
            tfs[b + 1] as f64,
            tfs[b] as f64,
        );
        let vd = _mm256_set_pd(
            fls[b + 3] as f64,
            fls[b + 2] as f64,
            fls[b + 1] as f64,
            fls[b] as f64,
        );
        let num = _mm256_mul_pd(vt, vk);
        let den = _mm256_add_pd(
            vt,
            _mm256_mul_pd(v1, _mm256_add_pd(vo, _mm256_mul_pd(vb, vd))),
        );
        let sc = _mm256_mul_pd(vi, _mm256_div_pd(num, den));
        let mut out = [0.0f64; 4];
        _mm256_storeu_pd(out.as_mut_ptr(), sc);
        for j in 0..4 {
            let did = *doc_ids.get_unchecked(b + j) as usize;
            let slot = scores.get_unchecked_mut(did);
            if *slot == 0.0 {
                scored_ids.push(did as u32);
            }
            *slot += *out.get_unchecked(j);
        }
    }
    for i in (chunks * 4)..n {
        let sc =
            idf * ((tfs[i] as f64 * k1p1) / (tfs[i] as f64 + k1 * (omb + bdav * fls[i] as f64)));
        let did = doc_ids[i] as usize;
        let slot = scores.get_unchecked_mut(did);
        if *slot == 0.0 {
            scored_ids.push(did as u32);
        }
        *slot += sc;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn score_neon(
    doc_ids: &[u64],
    tfs: &[u16],
    fls: &[u32],
    idf: f64,
    k1p1: f64,
    k1: f64,
    omb: f64,
    bdav: f64,
    scores: &mut [f64],
    scored_ids: &mut Vec<u32>,
) {
    use std::arch::aarch64::*;
    let n = doc_ids.len();
    let chunks = n / 2;
    let vi = vdupq_n_f64(idf);
    let vk = vdupq_n_f64(k1p1);
    let v1 = vdupq_n_f64(k1);
    let vo = vdupq_n_f64(omb);
    let vb = vdupq_n_f64(bdav);
    for c in 0..chunks {
        let b = c * 2;
        let ta = [tfs[b] as f64, tfs[b + 1] as f64];
        let da = [fls[b] as f64, fls[b + 1] as f64];
        let vt = vld1q_f64(ta.as_ptr());
        let vd = vld1q_f64(da.as_ptr());
        let num = vmulq_f64(vt, vk);
        let den = vaddq_f64(vt, vmulq_f64(v1, vaddq_f64(vo, vmulq_f64(vb, vd))));
        let sc = vmulq_f64(vi, vdivq_f64(num, den));
        let mut out = [0.0f64; 2];
        vst1q_f64(out.as_mut_ptr(), sc);
        for j in 0..2 {
            let did = *doc_ids.get_unchecked(b + j) as usize;
            let slot = scores.get_unchecked_mut(did);
            if *slot == 0.0 {
                scored_ids.push(did as u32);
            }
            *slot += *out.get_unchecked(j);
        }
    }
    if n % 2 != 0 {
        let i = n - 1;
        let sc =
            idf * ((tfs[i] as f64 * k1p1) / (tfs[i] as f64 + k1 * (omb + bdav * fls[i] as f64)));
        let did = doc_ids[i] as usize;
        let slot = scores.get_unchecked_mut(did);
        if *slot == 0.0 {
            scored_ids.push(did as u32);
        }
        *slot += sc;
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn score_generic(
    doc_ids: &[u64],
    tfs: &[u16],
    fls: &[u32],
    idf: f64,
    k1p1: f64,
    k1: f64,
    omb: f64,
    bdav: f64,
    scores: &mut [f64],
    scored_ids: &mut Vec<u32>,
) {
    for i in 0..doc_ids.len() {
        let sc =
            idf * ((tfs[i] as f64 * k1p1) / (tfs[i] as f64 + k1 * (omb + bdav * fls[i] as f64)));
        let did = doc_ids[i] as usize;
        let slot = scores.get_unchecked_mut(did);
        if *slot == 0.0 {
            scored_ids.push(did as u32);
        }
        *slot += sc;
    }
}

#[inline]
fn score_postings_simd(
    list: &PostingsList,
    total_docs: u64,
    avg_dl: f64,
    acc: &mut ScoreAccumulator,
) {
    if list.is_empty() {
        return;
    }
    let df = list.len() as u32;
    let idf = bm25_idf(df, total_docs);
    if idf <= 0.0 {
        return;
    }
    let k1 = 1.2f64;
    let b = 0.75f64;
    let k1p1 = k1 + 1.0;
    let omb = 1.0 - b;
    let avgdl = if avg_dl > 0.0 { avg_dl } else { 1.0 };
    let bdav = b / avgdl;
    match acc {
        ScoreAccumulator::Dense { scores, scored_ids } => unsafe {
            // All doc_ids in the postings list must fit within the scores array.
            debug_assert!(
                list.doc_ids.iter().all(|&d| (d as usize) < scores.len()),
                "doc_id exceeds dense score array bounds"
            );
            get_score_fn()(
                &list.doc_ids,
                &list.term_freqs,
                &list.field_lengths,
                idf,
                k1p1,
                k1,
                omb,
                bdav,
                scores,
                scored_ids,
            );
        },
        ScoreAccumulator::Sparse(map) => {
            for i in 0..list.len() {
                let tf_norm =
                    bm25_tf_norm(list.term_freqs[i], list.field_lengths[i], avg_dl, k1, b);
                *map.entry(list.doc_ids[i]).or_insert(0.0) += idf * tf_norm;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Merge intersection
// ---------------------------------------------------------------------------

/// Galloping (exponential) search for `target` in a sorted slice starting at `from`.
/// Returns Some(idx) where slice[idx] == target, or None if not found.
/// Advances exponentially then binary searches within the narrowed range.
#[inline]
fn gallop_search(slice: &[u64], target: u64, from: usize) -> Option<usize> {
    let len = slice.len();
    if from >= len {
        return None;
    }
    if slice[from] == target {
        return Some(from);
    }
    if slice[from] > target {
        return None;
    }

    // Exponential jump to find upper bound.
    let mut bound = 1usize;
    while from + bound < len && slice[from + bound] < target {
        bound *= 2;
    }
    // Binary search within [from + bound/2, min(from + bound, len-1)].
    let lo = from + bound / 2;
    let hi = (from + bound).min(len - 1) + 1; // exclusive upper bound
    match slice[lo..hi].binary_search(&target) {
        Ok(i) => Some(lo + i),
        Err(_) => None,
    }
}

fn intersect_sorted_ids(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut result = Vec::with_capacity(a.len().min(b.len()));
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Equal => {
                result.push(a[i]);
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    result
}

// ---------------------------------------------------------------------------
// InvertedIndex
// ---------------------------------------------------------------------------

pub struct InvertedIndex {
    postings: RwLock<HashMap<String, PostingsList>>,
    field_norms: RwLock<HashMap<DocId, u32>>,
    doc_count: AtomicU64,
    total_doc_length: AtomicU64,
    max_doc_id: AtomicU64,
    pub index_id: u32,
    pub table_id: u32,
    pub column_ids: Vec<u16>,
}

impl InvertedIndex {
    pub fn new(index_id: u32, table_id: u32, column_ids: Vec<u16>) -> Self {
        Self {
            postings: RwLock::new(HashMap::new()),
            field_norms: RwLock::new(HashMap::new()),
            doc_count: AtomicU64::new(0),
            total_doc_length: AtomicU64::new(0),
            max_doc_id: AtomicU64::new(0),
            index_id,
            table_id,
            column_ids,
        }
    }

    /// Indexes a document. Creates a temporary AnalysisBuffer per call.
    /// For bulk indexing, use `add_document_with_buf` with a reusable buffer.
    pub fn add_document(&self, doc_id: DocId, text: &str, analyzer: &dyn Analyzer) -> Result<()> {
        let mut buf = AnalysisBuffer::new();
        self.add_document_with_buf(doc_id, text, analyzer, &mut buf)
    }

    /// Indexes a document using a reusable AnalysisBuffer. After warmup,
    /// the analysis step performs zero heap allocations. The buffer should
    /// be created once and passed to each call.
    pub fn add_document_with_buf(
        &self,
        doc_id: DocId,
        text: &str,
        analyzer: &dyn Analyzer,
        buf: &mut AnalysisBuffer,
    ) -> Result<()> {
        analyzer.analyze_into(text, buf);
        let doc_length = buf.len() as u32;

        // Sort token indices by term text for grouping. Uses a u64 prefix key
        // for O(1) comparison in the common case, falling back to full string
        // comparison only when prefixes collide. This is ~5x faster than
        // sorting by string comparison directly.
        let mut indices: Vec<(u64, usize, u32)> = (0..buf.len())
            .map(|i| {
                let t = buf.term_at(i).as_bytes();
                let mut key = 0u64;
                for (j, &b) in t.iter().take(8).enumerate() {
                    key |= (b as u64) << (56 - j * 8);
                }
                (key, i, buf.position_at(i))
            })
            .collect();
        indices.sort_unstable_by(|a, b| {
            a.0.cmp(&b.0)
                .then_with(|| buf.term_at(a.1).cmp(buf.term_at(b.1)))
        });

        // Pre-compute term groups outside the lock: (group_start, group_end, tf).
        let mut groups: Vec<(usize, usize, u16)> = Vec::new();
        {
            let mut gs = 0;
            while gs < indices.len() {
                let mut ge = gs + 1;
                while ge < indices.len() && buf.term_at(indices[ge].1) == buf.term_at(indices[gs].1)
                {
                    ge += 1;
                }
                groups.push((gs, ge, (ge - gs) as u16));
                gs = ge;
            }
        }

        // Hold the write lock only for the final HashMap insertions.
        {
            let mut postings = self.postings.write();
            for &(gs, ge, tf) in &groups {
                let term_str = buf.term_at(indices[gs].1);
                if let Some(list) = postings.get_mut(term_str) {
                    list.begin_doc(doc_id, tf, doc_length);
                    for &(_, _, p) in &indices[gs..ge] {
                        list.push_position(p);
                    }
                } else {
                    let mut list = PostingsList::new();
                    list.begin_doc(doc_id, tf, doc_length);
                    for &(_, _, p) in &indices[gs..ge] {
                        list.push_position(p);
                    }
                    postings.insert(term_str.to_string(), list);
                }
            }
        }

        {
            let mut norms = self.field_norms.write();
            if let Some(old) = norms.insert(doc_id, doc_length) {
                self.total_doc_length
                    .fetch_sub(old as u64, Ordering::Relaxed);
            } else {
                self.doc_count.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.total_doc_length
            .fetch_add(doc_length as u64, Ordering::Relaxed);

        let mut cur = self.max_doc_id.load(Ordering::Relaxed);
        while doc_id > cur {
            match self.max_doc_id.compare_exchange_weak(
                cur,
                doc_id,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => cur = actual,
            }
        }
        Ok(())
    }

    pub fn delete_document(&self, doc_id: DocId) -> Result<()> {
        {
            let mut norms = self.field_norms.write();
            if let Some(old) = norms.remove(&doc_id) {
                self.doc_count.fetch_sub(1, Ordering::Relaxed);
                self.total_doc_length
                    .fetch_sub(old as u64, Ordering::Relaxed);
            }
        }
        // Scan all terms for this doc_id via binary search (cold path).
        let mut postings = self.postings.write();
        let mut empty_terms: Vec<String> = Vec::new();
        for (term, list) in postings.iter_mut() {
            if let Ok(idx) = list.doc_ids.binary_search(&doc_id) {
                list.remove(idx);
                if list.is_empty() {
                    empty_terms.push(term.clone());
                }
            }
        }
        for term in &empty_terms {
            postings.remove(term);
        }
        Ok(())
    }

    pub fn update_document(
        &self,
        doc_id: DocId,
        new_text: &str,
        analyzer: &dyn Analyzer,
    ) -> Result<()> {
        self.delete_document(doc_id)?;
        self.add_document(doc_id, new_text, analyzer)
    }

    pub fn search(
        &self,
        query: &FtsQuery,
        analyzer: &dyn Analyzer,
        scorer: &dyn RelevanceScorer,
        limit: usize,
    ) -> Result<Vec<(DocId, f64)>> {
        let max_id = self.max_doc_id.load(Ordering::Relaxed);
        let mut acc = ScoreAccumulator::new(max_id);
        self.score_query(query, analyzer, scorer, &mut acc)?;
        Ok(top_k_from_accumulator(acc, limit))
    }

    fn score_query(
        &self,
        query: &FtsQuery,
        analyzer: &dyn Analyzer,
        scorer: &dyn RelevanceScorer,
        acc: &mut ScoreAccumulator,
    ) -> Result<()> {
        let total_docs = self.doc_count();
        let avg_dl = self.avg_dl();
        match query {
            FtsQuery::Term(term) => {
                let analyzed = analyzer.analyze(term);
                let postings = self.postings.read();
                for token in &analyzed {
                    if let Some(list) = postings.get(&token.term) {
                        score_postings_simd(list, total_docs, avg_dl, acc);
                    }
                }
            }
            FtsQuery::Phrase(words) => self.search_phrase(words, total_docs, avg_dl, acc),
            FtsQuery::Boolean {
                must,
                should,
                must_not,
            } => self.search_boolean(must, should, must_not, analyzer, scorer, acc)?,
            FtsQuery::Fuzzy { term, max_edits } => {
                self.search_fuzzy(term, *max_edits, total_docs, avg_dl, acc)
            }
            FtsQuery::Prefix(prefix) => self.search_prefix(prefix, total_docs, avg_dl, acc),
            FtsQuery::Proximity { terms, distance } => {
                self.search_proximity(terms, *distance, total_docs, avg_dl, acc)
            }
            FtsQuery::Wildcard(pattern) => self.search_wildcard(pattern, total_docs, avg_dl, acc),
        }
        Ok(())
    }

    pub fn prefix_terms(&self, prefix: &str, limit: usize) -> Vec<(String, u32)> {
        let p = self.postings.read();
        let mut r: Vec<(String, u32)> = p
            .iter()
            .filter(|(t, _)| t.starts_with(prefix))
            .map(|(t, l)| (t.clone(), l.len() as u32))
            .collect();
        r.sort_by(|a, b| b.1.cmp(&a.1));
        r.truncate(limit);
        r
    }

    pub fn doc_count(&self) -> u64 {
        self.doc_count.load(Ordering::Relaxed)
    }
    pub fn avg_dl(&self) -> f64 {
        let c = self.doc_count();
        if c == 0 {
            0.0
        } else {
            self.total_doc_length.load(Ordering::Relaxed) as f64 / c as f64
        }
    }
    pub fn get_postings(&self, term: &str) -> Option<PostingsList> {
        self.postings.read().get(term).cloned()
    }
    pub fn doc_frequency(&self, term: &str) -> u32 {
        self.postings
            .read()
            .get(term)
            .map(|l| l.len() as u32)
            .unwrap_or(0)
    }
    pub fn field_length(&self, doc_id: DocId) -> u32 {
        self.field_norms.read().get(&doc_id).copied().unwrap_or(0)
    }

    // -----------------------------------------------------------------------
    // Phrase: multi-cursor merge walk with inline position checking.
    // Walks all postings lists in parallel, checking positions immediately
    // when a doc_id matches across all lists, avoiding intermediate candidate
    // vectors and redundant binary searches.
    // -----------------------------------------------------------------------

    fn search_phrase(
        &self,
        words: &[String],
        total_docs: u64,
        avg_dl: f64,
        acc: &mut ScoreAccumulator,
    ) {
        if words.is_empty() {
            return;
        }
        let postings = self.postings.read();

        // Collect references to each word's postings list with original phrase offset.
        let mut lists: Vec<(usize, &PostingsList)> = Vec::with_capacity(words.len());
        for (i, w) in words.iter().enumerate() {
            match postings.get(w) {
                Some(l) => lists.push((i, l)),
                None => return,
            }
        }

        // Sort by postings list length so the rarest term drives the outer loop.
        lists.sort_by_key(|(_, l)| l.len());

        let first_list = postings.get(&words[0]).unwrap();
        let df = first_list.len() as u32;
        let idf = bm25_idf(df, total_docs);

        let (lead_offset, lead_list) = lists[0];
        let rest = &lists[1..];

        // Cursors into each non-lead list, advanced monotonically.
        let mut cursors: Vec<usize> = vec![0; rest.len()];

        // Walk the lead (rarest) list document by document.
        for lead_idx in 0..lead_list.len() {
            let doc_id = lead_list.doc_ids[lead_idx];

            // Advance each cursor to match this doc_id using galloping search.
            let mut all_match = true;
            for (ci, &(_, ref list)) in rest.iter().enumerate() {
                let c = cursors[ci];
                match gallop_search(&list.doc_ids, doc_id, c) {
                    Some(idx) => cursors[ci] = idx,
                    None => {
                        all_match = false;
                        break;
                    }
                }
            }
            if !all_match {
                continue;
            }

            // All lists contain this doc_id. Check positions for phrase adjacency.
            // Build the index into each word's list for this doc_id.
            let lead_positions = lead_list.positions_for(lead_idx);

            let mut found = false;
            'outer: for &start_pos in lead_positions {
                // The phrase position for the lead word is start_pos.
                // For each other word at phrase offset `wo`, check if position
                // (start_pos - lead_offset + wo) exists in that word's positions.
                let base = start_pos as i64 - lead_offset as i64;
                if base < 0 {
                    continue;
                }

                for (ci, &(wo, ref list)) in rest.iter().enumerate() {
                    let target = base as u32 + wo as u32;
                    let pos_slice = list.positions_for(cursors[ci]);
                    if pos_slice.binary_search(&target).is_err() {
                        continue 'outer;
                    }
                }
                found = true;
                break;
            }

            if found {
                // Field length is the same across all lists for the same doc_id.
                let fl = lead_list.field_lengths[lead_idx];
                acc.add(doc_id, idf * bm25_tf_norm(1, fl, avg_dl, 1.2, 0.75));
            }
        }
    }

    // -----------------------------------------------------------------------
    // Boolean: bitset exclusion + hit-count intersection
    // -----------------------------------------------------------------------

    fn search_boolean(
        &self,
        must: &[FtsQuery],
        should: &[FtsQuery],
        must_not: &[FtsQuery],
        analyzer: &dyn Analyzer,
        scorer: &dyn RelevanceScorer,
        acc: &mut ScoreAccumulator,
    ) -> Result<()> {
        let max_id = self.max_doc_id.load(Ordering::Relaxed);
        let total_docs = self.doc_count();
        let avg_dl = self.avg_dl();

        // Build exclusion bitset from must_not sub-queries.
        let blen = ((max_id + 64) / 64) as usize + 1;
        let mut excluded = vec![0u64; blen];
        for sub in must_not {
            let mut sa = ScoreAccumulator::new(max_id);
            self.score_query(sub, analyzer, scorer, &mut sa)?;
            if let ScoreAccumulator::Dense { scored_ids, .. } = sa {
                for &idx in &scored_ids {
                    excluded[idx as usize / 64] |= 1u64 << (idx as usize % 64);
                }
            }
        }
        let is_excl = |d: DocId| -> bool {
            let i = d as usize;
            (excluded[i / 64] >> (i % 64)) & 1 != 0
        };

        // For must clauses that are simple Term queries, score all terms into
        // a single accumulator using SIMD, then use bitset intersection to
        // filter to docs matching all terms. This avoids per-sub-query 8MB
        // scratch accumulators while keeping SIMD scoring speed.
        if !must.is_empty() {
            let postings = self.postings.read();

            // Resolve must terms to postings lists when possible.
            let mut term_lists: Vec<&PostingsList> = Vec::new();
            let mut all_simple = true;
            for sub in must {
                if let FtsQuery::Term(term) = sub {
                    let analyzed = analyzer.analyze(term);
                    if let Some(token) = analyzed.first() {
                        if let Some(list) = postings.get(&token.term) {
                            term_lists.push(list);
                            continue;
                        }
                    }
                    // Term not found, no docs can satisfy must
                    return Ok(());
                }
                all_simple = false;
                break;
            }

            if all_simple && term_lists.len() >= 2 {
                // Multi-cursor merge walk with inline SIMD-style BM25 scoring.
                // Walk the shortest list and gallop-search the others. Score
                // matching docs inline. This uses O(1) extra memory per cursor
                // instead of 8MB ScoreAccumulator per sub-query.
                term_lists.sort_by_key(|l| l.len());
                let lead_list = term_lists[0];
                let rest = &term_lists[1..];
                let mut cursors: Vec<usize> = vec![0; rest.len()];
                let k1 = 1.2f64;
                let b = 0.75f64;

                // Pre-compute IDF for each term.
                let lead_idf = bm25_idf(lead_list.len() as u32, total_docs);
                let rest_idfs: Vec<f64> = rest
                    .iter()
                    .map(|l| bm25_idf(l.len() as u32, total_docs))
                    .collect();

                for lead_idx in 0..lead_list.len() {
                    let doc_id = lead_list.doc_ids[lead_idx];
                    if is_excl(doc_id) {
                        continue;
                    }

                    let mut all_match = true;
                    for (ci, list) in rest.iter().enumerate() {
                        match gallop_search(&list.doc_ids, doc_id, cursors[ci]) {
                            Some(idx) => cursors[ci] = idx,
                            None => {
                                all_match = false;
                                break;
                            }
                        }
                    }
                    if !all_match {
                        continue;
                    }

                    let fl = lead_list.field_lengths[lead_idx];
                    let mut score =
                        lead_idf * bm25_tf_norm(lead_list.term_freqs[lead_idx], fl, avg_dl, k1, b);
                    for (ci, list) in rest.iter().enumerate() {
                        let idx = cursors[ci];
                        score +=
                            rest_idfs[ci] * bm25_tf_norm(list.term_freqs[idx], fl, avg_dl, k1, b);
                    }
                    acc.add(doc_id, score);
                }
                drop(postings);
            } else {
                // Fallback: complex sub-queries. Use original per-sub-query approach.
                drop(postings);
                let mut combined = ScoreAccumulator::new(max_id);
                let mut hits: Vec<u8> = vec![0u8; (max_id + 1) as usize];
                let must_len = must.len() as u8;
                for sub in must {
                    let mut sa = ScoreAccumulator::new(max_id);
                    self.score_query(sub, analyzer, scorer, &mut sa)?;
                    if let ScoreAccumulator::Dense { scores, scored_ids } = sa {
                        for &idx in &scored_ids {
                            if is_excl(idx as DocId) {
                                continue;
                            }
                            hits[idx as usize] += 1;
                            combined.add(idx as DocId, scores[idx as usize]);
                        }
                    }
                }
                if let ScoreAccumulator::Dense { scores, scored_ids } = &combined {
                    for &idx in scored_ids {
                        if hits[idx as usize] >= must_len {
                            acc.add(idx as DocId, scores[idx as usize]);
                        }
                    }
                }
            }
        }

        let must_present = !must.is_empty();
        if !should.is_empty() {
            // Check if all should clauses are simple terms with must present.
            // If so, walk postings directly to avoid 8MB scratch allocators.
            let all_should_simple =
                must_present && should.iter().all(|s| matches!(s, FtsQuery::Term(_)));

            if all_should_simple {
                let postings = self.postings.read();
                let k1 = 1.2f64;
                let bv = 0.75f64;
                for sub in should {
                    if let FtsQuery::Term(term) = sub {
                        let analyzed = analyzer.analyze(term);
                        if let Some(token) = analyzed.first() {
                            if let Some(list) = postings.get(&token.term) {
                                let idf = bm25_idf(list.len() as u32, total_docs);
                                for i in 0..list.len() {
                                    let doc_id = list.doc_ids[i];
                                    if is_excl(doc_id) {
                                        continue;
                                    }
                                    if let Some(v) = acc.get_mut(doc_id) {
                                        *v += idf
                                            * bm25_tf_norm(
                                                list.term_freqs[i],
                                                list.field_lengths[i],
                                                avg_dl,
                                                k1,
                                                bv,
                                            );
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                // Fallback: complex should clauses or no must.
                for sub in should {
                    let mut sa = ScoreAccumulator::new(max_id);
                    self.score_query(sub, analyzer, scorer, &mut sa)?;
                    if let ScoreAccumulator::Dense { scores, scored_ids } = sa {
                        for &idx in &scored_ids {
                            if is_excl(idx as DocId) {
                                continue;
                            }
                            let score = scores[idx as usize];
                            if must_present {
                                if let Some(v) = acc.get_mut(idx as DocId) {
                                    *v += score;
                                }
                            } else {
                                acc.add(idx as DocId, score);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn search_fuzzy(
        &self,
        term: &str,
        max_edits: u8,
        total_docs: u64,
        avg_dl: f64,
        acc: &mut ScoreAccumulator,
    ) {
        let postings = self.postings.read();
        for (t, list) in postings.iter() {
            if edit_distance(term, t) <= max_edits as u32 {
                score_postings_simd(list, total_docs, avg_dl, acc);
            }
        }
    }

    fn search_prefix(
        &self,
        prefix: &str,
        total_docs: u64,
        avg_dl: f64,
        acc: &mut ScoreAccumulator,
    ) {
        let postings = self.postings.read();
        for (t, list) in postings.iter() {
            if t.starts_with(prefix) {
                score_postings_simd(list, total_docs, avg_dl, acc);
            }
        }
    }

    fn search_proximity(
        &self,
        terms: &[String],
        distance: u32,
        total_docs: u64,
        avg_dl: f64,
        acc: &mut ScoreAccumulator,
    ) {
        if terms.len() < 2 {
            return;
        }
        let postings = self.postings.read();
        for t in terms {
            if !postings.contains_key(t) {
                return;
            }
        }

        let mut st: Vec<(usize, &String)> = terms.iter().enumerate().collect();
        st.sort_by_key(|(_, t)| postings.get(*t).map_or(0, |l| l.len()));
        let first = postings.get(st[0].1).unwrap();
        let mut cands: Vec<u64> = first.doc_ids.clone();
        for &(_, t) in &st[1..] {
            cands = intersect_sorted_ids(&cands, &postings.get(t).unwrap().doc_ids);
            if cands.is_empty() {
                return;
            }
        }

        let fl = postings.get(&terms[0]).unwrap();
        let ll = postings.get(terms.last().unwrap()).unwrap();
        let df = fl.len() as u32;
        let idf = bm25_idf(df, total_docs);

        for &doc_id in &cands {
            let fi = match fl.doc_ids.binary_search(&doc_id) {
                Ok(i) => i,
                Err(_) => continue,
            };
            let li = match ll.doc_ids.binary_search(&doc_id) {
                Ok(i) => i,
                Err(_) => continue,
            };
            let fp = fl.positions_for(fi);
            let lp = ll.positions_for(li);
            // Merge-walk on sorted position arrays: O(n+m) instead of O(n*m).
            let mut found = false;
            let (mut i, mut j) = (0, 0);
            while i < fp.len() && j < lp.len() {
                let diff = if fp[i] > lp[j] {
                    fp[i] - lp[j]
                } else {
                    lp[j] - fp[i]
                };
                if diff <= distance {
                    found = true;
                    break;
                }
                if fp[i] < lp[j] {
                    i += 1;
                } else {
                    j += 1;
                }
            }
            if found {
                acc.add(
                    doc_id,
                    idf * bm25_tf_norm(1, fl.field_lengths[fi], avg_dl, 1.2, 0.75),
                );
            }
        }
    }

    fn search_wildcard(
        &self,
        pattern: &str,
        total_docs: u64,
        avg_dl: f64,
        acc: &mut ScoreAccumulator,
    ) {
        let postings = self.postings.read();
        for (t, list) in postings.iter() {
            if wildcard_match(pattern, t) {
                score_postings_simd(list, total_docs, avg_dl, acc);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Persistence
    // -----------------------------------------------------------------------

    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        let tmp = path.with_extension("zyfts.tmp");
        let mut f = std::fs::File::create(&tmp).map_err(|e| ZyronError::FtsIndexCorrupted {
            index: format!("index_{}", self.index_id),
            reason: format!("create: {e}"),
        })?;
        let p = self.postings.read();
        let mut terms: Vec<(&String, &PostingsList)> = p.iter().collect();
        terms.sort_by(|a, b| a.0.cmp(b.0));
        let n = self.field_norms.read();
        f.write_all(b"ZFTS").map_err(ZyronError::Io)?;
        f.write_all(&4u32.to_le_bytes()).map_err(ZyronError::Io)?;
        f.write_all(&(terms.len() as u32).to_le_bytes())
            .map_err(ZyronError::Io)?;
        f.write_all(&(self.doc_count() as u32).to_le_bytes())
            .map_err(ZyronError::Io)?;

        // Encode postings data into a contiguous buffer.
        let mut pbuf = Vec::new();
        let mut te: Vec<(&str, u32, u64, u32)> = Vec::new();
        for (term, list) in &terms {
            let off = pbuf.len() as u64;
            let enc = encode_postings(list);
            let len = enc.len() as u32;
            pbuf.extend_from_slice(&enc);
            te.push((term, list.len() as u32, off, len));
        }

        // Version 4: front-coded term dictionary. Terms are sorted, so
        // consecutive terms share common prefixes. For each term, write:
        // VByte(shared_prefix_len) + VByte(suffix_len) + suffix_bytes + df(4) + off(8) + len(4)
        let mut term_dict_buf = Vec::new();
        let mut prev_term: &[u8] = b"";
        for (t, df, off, len) in &te {
            let tb = t.as_bytes();
            let shared = tb
                .iter()
                .zip(prev_term.iter())
                .take_while(|(a, b)| a == b)
                .count();
            let suffix = &tb[shared..];
            encode_vbyte(shared as u64, &mut term_dict_buf);
            encode_vbyte(suffix.len() as u64, &mut term_dict_buf);
            term_dict_buf.extend_from_slice(suffix);
            term_dict_buf.extend_from_slice(&df.to_le_bytes());
            term_dict_buf.extend_from_slice(&off.to_le_bytes());
            term_dict_buf.extend_from_slice(&len.to_le_bytes());
            prev_term = tb;
        }
        f.write_all(&(term_dict_buf.len() as u32).to_le_bytes())
            .map_err(ZyronError::Io)?;
        f.write_all(&term_dict_buf).map_err(ZyronError::Io)?;
        f.write_all(&pbuf).map_err(ZyronError::Io)?;

        // Version 4: delta-encoded field norms. Sort by doc_id, then write
        // VByte(delta_doc_id) + VByte(field_length) for each entry.
        // Sequential doc_ids produce delta=1 (1 byte each). Field lengths
        // ~20-40 encode in 1 byte. Total: ~2 bytes/doc vs 12 bytes/doc raw.
        let mut sorted_norms: Vec<(DocId, u32)> = n.iter().map(|(&d, &fl)| (d, fl)).collect();
        sorted_norms.sort_unstable_by_key(|&(d, _)| d);
        let mut norm_buf = Vec::new();
        encode_vbyte(sorted_norms.len() as u64, &mut norm_buf);
        let mut prev_did = 0u64;
        for &(did, fl) in &sorted_norms {
            encode_vbyte(did - prev_did, &mut norm_buf);
            prev_did = did;
            encode_vbyte(fl as u64, &mut norm_buf);
        }
        f.write_all(&norm_buf).map_err(ZyronError::Io)?;

        drop(f);
        std::fs::rename(&tmp, path).map_err(ZyronError::Io)?;
        Ok(())
    }

    pub fn load_from_file(
        path: &Path,
        index_id: u32,
        table_id: u32,
        column_ids: Vec<u16>,
    ) -> Result<Self> {
        let mut f = std::fs::File::open(path).map_err(|e| ZyronError::FtsIndexCorrupted {
            index: format!("index_{index_id}"),
            reason: format!("open: {e}"),
        })?;
        let err = |r: String| ZyronError::FtsIndexCorrupted {
            index: format!("index_{index_id}"),
            reason: r,
        };
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic).map_err(ZyronError::Io)?;
        if &magic != b"ZFTS" {
            return Err(err("invalid magic".into()));
        }
        let mut b4 = [0u8; 4];
        f.read_exact(&mut b4).map_err(ZyronError::Io)?;
        let ver = u32::from_le_bytes(b4);
        f.read_exact(&mut b4).map_err(ZyronError::Io)?;
        let tc = u32::from_le_bytes(b4);
        f.read_exact(&mut b4).map_err(ZyronError::Io)?;

        let mut tes: Vec<(String, u32, u64, u32)> = Vec::new();

        if ver >= 4 {
            // Version 4: front-coded term dictionary.
            f.read_exact(&mut b4).map_err(ZyronError::Io)?;
            let dict_len = u32::from_le_bytes(b4) as usize;
            let mut dict_buf = vec![0u8; dict_len];
            f.read_exact(&mut dict_buf).map_err(ZyronError::Io)?;
            let mut off = 0usize;
            let mut prev_term = String::new();
            for _ in 0..tc {
                let shared = decode_vbyte(&dict_buf, &mut off)? as usize;
                let suffix_len = decode_vbyte(&dict_buf, &mut off)? as usize;
                if off + suffix_len > dict_buf.len() {
                    return Err(err("dict oob".into()));
                }
                let suffix = std::str::from_utf8(&dict_buf[off..off + suffix_len])
                    .map_err(|e| err(format!("utf8: {e}")))?;
                off += suffix_len;
                let mut term = prev_term[..shared].to_string();
                term.push_str(suffix);
                if off + 16 > dict_buf.len() {
                    return Err(err("dict entry oob".into()));
                }
                let df = u32::from_le_bytes(dict_buf[off..off + 4].try_into().unwrap());
                off += 4;
                let poff = u64::from_le_bytes(dict_buf[off..off + 8].try_into().unwrap());
                off += 8;
                let plen = u32::from_le_bytes(dict_buf[off..off + 4].try_into().unwrap());
                off += 4;
                prev_term = term.clone();
                tes.push((term, df, poff, plen));
            }
        } else {
            // Version 2-3: raw term dictionary.
            for _ in 0..tc {
                let mut b2 = [0u8; 2];
                f.read_exact(&mut b2).map_err(ZyronError::Io)?;
                let tl = u16::from_le_bytes(b2) as usize;
                let mut tb = vec![0u8; tl];
                f.read_exact(&mut tb).map_err(ZyronError::Io)?;
                let term = String::from_utf8(tb).map_err(|e| err(format!("utf8: {e}")))?;
                f.read_exact(&mut b4).map_err(ZyronError::Io)?;
                let df = u32::from_le_bytes(b4);
                let mut b8 = [0u8; 8];
                f.read_exact(&mut b8).map_err(ZyronError::Io)?;
                let poff = u64::from_le_bytes(b8);
                f.read_exact(&mut b4).map_err(ZyronError::Io)?;
                let plen = u32::from_le_bytes(b4);
                tes.push((term, df, poff, plen));
            }
        }

        let tsz: usize = tes
            .iter()
            .map(|(_, _, o, l)| *o as usize + *l as usize)
            .max()
            .unwrap_or(0);
        let mut pbuf = vec![0u8; tsz];
        if tsz > 0 {
            f.read_exact(&mut pbuf).map_err(ZyronError::Io)?;
        }
        let idx = Self::new(index_id, table_id, column_ids);
        let mut max_did: u64 = 0;
        {
            let mut postings = idx.postings.write();
            for (term, _, off, len) in &tes {
                let s = *off as usize;
                let e = s + *len as usize;
                if e > pbuf.len() {
                    return Err(err(format!("oob: \"{term}\"")));
                }
                let list = decode_postings(&pbuf[s..e], ver)?;
                if let Some(&last) = list.doc_ids.last() {
                    if last > max_did {
                        max_did = last;
                    }
                }
                postings.insert(term.clone(), list);
            }
        }

        let mut tl = 0u64;
        let nc;
        if ver >= 4 {
            // Version 4: delta-encoded field norms via VByte.
            let mut rest = Vec::new();
            f.read_to_end(&mut rest).map_err(ZyronError::Io)?;
            let mut off = 0usize;
            nc = decode_vbyte(&rest, &mut off)? as u32;
            let mut norms = idx.field_norms.write();
            let mut prev = 0u64;
            for _ in 0..nc {
                let delta = decode_vbyte(&rest, &mut off)?;
                prev += delta;
                let fl = decode_vbyte(&rest, &mut off)? as u32;
                norms.insert(prev, fl);
                tl += fl as u64;
                if prev > max_did {
                    max_did = prev;
                }
            }
        } else {
            // Version 2-3: raw field norms.
            f.read_exact(&mut b4).map_err(ZyronError::Io)?;
            nc = u32::from_le_bytes(b4);
            let mut norms = idx.field_norms.write();
            for _ in 0..nc {
                let mut b8 = [0u8; 8];
                f.read_exact(&mut b8).map_err(ZyronError::Io)?;
                let did = u64::from_le_bytes(b8);
                f.read_exact(&mut b4).map_err(ZyronError::Io)?;
                let fl = u32::from_le_bytes(b4);
                norms.insert(did, fl);
                tl += fl as u64;
                if did > max_did {
                    max_did = did;
                }
            }
        }

        idx.doc_count.store(nc as u64, Ordering::Relaxed);
        idx.total_doc_length.store(tl, Ordering::Relaxed);
        idx.max_doc_id.store(max_did, Ordering::Relaxed);
        if ver >= 3 {
            let norms = idx.field_norms.read();
            let mut postings = idx.postings.write();
            for list in postings.values_mut() {
                for i in 0..list.len() {
                    if let Some(&len) = norms.get(&list.doc_ids[i]) {
                        list.field_lengths[i] = len;
                    }
                }
            }
        }
        Ok(idx)
    }
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

/// Version 4 postings encoding. Two compact techniques:
/// 1. Position count omitted (equals tf), saves 1 VByte per posting.
/// 2. For tf=1 postings (the common case), doc_id delta is encoded with high
///    bit set in the first VByte byte to signal tf=1. This avoids a separate
///    tf VByte, saving 1 more byte per tf=1 posting.
///
/// Encoding per posting:
/// - tf=1: VByte(delta | 0x80 on first byte to signal tf=1), VByte(position)
/// - tf>1: VByte(delta, high bit clear on first byte), VByte(tf), delta-encoded positions
/// Version 4 postings encoding. Splits postings into tf=1 and tf>1 groups.
///
/// tf=1 postings use a bitset for doc_ids (1 bit per potential doc_id) and
/// 6-bit packed positions. For dense terms (appearing in most docs), the bitset
/// is much smaller than VByte delta encoding. A term in 500K of 1M docs uses
/// 125KB bitset vs ~500KB VByte deltas. Positions at 6 bits each save 25%
/// over 8-bit VByte.
///
/// tf>1 postings use VByte delta doc_ids, VByte tf, delta-encoded positions.
///
/// Layout: [total_count][max_doc_id][tf1_count][bitset][packed_positions][tf_n_section]
pub fn encode_postings(list: &PostingsList) -> Vec<u8> {
    let mut buf = Vec::new();
    let n = list.len();
    encode_vbyte(n as u64, &mut buf);

    if n == 0 {
        return buf;
    }

    let max_did = *list.doc_ids.last().unwrap();
    encode_vbyte(max_did, &mut buf);

    // Separate tf=1 and tf>1 postings.
    let mut tf1_indices: Vec<usize> = Vec::new();
    let mut tfn_indices: Vec<usize> = Vec::new();
    for i in 0..n {
        if list.term_freqs[i] == 1 {
            tf1_indices.push(i);
        } else {
            tfn_indices.push(i);
        }
    }

    encode_vbyte(tf1_indices.len() as u64, &mut buf);

    if !tf1_indices.is_empty() {
        // Bitset for tf=1 doc_ids. Each bit represents a doc_id in [0, max_did].
        let blen = (max_did as usize / 8) + 1;
        let mut bitset = vec![0u8; blen];
        for &i in &tf1_indices {
            let did = list.doc_ids[i] as usize;
            bitset[did / 8] |= 1u8 << (did % 8);
        }
        encode_vbyte(blen as u64, &mut buf);
        buf.extend_from_slice(&bitset);

        // Pack positions at 6 bits each. Groups of 4 positions = 3 bytes.
        let pos_count = tf1_indices.len();
        let full_groups = pos_count / 4;
        let remainder = pos_count % 4;
        for g in 0..full_groups {
            let base = g * 4;
            let p0 = list.positions_for(tf1_indices[base])[0] & 0x3F;
            let p1 = list.positions_for(tf1_indices[base + 1])[0] & 0x3F;
            let p2 = list.positions_for(tf1_indices[base + 2])[0] & 0x3F;
            let p3 = list.positions_for(tf1_indices[base + 3])[0] & 0x3F;
            buf.push((p0 as u8) | ((p1 as u8 & 0x03) << 6));
            buf.push(((p1 as u8) >> 2) | ((p2 as u8 & 0x0F) << 4));
            buf.push(((p2 as u8) >> 4) | ((p3 as u8) << 2));
        }
        // Remaining positions: 1 byte each (VByte).
        for r in 0..remainder {
            let idx = full_groups * 4 + r;
            buf.push(list.positions_for(tf1_indices[idx])[0] as u8 & 0x3F);
        }
    }

    // tf>1 postings: VByte delta encoding.
    encode_vbyte(tfn_indices.len() as u64, &mut buf);
    let mut prev = 0u64;
    for &i in &tfn_indices {
        let delta = list.doc_ids[i] - prev;
        prev = list.doc_ids[i];
        encode_vbyte(delta, &mut buf);
        encode_vbyte(list.term_freqs[i] as u64, &mut buf);
        let pos = list.positions_for(i);
        let mut pp = 0u32;
        for &p in pos {
            encode_vbyte((p - pp) as u64, &mut buf);
            pp = p;
        }
    }

    buf
}

pub fn decode_postings(data: &[u8], ver: u32) -> Result<PostingsList> {
    let mut off = 0;
    let count = decode_vbyte(data, &mut off)? as usize;
    let mut list = PostingsList {
        doc_ids: Vec::with_capacity(count),
        term_freqs: Vec::with_capacity(count),
        field_lengths: Vec::with_capacity(count),
        position_offsets: Vec::with_capacity(count),
        positions: Vec::with_capacity(count),
    };

    if ver >= 4 {
        if count == 0 {
            return Ok(list);
        }
        let _max_did = decode_vbyte(data, &mut off)?;
        let tf1_count = decode_vbyte(data, &mut off)? as usize;

        // Decode tf=1 postings: bitset doc_ids + 6-bit packed positions.
        let mut tf1_doc_ids: Vec<u64> = Vec::new();
        let mut tf1_positions: Vec<u32> = Vec::new();
        if tf1_count > 0 {
            let blen = decode_vbyte(data, &mut off)? as usize;
            let err_fn = || ZyronError::FtsIndexCorrupted {
                index: "postings".into(),
                reason: "oob".into(),
            };
            if off + blen > data.len() {
                return Err(err_fn());
            }
            let bitset = &data[off..off + blen];
            off += blen;

            tf1_doc_ids.reserve(tf1_count);
            for byte_idx in 0..blen {
                let b = bitset[byte_idx];
                if b == 0 {
                    continue;
                }
                for bit in 0..8u64 {
                    if (b >> bit) & 1 != 0 {
                        tf1_doc_ids.push(byte_idx as u64 * 8 + bit);
                    }
                }
            }

            let full_groups = tf1_count / 4;
            let remainder = tf1_count % 4;
            tf1_positions.reserve(tf1_count);
            for _ in 0..full_groups {
                if off + 3 > data.len() {
                    return Err(err_fn());
                }
                let b0 = data[off] as u32;
                let b1 = data[off + 1] as u32;
                let b2 = data[off + 2] as u32;
                off += 3;
                tf1_positions.push(b0 & 0x3F);
                tf1_positions.push(((b0 >> 6) | (b1 << 2)) & 0x3F);
                tf1_positions.push(((b1 >> 4) | (b2 << 4)) & 0x3F);
                tf1_positions.push(b2 >> 2);
            }
            for _ in 0..remainder {
                if off >= data.len() {
                    return Err(err_fn());
                }
                tf1_positions.push(data[off] as u32 & 0x3F);
                off += 1;
            }
        }

        // Decode tf>1 postings.
        let tfn_count = decode_vbyte(data, &mut off)? as usize;
        let mut tfn_doc_ids: Vec<u64> = Vec::with_capacity(tfn_count);
        let mut tfn_tfs: Vec<u16> = Vec::with_capacity(tfn_count);
        let mut tfn_pos_offsets: Vec<u32> = Vec::with_capacity(tfn_count);
        let mut tfn_positions: Vec<u32> = Vec::new();
        let mut prev = 0u64;
        for _ in 0..tfn_count {
            let delta = decode_vbyte(data, &mut off)?;
            let did = prev + delta;
            prev = did;
            let tf = decode_vbyte(data, &mut off)? as u16;
            tfn_doc_ids.push(did);
            tfn_tfs.push(tf);
            tfn_pos_offsets.push(tfn_positions.len() as u32);
            let mut pp = 0u32;
            for _ in 0..tf {
                let dp = decode_vbyte(data, &mut off)? as u32;
                pp += dp;
                tfn_positions.push(pp);
            }
        }

        // Merge tf=1 and tf>1 postings in doc_id order.
        let mut i = 0usize;
        let mut j = 0usize;
        while i < tf1_doc_ids.len() || j < tfn_doc_ids.len() {
            let take_tf1 = if i >= tf1_doc_ids.len() {
                false
            } else if j >= tfn_doc_ids.len() {
                true
            } else {
                tf1_doc_ids[i] <= tfn_doc_ids[j]
            };

            if take_tf1 {
                let ps = list.positions.len() as u32;
                list.doc_ids.push(tf1_doc_ids[i]);
                list.term_freqs.push(1);
                list.field_lengths.push(0);
                list.position_offsets.push(ps);
                list.positions.push(tf1_positions[i]);
                i += 1;
            } else {
                let ps = list.positions.len() as u32;
                list.doc_ids.push(tfn_doc_ids[j]);
                list.term_freqs.push(tfn_tfs[j]);
                list.field_lengths.push(0);
                list.position_offsets.push(ps);
                let start = tfn_pos_offsets[j] as usize;
                let end = if j + 1 < tfn_pos_offsets.len() {
                    tfn_pos_offsets[j + 1] as usize
                } else {
                    tfn_positions.len()
                };
                list.positions.extend_from_slice(&tfn_positions[start..end]);
                j += 1;
            }
        }
    } else {
        let mut prev = 0u64;
        for _ in 0..count {
            let d = decode_vbyte(data, &mut off)?;
            let did = prev + d;
            prev = did;
            let tf = decode_vbyte(data, &mut off)? as u16;
            let fl = if ver == 2 {
                decode_vbyte(data, &mut off)? as u32
            } else {
                0
            };
            let pc = decode_vbyte(data, &mut off)? as usize;
            let ps = list.positions.len() as u32;
            let mut pp = 0u32;
            for _ in 0..pc {
                let dp = decode_vbyte(data, &mut off)? as u32;
                pp += dp;
                list.positions.push(pp);
            }
            list.doc_ids.push(did);
            list.term_freqs.push(tf);
            list.field_lengths.push(fl);
            list.position_offsets.push(ps);
        }
    }
    Ok(list)
}

/// Encode doc_id delta with tf=1 signal. First byte layout:
/// bit 7 = continuation, bit 0 = tf1 flag (1), bits 1-6 = 6 data bits.
fn encode_vbyte_tf1(mut v: u64, buf: &mut Vec<u8>) {
    let data_bits = (v & 0x3F) as u8;
    v >>= 6;
    if v == 0 {
        buf.push((data_bits << 1) | 0x01);
    } else {
        buf.push((data_bits << 1) | 0x01 | 0x80);
        loop {
            let b = (v & 0x7F) as u8;
            v >>= 7;
            if v == 0 {
                buf.push(b);
                return;
            }
            buf.push(b | 0x80);
        }
    }
}

/// Encode doc_id delta with tf>1 signal. First byte layout:
/// bit 7 = continuation, bit 0 = tf1 flag (0), bits 1-6 = 6 data bits.
fn encode_vbyte_tfn(mut v: u64, buf: &mut Vec<u8>) {
    let data_bits = (v & 0x3F) as u8;
    v >>= 6;
    if v == 0 {
        buf.push(data_bits << 1);
    } else {
        buf.push((data_bits << 1) | 0x80);
        loop {
            let b = (v & 0x7F) as u8;
            v >>= 7;
            if v == 0 {
                buf.push(b);
                return;
            }
            buf.push(b | 0x80);
        }
    }
}

/// Decode doc_id delta and tf=1 flag. First byte: bit 0 = tf1, bits 1-6 = data.
fn decode_vbyte_with_flag(data: &[u8], off: &mut usize) -> Result<(u64, bool)> {
    if *off >= data.len() {
        return Err(ZyronError::FtsIndexCorrupted {
            index: "postings".into(),
            reason: "eof".into(),
        });
    }
    let first = data[*off];
    *off += 1;
    let is_tf1 = (first & 0x01) != 0;
    let mut v = ((first >> 1) & 0x3F) as u64;
    if first & 0x80 == 0 {
        return Ok((v, is_tf1));
    }
    let mut s = 6u32;
    loop {
        if *off >= data.len() {
            return Err(ZyronError::FtsIndexCorrupted {
                index: "postings".into(),
                reason: "eof".into(),
            });
        }
        let b = data[*off];
        *off += 1;
        v |= ((b & 0x7F) as u64) << s;
        if b & 0x80 == 0 {
            return Ok((v, is_tf1));
        }
        s += 7;
        if s > 63 {
            return Err(ZyronError::FtsIndexCorrupted {
                index: "postings".into(),
                reason: "overflow".into(),
            });
        }
    }
}

fn encode_vbyte(mut v: u64, buf: &mut Vec<u8>) {
    loop {
        let b = (v & 0x7F) as u8;
        v >>= 7;
        if v == 0 {
            buf.push(b);
            return;
        }
        buf.push(b | 0x80);
    }
}

fn decode_vbyte(data: &[u8], off: &mut usize) -> Result<u64> {
    let mut v = 0u64;
    let mut s = 0u32;
    loop {
        if *off >= data.len() {
            return Err(ZyronError::FtsIndexCorrupted {
                index: "postings".into(),
                reason: "eof".into(),
            });
        }
        let b = data[*off];
        *off += 1;
        v |= ((b & 0x7F) as u64) << s;
        if b & 0x80 == 0 {
            return Ok(v);
        }
        s += 7;
        if s > 63 {
            return Err(ZyronError::FtsIndexCorrupted {
                index: "postings".into(),
                reason: "overflow".into(),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::analyzer::SimpleAnalyzer;
    use crate::text::scoring::Bm25Scorer;
    fn ix() -> InvertedIndex {
        InvertedIndex::new(1, 100, vec![1, 2])
    }
    fn a() -> SimpleAnalyzer {
        SimpleAnalyzer
    }
    fn s() -> Bm25Scorer {
        Bm25Scorer::default()
    }

    #[test]
    fn test_add_and_search() {
        let i = ix();
        i.add_document(1, "the quick brown fox", &a()).unwrap();
        i.add_document(2, "the slow brown turtle", &a()).unwrap();
        i.add_document(3, "a quick red car", &a()).unwrap();
        let r = i
            .search(&FtsQuery::Term("quick".into()), &a(), &s(), 10)
            .unwrap();
        assert_eq!(r.len(), 2);
        let ids: Vec<DocId> = r.iter().map(|x| x.0).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&3));
    }
    #[test]
    fn test_delete() {
        let i = ix();
        i.add_document(1, "hello world", &a()).unwrap();
        i.add_document(2, "hello there", &a()).unwrap();
        i.delete_document(1).unwrap();
        let r = i
            .search(&FtsQuery::Term("hello".into()), &a(), &s(), 10)
            .unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0, 2);
    }
    #[test]
    fn test_update() {
        let i = ix();
        i.add_document(1, "hello world", &a()).unwrap();
        i.update_document(1, "goodbye world", &a()).unwrap();
        assert!(
            i.search(&FtsQuery::Term("hello".into()), &a(), &s(), 10)
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            i.search(&FtsQuery::Term("goodbye".into()), &a(), &s(), 10)
                .unwrap()
                .len(),
            1
        );
    }
    #[test]
    fn test_phrase() {
        let i = ix();
        i.add_document(1, "the quick brown fox", &a()).unwrap();
        i.add_document(2, "the brown quick fox", &a()).unwrap();
        let r = i
            .search(
                &FtsQuery::Phrase(vec!["quick".into(), "brown".into()]),
                &a(),
                &s(),
                10,
            )
            .unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0, 1);
    }
    #[test]
    fn test_prefix() {
        let i = ix();
        i.add_document(1, "database systems", &a()).unwrap();
        i.add_document(2, "datastore management", &a()).unwrap();
        i.add_document(3, "metadata extraction", &a()).unwrap();
        assert_eq!(
            i.search(&FtsQuery::Prefix("data".into()), &a(), &s(), 10)
                .unwrap()
                .len(),
            2
        );
    }
    #[test]
    fn test_fuzzy() {
        let i = ix();
        i.add_document(1, "database performance", &a()).unwrap();
        assert_eq!(
            i.search(
                &FtsQuery::Fuzzy {
                    term: "databse".into(),
                    max_edits: 1
                },
                &a(),
                &s(),
                10
            )
            .unwrap()
            .len(),
            1
        );
    }
    #[test]
    fn test_boolean() {
        let i = ix();
        i.add_document(1, "postgresql performance tuning", &a())
            .unwrap();
        i.add_document(2, "mysql performance tuning", &a()).unwrap();
        i.add_document(3, "postgresql backup strategy", &a())
            .unwrap();
        let r = i
            .search(
                &FtsQuery::Boolean {
                    must: vec![FtsQuery::Term("postgresql".into())],
                    should: vec![FtsQuery::Term("performance".into())],
                    must_not: vec![],
                },
                &a(),
                &s(),
                10,
            )
            .unwrap();
        let ids: Vec<DocId> = r.iter().map(|x| x.0).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&3));
        assert!(!ids.contains(&2));
        assert!(r.iter().find(|x| x.0 == 1).unwrap().1 > r.iter().find(|x| x.0 == 3).unwrap().1);
    }
    #[test]
    fn test_boolean_must_not() {
        let i = ix();
        i.add_document(1, "postgresql performance", &a()).unwrap();
        i.add_document(2, "mysql performance", &a()).unwrap();
        let r = i
            .search(
                &FtsQuery::Boolean {
                    must: vec![FtsQuery::Term("performance".into())],
                    should: vec![],
                    must_not: vec![FtsQuery::Term("mysql".into())],
                },
                &a(),
                &s(),
                10,
            )
            .unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0, 1);
    }
    #[test]
    fn test_wildcard() {
        let i = ix();
        i.add_document(1, "test text", &a()).unwrap();
        i.add_document(2, "toast roast", &a()).unwrap();
        assert_eq!(
            i.search(&FtsQuery::Wildcard("te?t".into()), &a(), &s(), 10)
                .unwrap()
                .len(),
            1
        );
    }
    #[test]
    fn test_doc_count_avg_dl() {
        let i = ix();
        i.add_document(1, "one two three", &a()).unwrap();
        i.add_document(2, "one two three four five", &a()).unwrap();
        assert_eq!(i.doc_count(), 2);
        assert!((i.avg_dl() - 4.0).abs() < 0.01);
    }
    #[test]
    fn test_prefix_terms() {
        let i = ix();
        i.add_document(1, "database datastore data", &a()).unwrap();
        assert!(i.prefix_terms("data", 10).len() >= 2);
    }
    #[test]
    fn test_vbyte() {
        let mut buf = Vec::new();
        for v in [0u64, 127, 128, 16384, u64::MAX] {
            encode_vbyte(v, &mut buf);
        }
        let mut off = 0;
        for v in [0u64, 127, 128, 16384, u64::MAX] {
            assert_eq!(decode_vbyte(&buf, &mut off).unwrap(), v);
        }
    }
    #[test]
    fn test_postings_roundtrip() {
        let mut list = PostingsList::new();
        list.push(10, 3, 50, &[0, 5, 12]);
        list.push(25, 1, 30, &[7]);
        list.push(100, 2, 80, &[3, 9]);
        let enc = encode_postings(&list);
        let dec = decode_postings(&enc, 4).unwrap();
        assert_eq!(dec.len(), 3);
        assert_eq!(dec.doc_ids[0], 10);
        assert_eq!(dec.positions_for(0), &[0, 5, 12]);
    }
    #[test]
    fn test_file_roundtrip() {
        let i = ix();
        i.add_document(1, "hello world", &a()).unwrap();
        i.add_document(2, "hello there friend", &a()).unwrap();
        i.add_document(3, "goodbye world", &a()).unwrap();
        let dir = std::env::temp_dir().join("zyron_fts_clean");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test.zyfts");
        i.save_to_file(&path).unwrap();
        let l = InvertedIndex::load_from_file(&path, 1, 100, vec![1, 2]).unwrap();
        assert_eq!(l.doc_count(), 3);
        assert_eq!(
            l.search(&FtsQuery::Term("hello".into()), &a(), &s(), 10)
                .unwrap()
                .len(),
            2
        );
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }
    #[test]
    fn test_empty() {
        assert!(
            ix().search(&FtsQuery::Term("x".into()), &a(), &s(), 10)
                .unwrap()
                .is_empty()
        );
    }
    #[test]
    fn test_proximity() {
        let i = ix();
        i.add_document(1, "query optimization techniques for databases", &a())
            .unwrap();
        i.add_document(2, "optimization is key and query matters", &a())
            .unwrap();
        let r = i
            .search(
                &FtsQuery::Proximity {
                    terms: vec!["query".into(), "optimization".into()],
                    distance: 2,
                },
                &a(),
                &s(),
                10,
            )
            .unwrap();
        assert!(!r.is_empty());
        assert!(r.iter().any(|x| x.0 == 1));
    }
    #[test]
    fn test_doc_id_encoding() {
        let id = encode_doc_id(1000, 42).unwrap();
        let (p, sl) = decode_doc_id(id);
        assert_eq!(p, 1000);
        assert_eq!(sl, 42);
        assert!(encode_doc_id(MAX_DOC_PAGE_ID + 1, 0).is_err());
    }
    #[test]
    fn test_deletion_cleanup() {
        let i = ix();
        i.add_document(1, "alpha beta gamma", &a()).unwrap();
        i.add_document(2, "beta gamma delta", &a()).unwrap();
        i.add_document(3, "gamma delta epsilon", &a()).unwrap();
        i.delete_document(2).unwrap();
        assert_eq!(
            i.search(&FtsQuery::Term("beta".into()), &a(), &s(), 10)
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            i.search(&FtsQuery::Term("delta".into()), &a(), &s(), 10)
                .unwrap()
                .len(),
            1
        );
    }
    #[test]
    fn test_top_k() {
        let i = ix();
        for n in 0..1000u64 {
            i.add_document(n, &format!("word{n} common"), &a()).unwrap();
        }
        assert_eq!(
            i.search(&FtsQuery::Term("common".into()), &a(), &s(), 5)
                .unwrap()
                .len(),
            5
        );
    }
    #[test]
    fn test_sequential_append() {
        let i = ix();
        for n in 0..100u64 {
            i.add_document(n, "test word", &a()).unwrap();
        }
        let l = i.get_postings("test").unwrap();
        assert_eq!(l.len(), 100);
        for w in l.doc_ids.windows(2) {
            assert!(w[0] < w[1]);
        }
    }
    #[test]
    fn test_dense_accumulator() {
        let mut acc = ScoreAccumulator::new(100);
        acc.add(5, 1.0);
        acc.add(10, 2.0);
        acc.add(5, 0.5);
        let r = top_k_from_accumulator(acc, 10);
        assert_eq!(r.len(), 2);
    }
    #[test]
    fn test_soa_positions() {
        let mut list = PostingsList::new();
        list.push(1, 2, 10, &[0, 3]);
        list.push(2, 1, 5, &[7]);
        list.push(3, 3, 15, &[1, 4, 9]);
        assert_eq!(list.len(), 3);
        assert_eq!(list.positions_for(0), &[0, 3]);
        assert_eq!(list.positions_for(1), &[7]);
        assert_eq!(list.positions_for(2), &[1, 4, 9]);
    }
    #[test]
    fn test_merge_intersection() {
        let a = vec![1u64, 3, 5, 7, 9];
        let b = vec![2u64, 3, 5, 8, 9, 10];
        assert_eq!(intersect_sorted_ids(&a, &b), vec![3, 5, 9]);
    }
}
