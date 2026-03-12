// ╔══════════════════════════════════════════════════════════════╗
// ║          TWIG SLM ENGINE  —  v2.0  (Rust)                  ║
// ║  Ultra-lightweight Small Language Model core                ║
// ║  Storage format: .twig  |  Zero external dependencies       ║
// ╚══════════════════════════════════════════════════════════════╝
//
// Twig File Syntax (v2)
// ─────────────────────
//   VOCAB section    →  @<id>:<word>
//   PATHWAY section  →  @<id1>+@<id2>->@<id3>:<weight>
//                        ^^^^^^^^ 2-word context (trigram)
//
// Build & run:
//   cargo build --release
//   cargo run --release -- teach   corpus.txt model.twig
//   cargo run --release -- predict model.twig "the" "cat" [top_n]
//   cargo run --release -- generate model.twig "the cat" [length] [temp]
//   cargo run --release -- demo

use std::collections::HashMap;
use std::fs;
use std::io::{self, BufRead, BufWriter, Write};
use std::time::{SystemTime, UNIX_EPOCH};

// ─────────────────────────────────────────────────────────────
//  Shared types
// ─────────────────────────────────────────────────────────────

/// word → id
type Vocab = HashMap<String, u32>;
/// id → word
type IdToWord = HashMap<u32, String>;
/// (context_id1, context_id2) → (next_id → weight)   [trigram]
type Pathways = HashMap<(u32, u32), HashMap<u32, u32>>;

// ─────────────────────────────────────────────────────────────
//  LCG — Linear Congruential Generator
//  Pure std, no rand crate needed.
//  Parameters: Knuth (MMIX), full 64-bit period.
// ─────────────────────────────────────────────────────────────

pub struct Lcg {
    state: u64,
}

impl Lcg {
    /// Seed from system time for non-deterministic generation.
    pub fn from_time() -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64 ^ d.subsec_nanos() as u64)
            .unwrap_or(0xDEAD_BEEF_CAFE_1337);
        Lcg { state: seed }
    }

    /// Advance the state and return the next raw u64.
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Return a uniformly distributed f64 in [0, 1).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        // Use top 53 bits for full double precision
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

// ─────────────────────────────────────────────────────────────
//  Stochastic sampling with temperature
// ─────────────────────────────────────────────────────────────

/// Pick one `to_id` from `targets` using temperature-scaled weighted sampling.
///
/// Temperature behaviour
/// ─────────────────────
///   temp → 0.0  : near-greedy (highest weight almost always wins)
///   temp = 1.0  : weights used as-is  (faithful to training data)
///   temp > 1.0  : weights flattened   (more creative / surprising)
///
/// Each raw weight w is scaled to w^(1/temp) before the raffle.
fn weighted_sample(targets: &[(u32, u32)], temp: f64, rng: &mut Lcg) -> Option<u32> {
    if targets.is_empty() {
        return None;
    }
    // Single candidate — no randomness needed
    if targets.len() == 1 {
        return Some(targets[0].0);
    }

    // Clamp temperature to a safe range to avoid 0-div or NaN
    let t = temp.max(1e-6);

    // Scale weights: w^(1/t)
    let scaled: Vec<f64> = targets
        .iter()
        .map(|&(_, w)| (w as f64).powf(1.0 / t))
        .collect();

    let total: f64 = scaled.iter().sum();
    let roll = rng.next_f64() * total;

    let mut cumsum = 0.0;
    for (i, &s) in scaled.iter().enumerate() {
        cumsum += s;
        if roll < cumsum {
            return Some(targets[i].0);
        }
    }
    // Floating-point edge case: roll == total exactly
    Some(targets.last().unwrap().0)
}

// ─────────────────────────────────────────────────────────────
//  SECTION 1 — THE TEACHER
//  Reads raw text, builds vocab + trigram pathway counts
// ─────────────────────────────────────────────────────────────

#[inline]
fn keep_char(c: char) -> Option<char> {
    if c.is_ascii_alphabetic() {
        Some(c.to_ascii_lowercase())
    } else if c == '\'' {
        Some(c)
    } else {
        None
    }
}

fn tokenise(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        match keep_char(ch) {
            Some(c) => current.push(c),
            None => {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
            }
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

pub struct TeachResult {
    pub vocab:      Vocab,
    pub id_to_word: IdToWord,
    pub pathways:   Pathways,
}

fn get_or_insert_id(
    word:       &str,
    vocab:      &mut Vocab,
    id_to_word: &mut IdToWord,
    next_id:    &mut u32,
) -> u32 {
    if let Some(&id) = vocab.get(word) {
        return id;
    }
    let id = *next_id;
    vocab.insert(word.to_string(), id);
    id_to_word.insert(id, word.to_string());
    *next_id += 1;
    id
}

/// Read a plain-text file and learn vocab + trigram pathways from it.
/// Requires at least 3 tokens (one context pair + one target).
pub fn teach(text_path: &str) -> Result<TeachResult, String> {
    let raw = fs::read_to_string(text_path)
        .map_err(|e| format!("Cannot read '{}': {}", text_path, e))?;

    let tokens = tokenise(&raw);
    if tokens.len() < 3 {
        return Err("Training file must contain at least 3 words for the trigram model.".into());
    }

    let mut vocab:      Vocab    = HashMap::new();
    let mut id_to_word: IdToWord = HashMap::new();
    let mut next_id:    u32      = 0;
    let mut pathways:   Pathways = HashMap::new();

    // Seed the sliding window with the first two tokens
    let mut ctx0 = get_or_insert_id(&tokens[0], &mut vocab, &mut id_to_word, &mut next_id);
    let mut ctx1 = get_or_insert_id(&tokens[1], &mut vocab, &mut id_to_word, &mut next_id);

    // Slide a 3-word window: (ctx0, ctx1) → token
    for token in &tokens[2..] {
        let next = get_or_insert_id(token, &mut vocab, &mut id_to_word, &mut next_id);
        *pathways
            .entry((ctx0, ctx1))
            .or_default()
            .entry(next)
            .or_insert(0) += 1;
        ctx0 = ctx1;
        ctx1 = next;
    }

    let pathway_count: usize = pathways.values().map(|m| m.len()).sum();
    println!("[TEACHER]  Learned {:>7} unique words",    format_num(vocab.len()));
    println!("[TEACHER]  Mapped  {:>7} trigram pathways", format_num(pathway_count));

    Ok(TeachResult { vocab, id_to_word, pathways })
}

// ─────────────────────────────────────────────────────────────
//  SECTION 2 — THE SAVER
// ─────────────────────────────────────────────────────────────

/// Write learned data to a compact .twig file (v2 trigram format).
pub fn save(
    id_to_word: &IdToWord,
    pathways:   &Pathways,
    out_path:   &str,
) -> Result<(), String> {
    let file = fs::File::create(out_path)
        .map_err(|e| format!("Cannot create '{}': {}", out_path, e))?;
    let mut w = BufWriter::new(file);

    // ── VOCAB ────────────────────────────────────
    writeln!(w, "[VOCAB]").map_err(|e| e.to_string())?;

    let mut entries: Vec<(u32, &str)> = id_to_word
        .iter()
        .map(|(&id, word)| (id, word.as_str()))
        .collect();
    entries.sort_unstable_by_key(|&(id, _)| id);

    for (id, word) in &entries {
        writeln!(w, "@{}:{}", id, word).map_err(|e| e.to_string())?;
    }

    // ── PATHWAYS ─────────────────────────────────
    // Format:  @<ctx1>+@<ctx2>->@<next>:<weight>
    writeln!(w, "[PATHWAYS]").map_err(|e| e.to_string())?;

    let mut contexts: Vec<(u32, u32)> = pathways.keys().copied().collect();
    contexts.sort_unstable();

    for (c1, c2) in contexts {
        let mut pairs: Vec<(u32, u32)> = pathways[&(c1, c2)]
            .iter()
            .map(|(&t, &wt)| (t, wt))
            .collect();
        // Heaviest path first — makes the file human-readable and speeds up top-1 load
        pairs.sort_unstable_by(|a, b| b.1.cmp(&a.1));

        for (to_id, weight) in pairs {
            writeln!(w, "@{}+@{}->@{}:{}", c1, c2, to_id, weight)
                .map_err(|e| e.to_string())?;
        }
    }

    w.flush().map_err(|e| e.to_string())?;

    let size_kb = fs::metadata(out_path)
        .map(|m| m.len() as f64 / 1024.0)
        .unwrap_or(0.0);
    println!("[SAVER]    Written → {}  ({:.2} KB)", out_path, size_kb);
    Ok(())
}

// ─────────────────────────────────────────────────────────────
//  SECTION 3 — THE SPEAKER  (TwigBrain)
// ─────────────────────────────────────────────────────────────

pub struct TwigBrain {
    vocab:      Vocab,
    id_to_word: IdToWord,
    /// Pre-sorted Vec<(to_id, weight)> per context pair, weight descending.
    pathways:   HashMap<(u32, u32), Vec<(u32, u32)>>,
}

impl TwigBrain {
    /// Load a v2 .twig file into memory.
    pub fn load(twig_path: &str) -> Result<Self, String> {
        let file = fs::File::open(twig_path)
            .map_err(|e| format!("Cannot open '{}': {}", twig_path, e))?;

        let mut vocab:        Vocab    = HashMap::new();
        let mut id_to_word:   IdToWord = HashMap::new();
        let mut raw_pathways: Pathways = HashMap::new();

        enum Section { None, Vocab, Pathways }
        let mut section = Section::None;

        for line_res in io::BufReader::new(file).lines() {
            let line = line_res.map_err(|e| e.to_string())?;
            let line = line.trim();
            if line.is_empty() { continue; }

            match line {
                "[VOCAB]"    => { section = Section::Vocab;    continue; }
                "[PATHWAYS]" => { section = Section::Pathways; continue; }
                _ => {}
            }

            match section {
                // @<id>:<word>
                Section::Vocab => {
                    if let Some(rest) = line.strip_prefix('@') {
                        if let Some((id_str, word)) = rest.split_once(':') {
                            if let Ok(id) = id_str.parse::<u32>() {
                                vocab.insert(word.to_string(), id);
                                id_to_word.insert(id, word.to_string());
                            }
                        }
                    }
                }
                // @<c1>+@<c2>->@<to>:<weight>
                Section::Pathways => {
                    if let Some(rest) = line.strip_prefix('@') {
                        if let Some((c1_str, rest2)) = rest.split_once("+@") {
                            if let Some((c2_str, rest3)) = rest2.split_once("->@") {
                                if let Some((to_str, w_str)) = rest3.split_once(':') {
                                    if let (Ok(c1), Ok(c2), Ok(to), Ok(wt)) = (
                                        c1_str.parse::<u32>(),
                                        c2_str.parse::<u32>(),
                                        to_str.parse::<u32>(),
                                        w_str.parse::<u32>(),
                                    ) {
                                        raw_pathways
                                            .entry((c1, c2))
                                            .or_default()
                                            .insert(to, wt);
                                    }
                                }
                            }
                        }
                    }
                }
                Section::None => {}
            }
        }

        // Sort each candidate list by weight descending once at load time
        let mut pathways: HashMap<(u32, u32), Vec<(u32, u32)>> = HashMap::new();
        for ((c1, c2), targets) in raw_pathways {
            let mut pairs: Vec<(u32, u32)> = targets.into_iter().collect();
            pairs.sort_unstable_by(|a, b| b.1.cmp(&a.1));
            pathways.insert((c1, c2), pairs);
        }

        let total: usize = pathways.values().map(|v| v.len()).sum();
        println!(
            "[SPEAKER]  Loaded {} words  |  {} trigram pathways  from {}",
            format_num(vocab.len()),
            format_num(total),
            twig_path
        );

        Ok(TwigBrain { vocab, id_to_word, pathways })
    }

    // ── Internal helpers ──────────────────────────────────────

    fn lookup_id(&self, word: &str) -> Option<u32> {
        self.vocab.get(word).copied()
    }

    // ── Public API ────────────────────────────────────────────

    /// Sample the next word stochastically given a 2-word context.
    ///
    /// `temp`  — temperature:  0 ≈ greedy,  1 = natural,  >1 = creative
    pub fn predict_one(
        &self,
        w1:   &str,
        w2:   &str,
        temp: f64,
        rng:  &mut Lcg,
    ) -> Option<&str> {
        let w1 = w1.trim().to_lowercase();
        let w2 = w2.trim().to_lowercase();

        let id1 = match self.lookup_id(&w1) {
            Some(id) => id,
            None => { println!("[SPEAKER]  '{}' not in vocabulary.", w1); return None; }
        };
        let id2 = match self.lookup_id(&w2) {
            Some(id) => id,
            None => { println!("[SPEAKER]  '{}' not in vocabulary.", w2); return None; }
        };

        let targets = match self.pathways.get(&(id1, id2)) {
            Some(t) => t,
            None => { println!("[SPEAKER]  No paths from '{}+{}'.", w1, w2); return None; }
        };

        let chosen_id = weighted_sample(targets, temp, rng)?;
        let result = self.id_to_word.get(&chosen_id)?.as_str();
        println!("[SPEAKER]  '{}' '{}' → '{}'  (temp: {:.2})", w1, w2, result, temp);
        Some(result)
    }

    /// Show the top-N candidates for a context pair (deterministic, for inspection).
    /// Displays raw weights — useful for understanding what the model learned.
    pub fn predict_top(&self, w1: &str, w2: &str, top: usize) -> Vec<(&str, u32)> {
        let w1_lc = w1.trim().to_lowercase();
        let w2_lc = w2.trim().to_lowercase();

        let Some(id1) = self.lookup_id(&w1_lc) else {
            println!("[SPEAKER]  '{}' not in vocabulary.", w1_lc);
            return vec![];
        };
        let Some(id2) = self.lookup_id(&w2_lc) else {
            println!("[SPEAKER]  '{}' not in vocabulary.", w2_lc);
            return vec![];
        };
        let Some(targets) = self.pathways.get(&(id1, id2)) else {
            println!("[SPEAKER]  No paths from '{}+{}'.", w1_lc, w2_lc);
            return vec![];
        };

        let results: Vec<(&str, u32)> = targets
            .iter()
            .take(top)
            .filter_map(|(tid, wt)| {
                self.id_to_word.get(tid).map(|w| (w.as_str(), *wt))
            })
            .collect();

        println!("[SPEAKER]  Top {} after '{}' '{}':", results.len(), w1_lc, w2_lc);
        for (w, weight) in &results {
            let bar = "█".repeat((*weight as usize).min(30));
            println!("           {:<20} {} {}", w, bar, weight);
        }
        results
    }

    /// Generate a word sequence using stochastic sampling + temperature.
    ///
    /// `seed`    — must contain at least 2 words (the initial context pair)
    /// `length`  — total words in the output (includes the seed words)
    /// `temp`    — temperature: 0 ≈ greedy, 1 = natural, >1 = creative
    pub fn generate(&self, seed: &str, length: usize, temp: f64) -> String {
        let seed_words: Vec<String> = seed
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();

        if seed_words.len() < 2 {
            eprintln!("[GENERATE] Need at least 2 seed words for the trigram model.");
            return seed.to_string();
        }

        let mut rng = Lcg::from_time();
        let mut sequence = seed_words.clone();

        // Use the last 2 words as the rolling context window
        let mut w1 = seed_words[seed_words.len() - 2].clone();
        let mut w2 = seed_words[seed_words.len() - 1].clone();

        while sequence.len() < length {
            match self.predict_one(&w1, &w2, temp, &mut rng) {
                None => break,
                Some(nxt) => {
                    w1 = w2;
                    w2 = nxt.to_string();
                    sequence.push(w2.clone());
                }
            }
        }

        let sentence = sequence.join(" ");
        println!("\n[GENERATE] (temp={:.2})  {}", temp, sentence);
        sentence
    }
}

// ─────────────────────────────────────────────────────────────
//  SECTION 4 — CLI + DEMO
// ─────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();

    match args.get(1).map(|s| s.as_str()) {

        // teach <corpus.txt> <model.twig>
        Some("teach") => {
            let txt  = args.get(2).expect("Usage: teach <text_file> <out.twig>");
            let twig = args.get(3).expect("Usage: teach <text_file> <out.twig>");
            let res  = teach(txt)
                .unwrap_or_else(|e| die(&e));
            save(&res.id_to_word, &res.pathways, twig)
                .unwrap_or_else(|e| die(&e));
        }

        // predict <model.twig> <word1> <word2> [top_n]
        Some("predict") => {
            let twig  = args.get(2).expect("Usage: predict <model.twig> <w1> <w2> [top_n]");
            let w1    = args.get(3).expect("Usage: predict <model.twig> <w1> <w2> [top_n]");
            let w2    = args.get(4).expect("Usage: predict <model.twig> <w1> <w2> [top_n]");
            let top   = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(5usize);
            let brain = TwigBrain::load(twig).unwrap_or_else(|e| die(&e));
            brain.predict_top(w1, w2, top);
        }

        // generate <model.twig> "word1 word2" [length] [temp]
        Some("generate") => {
            let twig   = args.get(2).expect("Usage: generate <model.twig> \"w1 w2\" [len] [temp]");
            let seed   = args.get(3).expect("Usage: generate <model.twig> \"w1 w2\" [len] [temp]");
            let length = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(20usize);
            let temp   = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(1.0f64);
            let brain  = TwigBrain::load(twig).unwrap_or_else(|e| die(&e));
            brain.generate(seed, length, temp);
        }

        Some("demo") | None => run_demo(),

        Some(cmd) => {
            eprintln!("Unknown command '{}'. Use: teach | predict | generate | demo", cmd);
            std::process::exit(1);
        }
    }
}

fn run_demo() {
    // Slightly longer corpus so trigrams have enough context to be meaningful
    const DEMO_TEXT: &str = "
        The cat sat on the mat and the cat ate the rat near the mat.
        The dog sat on the log and the dog bit the cat on the mat.
        A cat and a dog can be very good friends if they try hard.
        The rat ran fast from the cat and the cat ran after the rat.
        A good dog is a happy dog and a happy cat is a very fat cat.
        The fat cat sat on the mat. The sad rat ran far away today.
        The dog ran after the rat but the rat hid under the log fast.
        A happy dog and a happy cat sat on the mat together today.
    ";

    let sep = "=".repeat(62);
    println!("{}", sep);
    println!("  TWIG SLM  v2.0 — Trigrams + Temperature  (Rust)");
    println!("{}", sep);

    let txt_path  = "/tmp/twig_demo_corpus.txt";
    let twig_path = "/tmp/twig_demo_brain_v2.twig";
    fs::write(txt_path, DEMO_TEXT).expect("Could not write temp corpus");

    // ① Teach
    println!("\n── TEACH ──────────────────────────────────────");
    let res = teach(txt_path).expect("teach failed");

    // ② Save
    println!("\n── SAVE ────────────────────────────────────────");
    save(&res.id_to_word, &res.pathways, twig_path).expect("save failed");

    // ③ Preview
    println!("\n── TWIG FILE PREVIEW (first 24 lines) ──────────");
    let twig_content = fs::read_to_string(twig_path).unwrap();
    for (i, line) in twig_content.lines().enumerate() {
        if i >= 24 { println!("    ..."); break; }
        println!("    {}", line);
    }

    // ④ Load
    println!("\n── PREDICT (top-5, deterministic) ───────────────");
    let brain = TwigBrain::load(twig_path).expect("load failed");

    brain.predict_top("the", "cat", 5);
    println!();
    brain.predict_top("a", "happy", 5);

    // ⑤ Generate at three temperatures
    println!("\n── GENERATE ────────────────────────────────────");
    println!("  [cold  — temp 0.2, very predictable]");
    brain.generate("the cat", 12, 0.2);

    println!("  [warm  — temp 1.0, natural]");
    brain.generate("the cat", 12, 1.0);

    println!("  [hot   — temp 2.5, creative]");
    brain.generate("the cat", 12, 2.5);

    println!("  [hot   — different seed]");
    brain.generate("a happy", 12, 2.0);

    let _ = fs::remove_file(txt_path);
    println!("\n{}", sep);
    println!("  Done!  Brain saved to: {}", twig_path);
    println!("{}", sep);
}

// ─────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────

fn die(msg: &str) -> ! {
    eprintln!("Error: {}", msg);
    std::process::exit(1);
}

fn format_num(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    let offset  = s.len() % 3;
    for (i, ch) in s.chars().enumerate() {
        if i != 0 && (i % 3) == offset { out.push(','); }
        out.push(ch);
    }
    out
}
