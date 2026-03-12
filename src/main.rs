// ╔══════════════════════════════════════════════════════════════╗
// ║          TWIG SLM ENGINE  —  v1.1  (Rust)                  ║
// ║  Ultra-lightweight Small Language Model core                ║
// ║  Storage format: .twig  |  Zero external dependencies       ║
// ╚══════════════════════════════════════════════════════════════╝
//
// Twig File Syntax
// ────────────────
//   VOCAB section    →  @<id>:<word>
//   PATHWAY section  →  @<from_id>->@<to_id>:<weight>
//
// Build & run:
//   cargo build --release
//   cargo run --release -- teach  corpus.txt model.twig
//   cargo run --release -- predict model.twig "cat"
//   cargo run --release -- predict model.twig "the" 5
//   cargo run --release -- generate model.twig "the" 10
//   cargo run --release -- demo

use std::collections::HashMap;
use std::fs;
use std::io::{self, BufRead, BufWriter, Write};

// ─────────────────────────────────────────────────────────────
//  Shared types
// ─────────────────────────────────────────────────────────────

/// word → id
type Vocab = HashMap<String, u32>;
/// id → word
type IdToWord = HashMap<u32, String>;
/// from_id → (to_id → weight)
type Pathways = HashMap<u32, HashMap<u32, u32>>;

// ─────────────────────────────────────────────────────────────
//  SECTION 1 — THE TEACHER
//  Reads raw text, builds vocab + pathway counts
// ─────────────────────────────────────────────────────────────

/// Tokenise one character: keep a-z and apostrophe, map everything else to None.
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

/// Split `text` into lowercase word tokens (letters + apostrophes only).
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
    pub vocab: Vocab,
    pub id_to_word: IdToWord,
    pub pathways: Pathways,
}

// FIX 4: plain function instead of a macro — same logic, easier to read and debug.
fn get_or_insert_id(
    word: &str,
    vocab: &mut Vocab,
    id_to_word: &mut IdToWord,
    next_id: &mut u32,
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

/// Read a plain-text file and learn vocab + bigram pathways from it.
pub fn teach(text_path: &str) -> Result<TeachResult, String> {
    let raw = fs::read_to_string(text_path)
        .map_err(|e| format!("Cannot read '{}': {}", text_path, e))?;

    let tokens = tokenise(&raw);

    if tokens.len() < 2 {
        return Err("Training file must contain at least 2 words.".into());
    }

    let mut vocab: Vocab = HashMap::new();
    let mut id_to_word: IdToWord = HashMap::new();
    let mut next_id: u32 = 0;
    let mut pathways: Pathways = HashMap::new();

    let mut prev_id =
        get_or_insert_id(&tokens[0], &mut vocab, &mut id_to_word, &mut next_id);

    for token in &tokens[1..] {
        let curr_id =
            get_or_insert_id(token, &mut vocab, &mut id_to_word, &mut next_id);
        *pathways
            .entry(prev_id)
            .or_default()
            .entry(curr_id)
            .or_insert(0) += 1;
        prev_id = curr_id;
    }

    let pathway_count: usize = pathways.values().map(|m| m.len()).sum();
    println!(
        "[TEACHER]  Learned {:>7} unique words",
        format_num(vocab.len())
    );
    println!(
        "[TEACHER]  Mapped  {:>7} word-pair pathways",
        format_num(pathway_count)
    );

    Ok(TeachResult { vocab, id_to_word, pathways })
}

// ─────────────────────────────────────────────────────────────
//  SECTION 2 — THE SAVER
//  Serialises vocab + pathways into a .twig file
// ─────────────────────────────────────────────────────────────

/// Write learned data to a compact .twig file.
///
/// Space-saving decisions
/// ──────────────────────
/// • IDs replace words in the PATHWAYS section
/// • No padding, no whitespace beyond the separator
/// • Sections marked with single-line headers
///
// FIX 2: removed dead `vocab: &Vocab` parameter — only `id_to_word` was ever used.
// FIX 3: streams via BufWriter instead of building one giant String first,
//         which halves peak memory on large corpora.
// FIX 1 (indirect): push_u32 + unsafe entirely gone; write! handles formatting.
pub fn save(
    id_to_word: &IdToWord,
    pathways: &Pathways,
    out_path: &str,
) -> Result<(), String> {
    let file = fs::File::create(out_path)
        .map_err(|e| format!("Cannot create '{}': {}", out_path, e))?;
    let mut w = BufWriter::new(file);

    // ── VOCAB block ─────────────────────────────
    writeln!(w, "[VOCAB]").map_err(|e| e.to_string())?;

    // Sort by ID for deterministic, human-readable output
    let mut entries: Vec<(u32, &str)> = id_to_word
        .iter()
        .map(|(&id, word)| (id, word.as_str()))
        .collect();
    entries.sort_unstable_by_key(|&(id, _)| id);

    for (id, word) in &entries {
        writeln!(w, "@{}:{}", id, word).map_err(|e| e.to_string())?;
    }

    // ── PATHWAYS block ───────────────────────────
    writeln!(w, "[PATHWAYS]").map_err(|e| e.to_string())?;

    let mut from_ids: Vec<u32> = pathways.keys().copied().collect();
    from_ids.sort_unstable();

    for from_id in from_ids {
        // Sort by weight descending so the heaviest path is first
        let mut pairs: Vec<(u32, u32)> = pathways[&from_id]
            .iter()
            .map(|(&t, &wt)| (t, wt))
            .collect();
        pairs.sort_unstable_by(|a, b| b.1.cmp(&a.1));

        for (to_id, weight) in pairs {
            writeln!(w, "@{}->@{}:{}", from_id, to_id, weight)
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
//  SECTION 3 — THE SPEAKER (TwigBrain)
//  Loads a .twig file and predicts the next word
// ─────────────────────────────────────────────────────────────

pub struct TwigBrain {
    vocab: Vocab,
    id_to_word: IdToWord,
    /// Pathways pre-sorted by weight descending per source word.
    /// Stored as Vec<(to_id, weight)> for O(1) best-path lookup.
    pathways: HashMap<u32, Vec<(u32, u32)>>,
}

impl TwigBrain {
    /// Load a .twig file and build the in-memory brain.
    pub fn load(twig_path: &str) -> Result<Self, String> {
        let file = fs::File::open(twig_path)
            .map_err(|e| format!("Cannot open '{}': {}", twig_path, e))?;

        let mut vocab: Vocab = HashMap::new();
        let mut id_to_word: IdToWord = HashMap::new();
        // Raw unsorted map while loading; sort after
        let mut raw_pathways: Pathways = HashMap::new();

        // FIX 6: PartialEq derive removed — Section is only used in match arms.
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
                // Pattern: @<id>:<word>
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
                // Pattern: @<from>->@<to>:<weight>
                Section::Pathways => {
                    if let Some(rest) = line.strip_prefix('@') {
                        if let Some((from_str, rest2)) = rest.split_once("->@") {
                            if let Some((to_str, w_str)) = rest2.split_once(':') {
                                if let (Ok(f), Ok(t), Ok(wt)) = (
                                    from_str.parse::<u32>(),
                                    to_str.parse::<u32>(),
                                    w_str.parse::<u32>(),
                                ) {
                                    raw_pathways.entry(f).or_default().insert(t, wt);
                                }
                            }
                        }
                    }
                }
                Section::None => {}
            }
        }

        // Convert to sorted Vec<(to_id, weight)> per source, weight desc
        let mut pathways: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();
        for (from_id, targets) in raw_pathways {
            let mut pairs: Vec<(u32, u32)> = targets.into_iter().collect();
            pairs.sort_unstable_by(|a, b| b.1.cmp(&a.1));
            pathways.insert(from_id, pairs);
        }

        let total_paths: usize = pathways.values().map(|v| v.len()).sum();
        println!(
            "[SPEAKER]  Loaded {} words  |  {} pathways  from {}",
            format_num(vocab.len()),
            format_num(total_paths),
            twig_path
        );

        Ok(TwigBrain { vocab, id_to_word, pathways })
    }

    // ── Public API ─────────────────────────────────────────────

    /// Predict the single most likely next word after `word`.
    /// Returns `None` if the word is unknown or has no outgoing paths.
    // FIX 5: now prints diagnostic messages on every failure path,
    //         matching the behaviour of predict_top.
    pub fn predict_one(&self, word: &str) -> Option<&str> {
        let word_lc = word.trim().to_lowercase();

        let &wid = match self.vocab.get(&word_lc) {
            Some(id) => id,
            None => {
                println!("[SPEAKER]  '{}' not in vocabulary.", word_lc);
                return None;
            }
        };

        let targets = match self.pathways.get(&wid) {
            Some(t) => t,
            None => {
                println!("[SPEAKER]  No outgoing paths from '{}'.", word_lc);
                return None;
            }
        };

        let (best_id, best_w) = targets.first()?;
        let result = self.id_to_word.get(best_id)?.as_str();
        println!("[SPEAKER]  '{}' → '{}'  (weight: {})", word_lc, result, best_w);
        Some(result)
    }

    /// Predict the top-N most likely next words after `word`.
    /// Returns a Vec of `(&str, weight)` pairs, heaviest first.
    // FIX (minor): removed redundant explicit `'a` lifetime — elision handles it.
    pub fn predict_top(&self, word: &str, top: usize) -> Vec<(&str, u32)> {
        let word_lc = word.trim().to_lowercase();

        let Some(&wid) = self.vocab.get(&word_lc) else {
            println!("[SPEAKER]  '{}' not in vocabulary.", word_lc);
            return vec![];
        };

        let Some(targets) = self.pathways.get(&wid) else {
            println!("[SPEAKER]  No outgoing paths from '{}'.", word_lc);
            return vec![];
        };

        let results: Vec<(&str, u32)> = targets
            .iter()
            .take(top)
            .filter_map(|(tid, wt)| {
                self.id_to_word.get(tid).map(|w| (w.as_str(), *wt))
            })
            .collect();

        println!("[SPEAKER]  Top {} after '{}':", results.len(), word_lc);
        for (w, weight) in &results {
            let bar = "█".repeat((*weight as usize).min(30));
            println!("           {:<20} {} {}", w, bar, weight);
        }
        results
    }

    /// Generate a word sequence of up to `length` words starting from `seed`.
    /// Uses greedy best-path at each step; stops at dead-ends or self-loops.
    // FIX 7: removed the confusing double-binding
    //   (was: let mut current = …; let current = …trim()…; let mut cur = current)
    pub fn generate(&self, seed: &str, length: usize) -> String {
        let seed = seed.trim().to_lowercase();
        let mut sequence = vec![seed.clone()];
        let mut current = seed;

        for _ in 1..length {
            match self.predict_one(&current) {
                None => break,
                Some(nxt) if nxt == current => break, // self-loop guard
                Some(nxt) => {
                    current = nxt.to_string();
                    sequence.push(current.clone());
                }
            }
        }

        let sentence = sequence.join(" ");
        println!("\n[GENERATE] {}", sentence);
        sentence
    }
}

// ─────────────────────────────────────────────────────────────
//  SECTION 4 — CLI + DEMO
// ─────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();

    match args.get(1).map(|s| s.as_str()) {

        // cargo run -- teach <corpus.txt> <model.twig>
        Some("teach") => {
            let txt  = args.get(2).expect("Usage: teach <text_file> <out.twig>");
            let twig = args.get(3).expect("Usage: teach <text_file> <out.twig>");

            let res = teach(txt)
                .unwrap_or_else(|e| { eprintln!("Error: {}", e); std::process::exit(1); });
            // FIX 2 (call site): vocab argument removed.
            save(&res.id_to_word, &res.pathways, twig)
                .unwrap_or_else(|e| { eprintln!("Error: {}", e); std::process::exit(1); });
        }

        // cargo run -- predict <model.twig> <word> [top_n]
        Some("predict") => {
            let twig = args.get(2).expect("Usage: predict <model.twig> <word> [top_n]");
            let word = args.get(3).expect("Usage: predict <model.twig> <word> [top_n]");
            let top: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1);

            let brain = TwigBrain::load(twig)
                .unwrap_or_else(|e| { eprintln!("Error: {}", e); std::process::exit(1); });

            if top == 1 {
                brain.predict_one(word);
            } else {
                brain.predict_top(word, top);
            }
        }

        // cargo run -- generate <model.twig> <seed> [length]
        Some("generate") => {
            let twig   = args.get(2).expect("Usage: generate <model.twig> <seed> [length]");
            let seed   = args.get(3).expect("Usage: generate <model.twig> <seed> [length]");
            let length: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(10);

            let brain = TwigBrain::load(twig)
                .unwrap_or_else(|e| { eprintln!("Error: {}", e); std::process::exit(1); });

            brain.generate(seed, length);
        }

        // cargo run -- demo   (self-contained demo, no files needed)
        Some("demo") | None => run_demo(),

        Some(unknown) => {
            eprintln!(
                "Unknown command: '{}'. Use: teach | predict | generate | demo",
                unknown
            );
            std::process::exit(1);
        }
    }
}

fn run_demo() {
    const DEMO_TEXT: &str = "
        The cat sat on the mat. The cat ate the rat.
        The dog sat on the log. The dog bit the cat.
        A cat and a dog can be good friends.
        The rat ran from the cat. The cat ran after the rat.
        A good dog is a happy dog. A happy cat is a fat cat.
        The fat cat sat. The sad rat ran. The dog ran far.
    ";

    let sep = "=".repeat(60);
    println!("{}", sep);
    println!("  TWIG SLM  —  Quick Demo  (Rust)");
    println!("{}", sep);

    let txt_path  = "/tmp/twig_demo_corpus.txt";
    let twig_path = "/tmp/twig_demo_brain.twig";
    fs::write(txt_path, DEMO_TEXT).expect("Could not write temp corpus");

    // ① Teach
    println!("\n── TEACH ──────────────────────────────");
    let res = teach(txt_path).expect("teach failed");

    // ② Save
    println!("\n── SAVE ───────────────────────────────");
    save(&res.id_to_word, &res.pathways, twig_path).expect("save failed");

    // ③ Preview
    println!("\n── TWIG FILE PREVIEW (first 20 lines) ─");
    let twig_content = fs::read_to_string(twig_path).unwrap();
    for (i, line) in twig_content.lines().enumerate() {
        if i >= 20 { println!("    ..."); break; }
        println!("    {}", line);
    }

    // ④ Load + predict
    println!("\n── PREDICT ────────────────────────────");
    let brain = TwigBrain::load(twig_path).expect("load failed");

    brain.predict_one("cat");
    brain.predict_one("dog");
    brain.predict_one("notaword");    // now prints a diagnostic instead of silently failing
    brain.predict_top("the", 5);

    // ⑤ Generate
    println!("\n── GENERATE ───────────────────────────");
    brain.generate("the", 8);
    brain.generate("a", 6);

    let _ = fs::remove_file(txt_path);
    println!("\n{}", sep);
    println!("  Done! Brain saved to: {}", twig_path);
    println!("{}", sep);
}

// ─────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────

/// Format a usize with comma separators for readability.
fn format_num(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    let offset = s.len() % 3;
    for (i, ch) in s.chars().enumerate() {
        if i != 0 && (i % 3) == offset {
            out.push(',');
        }
        out.push(ch);
    }
    out
}
