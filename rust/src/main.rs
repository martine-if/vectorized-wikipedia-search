use std::collections::HashMap;
use std::io::{BufWriter, Write};
use std::{env, fs, io};
use std::fs::File;
use std::ops::Index;
use std::path::Path;
use bincode::config::standard;
use bincode::serde::{decode_from_std_read, encode_into_std_write};
use indexmap::IndexMap;
use indicatif::{ProgressBar, ProgressIterator};
use serde::de::DeserializeOwned;
use serde::Serialize;
use clap::Parser;
use log::error;
use crate::query_vectorizer::{filter_words, get_idf_scores, get_tf_score, get_tf_scores, parse_articles, parse_queries, Article, Query};

mod query_vectorizer;

struct Sim {
    doc_id: u32,
    score: f64,
}

#[derive(Parser, Debug)]
#[command(name = "wikisearch")]
struct Args {
    #[arg(long)]
    interactive: bool,
    #[arg(short = 's', long = "size", default_value_t = 100000)]
    size: usize
}

struct ArticleCorpus {
    article_map: IndexMap<u32, Article>,
    articles: Vec<Vec<String>>,
    idf: Vec<HashMap<String, f64>>,
    tf: Vec<IndexMap<String, f64>>,
}

fn save_to_cache<T: Serialize>(value: &T, path: &str) {
    let mut file = File::create(path).expect("Failed to create cache file");
    encode_into_std_write(value, &mut file, standard()).expect("Failed to encode");
}

fn load_from_cache<T: DeserializeOwned>(path: &str) -> T {
    let mut file = File::open(path).expect("Failed to open file");
    decode_from_std_read(&mut file, standard()).expect("Failed to decode")
}

fn main() {
    let args = Args::parse();

    let corpus_opt = build_article_corpus("../data/processed/all_articles.txt", args.size);
    let corpus = match corpus_opt {
        Some(c) => c,
        None => {
            error!("Failed to build article corpus");
            return
        },
    };

    if args.interactive {
        run_interactive(&corpus);
    } else {
        run_query_file(&corpus, "../data/processed/keysearch.qry");
    }
}

fn run_interactive(corpus: &ArticleCorpus) {
    loop {
        print!("Enter query: ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .expect("Failed to read line");

        let query = input.trim().to_string();
        let query_vec: Vec<String> = query
            .split_whitespace()
            .map(String::from)
            .collect();

        let query_tf = get_tf_score(&query_vec);
        let query_idf: HashMap<String, f64> = query_tf.keys().map(|k| (k.clone(), 0.1)).collect();

        let mut results = process_query(corpus, &query_vec, &query_tf, &query_idf);

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        let top_10 = &results[..10];

        for (rank, entry) in top_10.iter().enumerate() {
            let article = corpus.article_map.get(&entry.doc_id).unwrap();
            let score = entry.score;
            let rank = rank + 1;

            println!("{}. {} - {}", rank, article.title, score);
        }
    }
}

fn build_article_corpus(path: &str, corpus_size: usize) -> Option<ArticleCorpus> {
    let article_res = parse_articles(path);

    let article_map = match article_res {
        Ok(a) => a,
        Err(_) => return None,
    };

    let articles: Vec<Vec<String>> = article_map
        .values()
        .map(|article| article.text.clone())
        .collect();
    let articles = &articles;
    let articles = &articles[..articles.len().min(corpus_size)];

    let articles = filter_words(articles);

    let art_idf = get_idf_scores(&articles);
    let art_tf: Vec<IndexMap<String, f64>> = get_tf_scores(&articles);

    Some(ArticleCorpus {
        article_map,
        articles,
        idf: art_idf,
        tf: art_tf,
    })
}

fn run_query_file(corpus: &ArticleCorpus, path: &str) {
    let query_res = parse_queries(path);

    let query_map = match query_res {
        Ok(q) => q,
        Err(_) => return,
    };

    let queries: Vec<Vec<String>> = query_map
        .values()
        .map(|query| query.text.clone())
        .collect();
    let queries = filter_words(&queries);

    let query_idf_path = "../data/cache/query_idf.bin";
    let query_idf: Vec<HashMap<String, f64>>;
    if Path::new(query_idf_path).exists() {
        query_idf = load_from_cache(query_idf_path);
    } else {
        query_idf = get_idf_scores(&queries);
        save_to_cache(&query_idf, query_idf_path);
    }

    let query_tf: Vec<IndexMap<String, f64>> = get_tf_scores(&queries);

    let mut output_lines: Vec<String> = Vec::new();

    println!("Processing queries...");
    let pb = ProgressBar::new(queries.len() as u64);
    for (qid, query) in queries.iter().enumerate().progress_with(pb.clone()) {
        let mut results = process_query(corpus, query, &query_tf[qid], &query_idf[qid]);

        // Sort the results and get the top 10
        results.select_nth_unstable_by(10, |a, b| b.score.partial_cmp(&a.score).unwrap());
        let top10 = &mut results[..10];
        top10.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        // Append the top 10 to the output lines
        if let Some(entry) = query_map.get_index(qid) {
            let output_query_id = *entry.0;
            for (rank, sim) in top10.iter().enumerate() {
                let doc_id = sim.doc_id;
                let sim_score = sim.score;

                let display_rank = rank + 1;

                output_lines.push(format!("{:03} {doc_id} {display_rank} {sim_score}", output_query_id))
            }
        }
    }

    let output_path = Path::new("../data/results/ranking_output_rust.txt");

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).expect("Failed to create parent dirs");
    }

    let mut file = File::create(output_path).expect("Failed to create file");

    for line in output_lines {
        writeln!(file, "{}", line).expect("Failed to write line");
    }
}

// Returns UNSORTED vec of cosine similarities
fn process_query(
    corpus: &ArticleCorpus,
    query: &Vec<String>,
    query_tf: &IndexMap<String, f64>,
    query_idf: &HashMap<String, f64>,
) -> Vec<Sim> {
    let art_tf: &Vec<IndexMap<String, f64>> = &corpus.tf;
    let art_idf: &Vec<HashMap<String, f64>> = &corpus.idf;

    let mut query_vec: Vec<f64> = Vec::new();
    for word in query {
        let query_word_tf = query_tf[word];
        let query_word_idf = query_idf[word];
        let query_tf_idf = query_word_tf * query_word_idf;
        query_vec.push(query_tf_idf);
    }
    let a = &query_vec;
    let norm_a = norm(a);

    let mut sims: Vec<Sim> = Vec::new();
    for (art_idx, _) in corpus.articles.iter().enumerate() {
        let mut art_vec: Vec<f64> = Vec::with_capacity(query.len());
        for word in query {
            if let Some(tf) = art_tf[art_idx].get(word) {
                let idf = art_idf[art_idx].get(word).unwrap();
                art_vec.push(tf * idf);
            } else {
                art_vec.push(0.0);
            }
        }

        let b = &art_vec;
        let norm_b = norm(b);
        let mut cos_similarity = 0.0;
        if norm_a != 0.0 && norm_b != 0.0 {
            cos_similarity = dot(a, b) / (norm_a * norm_b);
        }

        if cos_similarity.is_nan() {
            cos_similarity = 0.0;
        }

        if let Some(entry) = corpus.article_map.get_index(art_idx) {
            let art_id = *entry.0;
            sims.push(Sim { doc_id: art_id, score: cos_similarity });
        }
    }

    sims
}

fn norm(v: &[f64]) -> f64 {
    let sum_sq: f64 = v.iter().map(|x| x * x).sum();
    sum_sq.sqrt()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .sum()
}


