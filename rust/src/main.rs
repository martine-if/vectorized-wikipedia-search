use std::collections::HashMap;
use std::io::{BufWriter, Write};
use std::fs;
use std::fs::File;
use std::ops::Index;
use std::path::Path;
use bincode::config::standard;
use bincode::serde::{decode_from_std_read, encode_into_std_write};
use indexmap::IndexMap;
use indicatif::{ProgressBar, ProgressIterator};
use serde::de::DeserializeOwned;
use serde::Serialize;
use crate::query_vectorizer::{filter_words, get_idf_scores, get_tf_scores, parse_articles, parse_queries, Article, Query};

mod query_vectorizer;

struct Sim(u32, f64);

fn save_to_cache<T: Serialize>(value: &T, path: &str) {
    let mut file = File::create(path).expect("Failed to create cache file");
    encode_into_std_write(value, &mut file, standard()).expect("Failed to encode");
}

fn load_from_cache<T: DeserializeOwned>(path: &str) -> T {
    let mut file = File::open(path).expect("Failed to open file");
    decode_from_std_read(&mut file, standard()).expect("Failed to decode")
}

fn main() {
    let query_res = parse_queries("../data/processed/keysearch.qry");
    let article_res = parse_articles("../data/processed/articles-1.txt");

    let (query_map, article_map) = match (query_res, article_res) {
        (Ok(q), Ok(a)) => (q, a),
        _ => return,
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

    let articles: Vec<Vec<String>> = article_map
        .values()
        .map(|article| article.text.clone())
        .collect();
    // let articles = &articles;
    let articles = &articles[..articles.len().min(1000)];

    let articles = filter_words(articles);

    let art_idf = get_idf_scores(&articles);
    let art_tf: Vec<IndexMap<String, f64>> = get_tf_scores(&articles);

    let mut output_lines: Vec<String> = Vec::new();

    println!("Processing queries...");
    let pb = ProgressBar::new(queries.len() as u64);
    for (qid, query) in queries.iter().enumerate().progress_with(pb.clone()) {
        process_query(qid, query, &articles, &art_tf, &art_idf, &query_tf, &query_idf, &article_map, &query_map, &mut output_lines);
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

fn process_query(
    qid: usize,
    query: &Vec<String>,
    articles: &Vec<Vec<String>>,
    art_tf: &Vec<IndexMap<String, f64>>,
    art_idf: &Vec<HashMap<String, f64>>,
    query_tf: &Vec<IndexMap<String, f64>>,
    query_idf: &Vec<HashMap<String, f64>>,
    article_map: &IndexMap<u32, Article>,
    query_map: &IndexMap<u32, Query>,
    output_lines: &mut Vec<String>
) {
    let mut sims: Vec<Sim> = Vec::new();
    for (art_idx, article) in articles.iter().enumerate() {
        let mut art_vec: Vec<f64> = Vec::new();
        for word in query {
            if article.contains(word) {
                let art_word_tf = art_tf[art_idx][word];
                let art_word_idf = art_idf[art_idx][word];
                let art_tf_idf = art_word_tf * art_word_idf;
                art_vec.push(art_tf_idf);
            } else {
                art_vec.push(0.0);
            }
        }

        let mut query_vec: Vec<f64> = Vec::new();
        for word in query {
            let query_word_tf = query_tf[qid][word];
            let query_word_idf = query_idf[qid][word];
            let query_tf_idf = query_word_tf * query_word_idf;
            query_vec.push(query_tf_idf);
        }

        let a = &query_vec;
        let b = &art_vec;

        let norm_a = norm(a);
        let norm_b = norm(b);
        let mut cos_similarity = 0.0;
        if norm_a != 0.0 && norm_b != 0.0 {
            cos_similarity = dot(a, b) / (norm_a * norm_b);
        }

        if cos_similarity.is_nan() {
            cos_similarity = 0.0;
        }

        if let Some(entry) = article_map.get_index(art_idx) {
            let art_id = *entry.0;
            sims.push(Sim(art_id, cos_similarity));
        }
    }

    sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    if let Some(entry) = query_map.get_index(qid) {
        let output_query_id = *entry.0;
        for (rank, sim) in sims.iter().enumerate() {
            let doc_id = sim.0;
            let sim_score = sim.1;

            if rank + 1 > 10 {
                break;
            }
            let display_rank = rank + 1;

            output_lines.push(format!("{:03} {doc_id} {display_rank} {sim_score}", output_query_id))
        }
    }
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


