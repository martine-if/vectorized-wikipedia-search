use std::collections::HashSet;
use std::fs::File;
use std::io;
use std::io::{BufRead, BufReader};
use indexmap::IndexMap;
use indicatif::{ProgressBar, ProgressIterator};
use once_cell::sync::Lazy;

#[derive(Debug)]
pub struct Query {
    id: u32,
    pub text: Vec<String>,
}

#[derive(Debug)]
pub struct Article {
    id: u32,
    pub title: String,
    pub text: Vec<String>,
}

static STOP_LIST_ARRAY: &[&str] = &["a", "the", "an", "and", "or", "but", "about", "above", "after", "along", "amid", "among",
    "as", "at", "by", "for", "from", "in", "into", "like", "minus", "near", "of", "off", "on",
    "onto", "out", "over", "past", "per", "plus", "since", "till", "to", "under", "until", "up",
    "via", "vs", "with", "that", "can", "cannot", "could", "may", "might", "must",
    "need", "ought", "shall", "should", "will", "would", "have", "had", "has", "having", "be",
    "is", "am", "are", "was", "were", "being", "been", "get", "gets", "got", "gotten",
    "getting", "seem", "seeming", "seems", "seemed",
    "enough", "both", "all", "your", "those", "this", "these",
    "their", "the", "that", "some", "our", "no", "neither", "my",
    "its", "his", "her", "every", "either", "each", "any", "another",
    "an", "a", "just", "mere", "such", "merely", "right", "no", "not",
    "only", "sheer", "even", "especially", "namely", "as", "more",
    "most", "less", "least", "so", "enough", "too", "pretty", "quite",
    "rather", "somewhat", "sufficiently", "same", "different", "such",
    "when", "why", "where", "how", "what", "who", "whom", "which",
    "whether", "why", "whose", "if", "anybody", "anyone", "anyplace",
    "anything", "anytime", "anywhere", "everybody", "everyday",
    "everyone", "everyplace", "everything", "everywhere", "whatever",
    "whenever", "whereever", "whichever", "whoever", "whomever", "he",
    "him", "his", "her", "she", "it", "they", "them", "its", "their", "theirs",
    "you", "your", "yours", "me", "my", "mine", "I", "we", "us", "much", "and/or"]; 

static STOP_LIST: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    STOP_LIST_ARRAY.iter().copied().collect()
});

static PUNCTUATION: &[&str] = &[".", ",", ":", "(", ")", "/", "\'", "=", "?", "!", ";", "\"", "&"];

pub fn get_tf_scores(queries: &Vec<Vec<String>>) -> Vec<IndexMap<String, f64>> {
    let mut query_tf: Vec<IndexMap<String, f64>> = Vec::new();
    println!("Getting TF scores...");
    let pb = ProgressBar::new(queries.len() as u64);
    for query in queries.iter().progress_with(pb.clone()) {
        let mut tf: IndexMap<String, f64> = IndexMap::new();
        for word in query {
            let instances = query.iter().filter(|w| w == &word).count();
            let query_size = query.len();
            tf.insert(word.clone(), if query_size > 0 { instances as f64 / query.len() as f64 } else { 0.0 });
        }
        query_tf.push(tf);
    }
    query_tf
}

pub fn get_idf_scores(documents: &Vec<Vec<String>>) -> Vec<IndexMap<String, f64>> {
    let num_docs = documents.len();
    let mut docs_as_sets: Vec<HashSet<String>> = Vec::new();
    for doc in documents.clone() {
        docs_as_sets.push(doc.into_iter().collect());
    }
    let mut idf_docs: Vec<IndexMap<String, f64>> = Vec::new();
    println!("Getting IDF scores...");
    let pb = ProgressBar::new(documents.len() as u64);
    for doc in documents.iter().progress_with(pb.clone()) {
        let mut doc_idf: IndexMap<String, f64> = IndexMap::new();
        for t in doc {
            let mut num_docs_containing_t = 0;
            for checked_doc in &docs_as_sets {
                if checked_doc.contains(t) {
                    num_docs_containing_t += 1;
                }
            }
            let idf = (num_docs as f64 / num_docs_containing_t as f64).ln();
            doc_idf.insert(t.to_string(), idf);
        }

        idf_docs.push(doc_idf);
    }

    idf_docs
}

pub fn filter_words(document_set: &[Vec<String>]) -> Vec<Vec<String>> {
    let mut filtered_docs: Vec<Vec<String>> = Vec::new();
    println!("Filtering words...");
    let pb = ProgressBar::new(document_set.len() as u64);
    for document in document_set.iter().progress_with(pb.clone()) {
        let mut filtered_doc: Vec<String> = Vec::new();
        for word in document {
            let mut word = word.clone();
            if PUNCTUATION.contains(&word.as_str()) {
                continue;
            }

            if word.chars().any(|c| c.is_ascii_digit()) {
                continue;
            }

            for punc in PUNCTUATION {
                word = word.replace(*punc, "");
            }

            if word == "" {
                continue;
            }

            if STOP_LIST.contains(&word.as_str()) {
                continue;
            }

            word = word.replace("--", "-");

            if word.contains("-") {
                for part in word.split("-") {
                    if part.trim() != "" {
                        filtered_doc.push(part.to_lowercase());
                    }
                }
            } else {
                filtered_doc.push(word.to_lowercase());
            }
        }
        filtered_docs.push(filtered_doc)
    }

    filtered_docs
}

pub fn parse_articles(file_path: &str) -> io::Result<IndexMap<u32, Article>> {
    let mut map: IndexMap<u32, Article> = IndexMap::new();

    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    let mut curr_id: Option<u32> = None;
    let mut curr_title: Option<String> = None;
    let mut curr_doc: Vec<String> = Vec::new();
    let mut is_text = false;
    let mut is_title = false;

    println!("Reading article file...");
    for line_res in reader.lines() {
        let line = line_res?;
        if line.starts_with(".I ") {
            complete_article(curr_doc, &curr_id, &curr_title, &mut map);

            let parsed = line
                .split_whitespace()
                .nth(1)
                .and_then(|v| v.parse::<u32>().ok());
            if parsed.is_some() {
                curr_id = parsed
            }
            curr_doc = Vec::new();
            is_text = false;
        } else if line == ".T" {
            is_title = true;
        } else if line == ".W" {
            is_text = true;
            is_title = false;
        } else if is_text {
            for word in line.split_whitespace() {
                curr_doc.push(word.trim().to_string())
            }
        } else if is_title {
            curr_title = Some(line);
        }
    }
    complete_article(curr_doc, &curr_id, &curr_title, &mut map);

    Ok(map)
}

fn complete_article(curr_doc: Vec<String>, curr_id: &Option<u32>, curr_title: &Option<String>, map: &mut IndexMap<u32, Article>) {
    if !curr_doc.is_empty() {
        if let (Some(id), Some(title)) = (curr_id, curr_title) {
            let article = Article {
                id: *id,
                title: title.clone(),
                text: curr_doc
            };
            map.insert(*id, article);
        }
    }
}

pub fn parse_queries(file_path: &str) -> io::Result<IndexMap<u32, Query>> {
    let mut map: IndexMap<u32, Query> = IndexMap::new();

    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    let mut curr_id: Option<u32> = None;

    println!("Reading query file...");
    for line_res in reader.lines() {
        let line = line_res?;
        if line.starts_with(".I ") {
            let parsed = line
                .split_whitespace()
                .nth(1)
                .and_then(|v| v.parse::<u32>().ok());

            if parsed.is_some() {
                curr_id = parsed
            }
        } else if line != ".W" {
            if let Some(id) = curr_id {
                let mut text: Vec<String> = Vec::new();
                for word in line.split_whitespace() {
                    text.push(word.trim().to_string());
                }
                let query = Query {
                    id,
                    text
                };
                map.insert(id, query);
            }
        }
    }

    Ok(map)
}