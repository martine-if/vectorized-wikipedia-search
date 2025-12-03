import json
import os

# paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")

# individual loaders
def load_documents():
    json_path = os.path.join(RAW_DIR, "KeySearchWiki-JSON.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"{json_path} does not exist.")
    
    with open(json_path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    return docs

def load_queries():
    path = os.path.join(RAW_DIR, "KeySearchWiki-queries-iri.txt")
    if not os.path.exists(path):
        return {}
    
    queries = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            qid = parts[0]
            text = " ".join(parts[1:])
            queries[qid] = text
    return queries

def load_natural_queries():
    path = os.path.join(RAW_DIR, "KeySearchWiki-queries-naturalized.txt")
    if not os.path.exists(path):
        return {}
    
    natural = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            qid = parts[0]
            text = " ".join(parts[1:])
            natural[qid] = text
    return natural

def load_query_labels():
    path = os.path.join(RAW_DIR, "KeySearchWiki-queries-label.txt")
    if not os.path.exists(path):
        return {}
    
    labels = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            qid = parts[0]
            label = parts[-1]
            text = " ".join(parts[1:-1])
            labels[qid] = {"text": text, "label": label}
    return labels

def load_qrels():
    path = os.path.join(RAW_DIR, "KeySearchWiki-qrels-trec.txt")
    if not os.path.exists(path):
        return {}
    
    qrels = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            qid, _, doc_id, rel = parts
            rel = int(rel)
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][doc_id] = rel
    return qrels

# store all loaded data
def load_all_raw():
    data = {
        "documents": load_documents(),
        "queries_iri": load_queries(),
        "queries_natural": load_natural_queries(),
        "query_labels": load_query_labels(),
        "qrels": load_qrels(),
    }
    print("Loaded:")
    for k, v in data.items():
        print(f"  {k}: {len(v)}")
    return data

if __name__ == "__main__":
    data = load_all_raw()
