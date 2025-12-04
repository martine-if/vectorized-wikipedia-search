import os
from data_loader import load_all_raw, DATA_DIR

class KeySearchWikiDataset:
    def __init__(self):
        data = load_all_raw()
        self.queries_natural = data["queries_natural"]
        self.queries_iri = data["queries_iri"]
        self.qrels = data["qrels"]
    
    def generate_query_file(self, output_path=None):
        if output_path is None:
            output_dir = os.path.join(DATA_DIR, "processed")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "keysearch.qry")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            sorted_qids = sorted(self.queries_natural.keys())
            
            for idx, qid in enumerate(sorted_qids, start=1):
                query_text = self.queries_natural[qid]
                f.write(f".I {idx:03d}\n")
                f.write(".W\n")
                f.write(f"{query_text}\n")
        return output_path
    
    def generate_query_id_mapping(self, output_path=None):
        # original ID to sequential mapping
        if output_path is None:
            output_dir = os.path.join(DATA_DIR, "processed")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "query_id_mapping.txt")
        
        sorted_qids = sorted(self.queries_natural.keys())
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Sequential_Number Original_ID Query_Text\n")
            for idx, qid in enumerate(sorted_qids, start=1):
                query_text = self.queries_natural[qid].replace('\n', ' ')
                f.write(f"{idx:03d} {qid} {query_text}\n")
        return output_path


if __name__ == "__main__":
    dataset = KeySearchWikiDataset()
    query_file = dataset.generate_query_file()
    mapping_file = dataset.generate_query_id_mapping()