import random
import pickle
import os
from data_loader import load_all_raw, DATA_DIR

class KeySearchWikiDataset:
    def __init__(self, shuffle=True, seed=42):
        data = load_all_raw()
        self.queries_natural = data["queries_natural"]
        self.qrels = data["qrels"]
        self.query_labels = data["query_labels"]
        self.documents_list = data["documents"]
        
        # reproducibility
        if seed is not None:
            random.seed(seed)

        # build doc_id map from the list of documents
        self.doc_id_map = {}
        
        for doc in self.documents_list:
            relevant_entities = doc.get('relevantEntities', [])
            for entity in relevant_entities:
                iri = entity.get('iri')
                label = entity.get('label', iri)
                if iri:
                    self.doc_id_map[iri] = label

        # list of doc_ids for negative sampling
        self.all_doc_ids = list(self.doc_id_map.keys())

        # generate training pairs: 1 positive + 1 negative per query
        self.training_pairs = self.create_training_pairs()
        
        # shuffler
        if shuffle:
            random.shuffle(self.training_pairs)

    def create_training_pairs(self):
        pairs = []
        queries_without_matches = 0

        for idx, (qid, target_id_map) in enumerate(self.qrels.items()):
            if qid not in self.queries_natural:
                continue
            
            query_text = self.queries_natural[qid]

            # positive example: randomly select 1 doc_id from relevant docs
            pos_ids = [doc_id for doc_id in target_id_map if doc_id in self.doc_id_map]
            if not pos_ids:
                queries_without_matches += 1
                continue
            
            pos_id = random.choice(pos_ids)
            pairs.append((query_text, self.doc_id_map[pos_id], 1))

            # negative example: randomly sample until non-relevant doc
            target_set = set(target_id_map.keys())
            max_attempts = 100
            neg_id = None
            
            for _ in range(max_attempts):
                candidate = random.choice(self.all_doc_ids)
                if candidate not in target_set:
                    neg_id = candidate
                    break
            
            if neg_id is None:
                continue
            
            pairs.append((query_text, self.doc_id_map[neg_id], 0))

        return pairs

    def train_test_split(self, test_size=0.1, seed=None):
        if seed is not None:
            random.seed(seed)
    
        pairs = self.training_pairs.copy()
        random.shuffle(pairs)
        
        split_idx = int(len(pairs) * (1 - test_size))
        train_data = pairs[:split_idx]
        test_data = pairs[split_idx:]
        
        return train_data, test_data

    def save_dataset(self, filename='training_pairs.pkl', save_dir=None):
        if save_dir is None:
            save_dir = os.path.join(DATA_DIR, "processed")
        
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.training_pairs, f)
        return filepath

    @staticmethod
    def load_dataset(filename='training_pairs.pkl', save_dir=None):
        if save_dir is None:
            save_dir = os.path.join(DATA_DIR, "processed")
        
        filepath = os.path.join(save_dir, filename)
        
        with open(filepath, 'rb') as f:
            training_pairs = pickle.load(f)
        
        return training_pairs

    def __len__(self):
        return len(self.training_pairs)

    def __getitem__(self, idx):
        return self.training_pairs[idx]


if __name__ == "__main__":
    # create the dataset, split, and save
    dataset = KeySearchWikiDataset(shuffle=True, seed=42)
    train_data, test_data = dataset.train_test_split(test_size=0.2, seed=42)
    dataset.save_dataset('training_pairs.pkl')

    with open(os.path.join(DATA_DIR, "processed", "train_pairs.pkl"), 'wb') as f:
        pickle.dump(train_data, f)
    
    with open(os.path.join(DATA_DIR, "processed", "test_pairs.pkl"), 'wb') as f:
        pickle.dump(test_data, f)
