import math
import numpy as np
from numpy.linalg import norm
from nltk.stem import PorterStemmer
import os
from data_loader import DATA_DIR
from tqdm import tqdm

stop_list = ['a','the','an','and','or','but','about','above','after','along','amid','among',\
                           'as','at','by','for','from','in','into','like','minus','near','of','off','on',\
                           'onto','out','over','past','per','plus','since','till','to','under','until','up',\
                           'via','vs','with','that','can','cannot','could','may','might','must',\
                           'need','ought','shall','should','will','would','have','had','has','having','be',\
                           'is','am','are','was','were','being','been','get','gets','got','gotten',\
                           'getting','seem','seeming','seems','seemed',\
                           'enough', 'both', 'all', 'your' 'those', 'this', 'these', \
                           'their', 'the', 'that', 'some', 'our', 'no', 'neither', 'my',\
                           'its', 'his' 'her', 'every', 'either', 'each', 'any', 'another',\
                           'an', 'a', 'just', 'mere', 'such', 'merely' 'right', 'no', 'not',\
                           'only', 'sheer', 'even', 'especially', 'namely', 'as', 'more',\
                           'most', 'less' 'least', 'so', 'enough', 'too', 'pretty', 'quite',\
                           'rather', 'somewhat', 'sufficiently' 'same', 'different', 'such',\
                           'when', 'why', 'where', 'how', 'what', 'who', 'whom', 'which',\
                           'whether', 'why', 'whose', 'if', 'anybody', 'anyone', 'anyplace', \
                           'anything', 'anytime' 'anywhere', 'everybody', 'everyday',\
                           'everyone', 'everyplace', 'everything' 'everywhere', 'whatever',\
                           'whenever', 'whereever', 'whichever', 'whoever', 'whomever' 'he',\
                           'him', 'his', 'her', 'she', 'it', 'they', 'them', 'its', 'their','theirs',\
                           'you','your','yours','me','my','mine','I','we','us','much','and/or'
                           ]

punctuation = ['.', ',', ':', '(', ')', '/', '\'', '=', '?', '!', ';', '"', '&']

def parse_documents(file_name):
    documents = []
    doc_ids = []
    with open(file_name, encoding='utf-8') as file:
        is_text = False
        curr_doc = []
        curr_id = None
        for line in file:
            if line.startswith('.I '):
                if curr_doc:
                    documents.append(curr_doc)
                    doc_ids.append(curr_id)
                curr_id = line.split()[1]
                curr_doc = []
                is_text = False
            elif line == '.W\n':
                is_text = True
            elif is_text:
                [curr_doc.append(word.strip()) for word in line.split(' ')]
        if curr_doc:
            documents.append(curr_doc)
            doc_ids.append(curr_id)
    return documents, doc_ids

def filter_words(document_set: list[list[str]]):
    filtered_docs = []
    ps = PorterStemmer()
    for doc in tqdm(document_set, desc="Filtering words"):
        filtered_doc = []
        for word in doc:
            if word in punctuation:
                continue
            # Skip anything with a digit
            if any(c.isdigit() for c in word):
                continue
            # Remove punctuation from larger word
            for punc in punctuation:
                word = word.replace(punc, '')

            if word == '':
                continue

            if word in stop_list:
                continue

            word = word.replace('--', '-')

            if '-' in word:
                for part in word.split('-'):
                    if part.strip() != '':
                        filtered_doc.append(ps.stem(part.lower()))
            else:
                filtered_doc.append(ps.stem(word.lower()))
        filtered_docs.append(filtered_doc)
    return filtered_docs

def get_idf_scores_dict(documents: list[list[str]]):
    num_docs = len(documents)
    docs_as_sets = []
    for doc in tqdm(documents, desc="Creating document sets"):
        docs_as_sets.append(set(doc))
    idf_docs = []
    for doc in tqdm(documents, desc="Calculating IDF scores"):
        doc_idf: dict[str, float] = {}
        for t in doc:
            num_docs_containing_t = 0
            for checked_doc in docs_as_sets:
                if t in checked_doc:
                    num_docs_containing_t += 1
            idf = math.log(num_docs / num_docs_containing_t)
            doc_idf[t] = idf

        idf_docs.append(doc_idf)
    return idf_docs

def main():
    queries, query_ids = parse_documents(os.path.join(DATA_DIR, "processed/keysearch.qry"))
    queries = filter_words(queries)
    
    query_idf = get_idf_scores_dict(queries)
    query_tf = []
    for query in queries:
        tf = {}
        for word in query:
            instances = query.count(word)
            tf[word] = instances / len(query) if len(query) > 0 else 0
        query_tf.append(tf)
    
    documents, doc_ids = parse_documents(os.path.join(DATA_DIR, "processed/articles-1.txt"))
    
    # TESTING
    documents = documents[:1000]
    doc_ids = doc_ids[:1000]
    print(f"Using subset of {len(documents)} documents for testing")
    
    documents = filter_words(documents)

    doc_idf = get_idf_scores_dict(documents)
    doc_tf = []
    for document in documents:
        tf = {}
        for word in document:
            instances = document.count(word)
            tf[word] = instances / len(document) if len(document) > 0 else 0
        doc_tf.append(tf)

    output_lines = []

    for qid, query in enumerate(tqdm(queries, desc="Processing queries")):
        sims = []
        for doc_idx, document in enumerate(documents):
            document_vec: list[float] = []
            for word in query:
                if word in document:
                    doc_word_tf = doc_tf[doc_idx][word]
                    doc_word_idf = doc_idf[doc_idx][word]
                    doc_tf_idf: float = doc_word_tf * doc_word_idf
                    document_vec.append(doc_tf_idf)
                else:
                    document_vec.append(0)
            
            query_vec = []
            for word in query:
                query_word_tf = query_tf[qid][word]
                query_word_idf = query_idf[qid][word]
                query_tf_idf = query_word_tf * query_word_idf
                query_vec.append(query_tf_idf)

            a = np.array(query_vec)
            b = np.array(document_vec)

            norm_a = norm(a)
            norm_b = norm(b)
            cos_similarity = 0
            if norm_a != 0 and norm_b != 0:
                cos_similarity = np.dot(a, b) / (norm_a * norm_b)

            if math.isnan(cos_similarity):
                cos_similarity = 0

            sims.append((doc_ids[doc_idx], float(cos_similarity)))

        sims.sort(key=lambda x: x[1], reverse=True)

        output_query_id = query_ids[qid]
        for rank, entry in enumerate(sims):
            doc_id = entry[0]
            sim_score = entry[1]

            if rank + 1 > 10:
                break

            output_lines.append(f'{output_query_id} Q0 {doc_id} {rank + 1} {sim_score}\n')

    output_path = os.path.join(DATA_DIR, "results/ranking_output.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as out:
        out.writelines(output_lines)

if __name__ == '__main__':
    main()