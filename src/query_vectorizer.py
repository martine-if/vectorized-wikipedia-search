import math
import numpy as np
from numpy.linalg import norm
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import os
from data_loader import DATA_DIR

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

def get_idf_scores(documents: list[list[str]]):
    num_docs = len(documents)
    docs_as_sets = []
    for doc in documents:
        docs_as_sets.append(set(doc))
    idf_docs = []
    for doc in documents:
        doc_idf: list[float] = []
        for t in doc:
            num_docs_containing_t = 0
            for checked_doc in docs_as_sets:
                if t in checked_doc:
                    num_docs_containing_t += 1
            idf = math.log(num_docs / num_docs_containing_t)
            doc_idf.append(idf)

        idf_docs.append(doc_idf)
    return idf_docs

def get_idf_scores_dict(documents: list[list[str]]):
    num_docs = len(documents)
    docs_as_sets = []
    for doc in documents:
        docs_as_sets.append(set(doc))
    idf_docs = []
    for doc in documents:
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

def get_tf_idf_vectors(documents: list[list[str]], idf_scores: list[list[float]]):
    all_tf_idf = []
    for i, doc in enumerate(documents):
        doc_tf_idf = []
        for j, word in enumerate(doc):
            tf = doc.count(word)
            normalized_tf = tf / len(doc)
            idf = idf_scores[i][j]
            tf_idf = normalized_tf * idf
            doc_tf_idf.append(tf_idf)
        all_tf_idf.append(doc_tf_idf)
    return all_tf_idf

def parse_documents(file_name):
    queries = []
    with open(file_name, encoding='utf-8') as file:
        is_text = False
        curr_query = []
        for line in file:
            if line == '.W\n':
                is_text = True
            elif is_text and line.startswith('.I '):
                queries.append(curr_query)
                curr_query = []
                is_text = False
            elif is_text:
                [curr_query.append(word.strip()) for word in line.split(' ')]
        queries.append(curr_query)
    return queries

def filter_words(document_set: list[list[str]]):
    filtered_docs = []
    ps = PorterStemmer()
    for doc in document_set:
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
                        filtered_doc.append(part)
            else:
                filtered_doc.append(word)
        filtered_docs.append(filtered_doc)
    return filtered_docs

def main():
    queries = parse_documents(os.path.join(DATA_DIR, "processed/keysearch.qry"))
    queries = filter_words(queries)
    idf = get_idf_scores(queries)
    tf_idf = get_tf_idf_vectors(queries, idf)
    
    output_dir = os.path.join(DATA_DIR, "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    vector_output = os.path.join(output_dir, "query_tf_idf_vectors.txt")
    with open(vector_output, 'w', encoding='utf-8') as f:
            for idx, (query, vector) in enumerate(zip(queries, tf_idf), start=1):
                f.write(f".I {idx:03d}\n")
                f.write(".W\n")
                f.write(f"{query}\n")
                f.write(f"{vector}\n")
    
    """"
    abstracts = parse_documents('cran.all.1400')
    abstracts = filter_words(abstracts)

    abs_idf = get_idf_scores_dict(abstracts)
    abs_tf = []
    for abstract in abstracts:
        tf = {}
        for word in abstract:
            instances = abstract.count(word)
            tf[word] = instances / len(abstract)
        abs_tf.append(tf)

    output_lines = []

    for qid, query in enumerate(queries):
        sims = []
        for abs_idx, abstract in enumerate(abstracts):
            abstract_vec: list[float] = []
            for word in query:
                if word in abstract:
                    abs_word_tf = abs_tf[abs_idx][word]
                    abs_word_idf = abs_idf[abs_idx][word]
                    abs_tf_idf: float = abs_word_tf * abs_word_idf
                    abstract_vec.append(abs_tf_idf)
                else:
                    abstract_vec.append(0)
            query_vec = tf_idf[qid]

            a = np.array(query_vec)
            b = np.array(abstract_vec)

            norm_a = norm(a)
            norm_b = norm(b)
            cos_similarity = 0
            if norm_a != 0 and norm_b != 0:
                cos_similarity = np.dot(a, b) / (norm_a * norm_b)

            if math.isnan(cos_similarity):
                cos_similarity = 0

            sims.append((abs_idx + 1, float(cos_similarity)))

        sims.sort(key=lambda x: x[1], reverse=True)

        output_query_id = qid + 1
        for rank, entry in enumerate(sims):
            abs_id = entry[0]
            sim_score = entry[1]

            if rank + 1 > 100:
                break

            output_lines.append(f'{output_query_id} {abs_id} {sim_score}\n')

    with open('output.txt', 'w') as out:
        out.writelines(output_lines)
    """

if __name__ == '__main__':
    main()