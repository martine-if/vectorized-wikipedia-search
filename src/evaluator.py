from nltk import precision

from data_loader import load_documents
from result_rewriter import parse_documents

# Map queryID from JSON dataset to query number in keysearch.qry (e.g. MH10 -> 001)
def load_query_id_mapping():
    mapping = {}
    with open('../data/processed/query_id_mapping.txt', 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            if i == 0:
                continue
            split = line.split(' ')
            query_num = int(split[0])
            original_id = split[1]
            mapping[original_id] = query_num

    return mapping

def write_expected_output_to_file(expected, articles):
    queries = parse_documents('../data/processed/keysearch.qry', has_title=False)

    with open('../data/results/expected_output.txt', 'w', encoding='utf-8') as file:
        expected_sorted = dict(sorted(expected.items()))
        for query_num, article_ids in expected_sorted.items():
            query_text = queries[int(query_num)]
            file.write(f'{query_num}. {query_text}\n  ')
            for i, article_id in enumerate(article_ids):
                article_title = articles[article_id]
                if i == len(article_ids) - 1:
                    file.write(article_title)
                else:
                    file.write(f'{article_title}, ')
            file.write('\n')

def main():
    docs = load_documents()

    query_id_map = load_query_id_mapping()
    articles = parse_documents('../data/processed/all_articles.txt')
    corpus_size = 100000
    articles = {k: v for i, (k, v) in enumerate(articles.items()) if i < corpus_size}

    reversed_articles = {v: k for k, v in articles.items()}

    expected = {}
    for doc in docs:
        next_query_id = doc['queryID']
        query_num = query_id_map[next_query_id]

        relevant = doc['relevantEntities']

        article_ids = []
        for entity in relevant:
            label = entity['label']
            if len(label) >= 2 and label[0] == 'Q' and label[1].isdigit():
                continue

            article_id = reversed_articles.get(label)
            if article_id is not None:
                article_ids.append(int(article_id))

        expected[query_num] = article_ids

    write_expected_output_to_file(expected, articles)

    total_precision = 0
    total_recall = 0
    total_queries = 0

    with open('../data/results/ranking_output_rust.txt', 'r', encoding='utf-8') as file:
        with open('../data/results/queries_scored.txt', 'w', encoding='utf-8') as out:
            curr_query_id = None
            curr_query_articles = []

            for line in file:
                split = line.split(' ')

                next_query_id = split[0]
                if curr_query_id is None:
                    curr_query_id = next_query_id
                # We moved to the next query
                if next_query_id != curr_query_id:
                    expected_articles = expected[int(curr_query_id)]

                    # Skip if there are no results expected for this query
                    if len(expected_articles) == 0:
                        curr_query_id = next_query_id
                        curr_query_articles = []
                        continue

                    num_relevant = 0
                    for generated_article in curr_query_articles:
                        if generated_article in expected_articles:
                            num_relevant += 1

                    precision_at_10 = num_relevant / 10

                    recall_at_10 = num_relevant / len(expected_articles)

                    if precision_at_10 == 0 and recall_at_10 == 0:
                        f1 = 0
                    else:
                        f1 = 2 * (precision_at_10 * recall_at_10) / (precision_at_10 + recall_at_10)

                    out.write(f'{curr_query_id} precision={precision_at_10} recall={recall_at_10} f1={f1}\n')

                    total_precision += precision_at_10
                    total_recall += recall_at_10
                    total_queries += 1

                    curr_query_id = next_query_id
                    curr_query_articles = []

                article_id = split[1]
                curr_query_articles.append(int(article_id))

    print(f'Average precision: {total_precision / total_queries}, Average recall: {total_recall / total_queries}')


if __name__ == '__main__':
    main()