from nltk import precision

from src.data_loader import load_documents
from src.result_rewriter import parse_documents

# Map queryID from JSON dataset to query number in keysearch.qry (e.g. MH10 -> 001)
def load_query_id_mapping():
    mapping = {}
    with open('../data/processed/query_id_mapping.txt') as file:
        for i, line in enumerate(file):
            if i == 0:
                continue
            split = line.split(' ')
            query_num = int(split[0])
            query_id = split[1]
            mapping[query_id] = query_num

    return mapping

def main():
    docs = load_documents()

    query_id_map = load_query_id_mapping()
    articles = parse_documents('../data/processed/all_articles.txt')
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

    total_precision = 0
    total_recall = 0
    total_queries = 0

    with open('../data/results/ranking_output_rust.txt') as file:
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

                    num_relevant = 0
                    for generated_article in curr_query_articles:
                        if generated_article in expected_articles:
                            num_relevant += 1

                    precision_at_10 = num_relevant / 10

                    num_expected = len(expected_articles)
                    if num_expected > 0:
                        recall_at_10 = num_relevant / num_expected
                    else:
                        recall_at_10 = 0

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