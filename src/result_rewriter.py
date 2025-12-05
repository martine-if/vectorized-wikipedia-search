def parse_documents(file_name, has_title=True):
    documents = {}
    with open(file_name, encoding='utf-8') as file:
        is_text = False
        curr_text = None
        curr_id = None
        for line in file:
            if has_title and line == '.W\n':
                if curr_text:
                    documents[curr_id] = curr_text
                is_text = False
            elif line.startswith('.I '):
                if not has_title and curr_text:
                    documents[curr_id] = curr_text
                    is_text = False
                curr_id = int(line.split()[1])
            elif has_title and line == '.T\n':
                is_text = True
            elif not has_title and line == '.W\n':
                is_text = True
            elif is_text:
                curr_text = line.strip()
        if curr_text:
            documents[curr_id] = curr_text
    return documents

def main():
    queries = parse_documents('../data/processed/keysearch.qry', has_title=False)
    articles = parse_documents('../data/processed/articles-1.txt')

    with open('../data/results/ranking_output.txt') as file:
        with open('../data/results/ranking_output_titles.txt', 'w', encoding='utf-8') as output:
            for line in file:
                split = line.split(' ')
                query_id = split[0]
                article_id = split[1]

                output.write('"' + queries[int(query_id)] + '" ')
                output.write('"' + articles[int(article_id)] + '" ')
                output.write(' '.join(split[2:]))

if __name__ == '__main__':
    main()
