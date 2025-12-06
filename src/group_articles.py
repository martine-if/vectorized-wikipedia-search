output_file = "../data/processed/all_articles.txt"

with open(output_file, "w", encoding="utf-8") as outfile:
    for i in range(1, 10):
        filename = f"../data/processed/articles-{i}.txt"
        try:
            with open(filename, "r", encoding="utf-8") as infile:
                outfile.write(infile.read())
        except FileNotFoundError:
            print(f"Warning: {filename} not found, skipping.")
