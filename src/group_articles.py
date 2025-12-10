output_file = "../data/processed/all_articles.txt"

def main():
    with open(output_file, "w", encoding="utf-8") as outfile:
        for i in range(1, 11):
            filename = f"../data/processed/articles_{i}.txt"
            try:
                with open(filename, "r", encoding="utf-8") as infile:
                    outfile.write(infile.read())
            except FileNotFoundError:
                print(f"Warning: {filename} not found, skipping.")

if __name__ == '__main__':
    main()
