import xml.etree.ElementTree as ET
import mwparserfromhell
import textwrap
import re

from tqdm import tqdm

from data_loader import load_documents
offset = 726184
category_pattern = re.compile(r'\[\[Category:([^\]]+)\]\]')

def save_pages_to_file(pages, fn):
    with open(fn, 'w', encoding='utf-8') as file:
        for page_num, (title, contents) in enumerate(pages.items(), start=1):
            file.write(f'.I {page_num+offset}\n')
            file.write('.T\n')
            file.write(f'{title}\n')
            file.write('.W\n')
            file.write(f'{contents}\n')

def load_dataset_article_titles():
    titles = set()
    docs = load_documents()
    for doc in docs:
        entities = doc['relevantEntities']
        for entity in entities:
            label = entity['label']
            if label.startswith('Q'):
                if len(label) >= 2 and label[1].isdigit():
                    continue
            titles.add(label)
    return titles

def should_keep_page(title: str, text: str):
    if text.startswith('#REDIRECT'):
        return False
    return True

def remove_file_section(line: str) -> str:
    result = []
    i = 0
    n = len(line)

    while i < n:
        if line.startswith('[[File', i):
            i += 6  # skip '[[File'
            depth = 1

            while i < n and depth > 0:
                if line.startswith('[[', i):
                    depth += 1
                    i += 2
                elif line.startswith(']]', i):
                    depth -= 1
                    i += 2
                else:
                    i += 1

            continue

        result.append(line[i])
        i += 1

    return ''.join(result)

def extract_categories(text):
    categories = []
    for line in text.splitlines():
        if '[[Category:' in line:
            cats = category_pattern.findall(line)
            categories.extend(cats)
    return categories

def extract_lead(text):
    truncated = []
    for line in text.splitlines():
        if line.startswith('[[File'):
            line = remove_file_section(line)
            if not line:
                continue

        if line.startswith('==') and line.endswith('=='):
            break

        truncated.append(line)

    return "\n".join(truncated)

def strip_page(text: str):
    categories = extract_categories(text)
    joined = extract_lead(text)

    joined = (joined.replace('}}</onlyinclude>', '')
                    .replace('<onlyinclude>{{#ifeq:', ''))
    wikicode = mwparserfromhell.parse(joined)
    stripped = wikicode.strip_code()

    stripped = (stripped.replace('( ; ) ', '')
                        .replace('( ) ', '')
                        .replace('() ', '')
                        .replace('(, ) ', '')
                        .replace('\u00A0', ' ')
                        .replace('. ', ' . ')
                        .replace('.\n', ' .\n')
                        .replace('  ', ' '))

    if stripped.endswith('.'):
        stripped = stripped[:-1] + ' .'

    stripped = stripped.lower().strip()

    clean_cats = []
    for cat in categories:
        cat = cat.strip().lower()
        if "|" in cat:
            clean_cats.extend(x.strip() for x in cat.split("|"))
        else:
            clean_cats.append(cat)

    wrapped_lines = []
    for line in stripped.splitlines():
        wrapped_lines.extend(textwrap.wrap(line, width=120, replace_whitespace=True) or [''])

    wrapped_lines.extend(clean_cats)
    return '\n'.join(wrapped_lines)


def parse_pages(path, titles_to_filter: set[str], batch_num):
    tree = ET.parse(path)

    root = tree.getroot()

    ns = {'mw': 'http://www.mediawiki.org/xml/export-0.11/'}

    pages = {}
    count = 0
    for page in tqdm(root.findall('./mw:page', ns), desc=f'Processing pages for batch {batch_num}'):
        title = page.find('mw:title', ns).text
        text = page.find('mw:revision/mw:text', ns).text

        if title is None or text is None:
            continue
        if title not in titles_to_filter:
            continue
        if not should_keep_page(title, text):
            continue

        pages[title] = strip_page(text)
        count += 1

    return pages

def main():
    titles_to_filter = load_dataset_article_titles()

    all_pages = {}
    for batch_num in range(25, 28):
        path = f'../data/raw-wiki/enwiki-latest-pages-articles-multistream{batch_num}.xml'

        pages = parse_pages(path, titles_to_filter, batch_num)
        all_pages.update(pages)

    save_pages_to_file(all_pages, f'../data/processed/articles-9.txt')


if __name__ == '__main__':
    main()
