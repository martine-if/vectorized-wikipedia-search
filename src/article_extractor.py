import xml.etree.ElementTree as ET
import mwparserfromhell
import textwrap

def save_pages_to_file(pages, fn):
    with open(fn, 'w', encoding='utf-8') as file:
        for page_num, (title, contents) in enumerate(pages.items(), start=1):
            file.write(f'.I {page_num}\n')
            file.write('.T\n')
            file.write(f'{title}\n')
            file.write('.W\n')
            file.write(f'{contents}\n')

def should_keep_page(title: str, text: str):
    if text.startswith('#REDIRECT'):
        return False
    return True

# Remove metadata, wiki formatting, and any subsections
def strip_page(text: str):
    truncated = []
    for line in text.splitlines():
        if line.startswith('[[File'):
            continue
        if line.startswith('==') and line.endswith('=='):
            break
        truncated.append(line)
    joined = '\n'.join(truncated)

    wikicode = mwparserfromhell.parse(joined)

    stripped = wikicode.strip_code()

    stripped = (stripped.replace('( ; ) ', '')
                .replace('( ) ', '')
                .replace('() ', '')
                .replace('\u00A0', ' ')
                .replace('. ', ' . ')
                .replace('  ', ' '))

    stripped = stripped.lower()

    wrapped_lines = []
    for line in stripped.splitlines():
        wrapped_lines.extend(textwrap.wrap(line, width=120, replace_whitespace=True) or [''])

    return '\n'.join(wrapped_lines)

def parse_pages(path):
    tree = ET.parse(path)

    root = tree.getroot()

    ns = {'mw': 'http://www.mediawiki.org/xml/export-0.11/'}

    pages = {}
    count = 0
    for page in root.findall('./mw:page', ns):

        title = page.find('mw:title', ns).text
        text = page.find('mw:revision/mw:text', ns).text

        if title is not None and text is not None:
            if should_keep_page(title, text):
                pages[title] = strip_page(text)
                count += 1

                if count % 1000 == 0:
                    print(f'Parsed {count} pages...')

    return pages

def main():
    batch_num = 3
    path = f'../data/raw-wiki/enwiki-latest-pages-articles-multistream{batch_num}.xml'

    pages = parse_pages(path)

    save_pages_to_file(pages, f'../data/processed/articles-{batch_num}.txt')


if __name__ == '__main__':
    main()