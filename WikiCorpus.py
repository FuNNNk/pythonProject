"""
Creates a corpus from Wikipedia dump file.
Inspired by:
https://github.com/panyang/Wikipedia_Word2vec/blob/master/v1/process_wiki.py
"""

import sys
from gensim.corpora import WikiCorpus
import warnings

warnings.filterwarnings(
    'ignore',
    'detected Windows; aliasing chunkize to chunkize_serial',
)


def make_corpus(in_f, out_f):
    """Convert Wikipedia xml dump file to text corpus"""
    print(1)
    wiki = WikiCorpus(in_f, processes = 1)
    print(2)
    i = 0
    with open(out_f, 'w', encoding='utf8') as output:
        print(3)
        for text in wiki.get_texts():
            output.write(bytes(' '.join(text), 'utf-8').decode('utf-8') + '\n')
            i = i + 1
            if i % 10000 == 0:
                print('Processed ' + str(i) + ' articles')
    print('Processing complete!')


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print('Usage: python make_wiki_corpus.py <wikipedia_dump_file> <processed_text_file>')
        sys.exit(1)
    in_f = sys.argv[1]
    out_f = sys.argv[2]
    make_corpus(in_f, out_f)
