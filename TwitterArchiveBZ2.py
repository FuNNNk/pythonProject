import sys
import bz2
import re
import time

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from timer.timer import Timer

text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
               'time', 'url', 'date', 'number'],
    # terms that will be annotated
    annotate={"hashtag", "allcaps", "elongated", "repeated",
              'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens

    # corpus from which the word statistics are going to be used
    # for word segmentation
    segmenter="twitter",

    # corpus from which the word statistics are going to be used
    # for spell correction
    corrector="twitter",

    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words

    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,

    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)


import timer

def twitter_archive(out_f):
    with open(out_f, 'wt') as output:
        false = False
        null = None
        true = True
        for i in range(1,24):
            for j in range(0,60):
                b={}
                with bz2.open("C:/Users/Alex/Desktop/Data/Twitter_Archive/04/{:02d}/{:02d}.json.bz2".format(i,j), 'rt') as file:
                    for line in file:
                        try:
                            b = eval(line)
                            text = b["text"]
                            output.write(" ".join(text_processor.pre_process_doc(text)) + "\n")
                        except:
                            continue
            print("{:02d}".format(i))



if __name__ == '__main__':
    t=time.perf_counter()
    if len(sys.argv) != 2:
        print('Usage: python twitter_archive.py <twitter_archive_file> <processed_text_file>')
        sys.exit(1)
    out_f = sys.argv[1]
    twitter_archive(out_f)
    toc = time.perf_counter()
    print(toc - t)
