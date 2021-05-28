import ast
import sys

import fasttext

# data = []
# with open(sys.argv[1], encoding='utf8') as file:
#     read_data = file.read()
#     dict = ast.literal_eval(read_data)
#     lista_propozitii = []
#     for i in dict.values():
#         new_data = i.replace("['", '')
#         new_data = new_data.replace("']", '')
#         new_data = new_data.replace("', '", ', ')
#         lista_propozitii.append(new_data[0:len(new_data) - 3])
#
# with open('f_test.txt','w', encoding='utf8') as f:
#   f.write('\n'.join(lista_propozitii))

# import re
# with open('drugsCom.txt','r', encoding='utf8') as f:
#    data_read = f.read()
#    data_read = data_read.lower()
#    regex = re.compile('[^a-zA-Z ]')
#
# with open('drugsCom.txt','w', encoding='utf8') as f:
#    f.write(regex.sub(' ', data_read))

# import re
# with open('C:/Users/Alex/Desktop/f2.txt','r', encoding='utf8') as f:
#    data_read = f.read()
#    regex = re.compile('[^a-zA-Z ]')
#
# with open('f3.txt','w', encoding='utf8') as f:
#    f.write(regex.sub('', data_read))

skipgram_model = fasttext.train_unsupervised('WordRepresentationData/data_merged.txt', model="skipgram", loss='ns', minn=3, maxn=6, dim=300, lr=0.025, epoch=500)
skipgram_model.save_model('skipgram_model.bin')

# model = fasttext.load_model("skipgram_model.bin")
# print(model.get_nearest_neighbors('cocaine'))

# from ekphrasis.classes.preprocessor import TextPreProcessor
# from ekphrasis.classes.tokenizer import SocialTokenizer
# from ekphrasis.dicts.emoticons import emoticons
#
# text_processor = TextPreProcessor(
#     # terms that will be normalized
#     normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
#                'time', 'url', 'date', 'number'],
#     # terms that will be annotated
#     annotate={"hashtag", "allcaps", "elongated", "repeated",
#               'emphasis', 'censored'},
#     fix_html=True,  # fix HTML tokens
#
#     # corpus from which the word statistics are going to be used
#     # for word segmentation
#     segmenter="english",
#
#     # corpus from which the word statistics are going to be used
#     # for spell correction
#     corrector="english",
#
#     unpack_hashtags=True,  # perform word segmentation on hashtags
#     unpack_contractions=True,  # Unpack contractions (can't -> can not)
#     spell_correct_elong=False,  # spell correction for elongated words
#
#     # select a tokenizer. You can use SocialTokenizer, or pass your own
#     # the tokenizer, should take as input a string and return a list of tokens
#     tokenizer=SocialTokenizer(lowercase=True).tokenize,
#
#     # list of dictionaries, for replacing tokens extracted from the text,
#     # with other expressions. You can pass more than one dictionaries.
#     dicts=[emoticons]
# )
#
# with open('CHQA_data.txt', 'w', encoding='utf8') as output:
#     with open("C:/Users/Alex/Desktop/Data/merged_CHQA.txt", 'r', encoding='utf8') as file:
#         for line in file:
#             output.write(" ".join(text_processor.pre_process_doc(line)) + "\n")
