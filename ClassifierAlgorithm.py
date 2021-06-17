# eu as avea 2 clase, abuz si non abuz
# x = text
# y = fisierul 0 si 1
import numpy
import pandas as pd
import regex as re
import scispacy
import spacy
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
## Vader Sentiment Score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

listaVS = []
listaBF = []
listaUniqueDrugs = set()
analyzer = SentimentIntensityAnalyzer()

import xlrd
import xlwt
# loc = ("TrainTestData\AllData.xls")
# workbook = xlwt.Workbook()
# sheetNew = workbook.add_sheet('test')
# wb = xlrd.open_workbook(loc)
# sheet = wb.sheet_by_index(0)
# sheet.cell_value(0, 0)
# for i in range(sheet.nrows):
#     vs = analyzer.polarity_scores(sheet.cell_value(i, 0))
#     sheetNew.write(i, 0, vs["compound"])
# workbook.save("TrainTestData\AllDataVader.xls")

## Binary flag features including presence of commonly abused drug terms, (see 2.3) and presence of emojis commonly
## associated with drug abuse

# loc = ("TrainTestData\AllData.xls")
# workbook = xlwt.Workbook()
# sheetNew = workbook.add_sheet('test')
# wb = xlrd.open_workbook(loc)
# sheet = wb.sheet_by_index(0)
# sheet.cell_value(0, 0)
# for i in range(sheet.nrows):
#     with open("Drugs", 'r', encoding='utf8') as DrugFile:
#         ok = 0
#         for line2 in DrugFile:
#             if re.search(r"\b" + line2.replace('\n', '').lower() + r"\b", sheet.cell_value(i, 0).lower()):
#                 ok = 1
#                 break
#         sheetNew.write(i, 0, ok)
# workbook.save("TrainTestData\AllDataBinary.xls")

## Count-based features such as number of hashtags, username mentions, emojis/emoticons and number of drug terms
## identified
#
# loc = ("TrainTestData\AllData.xls")
# workbook = xlwt.Workbook()
# sheetNew = workbook.add_sheet('test')
# wb = xlrd.open_workbook(loc)
# sheet = wb.sheet_by_index(0)
# sheet.cell_value(0, 0)
# for i in range(sheet.nrows):
#     drugCount = 0
#     countHashtags = sheet.cell_value(i, 0).count("<hashtag>")
#     countUsernameMentions = sheet.cell_value(i, 0).count("<user>")
#     emoticons = re.finditer(r'[\U0001f600-\U0001f650]', sheet.cell_value(i, 0))
#     count = sum(1 for _ in emoticons)
#     countEmojis = sheet.cell_value(i, 0).count("<annoyed>") + sheet.cell_value(i, 0).count("<sad>") + sheet.cell_value(i, 0).count("<happy>")
#     with open("Drugs", 'r', encoding='utf8') as DrugFile:
#         for line2 in DrugFile:
#             if re.search(r"\b" + line2.replace('\n', '').lower() + r"\b", sheet.cell_value(i, 0).lower()):
#                 drugCount = drugCount + 1
#     counter = countEmojis + count + countHashtags + countUsernameMentions+ drugCount
#     sheetNew.write(i, 0, counter)
# workbook.save("TrainTestData\AllDataCounter.xls")

##Co-occurrence of chemical and disease entities, where entity extraction was performed using ennerbc5cdrmd

# loc = ("TrainTestData\AllData.xls")
# workbook = xlwt.Workbook()
# sheetNew = workbook.add_sheet('test')
# wb = xlrd.open_workbook(loc)
# sheet = wb.sheet_by_index(0)
# sheet.cell_value(0, 0)
# nlp = spacy.load("en_ner_bc5cdr_md")
# for i in range(sheet.nrows):
#     counterEntity = 0
#     doc = nlp(sheet.cell_value(i, 0))
#     for entity in doc.ents:
#         counterEntity = counterEntity + 1
#     sheetNew.write(i, 0, counterEntity)
# workbook.save("TrainTestData\AllDataEntity.xls")


## tf-idf of tokens (unigrams and bigrams)

# loc = ("TrainTestData\AllData.xls")
# workbook = xlwt.Workbook()
# sheetNew = workbook.add_sheet('test')
# wb = xlrd.open_workbook(loc)
# sheet = wb.sheet_by_index(0)
# sheet.cell_value(0, 0)
# with open("WordRepresentationData/output_tweets_original.txt", encoding='utf8') as f:
#     vectorizer = TfidfVectorizer()
#     vectors = vectorizer.fit_transform(f)
#     feature_names = vectorizer.get_feature_names()
#     dense = vectors.todense()
#     denselist = dense.tolist()
#     df = pd.DataFrame(denselist, columns=feature_names)
    # for i in range(len(denselist)):
    #     for j in range(len(denselist[i])):
    #         sheetNew.write(i, j, denselist[i][j])
# workbook.save("TrainTestData\AllDataTfIdf.xls")


# vectorizer = TfidfVectorizer()
# vectors = vectorizer.fit_transform(f)
# feature_names = vectorizer.get_feature_names()
# dense = vectors.todense()
# denselist = dense.tolist()
# df = pd.DataFrame(denselist, columns=feature_names)
# print(df['seroquel'])

from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#
# X = []
# with open("WordRepresentationData\output_tweets_original.txt", 'r', encoding='utf8') as f:
#     for line in f:
#         X.append(line)
# Y = [0, 1]
#
import fasttext

loc = ("TrainTestData\AllData.xls")
wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(0)
listaOffset = []
listaClasa = []
data = fasttext.load_model("skipgram_model.bin")
for i in range(sheet.nrows):
    listafasttext = []
    listafasttext = np.append(listafasttext, [data.get_sentence_vector(sheet.cell(i,0).value)])
    lista1 = numpy.append(listafasttext, [sheet.cell(i, 2).value, sheet.cell(i, 3).value, sheet.cell(i, 4).value,
                                           sheet.cell(i, 5).value])
    listaOffset = numpy.append(listaOffset, [lista1])
    listaClasa.append([sheet.cell(i, 1).value])
    print(i)
listaOffset = listaOffset.reshape(5610, 304)
    # listadenselist = []
    # for j in range(len(denselist[i])):
    #     listadenselist= np.append(listadenselist, [denselist[i][j]])
    # lista1= numpy.append(listadenselist,[sheet.cell(i,2).value,sheet.cell(i,3).value, sheet.cell(i,4).value, sheet.cell(i,5).value])
    # listaOffset=numpy.append(listaOffset, [lista1])
    # listaClasa.append([sheet.cell(i, 1).value])
    # print(i)
# listaOffset= listaOffset.reshape(5610,6727)

from sklearn.ensemble import RandomForestClassifier

lsvc = LinearSVC(verbose=0)
clf = RandomForestClassifier(max_depth=2, random_state = 0)

LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=10000,
          multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
          verbose=0)

#
# # xtrain si xtest sunt textul meu
# # ytrain si ytest sunt offsetul
#
# # n features = how many columns are / pun toate featurile de mai sus intr-un csv
xtrain, xtest, ytrain, ytest = train_test_split(listaOffset,np.ravel(listaClasa),test_size=0.2)
# print("Xtrain is ")
# print(xtrain)
# print("Xtest is ")
# print(xtest)
# print("Ytrain is ")
# print(ytrain)
# print("Ytrain is ")
# print(ytest)

lsvc.fit(xtrain, ytrain)

score = lsvc.score(xtrain, ytrain)
print("Training score: ", score)

pred = lsvc.predict(xtest)
score_test = metrics.accuracy_score(ytest, pred)
print("Test score: ", score_test)

clf.fit(xtrain, ytrain)

score_clf = clf.score(xtrain, ytrain)
print("Training score RFC: ", score_clf)

pred_clf = clf.predict(xtest)
score_test_clf = metrics.accuracy_score(ytest, pred_clf)
print("Test score RFC: ", score_test_clf)

#use case
# def use_case():
#     date_intrare = input("Introdu fraza: ")
#     # abuz = input("Este abuz: ")
#
#     vs = analyzer.polarity_scores(date_intrare)
#
#     with open("Drugs", 'r', encoding='utf8') as DrugFile:
#         ok = 0
#         for line2 in DrugFile:
#             if re.search(r"\b" + line2.replace('\n', '').lower() + r"\b", sheet.cell_value(i, 0).lower()):
#                 ok = 1
#                 break
#
#     counterEntity = 0
#     doc = nlp(date_intrare)
#     for entity in doc.ents:
#         counterEntity = counterEntity + 1

# Score:  0.8602352941176471

# Score:  0.6546345811051694
# Test score:  0.6331255565449688

###LinearSVC with TF-IDF
# Training score:  0.9621212121212122
# Test score:  0.8431372549019608

###LinearSVC with fasttext
# Training score:  0.8469251336898396
# Test score:  0.8315508021390374


# ar trebui sa fac pasii pe care ii descrie aici?
# https://towardsdatascience.com/sentiment-analysis-on-swachh-bharat-using-twitter-216369cfa534
