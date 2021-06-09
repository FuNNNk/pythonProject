# eu as avea 2 clase, abuz si non abuz
# x = text
# y = fisierul 0 si 1
import pandas as pd
import regex as re
import scispacy
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
## Vader Sentiment Score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

listaVS = []
listaBF = []
listaUniqueDrugs = set()
analyzer = SentimentIntensityAnalyzer()


with open("WordRepresentationData\output_tweets_original.txt", 'r', encoding='utf8') as f:
    # for line in f:
    #     vs = analyzer.polarity_scores(line)
    #     listaVS.append("{}".format(str(vs["compound"])))

    ## Binary flag features including presence of commonly abused drug terms, (see 2.3) and presence of emojis commonly
    ## associated with drug abuse

    # for line in f:
    #     with open("Drugs", 'r', encoding='utf8') as DrugFile:
    #         ok = 0
    #         for line2 in DrugFile:
    #             if re.search(r"\b" + line2.replace('\n', '').lower() + r"\b", line.lower()):
    #                 listaUniqueDrugs.add(line2.replace('\n', '').lower())
    #                 ok = 1
    #                 break
    #         listaBF.append(ok)

    ## Count-based features such as number of hashtags, username mentions, emojis/emoticons and number of drug terms
    ## identified
    data = f.read()
    # countHashtags = data.count("<hashtag>")
    # countUsernameMentions = data.count("<user>")
    # emoticons = re.finditer(r'[\U0001f600-\U0001f650]', data)
    # count = sum(1 for _ in emoticons)
    # countEmojis = data.count("<annoyed>") + data.count("<sad>") + data.count("<happy>")
    # print(count + countEmojis)
    # print(countHashtags)
    # print(countUsernameMentions)
    # print(len(listaUniqueDrugs))

    ##Co-occurrence of chemical and disease entities, where entity extraction was performed using ennerbc5cdrmd
    #error
    nlp = spacy.load("en_ner_bc5cdr_md")
    doc = nlp(data)
    for entity in doc.ents:
        print(entity.text + ' : ' + entity.label_)

    ## tf-idf of tokens (unigrams and bigrams)
    # vectorizer = TfidfVectorizer()
    # vectors = vectorizer.fit_transform(f)
    # feature_names = vectorizer.get_feature_names()
    # dense = vectors.todense()
    # denselist = dense.tolist()
    # df = pd.DataFrame(denselist, columns=feature_names)
    # print(df['seroquel'])

# from sklearn.svm import LinearSVC
#
#
# X = []
# with open("WordRepresentationData\output_tweets_original.txt", 'r', encoding='utf8') as f:
#     for line in f:
#         X.append(line)
# Y = [0, 1]
#
# lin_clf = LinearSVC()
# lin_clf.fit(X, Y)
#
# dec = lin_clf.decision_function([[1]])
# print(dec.shape[1])


# ar trebui sa fac pasii pe care ii descrie aici?
# https://towardsdatascience.com/sentiment-analysis-on-swachh-bharat-using-twitter-216369cfa534