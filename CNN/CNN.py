# conv depth of 4, cosine loss function, drop out of 0.4, andbatch size of 64
# raw Task 4 dataset with the word embeddings from section 2.

with open('../WordRepresentationData/output_tweets_original.txt', 'r', encoding='utf8') as f:
    dataset = f.read()