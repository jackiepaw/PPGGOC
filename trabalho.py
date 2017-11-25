#!/usr/bin/env python
# coding=UTF-8


import re
import os
import glob
import codecs
import string
import numpy as np
import pandas as pd
import gensim
from gensim import corpora

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import feature_extraction
import mpld3

import pyLDAvis
import pyLDAvis.gensim
from IPython.display import Image

#executar na primeira vez
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')


#Specifying the path to the files
datapath = "./arquivos/"
outputs = "./outputs/"


# Lendo os documentos txt
datafile = {}
for file in glob.glob("arquivos/*.txt"):
    f = codecs.open(file, 'r', 'utf-8-sig')
    datafile.update({file: f.read().replace('\n', ' ')})
    f.close()
data = datafile.values()

# We will use NLTK to tokenize.
# A document will now be a list of tokens.
print("Number of documents:",len(data))
gen_docs = [[w.lower() for w in word_tokenize(text)] for text in data]
print(gen_docs)

# We will create a dictionary from a list of documents.
# A dictionary maps every word to a number.
dictionary = gensim.corpora.Dictionary(gen_docs)
print("Number of words in dictionary:",len(dictionary))
for i in range(len(dictionary)):
    print(i, dictionary[i])

#
# Now we will create a corpus. A corpus is a list of bags of words.
# A bag-of-words representation for a document just lists the number of times each word occurs in the document.
#
corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
for d in corpus:
    print(d)

#
# Now we create a tf-idf model from the corpus.
#
tf_idf = gensim.models.TfidfModel(corpus)
print(tf_idf)
s = 0
for i in corpus:
    s += len(i)
print(s)


#
# Now we will create a similarity measure object in tf-idf space.
#
sims = gensim.similarities.Similarity(outputs,tf_idf[corpus],num_features=len(dictionary))
print(sims)
print(type(sims))


#
# Now create a query document and convert it to tf-idf.
#
query_doc = [w.lower() for w in word_tokenize("Agradeço à toda equipe.")]
print(query_doc)
query_doc_bow = dictionary.doc2bow(query_doc)
print(query_doc_bow)
query_doc_tf_idf = tf_idf[query_doc_bow]
print(query_doc_tf_idf)

# We show an array of document similarities to query.
# We see that the second document is the most similar with the overlapping of socks and force.
sims[query_doc_tf_idf]

# compile documents. data is all documents combining together to form a corpus.
doc_complete = data

#Cleaning is an important step before any text mining task, in this step, we will remove the punctuations, stopwords and normalize the corpus.
stop = set(stopwords.words('portuguese'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in doc_complete]

# Creating the term dictionary of our courpus, where every unique term is assigned an index.
dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50)

# Show the matrix
print(ldamodel.print_topics(num_topics=2, num_words=2))