#####################################################
# w2v tutorial - re-used from Kaggle
#####################################################

# Re-used from Kaggle.com
# Full tutorial can be found here: https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial


#####################################################
# set upo libraries and upload data
#####################################################

import re  # For preprocessing
import pandas as pd  # For data handling
from time import time  # To time our operations
from collections import defaultdict  # For word frequency

import spacy  # For preprocessing

from gensim.models.phrases import Phrases, Phraser # for bigrams
import multiprocessing
from gensim.models import Word2Vec

import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

my_path = '/Users/Petra_Kummerova/Desktop/Python/NLP/Czechitas/'
cd '/Users/Petra_Kummerova/Desktop/Python/NLP/Czechitas'


df = pd.read_csv('simpsons_dataset.csv')
df.shape
df.head()


#####################################################
# data pre-processing
#####################################################

# remove null values 
# df.isnull().sum()
df = df.dropna().reset_index(drop=True)
#df.isnull().sum()

# remove stopwords
# python3 -m spacy download en_core_web_sm
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser']) # disabling Named Entity Recognition for speed

def cleaning(doc):
    # Lemmatizes and removes stopwords
    # doc needs to be a spacy Doc object
    txt = [token.lemma_ for token in doc if not token.is_stop]
    # Word2Vec uses context words to learn the vector representation of a target word,
    # if a sentence is only one or two words long,
    # the benefit for the training is very small
    if len(txt) > 2:
        return ' '.join(txt)

# Removes non-alphabetic characters:
brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in df['spoken_words'])

t = time()
txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000)]
print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))
# Time to clean up everything: 1.33 mins

df_clean = pd.DataFrame({'clean': txt})
df_clean = df_clean.dropna().drop_duplicates()
df_clean.shape


#####################################################
# train w2v
#####################################################

# As Phrases() takes a list of list of words as input:
sent = [row.split() for row in df_clean['clean']]
phrases = Phrases(sent, min_count=30, progress_per=10000)
# Transform the corpus based on the bigrams detected:
bigram = Phraser(phrases)
sentences = bigram[sent]


cores = multiprocessing.cpu_count() # Count the number of cores in a computer

# define w2v model
w2v_model = Word2Vec(min_count=20,
                     window=2,
                     size=300,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=cores-1)

# build the vocabulary
t = time()
w2v_model.build_vocab(sentences, progress_per=10000)
print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
# Time to build vocab: 0.03 mins

# train the model
t = time()
w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

# As we do not plan to train the model any further, we are calling init_sims(), which will make the model much more memory-efficient:
w2v_model.init_sims(replace=True)
# Time to train the model: 1.07 mins

#####################################################
# explore trained w2v
####################################################

w2v_model.wv.most_similar(positive=["homer"]) #check out also marge, bart...
w2v_model.wv.most_similar(positive=["homer_simpson"]) # n-gram / bigram
w2v_model.wv.similarity('maggie', 'baby') # similarity between two terms
w2v_model.wv.doesnt_match(['homer', 'patty', 'selma']) # which one is the furthest away from others
w2v_model.wv.most_similar(positive=["woman", "bart"], negative=["man"], topn=3) # analogy difference


w2v_model.save("word2vec.model")
w2v_model = Word2Vec.load("word2vec.model")





