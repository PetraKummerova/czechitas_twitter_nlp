#!/usr/bin/env python
# coding: utf-8

# ---------------------------------------------------------------------
# # twitter app data - download all hashtag data
# ---------------------------------------------------------------------
# 
# ### There are limitations in using Tweepy for scraping tweets. 
# ### The standard API only allows you to retrieve tweets up to 7 days ago 
# ### and is limited to scraping 18,000 tweets per a 15 minute window.
# 


import tweepy as tw
import pandas as pd 
import re
import pickle
from tqdm import tqdm

# my_path = '/Users/Petra_Kummerova/Desktop/Python/NLP/Czechitas/'
# my_out_USE = "my_out_USE.csv"
# my_out_USE_all_columns = "my_out_USE_all_columns.csv"
# cd '/Users/Petra_Kummerova/Desktop/Python/NLP/Czechitas'

# Twitter API deatils
consumer_key= 'XXX'
consumer_secret= 'XXX'
access_token= 'XXX'
access_token_secret= 'XXX'

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

# clean tweets list for special characters
def clean_tweets(tweets_list_all):
    pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    messages = [pattern.sub('', s) for s in tweets_list_all]
    return messages

#download tweet data
def tweets_dwnld(new_search,date_since):
    tweets = tw.Cursor(api.search,
                q=new_search,
                tweet_mode='extended',  # to get the full tweet text, instead of just shortened version
                # geocode=geocodes,
                lang="en",
                since=date_since).items(1000)
    tweets_list = []
    tweets_authors = []
    tweets_geo = []
    tweets_date = []
    tweets_id = []
    tweets_url = []
    

    for tweet in tweets:
            # tweets_list = tweets_list + [tweet.text]
            tweets_list = tweets_list + [tweet.full_text]
            tweets_authors = tweets_authors + [tweet.user.name]
            tweets_geo = tweets_geo + [tweet.place]
            tweets_date = tweets_date + [tweet.created_at]
            tweets_id = tweets_id + [tweet.id]
            tweets_url = tweets_url + ['https://twitter.com/twitter/statuses/' + str(tweet.id)]
    # clean tweets for special characters
    messages = clean_tweets(tweets_list)
    # messages = tweets_list
    df_tweets = pd.DataFrame({'topic': new_search, 'tweet': messages, 'author': tweets_authors, 'geo': tweets_geo, 'twdate': tweets_date, 'tw_id': tweets_id, 'tw_url': tweets_url})
    return df_tweets

def search_tweets(search_word_list, date_since):

    df_tweets_all = []
    for search_word in tqdm(search_word_list):  
        print(f"Currently searching for {search_word}",end='\r')
        new_search = search_word + " -filter:retweets"
        df_tweets = tweets_dwnld(new_search,date_since)
        df_tweets_all.append(df_tweets)
      #  print("end of one loop")
    return df_tweets_all




# Define the search term and the date_since date as variables
search_word_list = ["#Keboola","#python","#Tableau","#Snowflake","#R"]
geocodes = "50.0755, 14.4378°, 500km"
# Prague 50.0755° N, 14.4378° E
# search_word = "#Keboola" 
# new_search = search_word + " -filter:retweets"


date_since = "2021-08-20" # only goes back max around 7 days anyways..


df_tweets_all = search_tweets(search_word_list, date_since = date_since) # takes approx. 7 mins, depending on ionternet connection and speed


# #### Save into a pickle file.





                                                                                                                                                                                                                   
pickle.dump( df_tweets_all, open( "df_tweets_all.p", "wb" ) )


# #### Load the dictionary back from the pickle file.




df_tweets_all_load = pickle.load( open( "df_tweets_all.p", "rb" ) )


# #### Check one-by-one tweet categories (#Keboola","#python","#Tableau","#Snowflake","#R)



print(df_tweets_all_load[0])
print(df_tweets_all_load[1])
print(df_tweets_all_load[2])
print(df_tweets_all_load[3])
print(df_tweets_all_load[4])

df_tweets_all_load_joined = pd.concat([df_tweets_all_load[0], df_tweets_all_load[1],df_tweets_all_load[2],df_tweets_all_load[3],df_tweets_all_load[4]], ignore_index=True)
df_tweets_all_load_joined['tweet_ID'] = df_tweets_all_load_joined.index
df_tweets_all_load_joined.to_csv('df_tweets_all.csv')
# df_tweets_all_load_joined = pd.read_csv('df_tweets_all.csv')


df_tweets_all_load_joined


# ------------------------------------------------------------
# #### Remove http and https link (find https until first space)
# ------------------------------------------------------------
# 
# 



# example one data.frame (on python tweets)
# messages = df_tweets_all_load[1]["tweet"]
messages = df_tweets_all_load_joined["tweet"]


# # Sentence Embeddings using USE 




import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns



# might take up to 10ins for first download/load, later on shall be ulpoaded within seconds from cache

def generate_sentence_embeddings(sentences_list, module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"):
    
    embed = hub.load(module_url)
    sentence_embeddings = embed(sentences_list)
    
    return sentence_embeddings


# ### Generate embeddings for tweets




message_embeddings = generate_sentence_embeddings(messages)





print(message_embeddings)

# convert into numpy array
message_embeddings = message_embeddings.numpy()


# #### Save into a pickle file.





                                                                                                                                                                                                                   
pickle.dump( message_embeddings, open( "message_embeddings.p", "wb" ) )


# #### Load the dictionary back from the pickle file.




message_embeddings = pickle.load( open( "message_embeddings.p", "rb" ) )




# # Clustering the embeddings 

# 
# 
# ### K-means clustering 
# 






import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

import pandas as pd

from sklearn.cluster import KMeans
import collections

# convert message embeddings to DataFrame with numeric values
myX = pd.DataFrame(message_embeddings)
X = myX
X = X.apply(pd.to_numeric)

# k-means clustering from text embeddings
# number of clusters to have on averahe 30 tweets in each - can be tested for more options / better fit 
n_clusters_proportionate = int(round(len(X)/30,0))

# apply k-means clustering 
kmeans_plus = KMeans(init='k-means++', n_clusters = n_clusters_proportionate, n_init=10)
kmeans_plus.fit(X)
cluster_labels = kmeans_plus.labels_
# print(kmeans_plus.labels_)
collections.Counter(kmeans_plus.labels_)

# join labels with predicted clusters
out = pd.DataFrame(columns = ['cluster_ID'])
out['cluster_ID'] = cluster_labels 
out_merged = pd.merge(out, df_tweets_all_load_joined, left_index=True, right_index=True) #  to check output clusters merged with raw data
print(out_merged)

# add tweet_ID to mapoing table as output for Tableau
out['tweet_ID'] = out.index 





# save output to csv
out.to_csv(my_path + "out_cluster_twitter_ID_mapping.csv", sep = ';')


# check some random clusters
print(out_merged[out_merged.cluster_ID == 5])


# ## TF-IDF to determine top topics within in each cluster

# ### Data prep




def tfidf_preproc(text):

    text = text.replace('.',' ')
    text = text.replace('#','')
    text = re.sub(r'\s+',' ',re.sub(r'[^\w \s]','',re.sub(r'[0-9]','',text) ) ).lower() # remove white space characters and all numbers 
    # more regex ppatterns can be found here: https://docs.python.org/3/library/re.html

    return text




out_merged['cleaned'] = [tfidf_preproc(i) for i in out_merged.tweet]                





out_merged


# #### Split the dataframe into clusters
# Each cluster to form one document for TF-IDF. Such as; All tweets from one cluster concatenated into one doc.



gb = out_merged.groupby('cluster_ID')    
[gb.get_group(x) for x in gb.groups]






out_groups = [gb.get_group(x) for x in gb.groups]





final = [''.join(i.cleaned) for i in out_groups]

print(final[0])
print(final[20])


# #### TF-IDF Vectorization 




import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(final)
names = vectorizer.get_feature_names()
data = vectors.todense().tolist()
# Create a dataframe with the results
df = pd.DataFrame(data, columns=names)




# nrows = no. clusters, ncols = no. words in tfidf, matrix values = tfidf of each word in each cluster
df



# ### Remove all columns containing a stop word from the resultant dataframe. 




import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
st = set(stopwords.words('english'))

df = df[filter(lambda x: x not in list(st) , df.columns)]


# ### Check top 10 topics within each cluster 


def add_top_X_topics(df, X_topics):

    out_cluster_topics = pd.DataFrame(columns = ['tfidf', 'topics','cluster_ID'])

    n = X_topics;
    for i in df.iterrows():
        print(i[1].sort_values(ascending=False)[:n])
        my_df = pd.DataFrame(i[1].sort_values(ascending=False)[:n])
        cluster_id = list(my_df)[0]
        my_df['topics'] = my_df.index
        my_df.columns = ['tfidf', 'topics']
        my_df['cluster_ID'] = cluster_id
        out_cluster_topics = pd.concat([out_cluster_topics,my_df], ignore_index=True)

    return out_cluster_topics


# join with full data to export to Tableau
out_cluster_topics = add_top_X_topics(df,10)
out_cluster_topics.to_csv(my_path + 'out_cluster_metadata.csv', sep = ';')

# check out joined in master table with raw data
out_all_columns = pd.merge(out_merged, out_cluster_topics, right_on = 'cluster_ID', left_on = 'cluster_ID')




# # Dimensionality reduction using PCA for cluster plotting 




import numpy as np
from sklearn.decomposition import PCA




def generate_pca_coordinates(original_vectors_array, num_components = 2):
    X = original_vectors_array
    pca = PCA(n_components=num_components)
    pca_embeddings = pca.fit_transform(X)
    pca_coords = pd.DataFrame(pca_embeddings,
                          columns=['x','y'])
    return pca_coords


out_merged
out_plot = out_merged.copy()
# reduce 512-dim to 2-dim to be able to visualize in 2D chart
pca_coords = generate_pca_coordinates(message_embeddings, num_components=2)
out_plot['x'] = pca_coords['x']
out_plot['y'] = pca_coords['y']

out_plot_pca = out_plot[['x','y','cluster_ID']]
out_plot_pca.to_csv(my_path + 'out_plot_pca.csv', sep = ';')


# create chart wtih plotly - to be updated
import matplotlib.pyplot as plt
import matplotlib.cm as cm # to hve nicer and more colors


N = len(out_plot)
x = out_plot['x']
y = out_plot['y']
colors = out_plot['cluster_ID']/100 # divide by 100 to have decimal numbners to have broader scale of colors

plt.scatter(x, y, c=cm.rainbow(colors), alpha=0.5)
plt.show()



# # TO DO hierarchical clustering




# hierarchical clustering
from scipy.cluster.hierarchy import dendrogram, linkage  
from matplotlib import pyplot as plt

# linked = linkage(np.array(X), 'single')
linked = linkage(np.array(X), 'ward')
labelList = out.index.values

plt.figure(figsize=(10, 7))  
dendrogram(linked,  
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()  





####### Tableau input #######
# 1. Raw data (location, date, http link, twitter text)  + ID twitter + added hyperlinks with https = df_tweets_all.csv
# 2. ID twitter + ID cluster = out_cluster_twitter_ID_mapping.csv
# 3. ID cluster + Custer data (top 10 keywords) + ID keyword  = out_cluster_metadata.csv
# 4. ID cluster + x + y + ID Twitter (optional) = out_plot_pca.csv
 

