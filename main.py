import pandas as pd
import numpy as np
import csv
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from difflib import get_close_matches
df = pd.read_csv('movies.csv')                   #reading the data
df=df[['movieId','title','genres']]

df['genres']=df['genres'].str.lower()            #lowering the case genres column

tf=TfidfVectorizer(analyzer='word',ngram_range=(1,2),min_df=0,stop_words='english')
tmatrix=tf.fit_transform(df['genres'])

cos_sim_matrix=linear_kernel(tmatrix,tmatrix)

rec_movies=df['title']
index=pd.Series(df.index,index=df['title'])

def recommend_movies(title):
    idx=index[title]
    sim_score=list(enumerate(cos_sim_matrix[idx]))
    sim_score=sorted(sim_score,key=lambda x:x[1],reverse=True)
    sim_score=sim_score[2:21]
    movie_indices=[i[0] for i in sim_score]
    return rec_movies.iloc[movie_indices]



enter_movie=input("Enter a movie: ")
close_match=get_close_matches(enter_movie,rec_movies,1,0.4)

enter_movie=''.join(close_match)
print(enter_movie)
x=recommend_movies(enter_movie)
print(x)




