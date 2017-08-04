
# coding: utf-8

# In[ ]:

import sklearn
import matplotlib.pyplot as plot
import pandas
from sklearn.model_selection import train_test_split
import numpy
from wordcloud import WordCloud, STOPWORDS
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

#load the Techcrunch post compilation dataset

dataframe = pandas.read_csv('techcrunch_posts.csv')
#print (dataframe.shape)
dataframe.head()

#pre-processing the data in order to format the .csv file

dataframe['authors']= dataframe['authors'].apply(lambda l: str(l).split(','))
dataframe['content']=dataframe['content'].fillna(0)
dataframe=dataframe[dataframe['content']!=0]
dataframe['tags']= dataframe['tags'].apply(lambda l:['No tag']
if str(l)=='NaN'
else str(l).split(','))
dataframe['topics']= dataframe['topics'].apply(lambda l: str(l).split(','))
dataframe=dataframe.reset_index(drop=True)
print (dataframe)

#Selecting classification algorithm as K-Nearest Neighbors algorithm


def words(content):
    let = re.sub("[^a-zA-Z]"," ",content)
    words = let.lower().split()
    stop_words = set(stopwords.words("english"))
    words_meaning = [w for w in words if not w in stop_words]
    return(" ".join(words_meaning))
    print(content)

# a numerical statistic that is intended to reflect how important a word in the TechCrunch posts using tfidf matrix.

content_cleaning=[]
for every_word in dataframe['content']:
    content_cleaning.append(words(every_word))

tfidf = TfidfVectorizer()
feat = tfidf.fit_transform(content_cleaning)

knn = NearestNeighbors(n_neighbors=30, algorithm='brute', metric='cosine')
f = knn.fit(feat)
print (f)

# First finding most prominent words in articles of a particular author from the dataset to create a WordCloud

def recommendation_sys(author):
    index=[]
    for d in range(len(dataframe)):
        if author in dataframe['authors'][d]:
            index.append(d)
    temp_dataframe = dataframe.iloc[index,:]
    cont_author=[]
    for each in temp_dataframe['content']:
        cont_author.append(words(each))
        w_c = WordCloud(background_color = 'white', width = 2000, height = 1500).generate(cont_author[0])

# classifying articles of authors using KNN

    neighbors = f.kneighbors(feat[index[0]])[1].tolist()[0][2:]

#Find the articles that are not authored/co-authored  by the author

    all = dataframe.iloc[neighbors]
    all = all.reset_index(drop = True)
    remain_index = []
    for j in range(len(all)):
        if author in all['authors'][j]:
            pass
        else:
            remain_index.append(j)
    f_frame = all.iloc[remain_index,:]
    f_frame = f_frame.reset_index(drop = True)
    articles_selected = f_frame.iloc[0:5,:]

    print ('==='*30)
    print ('The Follwoing words are always featured in ' +author+ ' articles.')
    plot.figure(1,figsize=(8,8))
    plot.imshow(w_c)
    plot.axis('off')
    plot.show()
    print ('==='*30)
    print('Based on your readings, following are the top 5 articles for you')
    for k in range(len(articles_selected)):
        print(articles_selected['title'][k]+', authored by\n ' +articles_selected['authors'][k][0]+ ' ,follow the link below to visit the main article \n' +articles_selected['url'][k])

#Enter the Name of the Author
recommendation_sys('Sarah Perez')





# In[ ]:



