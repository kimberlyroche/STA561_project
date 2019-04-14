# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 21:12:05 2019

@author: harsh
"""

#%%
import pandas as pd
import numpy as np

def splitDataFrameList(df,target_column,separator):
    ''' df = dataframe to split,
    target_column = the column containing the values to split
    separator = the symbol used to perform the split
    returns: a dataframe with each entry for the target column separated, with each element moved into a new row. 
    The values in the other columns are duplicated across the newly divided rows.
    '''
    def splitListToRows(row,row_accumulator,target_column,separator):
        split_row = row[target_column].split(separator)
        for s in split_row:
            new_row = row.to_dict()
            new_row[target_column] = s
            row_accumulator.append(new_row)
    new_rows = []
    df.apply(splitListToRows,axis=1,args = (new_rows,target_column,separator))
    new_df = pd.DataFrame(new_rows)
    return new_df



data=pd.read_csv("movies_with_overviews.txt",sep="\t",header=None,encoding='latin-1')
data.columns=['Title','Overview','Genre']

#data = data.pivot(columns='Genre').fillna(0)
data=data.dropna(axis=0,how='any')

for index,row in data.iterrows():
    data.at[index,'Genre']=row['Genre'][2:len(row['Genre'])-1].replace(" ","")
    
for index,row in data.iterrows():
    data.at[index,'Primary Genre']=row['Genre'].split(",")[0]



check=splitDataFrameList(data,'Genre',',')

check['Value']=1

final_df = check.pivot_table(index=['Title','Overview'], columns='Genre', values='Value').fillna(0).reset_index()

final_df=final_df[final_df.Overview != '  ']
        
#%%

#%%
df=final_df
# check language of plots, here all english
from langdetect import detect
df['plot_lang'] = df.apply(lambda row: detect(row['Overview']), axis=1)
print(df['plot_lang'].value_counts())
#%%

#%%
# remove all stop words

# need to install stopwords and punkt before

#import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize,RegexpTokenizer
stop_words = set(stopwords.words('english')) 
example=df['Overview'][1].lower()
tokenizer=RegexpTokenizer(r'\w+')
word_tokens =tokenizer.tokenize(example) 
filtered_sentence = [w for w in word_tokens if not w in stop_words] 



#%%


#%%
from gensim import models
# model2 = models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True) 
model_nlp = models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
#%%


#%%
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,RegexpTokenizer


df.index=range(1514)     
mean_wordvec=np.zeros((len(df),300))

for index,row in df.iterrows():
    overview=row['Overview'].lower()
    tokenizer=RegexpTokenizer(r'\w+')
    tokens =tokenizer.tokenize(overview)
    filtered_tokens=[w for w in word_tokens if not w in stop_words] 
    count=0
    sum_vec=0
    if len(filtered_tokens)!=0:
        for token in filtered_tokens:
            if token in model_nlp.vocab:
                count+=1
                sum_vec+=model_nlp[token]
            
        if count!=0:
            mean_wordvec[index]=sum_vec/float(count)
        else:
            print("no tokens in word2vec for sample",index)
    else:
        print("no tokens for sample",index)
    
    if(index%50==0):
        print('Done with index',index)
            

#%%

#%%
from sklearn.model_selection import train_test_split

genres_set=df.drop(['Title','Overview','plot_lang'],axis=1)

genres_set=primary_only_genres
features_set=mean_wordvec


indices = np.random.permutation(features_set.shape[0])
training_idx, test_idx = indices[:1135], indices[1135:]

features_train, features_test = features_set[training_idx,:], features_set[test_idx,:]
genres_train,genres_test = np.array(genres_set)[training_idx,:], np.array(genres_set)[test_idx,:]
#%%

#%%


from keras.models import Sequential
from keras.layers import Dense, Activation

model_textual = Sequential([
    Dense(300, input_shape=(300,)),
    Activation('relu'),
    Dense(19),
    Activation('softmax'),
])

model_textual.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model_textual.fit(features_train, genres_train, epochs=10, batch_size=32)

#%%

#%%
import sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
genres_pred=model_textual.predict(features_test)
genres_pred_train=model_textual.predict(features_train)
matrix=confusion_matrix(train.argmax(axis=1), genres_pred_train.argmax(axis=1))
data=pd.DataFrame(matrix)
cm = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
plt.imshow(data, vmax=10,cmap='viridis')
plt.xticks([])
#%%

#%%

check2=data.drop(['Genre'], axis=1)

check2['Value']=1

primary = check2.pivot_table(index=['Title','Overview'], columns='Primary Genre', values='Value').fillna(0).reset_index()

primary = primary[primary.Overview != '  ']
 
primary_only_genres=primary.drop(['Title','Overview'],axis=1)


test=np.array(primary_only_genres)[test_idx,:]
train=np.array(primary_only_genres)[training_idx,:]
#%%


#%%
score = model_textual.evaluate(features_test,genres_test,batch_size=249)
#%%

    
