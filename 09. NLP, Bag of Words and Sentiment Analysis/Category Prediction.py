#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


# In[2]:


dict_cat = {'talk.religion.misc': 'Religious Content', 'rec.autos': 'Automobile and Transport','rec.sport.hockey':'Sport: Hockey','sci.electronics':'Content: Electronics', 'sci.space': 'Content: Space'}


# In[3]:


data_train = fetch_20newsgroups(subset='train', categories = dict_cat.keys(), shuffle=True, random_state=3)


# In[4]:


cv_vector = CountVectorizer()
data_train_fit = cv_vector.fit_transform(data_train.data)
print("\nTraining Data Dimensions:", data_train_fit.shape)


# In[5]:


tfidf_transformer = TfidfTransformer()
train_tfidf_transformer = tfidf_transformer.fit_transform(data_train_fit)


# In[6]:


sample_input_data = [
'The Apollo Series were a bunch of space shuttles',
'Islamism, Hinduism, Christianity, Sikhism are all major religions of the world',
'It is a necessity to drive safely',
'Gloves are made of rubber',
'Gadgets like TV, Refrigerator and Grinders, all use electricity'
]


# In[7]:


input_classifier = MultinomialNB().fit(train_tfidf_transformer, data_train.target)
input_cv = cv_vector.transform(sample_input_data)
tfidf_input = tfidf_transformer.transform(input_cv)
predictions_sample = input_classifier.predict(tfidf_input)


# In[8]:


for inp, cat in zip(sample_input_data, predictions_sample):
    print('\nInput Data:', inp, '\n Category:',         dict_cat[data_train.target_names[cat]])


# In[ ]:




