#!/usr/bin/env python
# coding: utf-8

# In[1]:


# We will use the CountVectorizer from Scikit learn to convert the text into numeric vectors


# In[2]:


from sklearn.feature_extraction.text import CountVectorizer


# In[3]:


input_sent = ['Demonstration of the BoW NLTK model', 'This model builds numerical features for text input']
input_cv = CountVectorizer()
features_text = input_cv.fit_transform(input_sent).todense()
print(input_cv.vocabulary_)


# In[4]:


# This allows us to build feature vectors that will successfully be used in Machine Learning algorithms.

