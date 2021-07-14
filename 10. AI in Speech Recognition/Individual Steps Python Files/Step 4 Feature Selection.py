#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install python_speech_features


# In[1]:


# Import the necessary pacakges
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank


# In[2]:


sampling_freq, sig_audio = wavfile.read("Welcome.wav")


# In[3]:


# We will now be taking the first 15000 samples from the signal for analysis
sig_audio = sig_audio[:15000]


# In[4]:


# Using MFCC to extract features from the signal
mfcc_feat = mfcc(sig_audio, sampling_freq)


# In[5]:


print('\nMFCC Parameters\nWindow Count =', mfcc_feat.shape[0])
print('Individual Feature Length =', mfcc_feat.shape[1])


# In[6]:


mfcc_feat = mfcc_feat.T
plt.matshow(mfcc_feat)
plt.title('MFCC Features')


# In[7]:


# Generating filter bank features
fb_feat = logfbank(sig_audio, sampling_freq)


# In[8]:


print('\nFilter bank\nWindow Count =', fb_feat.shape[0])
print('Individual Feature Length =', fb_feat.shape[1])


# In[9]:


fb_feat = fb_feat.T
plt.matshow(fb_feat)
plt.title('Features from Filter bank')
plt.show()


# In[ ]:




