#!/usr/bin/env python
# coding: utf-8

# In[28]:


# Characterization of the signal from the input file
# We will be using Fourier Transforms to convert the signals to a frequency domain distribution


# In[29]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


# In[30]:


freq_sample, sig_audio = wavfile.read("Welcome.wav")


# In[31]:


print('\nShape of the Signal:', sig_audio.shape)
print('Signal Datatype:', sig_audio.dtype)
print('Signal duration:', round(sig_audio.shape[0] / float(freq_sample), 2), 'seconds')


# In[32]:


sig_audio = sig_audio / np.power(2, 15)


# In[33]:


# Extracting the length and the half-length of the signal to input to the foruier transform
sig_length = len(sig_audio)
half_length = np.ceil((sig_length + 1) / 2.0).astype(np.int)


# In[34]:


# We will now be using the Fourier Transform to form the frequency domain of the signal
signal_freq = np.fft.fft(sig_audio)
# Normalize the frequency domain and square it 
signal_freq = abs(signal_freq[0:half_length]) / sig_length
signal_freq **= 2


# In[35]:


transform_len = len(signal_freq)
# The Fourier transformed signal now needs to be adjusted for both even and odd cases


# In[36]:


if sig_length % 2:
    signal_freq[1:transform_len] *= 2
else:
    signal_freq[1:transform_len-1] *= 2


# In[37]:


# Extract the signal's strength in decibels (dB)
exp_signal = 10 * np.log10(signal_freq)


# In[38]:


x_axis = np.arange(0, half_length, 1) * (freq_sample / sig_length) / 1000.0


# In[39]:


plt.figure()
plt.plot(x_axis, exp_signal, color='green', linewidth=1)
plt.xlabel('Frequency Representation (kHz)')
plt.ylabel('Power of Signal (dB)')
plt.show()


# In[ ]:




