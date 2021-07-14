#!/usr/bin/env python
# coding: utf-8

# In[15]:


# Import the packages needed for this analysis
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


# In[16]:


# We will now read the audio file and determine the audio signal and sampling frequency 
# Give the path of the file
freq_sample, sig_audio = wavfile.read("Welcome.wav")


# In[17]:


# Output the parameters: Signal Data Type, Sampling Frequency and Duration
print('\nShape of Signal:', sig_audio.shape)
print('Signal Datatype:', sig_audio.dtype)
print('Signal duration:', round(sig_audio.shape[0] / float(freq_sample), 2), 'seconds')


# In[18]:


# Normalize the signal values
pow_audio_signal = sig_audio / np.power(2, 15)


# In[19]:


# We shall now extract the first 100 values from the signal 
pow_audio_signal = pow_audio_signal [:100]
time_axis = 1000 * np.arange(0, len(pow_audio_signal), 1) / float(freq_sample)


# In[20]:


# Visualize the signal
plt.plot(time_axis, pow_audio_signal, color='blue')
plt.xlabel('Time (milliseconds)')
plt.ylabel('Amplitude')
plt.title('Input audio signal')
plt.show()


# In[ ]:




