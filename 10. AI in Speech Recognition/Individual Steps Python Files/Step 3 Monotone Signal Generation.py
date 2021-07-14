#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write


# In[18]:


# Specify the output file where this data needs to be stored
output_file = 'generated_signal_audio.wav'


# In[19]:


# Duration in seconds, Sampling Frequency in Hz
sig_duration = 8 
sig_frequency_sampling = 74100 
sig_frequency_tone = 802
sig_min_val = -5 * np.pi
sig_max_val = 5 * np.pi


# In[20]:


# Generating the audio signal
temp_signal = np.linspace(sig_min_val, sig_max_val, sig_duration * sig_frequency_sampling)
temp_audio_signal = np.sin(2 * np.pi * sig_frequency_tone * temp_signal)


# In[21]:


# The write() function creates a frequency based sound signal and writes it to the created file
write(output_file, sig_frequency_sampling, temp_audio_signal)


# In[22]:


sig_audio = temp_audio_signal[:100]
def_time_axis = 1000 * np.arange(0, len(sig_audio), 1) / float(sig_frequency_sampling)


# In[23]:


plt.plot(def_time_axis, sig_audio, color='green')
plt.xlabel('Time (milliseconds)')
plt.ylabel('Sound Amplitude')
plt.title('Audio Signal Generation')
plt.show()


# In[ ]:




