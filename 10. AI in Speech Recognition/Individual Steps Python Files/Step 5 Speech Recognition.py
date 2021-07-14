#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install SpeechRecognition


# In[2]:


pip install pipwin


# In[7]:


# https://anaconda.org/anaconda/pyaudio
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
# Run in the Anaconda Terminal CMD: conda install -c anaconda pyaudio
# Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/

pip install pyaudio


# In[7]:


import speech_recognition as speech_recog


# In[8]:


# Creating a recording object to store input
rec = speech_recog.Recognizer()


# In[9]:


# Importing the microphone class to check availabiity of microphones
mic_test = speech_recog.Microphone()


# In[10]:


# List the available microphones
speech_recog.Microphone.list_microphone_names()


# In[14]:


# We will now directly use the microphone module to capture voice input
# Specifying the second microphone to be used for a duration of 3 seconds
# The algorithm will also adjust given input and clear it of any ambient noise
with speech_recog.Microphone(device_index=1) as source: 
    rec.adjust_for_ambient_noise(source, duration=3)
    print("Reach the Microphone and say something!")
    audio = rec.listen(source)


# In[15]:


try:
    print("I think you said: \n" + rec.recognize_google(audio))
except Exception as e:
    print(e)


# In[ ]:




