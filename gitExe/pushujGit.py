#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %load pushujGit.py
import os
from datetime import date

repoPath = r'C:\Users\bogdan.sliskovic\Desktop\bole\ML'

os.chdir(repoPath)
os.system("git add .")

today = date.today()

x = input('Kako zelis da nazoves commit?')
os.system(f'git commit -m"{today.day}/{today.month} {x}"')

os.system("git push")

get_ipython().run_line_magic('save', '')

