#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

# Putanja do Git repozitorijuma
repo_path = r'C:\Users\Jelena\Desktop\Freestyle\GitHub\ML'

# Prelazak u repozitorijum
os.chdir(repo_path)

# Fetch-uje izmene sa remote-a (da vidi šta se promenilo)
print("📢 Proveravam promene u repozitorijumu...\n")
os.system("git fetch")

# Prikazuje detaljan status i promene fajlova
print("\n📝 *** STATUS REPOZITORIJUMA ***\n")
os.system("git status")

print("\n🔍 *** DETALJNE PROMENE ***\n")
os.system("git diff --stat")  # Prikazuje listu promenjenih fajlova

# Pita korisnika da li želi da pull-uje izmene
odgovor = input("\n❓ Želiš li da pull-uješ najnovije promene? (y/n): ").strip().lower()

if odgovor == 'y':
    print("\n📥 Povlačim najnovije izmene sa Gita...\n")
    os.system("git pull")
else:
    print("\n🚫 Povlačenje izmena je preskočeno.")

