import os
from datetime import date

repo_path = r'C:\Users\Jelena\Desktop\Freestyle\GitHub\ML'

os.chdir(repo_path)
os.system("git add .")

today = date.today()

x = input('Kako zelis da nazoves commit?')
os.system(f'git commit -m"{today.day}/{today.month} {x}"')

os.system("git push")