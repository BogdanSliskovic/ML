import os

repo_path = r'C:\Users\Jelena\Desktop\Freestyle\GitHub\ML'

os.chdir(repo_path)

print("Proveravam promene u repozitorijumu...\n")
os.system("git fetch")

print("\nSTATUS REPOZITORIJUMA ***\n")
os.system("git status")

print("\nDETALJNE PROMENE ***\n")
os.system("git diff --stat") 

odgovor = input("\nŽeliš li da pull-uješ najnovije promene? (y/n): ").strip().lower()

if odgovor == 'y':
    print("\nPovlačim najnovije izmene sa Gita...\n")
    os.system("git pull")
else:
    print("\nPovlačenje izmena je preskočeno.")

