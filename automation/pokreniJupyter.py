import tkinter as tk
from tkinter import ttk
import subprocess
import os


notebook_dir = r"C:\Users\bogdan.sliskovic\Desktop\bole\ML"
notebook_files = []
for root_dir, dirs, files in os.walk(notebook_dir):
    if ".ipynb_checkpoints" in dirs:
        dirs.remove(".ipynb_checkpoints")
    for file in files:
        if file.endswith(".ipynb"):
            rel_dir = os.path.relpath(root_dir, notebook_dir)
            if rel_dir == ".":
                notebook_files.append(file)
            else:
                notebook_files.append(os.path.join(rel_dir, file))
if not notebook_files:
    print("Nema .ipynb fajlova u direktorijumu!")
else:
    root = tk.Tk()
    root.title("Izaberi Notebook")
    root.geometry("300x200")

    label = tk.Label(root, text="Izaberi Jupyter Notebook:")
    label.pack(pady=5)

    selected_notebook = tk.StringVar()
    dropdown = ttk.Combobox(root, textvariable=selected_notebook, values=notebook_files, state="readonly", width = 50)
    dropdown.configure(font = ('Helvetica', 12))
    dropdown.pack(pady=5)
    dropdown.set(notebook_files[0])

    def open_notebook():
        notebook_name = dropdown.get()
        notebook_path = os.path.join(notebook_dir, notebook_name)
        print(f"Otvoriću: {notebook_path}")
        subprocess.Popen([r"C:\Users\bogdan.sliskovic\AppData\Local\Programs\Python\Python313\Scripts\jupyter.exe", "notebook", notebook_path])

    open_button = tk.Button(root, text="Otvori Notebook", command=open_notebook)
    open_button.pack(pady=10)

    root.mainloop()




