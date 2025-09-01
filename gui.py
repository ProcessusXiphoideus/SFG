import os
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox
from analyze_data import analyze_and_plot

# Vytvoření okna
app = ttk.Window(themename="flatly")
app.title("GUI pro analyze_data")
app.geometry("1200x600")


def vyber_slozku(entry_widget):
    cesta = filedialog.askdirectory()
    if cesta:
        entry_widget.delete(0, "end")
        entry_widget.insert(0, cesta)


def spustit():
    folder_path_up = entry1.get()
    folder_path_down = entry2.get()
    
    name = entry3.get()
    date = entry4.get()
    variables_selected = [v.get() for v in variables]
    
    analyze_and_plot(
        variables=variables_selected,
        folder_path_up=folder_path_up,
        folder_path_down=folder_path_down,
        component=name,
        date=date,
        per_chip=var1.get(),
        plot_hist=var2.get(),
        under=var3.get(),
        away=var4.get(),
        H0=var5.get(),
        H1=var6.get()
    )
    messagebox.showinfo("Done", "Great success!")

variables = []

def add_variable():
    var = ttk.StringVar(value=variables_list[0])
    variables.append(var)
    menu = ttk.OptionMenu(app, var, *variables_list)
    menu.grid(row=len(variables)-1, column=1, padx=10, pady=5, sticky="ew", columnspan=1)

variables_list = ["Input Noise", "Input Noise", "Gain", "Vt50"]
ttk.Label(app, text="Variable:").grid(row=0, column=0, padx=10, pady=(15,5), sticky="w")
add_variable()  # Přidání prvního OptionMenu
ttk.Button(app, text="Add Variable", command=add_variable).grid(row=0, column=2, padx=10, pady=5, sticky="w")

# Entry a tlačítka pro složky
ttk.Label(app, text="Folder with data:").grid(row=len(variables) + 2, column=0, padx=10, pady=5, sticky="w")
entry1 = ttk.Entry(app, width=35)
entry1.grid(row=len(variables) + 2, column=1, padx=5, pady=5, sticky="ew")
ttk.Button(app, text="Search...", command=lambda: vyber_slozku(entry1)).grid(row=len(variables) + 2, column=2, padx=5, pady=5)

ttk.Label(app, text="Folder for output:").grid(row=len(variables) + 3, column=0, padx=10, pady=5, sticky="w")
entry2 = ttk.Entry(app, width=35)
entry2.grid(row=len(variables) + 3, column=1, padx=5, pady=5, sticky="ew")
ttk.Button(app, text="Search...", command=lambda: vyber_slozku(entry2)).grid(row=len(variables) + 3, column=2, padx=5, pady=5)

# Entry pro Name a Date
ttk.Label(app, text="Name:").grid(row=len(variables) + 4, column=0, padx=10, pady=5, sticky="w")
entry3 = ttk.Entry(app, width=40)
entry3.grid(row=len(variables) + 4, column=1, columnspan=2, padx=10, pady=5, sticky="ew")

ttk.Label(app, text="Date:").grid(row=len(variables) + 5, column=0, padx=10, pady=5, sticky="w")
entry4 = ttk.Entry(app, width=40)
entry4.grid(row=len(variables) + 5, column=1, columnspan=2, padx=10, pady=5, sticky="ew")

# Checkboxy
var1 = ttk.BooleanVar()
check1 = ttk.Checkbutton(app, text="Analyze single chips", variable=var1)
check1.grid(row=len(variables) + 6, column=0, columnspan=3, padx=10, pady=5, sticky="w")

var2 = ttk.BooleanVar()
check2 = ttk.Checkbutton(app, text="Plot Histogram", variable=var2)
check2.grid(row=len(variables) + 7, column=0, columnspan=3, padx=10, pady=5, sticky="w")

var3 = ttk.BooleanVar(value=True)
check3 = ttk.Checkbutton(app, text = "Under", variable=var3)
check3.grid(row=len(variables) + 8, column=0, columnspan=3, padx=10, pady=5, sticky="w")

var4 = ttk.BooleanVar(value=True)
check4 = ttk.Checkbutton(app, text = "Away", variable=var4)
check4.grid(row=len(variables) + 8, column=1, columnspan=3, padx=10, pady=5, sticky="w")

var5 = ttk.BooleanVar(value=True)
check5 = ttk.Checkbutton(app, text = "H0", variable=var5)
check5.grid(row=len(variables) + 9, column=0, columnspan=3, padx=10, pady=5, sticky="w")

var6 = ttk.BooleanVar(value=True)
check6 = ttk.Checkbutton(app, text = "H1", variable=var6)
check6.grid(row=len(variables) + 9, column=1, columnspan=3, padx=10, pady=5, sticky="w")

# Tlačítko Spustit
tlacitko = ttk.Button(app, text="Analyze", bootstyle="SUCCESS", command=spustit)
tlacitko.grid(row=len(variables) + 10, column=0, columnspan=3, padx=10, pady=40)

# Grid konfigurace
app.grid_columnconfigure(1, weight=1)
app.mainloop()



