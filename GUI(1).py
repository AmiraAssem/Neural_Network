import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import Task1

window = tk.Tk()
window.title('Task1')
window.geometry('500x500')

# label
ttk.Label(window, text="Select Features :",
          font=("Times New Roman", 10)).grid(column=0,
                                             row=5, padx=10, pady=25)
# label
ttk.Label(window, text="Select Classes :",
          font=("Times New Roman", 10)).grid(column=0,
                                             row=10, padx=10, pady=25)
# label
ttk.Label(window, text="Enter learning rate :",
          font=("Times New Roman", 10)).grid(column=0,
                                             row=15, padx=10, pady=25)
# label
ttk.Label(window, text="Enter number of epochs :",
          font=("Times New Roman", 10)).grid(column=0,
                                             row=20, padx=10, pady=25)
# label
ttk.Label(window, text="Add Bias ? ",
          font=("Times New Roman", 10)).grid(column=0,
                                             row=25, padx=10, pady=25)

Feature_values = ('x1 & x2',
                  'x1 & x3',
                  'x1 & x4',
                  'x2 & x3',
                  'x2 & x4',
                  'x3 & x4')

Class_values = ('C1 & C2', 'C1 & C3', 'C2 & C3')

Features = tk.StringVar()
Feature_choice = ttk.Combobox(window, width=27, textvariable=Features)
Feature_choice['values'] = Feature_values
Feature_choice.grid(column=1, row=5)
Feature_choice.current()

Classes = tk.StringVar()
Class_choice = ttk.Combobox(window, width=27, textvariable=Classes)
Class_choice['values'] = Class_values
Class_choice.grid(column=1, row=10)
Class_choice.current()

learning_rate = tk.StringVar()
learning_rate_entry = ttk.Entry(window, textvariable=learning_rate, font=("Times New Roman", 10))
learning_rate_entry.grid(column=1, row=15)

epochs = tk.StringVar()
epochs_entry = ttk.Entry(window, textvariable=epochs, font=("Times New Roman", 10))
epochs_entry.grid(column=1, row=20)


IsBias = tk.IntVar()
IsBias.set(0)
Bias_values = {1: "Yes",
               0: "No"}
col = 1
for (value, text) in Bias_values.items():
    ttk.Radiobutton(window, text=text, variable=IsBias,
                    value=value).grid(column=col, row=25)
    col = col + 1


def Process():
    acc = Task1.Call_back(str(Feature_choice.get()), str(Class_choice.get()),
                          learning_rate.get(), epochs.get(), IsBias.get())
    tk.messagebox.showinfo("Classification Accuracy", str(acc[0]))
    tk.messagebox.showinfo("Training Accuracy", str(acc[1]))


btn = tk.Button(window, text='Process!', width=15
                , command=Process).grid(column=2, row=30)
window.mainloop()
