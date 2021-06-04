import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import Task3


window = tk.Tk()
window.title('Task3')
window.geometry('600x500')

# label
ttk.Label(window, text="Enter number of hidden layers :",
          font=("Times New Roman", 10)).grid(column=0,
                                             row=5, padx=10, pady=25)
# label
ttk.Label(window, text="Enter number of neurons in each hidden layer :",
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
ttk.Label(window, text="Choose activation function :",
          font=("Times New Roman", 10)).grid(column=0,
                                             row=25, padx=10, pady=25)
# label
ttk.Label(window, text="Add Bias ? ",
          font=("Times New Roman", 10)).grid(column=0,
                                             row=30, padx=10, pady=25)

Activation_values = ('Sigmoid',
                     'Hyperbolic Tangent sigmoid')

layers = tk.StringVar()
epochs_entry = ttk.Entry(window, textvariable=layers, font=("Times New Roman", 10))
epochs_entry.grid(column=1, row=5)

neurons = tk.StringVar()
epochs_entry = ttk.Entry(window, textvariable=neurons, font=("Times New Roman", 10))
epochs_entry.grid(column=1, row=10)

learning_rate = tk.StringVar()
learning_rate_entry = ttk.Entry(window, textvariable=learning_rate, font=("Times New Roman", 10))
learning_rate_entry.grid(column=1, row=15)

epochs = tk.StringVar()
epochs_entry = ttk.Entry(window, textvariable=epochs, font=("Times New Roman", 10))
epochs_entry.grid(column=1, row=20)

activation = tk.StringVar()
Class_choice = ttk.Combobox(window, width=27, textvariable=activation)
Class_choice['values'] = Activation_values
Class_choice.grid(column=1, row=25)
Class_choice.current()

IsBias = tk.IntVar()
IsBias.set(0)
Bias_values = {1: "Yes",
               0: "No"}
col = 1
for (value, text) in Bias_values.items():
    ttk.Radiobutton(window, text=text, variable=IsBias,
                    value=value).grid(column=col, row=30)
    col = col + 1


def Process():
    acc = Task3.Call_back(int(layers.get()), str(neurons.get()),
                          float(learning_rate.get()), int(epochs.get()), activation.get(), IsBias.get())
    tk.messagebox.showinfo("Classification Accuracy", str(acc[0]))
    tk.messagebox.showinfo("Training Accuracy", str(acc[1]))


btn = tk.Button(window, text='Process!', width=15
                , command=Process).grid(column=2, row=35)
window.mainloop()
