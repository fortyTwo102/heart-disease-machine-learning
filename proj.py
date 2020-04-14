import tkinter as tk
from tkinter import ttk

import time

LARGE_FONT = ("Helvetica", 40, "bold italic")


class gui(tk.Tk):
    def __init__(self):


        tk.Tk.__init__(self)

        container = tk.Frame(self)

        container.pack(side="top", fill="both", expand = True)


        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, LoadPage, PageOne, EndPage):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()

def popupmsg():
    popup=tk.Tk()

    def leavemini():
        popup.destroy()

    popup.wm_title("Info")
    label=ttk.Label(popup,text="Information",font = ("Helvetica", 20, "bold"))
    label.pack(side="top",fill="x",pady=10)
    b1=ttk.Button(popup,text="Okay",command=leavemini)
    b1.pack()
    popup.mainloop()
    


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        
        label = tk.Label(self, text="LIVER DISEASE PREDICTION\nUSING\nLOGISTIC REGRESSION", font = LARGE_FONT)

        label.grid(row=0, column=0, pady=40,padx=40)

        button = tk.Button(self, text="START",font=("Helvetica", 16, "bold"),
                            command=lambda: controller.show_frame(PageOne))
        button.grid(row=2, column=0,padx=20,pady=20)



class PageOne(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)

        label = tk.Label(self, text="Enter Feature Fields", font = ("Helvetica", 20, "bold"))
        label.grid(column = 3, row = 0, padx=10, pady=10)

        name = tk.StringVar()

        label = tk.Label(self, text = "LABEL")
        label.grid(column = 0, row = 2, padx=10, pady=10)

        nameEntered = ttk.Entry(self, width = 15, textvariable = name)
        nameEntered.grid(column = 1, row = 2, padx=10, pady=10)

        label = tk.Label(self, text = "LABEL")
        label.grid(column = 4, row = 2, padx=10, pady=10)

        nameEntered = ttk.Entry(self, width = 15, textvariable = name)
        nameEntered.grid(column =5, row = 2, padx=10, pady=10)

        

        label = tk.Label(self, text = "LABEL")
        label.grid(column = 0, row = 3, padx=10, pady=10)

        nameEntered = ttk.Entry(self, width = 15, textvariable = name)
        nameEntered.grid(column = 1, row = 3, padx=10, pady=10)

        label = tk.Label(self, text = "LABEL")
        label.grid(column = 4, row = 3, padx=10, pady=10)

        nameEntered = ttk.Entry(self, width = 15, textvariable = name)
        nameEntered.grid(column =5, row = 3, padx=10, pady=10)

        

        label = tk.Label(self, text = "LABEL")
        label.grid(column = 0, row = 4, padx=10, pady=10)

        nameEntered = ttk.Entry(self, width = 15, textvariable = name)
        nameEntered.grid(column = 1, row = 4, padx=10, pady=10)

        label = tk.Label(self, text = "LABEL")
        label.grid(column = 4, row = 4, padx=10, pady=10)

        nameEntered = ttk.Entry(self, width = 15, textvariable = name)
        nameEntered.grid(column =5, row = 4, padx=10, pady=10)

        

        label = tk.Label(self, text = "LABEL")
        label.grid(column = 0, row = 5, padx=10, pady=10)

        nameEntered = ttk.Entry(self, width = 15, textvariable = name)
        nameEntered.grid(column = 1, row = 5, padx=10, pady=10)

        label = tk.Label(self, text = "LABEL")
        label.grid(column = 4, row = 5, padx=10, pady=10)

        nameEntered = ttk.Entry(self, width = 15, textvariable = name)
        nameEntered.grid(column =5, row = 5, padx=10, pady=10)

        

        label = tk.Label(self, text = "LABEL")
        label.grid(column = 0, row = 6, padx=10, pady=10)

        nameEntered = ttk.Entry(self, width = 15, textvariable = name)
        nameEntered.grid(column = 1, row = 6, padx=10, pady=10)

        label = tk.Label(self, text = "LABEL")
        label.grid(column = 4, row = 6, padx=10, pady=10)

        nameEntered = ttk.Entry(self, width = 15, textvariable = name)
        nameEntered.grid(column =5, row = 6, padx=10, pady=10)

        

        label = tk.Label(self, text = "LABEL")
        label.grid(column = 0, row = 7, padx=10, pady=10)

        nameEntered = ttk.Entry(self, width = 15, textvariable = name)
        nameEntered.grid(column = 1, row = 7, padx=10, pady=10)

        label = tk.Label(self, text = "LABEL")
        label.grid(column = 4, row = 7, padx=10, pady=10)

        nameEntered = ttk.Entry(self, width = 15, textvariable = name)
        nameEntered.grid(column =5, row = 7, padx=10, pady=10)

        

        label = tk.Label(self, text = "LABEL")
        label.grid(column = 0, row = 8, padx=10, pady=10)

        nameEntered = ttk.Entry(self, width = 15, textvariable = name)
        nameEntered.grid(column = 1, row = 8, padx=10, pady=10)

        label = tk.Label(self, text = "LABEL")
        label.grid(column = 4, row = 8, padx=10, pady=10)

        nameEntered = ttk.Entry(self, width = 15, textvariable = name)
        nameEntered.grid(column =5, row = 8, padx=10, pady=10)
        
        

        button = tk.Button(self, text="Predict!",font=("Helvetica", 16, "bold"),
                            command=lambda: controller.show_frame(LoadPage))
        button.grid(row=10,column=3, padx=10, pady=20)

        button1 = tk.Button(self, text="inf",command=popupmsg)
        button1.grid(row=2,column=2, padx=10, pady=10)
        button1 = tk.Button(self, text="inf",command=popupmsg)
        button1.grid(row=2,column=6, padx=10, pady=10)
        button1 = tk.Button(self, text="inf",command=popupmsg)
        button1.grid(row=3,column=2, padx=10, pady=10)
        button1 = tk.Button(self, text="inf",command=popupmsg)
        button1.grid(row=3,column=6, padx=10, pady=10)
        button1 = tk.Button(self, text="inf",command=popupmsg)
        button1.grid(row=4,column=2, padx=10, pady=10)
        button1 = tk.Button(self, text="inf",command=popupmsg)
        button1.grid(row=4,column=6, padx=10, pady=10)
        button1 = tk.Button(self, text="inf",command=popupmsg)
        button1.grid(row=5,column=2, padx=10, pady=10)
        button1 = tk.Button(self, text="inf",command=popupmsg)
        button1.grid(row=5,column=6, padx=10, pady=10)
        button1 = tk.Button(self, text="inf",command=popupmsg)
        button1.grid(row=6,column=2, padx=10, pady=10)
        button1 = tk.Button(self, text="inf",command=popupmsg)
        button1.grid(row=6,column=6, padx=10, pady=10)
        button1 = tk.Button(self, text="inf",command=popupmsg)
        button1.grid(row=7,column=2, padx=10, pady=10)
        button1 = tk.Button(self, text="inf",command=popupmsg)
        button1.grid(row=7,column=6, padx=10, pady=10)
        button1 = tk.Button(self, text="inf",command=popupmsg)
        button1.grid(row=8,column=2, padx=10, pady=10)
        button1 = tk.Button(self, text="inf",command=popupmsg)
        button1.grid(row=8,column=6, padx=10, pady=10)



class LoadPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        

        label = tk.Label(self, text="Generating Rreport...", font = LARGE_FONT)
        label.pack(pady=40,padx=40)
        
        button = tk.Button(self, text="Show Result!",font=("Helvetica", 16, "bold"),
                            command=  lambda: controller.show_frame(EndPage))
        
        button.pack(pady=20,padx=20)



class EndPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)

        label = tk.Label(self, text="92%", font = LARGE_FONT)
        label.pack(pady=40,padx=40)

        label1 = tk.Label(self, text="This is the probability of the\npatient having liver disease", font = ("Helvetica", 16))
        label1.pack(pady=20,padx=20)

        button1 = tk.Button(self, text="Again",font=("Helvetica", 16, "bold"),
                            command=lambda: controller.show_frame(PageOne))
        button1.pack(padx=20,pady=20)


        
        
        

        
app = gui()
app.mainloop()












        
