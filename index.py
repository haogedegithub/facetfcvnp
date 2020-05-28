import tkinter as tk
import tkinter.filedialog
from PIL import Image, ImageTk
import re
import egf1
def img():
    top = tk.Toplevel()
    top.title = 'new'
    top.geometry('350x400')
    def choose_fiel():
        selectFileName = tkinter.filedialog.askopenfilename(title='选择文件')  # 选择文件
        e.set(selectFileName)
        load = Image.open(e.get())
        render = ImageTk.PhotoImage(load)
        img = tk.Label(top,image=render)
        img.image = render
        img.place(x=200, y=100)
        egf = egf1.EigenFace(20,50,(50,50))
        strg = egf.predict('F:/Disface/image/trainfaces/',e.get())
        j = re.findall(r"F:/Disface/image/trainfaces/(.+?)/", strg)  #正则匹配
        lab = tk.Label(top, text=j)
        lab.pack()
    e = tk.StringVar()
    e_entry = tk.Entry(top, width=68, textvariable=e)
    e_entry.pack()

    submit_button = tk.Button(top, text ="选择文件", command = choose_fiel)
    submit_button.pack()
    top.mainloop()