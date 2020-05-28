import tkinter as tk
import getface
import os
import index
import tranmodal
import dis_face
window = tk.Tk()
window.title('人工**')
window.geometry('740x400')
# 新建数据集
def create():
    def folder():
        vyn = enter1.get()
        if vyn == '':
            print ('kong')
        else:
            getface.mkdir(vyn)
            getface.generate(r'F:\Disface\image\trainfaces/' + vyn)
    window1 = tk.Toplevel(window)
    window1.title('输入姓名开始保存')
    window1.geometry('100x100')

    var_your_name = tk.StringVar()
    enter1 = tk.Entry(window1,textvariable = var_your_name)
    enter1.place(x = 20, y = 20)
    btn2 = tk.Button(window1,text = '新建数据集',command = folder)
    btn2.place(x = 50, y = 50)

# 这里是窗口的内容
img_gif = tk.PhotoImage(file = 'abcdef.gif')
label_img = tk.Label(image = img_gif)
label_img.pack(side = 'left')

bth1 = tk.Button(window,
    text='新建数据集',      # 显示在按钮上的文字
    height = 3,
    width = 18,
    command=create)     # 点击按钮式执行的命令
bth1.place(x = 603,y =20)    # 按钮位置


bth2 = tk.Button(window,
    text='图片识别',      # 显示在按钮上的文字
    height=3,
    width=18,
    command=index.img)     # 点击按钮式执行的命令
bth2.place(x = 603,y = 100)    # 按钮位置

bth3 = tk.Button(window,
    text='训练数据集',      # 显示在按钮上的文字
    height=3,
    width=18,
    command=tranmodal.mainm)     # 点击按钮式执行的命令
bth3.place(x = 603,y = 180)    # 按钮位置

bth4 = tk.Button(window,
    text='动态识别',      # 显示在按钮上的文字
    height=3,
    width=18,
    command=dis_face.mainm)     # 点击按钮式执行的命令
bth4.place(x = 603,y = 260)    # 按钮位置

window.mainloop()
