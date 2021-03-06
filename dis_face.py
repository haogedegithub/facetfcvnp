import os
import logging as log
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import cv2
import tf_face as myconv


def createdir(*args):

    for item in args:
        if not os.path.exists(item):  # 判断是否存在
            os.makedirs(item)  # 递归文件目录


IMGSIZE = 64


def getpaddingSize(shape):

    h, w = shape
    longest = max(h, w)
    result = (np.array([longest] * 4, int) - np.array([h, h, w, w], int)) // 2
    return result.tolist()


def dealwithimage(img, h=64, w=64):

    top, bottom, left, right = getpaddingSize(img.shape[0:2])
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img = cv2.resize(img, (h, w))
    return img


def relight(imgsrc, alpha=1, bias=0):

    imgsrc = imgsrc.astype(float)
    imgsrc = imgsrc * alpha + bias
    imgsrc[imgsrc < 0] = 0
    imgsrc[imgsrc > 255] = 255
    imgsrc = imgsrc.astype(np.uint8)
    return imgsrc


def getface(imgpath, outdir):

    filename = os.path.splitext(os.path.basename(imgpath))[0]
    img = cv2.imread(imgpath)
    haar = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray_img, 1.3, 5)
    n = 0
    for f_x, f_y, f_w, f_h in faces:
        n += 1
        face = img[f_y:f_y + f_h, f_x:f_x + f_w]

        face = dealwithimage(face, IMGSIZE, IMGSIZE)
        for inx, (alpha, bias) in enumerate([[1, 1], [1, 50], [0.5, 0]]):
            facetemp = relight(face, alpha, bias)
            cv2.imwrite(os.path.join(outdir, '%s_%d_%d.jpg' % (filename, n, inx)), facetemp)


def getfilesinpath(filedir):

    for (path, dirnames, filenames) in os.walk(filedir):
        for filename in filenames:
            if filename.endswith('.jpg'):
                yield os.path.join(path, filename)
        for diritem in dirnames:
            getfilesinpath(os.path.join(path, diritem))


def generateface(pairdirs):

    for inputdir, outputdir in pairdirs:
        for name in os.listdir(inputdir):
            inputname, outputname = os.path.join(inputdir, name), os.path.join(outputdir, name)
            if os.path.isdir(inputname):
                createdir(outputname)
                for fileitem in getfilesinpath(inputname):
                    getface(fileitem, outputname)


def readimage(pairpathlabel):

    imgs = []
    labels = []
    for filepath, label in pairpathlabel:
        for fileitem in getfilesinpath(filepath):
            img = cv2.imread(fileitem)
            imgs.append(img)
            labels.append(label)
    return np.array(imgs), np.array(labels)


def onehot(numlist):

    b = np.zeros([len(numlist), max(numlist) + 1])
    b[np.arange(len(numlist)), numlist] = 1
    return b.tolist()


def getfileandlabel(filedir):

    dictdir = dict([[name, os.path.join(filedir, name)] \
                    for name in os.listdir(filedir) if os.path.isdir(os.path.join(filedir, name))])

    dirnamelist, dirpathlist = dictdir.keys(), dictdir.values()
    indexlist = list(range(len(dirnamelist)))

    return list(zip(dirpathlist, onehot(indexlist))), dict(zip(indexlist, dirnamelist))


def mainm():
    savepath = './checkpoint/face.ckpt'
    tesfromcamera(savepath)

def tesfromcamera(chkpoint):
    camera = cv2.VideoCapture(0)
    haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    pathlabelpair, indextoname = getfileandlabel('./image/trainfaces')
    output = myconv.cnnLayer(len(pathlabelpair))
    predict = output

    saver = tf.train.Saver()
    with tf.Session() as sess:
        #读取chkpoint
        saver.restore(sess, chkpoint)

        n = 1
        while 1:
            if (n <= 20000):
                success, img = camera.read()

                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = haar.detectMultiScale(gray_img, 1.3, 5)
                for f_x, f_y, f_w, f_h in faces:
                    face = img[f_y:f_y + f_h, f_x:f_x + f_w]
                    face = cv2.resize(face, (IMGSIZE, IMGSIZE))
                    test_x = np.array([face])
                    test_x = test_x.astype(np.float32) / 255.0

                    res = sess.run([predict, tf.argmax(output, 1)], \
                                   feed_dict={myconv.x_data: test_x, \
                                              myconv.keep_prob_5: 1.0, myconv.keep_prob_75: 1.0})
                    print(res)

                    cv2.putText(img, indextoname[res[1][0]], (f_x, f_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255,
                                2)  # 显示名字
                    img = cv2.rectangle(img, (f_x, f_y), (f_x + f_w, f_y + f_h), (255, 0, 0), 2)
                    n += 1
                cv2.imshow('img', img)
                if cv2.waitKey(100) & 0xff == ord("q"):
                    break
            else:
                break
    camera.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':

    mainm()
