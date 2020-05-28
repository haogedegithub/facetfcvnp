import cv2
import os
import time
import random
def generate(dirname):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # 创建目录
    if (not os.path.isdir(dirname)):
        os.makedirs(dirname)
        # 打开摄像头进行人脸图像采集
    camera = cv2.VideoCapture(0)
    count = 0
    while (True):
        if (count<199):
            ret, frame = camera.read()

            faces = face_cascade.detectMultiScale(frame, 1.3, 5)#1.3表示在前后两次相继的扫描中，搜索窗口的比例系数。默认为1.1即每次搜索窗口依次扩大10%;5表示构成检测目标的相邻矩形的最小个数(默认为3个)
            for (x, y, w, h) in faces:
                img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                listStr = [str(int(time.time())), str(count)]  # 以时间戳和读取的排序作为文件名称
                fileName = ''.join(listStr)
                f = cv2.resize(frame[y:y + h, x:x + w], (64, 64))
                cv2.imwrite(dirname + os.sep + '%s.jpg' % fileName, f)
                count += 1
            cv2.imshow("camera", frame)
            if cv2.waitKey(100) & 0xff == ord("q"):
                break
        else:
            break
    camera.release()
    cv2.destroyAllWindows()


def mkdir(path):
    folder = os.path.exists(r'F:\\image\trainfaces/' + path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(r'F:\Disface\image\trainfaces/' + path)  # makedirs 创建文件时如果路径不存在会创建这个路径

    else:
        os.makedirs(r'F:\Disface\image\trainfaces/' + path +str(int(random.randint(0,5))) )

