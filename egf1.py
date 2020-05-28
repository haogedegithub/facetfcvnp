import numpy as np
import cv2
import os


class EigenFace(object):
    def __init__(self, threshold, dimNum, dsize):
        self.threshold = threshold  # 阈值暂未使用

        self.dimNum = dimNum
        self.dsize = dsize
        print ('thresh',threshold)
    def loadImg(self, fileName, dsize):
        '''''
        载入图像，灰度化处理，统一尺寸，直方图均衡化
        :param fileName: 图像文件名
        :param dsize: 统一尺寸大小。元组形式
        :return: 图像矩阵
        '''

        img = cv2.imread(fileName)
        retImg = cv2.resize(img, dsize)#将相近的几个像素点相加乘以所占比例
        #print('retImg3',retImg)
        retImg = cv2.cvtColor(retImg, cv2.COLOR_RGB2GRAY)# r，g，b=0.299*R +0.587*g + 0.114*B
        #print('retImg2', retImg)
        retImg = cv2.equalizeHist(retImg)               #直方图均衡化
        #print('retImg',retImg)
        #cv2.imshow('img',retImg)
        #cv2.waitKey()
        #print('dsize', dsize)
        return retImg

    def createImgMat(self, dirName):
        '''''
        生成图像样本矩阵，组织形式为行为属性，列为样本
        :param dirName: 包含训练数据集的图像文件夹路径
        :return: 样本矩阵，标签矩阵
        '''

        dataMat = np.zeros((10, 1))         #10行1列的二维数组
        label = []
        for parent, dirnames, filenames in os.walk(dirName):
            '''
            :parent:包含训练数据集图像的文件夹路径
            :dirnames：各个分类的名称
            :filenames：各图片
            '''
            print (parent)
            print (dirnames)
            print (filenames)
            index = 0
            for dirname in dirnames:
                for subParent, subDirName, subFilenames in os.walk(parent + '/' + dirname):
                    for filename in subFilenames:
                        img = self.loadImg(subParent + '/' + filename, self.dsize)
                        tempImg = np.reshape(img, (-1, 1))               #转换为n*1数组
                        #print('reshape',tempImg,tempImg.shape,tempImg.ndim,filename)
                        if index == 0:
                            dataMat = tempImg
                        else:
                            dataMat = np.column_stack((dataMat, tempImg)) #colum_stack行不变列扩展
                            label.append(subParent + '/' + filename)
                        index += 1
        return dataMat, label

    def PCA(self, dataMat, dimNum):
        '''''
        PCA函数，用于数据降维
        :param dataMat: 样本矩阵
        :param dimNum: 降维后的目标维度
        :return: 降维后的样本矩阵和变换矩阵
        '''

        # 均值化矩阵
        print('dataMat',dataMat,dataMat.shape)
        meanMat = np.mat(np.mean(dataMat, 1)).T
        # a = meanMat.reshape(50,50)
        # cv2.imshow('a',a)
        # cv2.waitKey()
        print ('平均值矩阵维度', meanMat.shape,meanMat)
        diffMat = dataMat - meanMat
        print ('diffMat',diffMat,diffMat.shape,diffMat.shape[1])
        # 求协方差矩阵，由于样本维度远远大于样本数目，所以不直接求协方差矩阵，采用下面的方法
        covMat = (diffMat.T * diffMat) / float(diffMat.shape[1])  # 归一化
        covMat2 = np.cov(dataMat,bias=True)
        print ('基本方法计算协方差矩阵为',covMat2,covMat2.shape)
        print ('协方差矩阵维度', covMat,covMat.shape)
        eigVals, eigVects = np.linalg.eig (np.mat(covMat))
        print ('特征向量维度', eigVects.shape)

        print ('特征值', eigVals,eigVals.shape)
        print ('特征向量', eigVects,eigVects.shape)
        eigVects = diffMat * eigVects                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
        print('#'*20,eigVects,eigVects.shape,eigVects.reshape)
        eigValInd = np.argsort(eigVals)#索引值从小到大排序
        #print('eigValInd',eigValInd)
        eigValInd = eigValInd[::-1]
        #print ('eivvvvvv',eigValInd)
        eigValInd = eigValInd[:dimNum]  # 取出指定个数的前n大的特征值
        print ('选取的特征值', eigValInd)
        eigVects = eigVects / np.linalg.norm(eigVects, axis=0)  # 归一化特征向量
        redEigVects = eigVects[:, eigValInd]
        print ('归一化后的特征向量', eigVects,eigVects.shape)
        print ('均值矩阵',diffMat, diffMat.shape)
        lowMat = redEigVects.T * diffMat
        print ('低 维矩阵',lowMat, lowMat.shape)
        print ('dimnum', dimNum,redEigVects)
        return lowMat, redEigVects
    def compare(self, dataMat, testImg, label):
        '''''
        比较函数，这里用了最简单的欧氏距离比较
        :param dataMat: 样本矩阵
        param testImg: 测试图像矩阵，最原始形式
        :param label: 标签矩阵
        :return: 与测试图片最相近的图像文件名
        '''

        testImg = cv2.resize(testImg, self.dsize)
        testImg = cv2.cvtColor(testImg, cv2.COLOR_RGB2GRAY)
        testImg = np.reshape(testImg, (-1, 1))
        lowMat, redVects = self.PCA(dataMat, self.dimNum)
        testImg = redVects.T * testImg
        print ('检测样本变换后的维度', testImg.shape)
        disList = []
        testVec = np.reshape(testImg, (1, -1)) #变换为1*n的矩阵
        for sample in lowMat.T:
            disList.append(np.linalg.norm(testVec - sample))
        print('#'*20)
        print ('disList',disList)
        sortIndex = np.argsort(disList)
        print('sortIndex',sortIndex)
        return label[sortIndex[0]]

    def predict(self, dirName, testFileName):
        '''''
        预测函数
        :param dirName: 包含训练数据集的文件夹路径
        :param testFileName: 测试图像文件名
        :return: 预测结果
        '''

        testImg = cv2.imread(testFileName)
        dataMat, label = self.createImgMat(dirName)
        print ('加载图片标签', label)
        ans = self.compare(dataMat, testImg, label)
        return ans


if __name__ == '__main__':
    eigenface = EigenFace(20, 50, (50, 50)) #阈值  降维后维度  降维后
    # 图片大小
    print (eigenface.predict('F:/Disface/face_data', 'F:/Disface/face_data/Nigen/1559544513122.jpg'))