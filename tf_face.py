import os
import logging as log
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import cv2

SIZE = 64
x_data = tf.placeholder(tf.float32, [None, SIZE, SIZE, 3])#4维  总图片数量 * 图片宽 * 高 *3（颜色通道） #传入值
y_data = tf.placeholder(tf.float32, [None, None])         #2维 不指定行和列

keep_prob_5 = tf.placeholder(tf.float32)                  #1维
keep_prob_75 = tf.placeholder(tf.float32)

def weightVariable(shape):
    init = tf.random_normal(shape, stddev=0.01) #从符合
    return tf.Variable(init)        #传入变量

def biasVariable(shape):
    init = tf.random_normal(shape)
    return tf.Variable(init)

def conv2d(x, W):
    #卷积
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') #步长为1

def maxPool(x):
    #最大池化
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def dropout(x, keep):
    #防止过拟合
    return tf.nn.dropout(x, keep)

def cnnLayer(classnum):
    ''' create cnn layer'''
    # 第一层
    W1 = weightVariable([3, 3, 3, 32]) # 卷积核大小(3,3)， 输入通道(3)， 输出通道(32)
    b1 = biasVariable([32])            #
    conv1 = tf.nn.relu(conv2d(x_data, W1) + b1) #返回64*64*32   x_data=tf.placeholder(tf.float32, [None, SIZE, SIZE, 3])
    pool1 = maxPool(conv1)                      #步长为2 所以维度缩小了一半 32*32*32
    # 减少过拟合，随机让某些权重不更新
    drop1 = dropout(pool1, keep_prob_5) # 32 * 32 * 32 多个输入channel 被filter内积掉了

    # 第二层
    W2 = weightVariable([3, 3, 32, 64])
    b2 = biasVariable([64])
    conv2 = tf.nn.relu(conv2d(drop1, W2) + b2)  #32*32*64
    pool2 = maxPool(conv2)
    drop2 = dropout(pool2, keep_prob_5) # 64 * 16 * 16

    # 第三层
    W3 = weightVariable([3, 3, 64, 64])
    b3 = biasVariable([64])
    conv3 = tf.nn.relu(conv2d(drop2, W3) + b3)  #16*16*64
    pool3 = maxPool(conv3)
    drop3 = dropout(pool3, keep_prob_5) # 64 * 8 * 8

    # 全连接层
    Wf = weightVariable([8*16*32, 512])
    bf = biasVariable([512])
    drop3_flat = tf.reshape(drop3, [-1, 8*16*32])   #转1维 1*(8*8*64
    dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)      #matmul矩阵相乘  tf.nn.relu大于0的数不变其他的变成0  1*512
    dropf = dropout(dense, keep_prob_75)

    # 输出层
    Wout = weightVariable([512, classnum])  #512,4
    bout = biasVariable([classnum])
    #out = tf.matmul(dropf, Wout) + bout
    out = tf.add(tf.matmul(dropf, Wout), bout)      #1x4
    return out

def train(train_x, train_y, tfsavepath):
    #训练
    out = cnnLayer(train_y.shape[1])    #图片集列表最后一个有几个集就是几
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y_data))#对所有损失值求平均
    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)   #最速下降法让交叉熵下降，步长为0.01
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y_data, 1)), tf.float32))   #equal相等返回true   cast数据类型转换

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) #初始化模型
        batch_size = 10
        num_batch = len(train_x) // 10
        for n in range(10):
            r = np.random.permutation(len(train_x))     #返回一个新的打乱的数组，不改变原数组
            train_x = train_x[r, :]
            train_y = train_y[r, :]

            for i in range(num_batch):
                batch_x = train_x[i*batch_size : (i+1)*batch_size]
                batch_y = train_y[i*batch_size : (i+1)*batch_size]
                _, loss = sess.run([train_step, cross_entropy],\
                                   feed_dict={x_data:batch_x, y_data:batch_y,
                                              keep_prob_5:0.75, keep_prob_75:0.75})

                print(n*num_batch+i, loss)

        # 获取测试数据的准确率
        acc = accuracy.eval({x_data:train_x, y_data:train_y, keep_prob_5:1.0, keep_prob_75:1.0})
        print('after 10 times run: accuracy is ', acc)
        saver.save(sess, tfsavepath)

def validate(test_x, tfsavepath):
    ''' validate '''
    output = cnnLayer(2)
    predict = output

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tfsavepath)
        res = sess.run([predict, tf.argmax(output, 1)],
                       feed_dict={x_data: test_x,
                                  keep_prob_5:1.0, keep_prob_75: 1.0})
        return res

if __name__ == '__main__':
    pass