import numpy as np
import pandas as pd
import tensorflow as tf
%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
import math
import idx2numpy

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置全局字体为微软雅黑(避免出现某些中文显示不了的情况)

train_images = idx2numpy.convert_from_file(r'D:\OneDrive\桌面\database\train-images.idx3-ubyte')

train_labels = idx2numpy.convert_from_file(r'D:\OneDrive\桌面\database\train-labels.idx1-ubyte')

test_images = idx2numpy.convert_from_file(r'D:\OneDrive\桌面\database\t10k-images.idx3-ubyte')

test_labels = idx2numpy.convert_from_file(r'D:\OneDrive\桌面\database\t10k-labels.idx1-ubyte')

np.random.seed(1)
train_images,test_images = train_images/255,test_images/255

index=np.arange(len(train_images))
# index中的数据被打乱
np.random.shuffle(index)

valid_images,valid_labels = train_images[index[-10000:]],train_labels[index[-10000:]]#验证数据
train_images,train_labels=train_images[index[:50000]],train_labels[index[:50000]]#训练数据

#建立全连接神经网络
mnist_model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),\
                                           tf.keras.layers.Dense(128,activation='relu'),\
                                           tf.keras.layers.Dense(256,activation='relu'),\
                                           tf.keras.layers.Dense(10,activation='softmax')])

#设置优化方法，损失函数和评价指标
mnist_model.compile(optimizer=tf.keras.optimizers.Adam(),\
                    loss='sparse_categorical_crossentropy',\
                    metrics=['accuracy'])

#使用mnist_model.fit()训练模型,用mnist_model.evaluate()计算测试准确率
mnist_model_history = mnist_model.fit(train_images,train_labels,epochs=20,verbose=0,validation_data=(valid_images,valid_labels),batch_size=128)
mnist_model.evaluate(test_images,test_labels)

#作出准确率随着每一次迭代的情况
plt.plot(mnist_model_history.epoch,\
        mnist_model_history.history['accuracy'],\
        label="训练准确率")
plt.plot(mnist_model_history.epoch,\
        mnist_model_history.history['val_accuracy'],\
        label="验证准确率")

plt.xlabel("迭代次数",fontsize=16)
plt.ylabel("预测准确率",fontsize=16)
_ =plt.legend(fontsize=16)







