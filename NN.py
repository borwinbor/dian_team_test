import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import idx2numpy
from sklearn.metrics import precision_score, recall_score, f1_score

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置全局字体为微软雅黑(避免出现某些中文显示不了的情况)

# 加载数据
train_images = idx2numpy.convert_from_file(r'D:\OneDrive\桌面\database\train-images.idx3-ubyte')
train_labels = idx2numpy.convert_from_file(r'D:\OneDrive\桌面\database\train-labels.idx1-ubyte')
test_images = idx2numpy.convert_from_file(r'D:\OneDrive\桌面\database\t10k-images.idx3-ubyte')
test_labels = idx2numpy.convert_from_file(r'D:\OneDrive\桌面\database\t10k-labels.idx1-ubyte')

# 数据预处理
np.random.seed(1)
train_images, test_images = train_images / 255, test_images / 255

index = np.arange(len(train_images))
np.random.shuffle(index)

valid_images, valid_labels = train_images[index[-10000:]], train_labels[index[-10000:]]  # 验证数据
train_images, train_labels = train_images[index[:50000]], train_labels[index[:50000]]  # 训练数据

# 建立全连接神经网络
mnist_model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 自定义评价指标函数
def precision(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
    precision_val = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision_val

def recall(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    actual_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    recall_val = true_positives / (actual_positives + tf.keras.backend.epsilon())
    return recall_val

def f1_score_val(y_true, y_pred):
    precision_val = precision(y_true, y_pred)
    recall_val = recall(y_true, y_pred)
    f1_score_val = 2 * ((precision_val * recall_val) / (precision_val + recall_val + tf.keras.backend.epsilon()))
    return f1_score_val

# 自定义回调函数来计算评价指标历史值
class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.precision_scores = []
        self.recall_scores = []
        self.f1_scores = []

    def on_epoch_end(self, epoch, logs=None):
        val_pred = np.argmax(self.model.predict(valid_images), axis=1)
        precision_val = precision_score(valid_labels, val_pred, average='weighted')
        recall_val = recall_score(valid_labels, val_pred, average='weighted')
        f1_val = f1_score(valid_labels, val_pred, average='weighted')
        
        self.precision_scores.append(precision_val)
        self.recall_scores.append(recall_val)
        self.f1_scores.append(f1_val)

# 编译模型时使用默认的评测指标，但添加自定义回调函数
mnist_model.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

# 创建并使用回调函数来计算评价指标历史值
metrics_callback = MetricsCallback()

# 使用自定义回调函数训练模型
mnist_model_history = mnist_model.fit(train_images, train_labels, epochs=20, verbose=0,
                                      validation_data=(valid_images, valid_labels), batch_size=128,
                                      callbacks=[metrics_callback])

# 获取评价指标历史值
precision_history = metrics_callback.precision_scores
recall_history = metrics_callback.recall_scores
f1_score_history = metrics_callback.f1_scores

# 输出评价指标历史值
print('Precision History:', precision_history)
print('Recall History:', recall_history)
print('F1 Score History:', f1_score_history)

# 作出准确率、精确率、召回率和 F1 值随迭代次数变化的曲线图
plt.figure(figsize=(10, 6))

plt.plot(mnist_model_history.epoch, mnist_model_history.history['accuracy'], label="训练准确率")
plt.plot(mnist_model_history.epoch, mnist_model_history.history['val_accuracy'], label="验证准确率")
plt.plot(mnist_model_history.epoch, precision_history, label="训练精确率")
plt.plot(mnist_model_history.epoch, recall_history, label="训练召回率")
plt.plot(mnist_model_history.epoch, f1_score_history, label="训练F1")

plt.xlabel("迭代次数", fontsize=16)
plt.ylabel("评价指标", fontsize=16)
plt.legend(fontsize=12)
plt.title("评价指标随迭代次数的变化", fontsize=18)
plt.show()
