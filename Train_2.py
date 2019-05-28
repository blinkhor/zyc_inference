import os
import numpy as np
import tensorflow as tf
from time import time
from Label import get_files, get_batch
from Model import inference, loss_function, a_optimizer, evaluation
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# --------------------------------------------------------------------
# 变量声明
n_class = 10  # 10分类
img_W = 128  # 图像的宽度
img_H = 128  # 图像的高度
batch_size = 20  # 单次训练样本数
total_batch = 80  # 训练图像总数20*80=1600，测试图像总数400
display_step = 100  # 显示粒度
capacity = 200  # 2000张图片
train_epochs = 1000  # 迭代次数
learning_rate = 0.0001  # 学习率
startTime = time()  # 记录训练开始的时间
epoch_list = []  # 迭代次数列表（用于数据可视化）
accuracy_list = []  # 正确率列表（用于数据可视化）
loss_list = []  # 损失值列表（用于数据可视化）
epoch = tf.Variable(0, name='epoch', trainable=False)  # 迭代次数
train_dir = 'E:/Medical_train_image_data/input_data'  # 训练样本的读入路径
logs_train_dir = 'E:/Medical_train_logs'  # logs日志存储路径


# --------------------------------------------------------------------
# 获取文件files
train, train_label, test, test_label = get_files(train_dir, 0.8)

# # 训练操作定义
# x = tf.placeholder(dtype='float32', shape=[20, 128, 128, 3], name="x")
# y = tf.placeholder(dtype='int32', shape=[20, ], name="y")
#
# train_logits = inference(x, batch_size, n_class)  # 训练集 神经网络输出层的输出
# train_loss = loss_function(train_logits, y)  # 训练集 loss计算
# train_op = a_optimizer(train_loss, learning_rate)  # 选择optimizer优化器adam
# train_acc = evaluation(train_logits, y)  # 训练集 准确率计算
# #  测试操作定义
# test_logits = inference(x, batch_size, n_class)  # 测试集 神经网络输出层的输出
# test_loss = loss_function(test_logits, y)  # 测试集 loss计算
# test_acc = evaluation(test_logits, y)  # 测试集 准确率计算


def get_next_batch(a):
    # 获取训练数据及标签
    train_b, train_label_b = get_batch(train, train_label, img_W, img_H, batch_size, capacity, a)
    # 获取测试数据及标签
    test_b, test_label_b = get_batch(test, test_label, img_W, img_H, batch_size, capacity, a)
    print("reading batch...")
    #  训练操作定义
    train_logits = inference(train_b, batch_size, n_class)  # 训练集 神经网络输出层的输出
    train_loss = loss_function(train_logits, train_label_b)  # 训练集 loss计算
    train_op = a_optimizer(train_loss, learning_rate)  # 选择optimizer优化器adam
    train_acc = evaluation(train_logits, train_label_b)  # 训练集 准确率计算
    #  测试操作定义
    test_logits = inference(test_b, batch_size, n_class)  # 测试集 神经网络输出层的输出
    test_loss = loss_function(test_logits, test_label_b)  # 测试集 loss计算
    test_acc = evaluation(test_logits, test_label_b)  # 测试集 准确率计算
    print("loading batch...")
    return train_loss, train_op, train_acc, test_loss, test_acc


# 加载会话
sess = tf.Session()
sess.run(epoch.initializer)
sess.run(tf.local_variables_initializer())
# log汇总记录
summary_op = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)

# 断点续训
checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
saver = tf.train.Saver(max_to_keep=1)
# ckpt = tf.train.latest_checkpoint(checkpoint_path)
ckpt = tf.train.get_checkpoint_state(logs_train_dir)
if ckpt and ckpt.model_checkpoint_path:
    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('Loading success, global_step is %s' % global_step)
else:
    print("Training from scratch")
start = sess.run(epoch)  # 获取续训参数
print("Training starts from {} epoch.".format(start+1))

# 队列监控
coord = tf.train.Coordinator()  # 设置多线程协调器
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# --------------------------------------------------------------------
# 迭代训练
try:
    for ep in np.arange(start, train_epochs):
        if coord.should_stop():
            break
        a, b, c, d, e = get_next_batch(ep % 80)
        _, tra_loss, tra_acc = sess.run([b, a, c])
        epoch_list.append(ep+1)
        loss_list.append(tra_loss)
        accuracy_list.append(tra_acc)
        # 100步输出一次损失率与当前准确率
        if ep % display_step == 0 and ep != 0:
            print("Epoch {}".format(ep)+" finished!")
            print('train loss = %.2f, train accuracy = %.2f' % (tra_loss, tra_acc))
            summary_str = sess.run(summary_op)
            train_writer.add_summary(summary_str, ep)
        # 保存最后一次的网络参数
        sess.run(epoch.assign(ep+1))
        if ep == train_epochs:
            saver.save(sess, checkpoint_path, global_step=ep)
    duration = time()-startTime
    print('Train Finished, takes : ', '{:.2f}'.format(duration) + 's')

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')

finally:
    coord.request_stop()
coord.join(threads)
sess.close()

