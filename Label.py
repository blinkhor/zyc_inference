import os
import numpy as np
import tensorflow as tf
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

train_dir = 'E:/Medical_train_image_data/input_data/'

baihe = []
label_baihe = []

biejia = []
label_biejia = []

caoguo = []
label_caoguo = []

chantui = []
label_chantui = []

diyu = []
label_diyu = []

jinyinhua = []
label_jinyinhua = []

jinjie = []
label_jinjie = []

tongcao = []
label_tongcao = []

wulingzhi = []
label_wulingzhi = []

xixin = []
label_xixin = []


# --------------------------------------------------------------------
def get_files(file_dir, ratio):

    for file in os.listdir(file_dir + '/baihe'):
        baihe.append(file_dir + '/baihe' + '/' + file)
        label_baihe.append(0)
    for file in os.listdir(file_dir + '/biejia'):
        biejia.append(file_dir + '/biejia' + '/' + file)
        label_biejia.append(1)
    for file in os.listdir(file_dir + '/caoguo'):
        caoguo.append(file_dir + '/caoguo' + '/' + file)
        label_caoguo.append(2)
    for file in os.listdir(file_dir + '/chantui'):
        chantui.append(file_dir + '/chantui' + '/' + file)
        label_chantui.append(3)
    for file in os.listdir(file_dir + '/diyu'):
        diyu.append(file_dir + '/diyu' + '/' + file)
        label_diyu.append(4)
    for file in os.listdir(file_dir + '/jinyinhua'):
        jinyinhua.append(file_dir + '/jinyinhua' + '/' + file)
        label_jinyinhua.append(5)
    for file in os.listdir(file_dir + '/jinjie'):
        jinjie.append(file_dir + '/jinjie' + '/' + file)
        label_jinjie.append(6)
    for file in os.listdir(file_dir + '/tongcao'):
        tongcao.append(file_dir + '/tongcao' + '/' + file)
        label_tongcao.append(7)
    for file in os.listdir(file_dir + '/wulingzhi'):
        wulingzhi.append(file_dir + '/wulingzhi' + '/' + file)
        label_wulingzhi.append(8)
    for file in os.listdir(file_dir + '/xixin'):
        xixin.append(file_dir + '/xixin' + '/' + file)
        label_xixin.append(9)

    # print("There are %d 百合\nThere are %d 鳖甲\nThere are %d 草果\n""There are %d 蝉蜕\n"
    #       "There are %d 地榆\n" % (len(baihe), len(biejia), len(caoguo), len(chantui), len(diyu)), end="")
    # print("There are %d 金银花\nThere are %d 荆芥\nThere are %d 通草\n""There are %d 五灵脂\n"
    #       "There are %d 细辛\n" % (len(jinyinhua), len(jinjie), len(tongcao), len(wulingzhi), len(xixin)), end="")

    # 对生成的图片路径和标签List做打乱处理把所有的合起来组成一个list（img和lab）
    # 合并数据numpy.hstack(tup)
    # tup可以是python中的元组（tuple）、列表（list），或者numpy中数组（array），函数作用是将tup在水平方向上（按列顺序）合并
    image_list = np.hstack((baihe, biejia, caoguo, chantui, diyu, jinyinhua, jinjie, tongcao, wulingzhi, xixin))
    label_list = np.hstack((label_baihe, label_biejia, label_caoguo, label_chantui, label_diyu, label_jinyinhua,
                            label_jinjie, label_tongcao, label_wulingzhi, label_xixin))

    # 利用shuffle，转置、随机打乱
    temp = np.array([image_list, label_list])  # 转换成2维矩阵
    temp = temp.transpose()  # 转置：numpy.transpose(a, axes=None) 作用：将输入的array转置，并返回转置后的array
    np.random.shuffle(temp)  # 按行随机打乱顺序函数

    # 将所有的img和lab转换成list
    all_image_list = list(temp[:, 0])  # 取出第0列数据，即图片路径
    all_label_list = list(temp[:, 1])  # 取出第1列数据，即图片标签

    # 将所得List分为两部分，一部分用来训练train，一部分用来测试test
    ratio = ratio
    sum_sample = len(all_label_list)
    n_train = int(math.ceil(sum_sample * ratio))
    train_images = all_image_list[0:n_train]
    train_labels = all_label_list[0:n_train]
    train_labels = [int(float(i)) for i in train_labels]
    test_images = all_image_list[n_train:-1]
    test_labels = all_label_list[n_train:-1]
    test_labels = [int(float(i)) for i in test_labels]
    return train_images, train_labels, test_images, test_labels


# --------------------------------------------------------------------
def get_batch(image, label, image_W, image_H, batch_size, capacity):

    # step1：将上面生成的List传入get_batch() ，转换类型，产生一个输入队列queue
    # image = image[batch_size*(number-1):batch_size*number]
    # label = label[batch_size*(number-1):batch_size*number]

    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # tf.train.slice_input_producer是一个tensor生成器,作用是按照设定，每次从一个tensor列表中按顺序或者随机抽取出一个tensor放入文件名队列。
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])  # tf.read_file()从队列中读取图像

    # step2：将图像解码
    image = tf.image.decode_jpeg(image_contents, channels=3)  # jpeg或者jpg格式都用decode_jpeg函数

    # step3：数据预处理，对图像进行旋转、缩放、裁剪、归一化等操作
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)

    # 对resize后的图片进行标准化处理
    image = tf.image.per_image_standardization(image)

    # step4：生成batch
    # image_batch: 4D tensor [batch_size, width, height, 3], dtype = tf.float32
    # label_batch: 1D tensor [batch_size], dtype = tf.int32
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=32, capacity=capacity)

    # 重新排列label，行数为[batch_size]
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch


# def PreWork():
#     # 对预处理的数据进行可视化，查看预处理的效果
#     IMG_W = 128
#     IMG_H = 128
#     BATCH_SIZE = 10
#     CAPACITY = 64
#
#     train_image_list, train_label_list, test_image_list, test_label_list = get_files(train_dir, 0.5)
#     print(train_image_list)
#     print(train_label_list)
#     image_batch, label_batch = get_batch(train_image_list, train_label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
#     print(label_batch.shape)
#
#
# if __name__ == '__main__':
#     PreWork()
