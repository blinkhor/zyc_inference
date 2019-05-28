from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import Model
from Label import get_files


# --------------------------------------------------------------------
# 获取一张图片
def get_one_image(train):
    # 输入参数：train,训练图片的路径
    # 返回参数：image，从训练图片中随机抽取一张图片
    n = len(train)
    ind = np.random.randint(0, n)
    img_dir = train[ind]  # 随机选择测试的图片

    img = Image.open(img_dir)
    plt.imshow(img)
    imag = img.resize([128, 128])
    image = np.array(imag)
    return image


# --------------------------------------------------------------------
# 测试图片
def evaluate_one_image(image_array):
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 10

        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 128, 128, 3])

        logit = Model.inference(image, BATCH_SIZE, N_CLASSES)

        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[128, 128, 3])

        logs_train_dir = 'E:/Medical_train_logs'

        saver = tf.train.Saver()

        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')

            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)  # 分类
            if max_index == 0:
                print('This is a baihe(百合) with possibility %.6f' % prediction[:, 0])
            elif max_index == 1:
                print('This is a biejia(鳖甲) with possibility %.6f' % prediction[:, 1])
            elif max_index == 2:
                print('This is a caoguo(草果) with possibility %.6f' % prediction[:, 2])
            elif max_index == 3:
                print('This is a chantui(蝉蜕) with possibility %.6f' % prediction[:, 3])
            elif max_index == 4:
                print('This is a diyu(地榆) with possibility %.6f' % prediction[:, 4])
            elif max_index == 5:
                print('This is a jinyinhua(金银花) with possibility %.6f' % prediction[:, 5])
            elif max_index == 6:
                print('This is a jinjie(荆芥) with possibility %.6f' % prediction[:, 6])
            elif max_index == 7:
                print('This is a tongcao(通草) with possibility %.6f' % prediction[:, 7])
            elif max_index == 8:
                print('This is a wulingzhi(五灵脂) with possibility %.6f' % prediction[:, 8])
            else:
                print('This is a xixin(细辛) with possibility %.6f' % prediction[:, 9])


# ------------------------------------------------------------------------

if __name__ == '__main__':
    train_dir = 'E:/Medical_train_image_data/input_data/'
    train, train_label, test, test_label = get_files(train_dir, 0.3)
    img = get_one_image(test)  # 通过改变参数train or val，进而验证训练集或测试集
    evaluate_one_image(img)
