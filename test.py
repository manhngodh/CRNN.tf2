import os
import shutil
import time
import tensorflow as tf

import cv2

from model import vgg_style


def split_image():
    image_names = os.listdir("data/src_images")
    src_path = 'data/src_annotation.txt'
    dst_path = "data/annotation.txt"
    try:
        shutil.rmtree("data/images")
    except Exception as e:
        print(e)
    os.makedirs("data/images", exist_ok=True)
    with open(src_path, "r", encoding="utf8") as src_file:
        print('[image path] label')
        content = [l.strip('\n').split(" ", 1) for l in src_file.readlines()]
        img_paths, labels = zip(*content)
        dirname = os.path.dirname(src_path)
        img_paths = [os.path.join(dirname, 'src_images', img_path) for img_path in img_paths]
        with open(dst_path, "w", encoding="utf8") as dst_file:
            for img_path, label in zip(img_paths, labels):
                start_view = 840 + 420
                image = cv2.imread(img_path)
                ratio = image.shape[0] / image.shape[1]

                image_src = cv2.resize(image, (840, int(840 * ratio)))

                start_view = start_view + 420
                dst_img_path = f'data/images/{round(time.time() * 1000000)}.jpg'
                image = image_src[start_view:start_view + 840, :]
                cv2.imwrite(dst_img_path, image)
                dst_file.write(f'{dst_img_path} {label}\n')

                start_view += 420
                dst_img_path = f'data/images/{round(time.time() * 1000000)}.jpg'
                image = image_src[start_view:start_view + 840, :]
                cv2.imwrite(dst_img_path, image)
                dst_file.write(f'{dst_img_path} {label}\n')


def hash_table():
    letters = " #'\"()[]+,-./:0123456789ABCDEFGHIJKLMNOPQRSTUVWXYabcdeghiklmnopqrstuvxyzÂÊÔàáâãèéêìíòóôõùúýăĐđĩũƠơưạảấầẩậắằẵặẻẽếềểễệỉịọỏốồổỗộớờởỡợụủỨứừửữựỳỵỷỹ"
    # labels = '["Bán căn hộ chung cư","162m²","Số 253 đường Trần Phú, Thị trấn Nam Sách, Huyện Nam Sách, Hải Dương","0777777284","12-07-2020","22-07-2020"]'
    # x=''.join(list(map(lambda x: letters[x] if x < len(letters) else "", labels)))
    # print(x)
    letters = list(letters)
    print(letters)
    indices = tf.range(len(letters), dtype=tf.int64)
    table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(letters, indices), -1)
    print(table.lookup(tf.constant([" "])))


def build_model(num_classes, image_width=None, image_height=None, channels=1):
    """build CNN-RNN model"""

    img_input = tf.keras.Input(shape=(image_height, image_width, channels))
    backbone = tf.keras.applications.ResNet50V2(
        include_top=False, weights='imagenet', input_tensor=img_input
    )
    # x = backbone(img_input)
    # x = vgg_style(img_input)
    x = tf.keras.layers.Reshape((-1, 2048))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=256, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=256, return_sequences=True))(x)
    x = tf.keras.layers.Dense(units=num_classes)(x)
    return tf.keras.Model(inputs=img_input, outputs=x, name='CRNN')


if __name__ == '__main__':
    # backbone = tf.keras.applications.ResNet50V2(
    #     input_shape=(840, 840, 3),
    #     include_top=False, weights='imagenet'
    # )
    # img_input = tf.keras.Input(shape=(840, 840, 3))
    # out = vgg_style(img_input)
    # model = tf.keras.Model(inputs=img_input, outputs=out)
    # backbone.summary()
    split_image()
