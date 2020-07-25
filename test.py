import os
import time
import tensorflow as tf

import cv2


def split_image(image):
    image_names = os.listdir("our_data/images_src")
    src_path = 'our_data/src_annotation.txt'
    dst_path = "our_data/annotation.txt"
    with open(src_path, "r") as src_file:
        print('[image path] label')
        content = [l.strip('\n').split(" ", 1) for l in src_file.readlines()]
        img_paths, labels = zip(*content)
        dirname = os.path.dirname(src_path)
        img_paths = [os.path.join(dirname, 'images_src', img_path) for img_path in img_paths]
        with open(dst_path, "w") as dst_file:
            for img_path, label in zip(img_paths, labels):
                image = cv2.imread(img_path)
                ratio = image.shape[0] / image.shape[1]

                image_src = cv2.resize(image, (840, int(840 * ratio)))

                start = 840 + 420
                dst_img_path = f'our_data/images/{round(time.time() * 1000000)}.jpg'
                image = image_src[start:start + 840, :]
                cv2.imwrite(dst_img_path, image)
                dst_file.write(f'{dst_img_path} {label}\n')

                start += 420
                dst_img_path = f'our_data/images/{round(time.time() * 1000000)}.jpg'
                image = image_src[start:start + 840, :]
                cv2.imwrite(dst_img_path, image)
                dst_file.write(f'{dst_img_path} {label}\n')


if __name__ == '__main__':
    letters = " #'\"()[]+,-./:0123456789ABCDEFGHIJKLMNOPQRSTUVWXYabcdeghiklmnopqrstuvxyzÂÊÔàáâãèéêìíòóôõùúýăĐđĩũƠơưạảấầẩậắằẵặẻẽếềểễệỉịọỏốồổỗộớờởỡợụủỨứừửữựỳỵỷỹ"
    # labels = '["Bán căn hộ chung cư","162m²","Số 253 đường Trần Phú, Thị trấn Nam Sách, Huyện Nam Sách, Hải Dương","0777777284","12-07-2020","22-07-2020"]'
    # x=''.join(list(map(lambda x: letters[x] if x < len(letters) else "", labels)))
    # print(x)
    letters = list(letters)
    print(letters)
    indices = tf.range(len(letters), dtype=tf.int64)
    table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(letters, indices), -1)
    print(table.lookup(tf.constant([" "])))
