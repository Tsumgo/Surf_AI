import numpy as np
import cv2
import time
import os
from utils.grabscreen import grab_screen, screen_mask, pixel_filter
from utils.getkeys import key_check

file_name = "./data/training_data.npy"
file_name2 = "./data/target_data.npy"

# 定义获取数据的函数
def get_data():

    if os.path.isfile(file_name):
        print('File exists, loading previous data!')
        image_data = list(np.load(file_name, allow_pickle=True))
        targets = list(np.load(file_name2, allow_pickle=True))
    else:
        print('File does not exist, starting fresh!')
        image_data = []
        targets = []
    return image_data, targets


def save_data(image_data, targets):
    np.save(file_name, image_data)
    np.save(file_name2, targets)


image_data, targets = get_data()
while True:
    keys = key_check()
    print("waiting press B to start")
    if keys == "B":
        print("Starting")
        break

count = 0

while True:
    count +=1
    last_time = time.time()
    image = grab_screen(region=(270, 600, 2286, 1299))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = screen_mask(image, region=(270, 600, 2286, 1299))
    image = pixel_filter(image)
    image = cv2.resize(image, (224, 224))

    # 可视化
    cv2.imshow("Visualization", image)
    cv2.waitKey(1)

    # 保存数据
    image = np.array(image)
    image_data.append(image)

    keys = key_check()
    targets.append(keys)
    if keys == "H":
        break

    print('loop took {} seconds'.format(time.time()-last_time))

save_data(image_data, targets)
