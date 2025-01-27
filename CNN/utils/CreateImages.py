import cv2
import numpy as np
import os

# 加载训练数据和目标数据
data = np.load("./data/training_data.npy", allow_pickle=True)
targets = np.load("./data/target_data.npy", allow_pickle=True)

# 定义迭代次数
iteration = 6
iteration1 = 8

os.makedirs(f"D:/GC{iteration}/Nothing/", exist_ok=True)
os.makedirs(f"D:/GC{iteration}/Left/", exist_ok=True)
os.makedirs(f"D:/GC{iteration}/Right/", exist_ok=True)
os.makedirs(f"D:/GC{iteration}/Down/", exist_ok=True)

print(f'Image Data Shape: {data.shape}')
print(f'targets Shape: {targets.shape}')

# 统计每种动作的数量
unique_elements, counts = np.unique(targets, return_counts=True)
print(np.asarray((unique_elements, counts)))

holder_list = []
for i, image in enumerate(data):
    holder_list.append([data[i], targets[i]])

count_nothing = 0
count_left = 0
count_right = 0
count_down = 0

# 根据目标类别保存图像
for data in holder_list:
    if data[1] == 'Q':
        count_nothing += 1
        cv2.imwrite(f"D:/GC{iteration}/Nothing/H{iteration1}-n{count_nothing}.png", data[0]) 
    elif data[1] == 'A':
        count_left += 1
        cv2.imwrite(f"D:/GC{iteration}/Left/H{iteration1}-l{count_left}.png", data[0]) 
    elif data[1] == 'D':
        count_right += 1
        cv2.imwrite(f"D:/GC{iteration}/Right/H{iteration1}-r{count_right}.png", data[0]) 
    elif data[1] == 'S':
        count_down += 1
        cv2.imwrite(f"D:/GC{iteration}/Down/H{iteration1}-d{count_down}.png", data[0]) 
