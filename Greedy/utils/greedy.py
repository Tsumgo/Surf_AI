import os
import cv2
import numpy as np
import time

class ActionDeterminer:
    def __init__(self):
        self.offset = 30
        self.linewidth1 = 60
        self.linewidth2 = 80
        self.masks_computed = False  # 判断是否已经计算过mask
        self.masks = {}  # 存储mask
        self.counter = 0

    def _compute_masks(self, image):
        height, width = image.shape
        center_x = width // 2
        
        # Down
        top_left = (center_x - self.offset, 40)
        bottom_right = (center_x + self.offset, height)
        mask1 = np.zeros((height, width), dtype=np.uint8)
        mask1[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = 1
        self.masks['mask1'] = mask1

        # right
        start_point = (center_x, 22)
        end_point = (1328, height)
        mask2 = np.zeros((height, width), dtype=np.uint8)
        cv2.line(mask2, start_point, end_point, 1, thickness=self.linewidth1)
        self.masks['mask2'] = mask2

        # left
        start_point = (center_x, 22)
        end_point = (690, height)
        mask3 = np.zeros((height, width), dtype=np.uint8)
        cv2.line(mask3, start_point, end_point, 1, thickness=self.linewidth1)
        self.masks['mask3'] = mask3

        # right right
        start_point = (center_x, 22)
        end_point = (1770, height)
        mask4 = np.zeros((height, width), dtype=np.uint8)
        cv2.line(mask4, start_point, end_point, 1, thickness=self.linewidth2)
        self.masks['mask4'] = mask4

        # left left
        start_point = (center_x, 22)
        end_point = (248, height)
        mask5 = np.zeros((height, width), dtype=np.uint8)
        cv2.line(mask5, start_point, end_point, 1, thickness=self.linewidth2)
        self.masks['mask5'] = mask5

        self.masks_computed = True 

    def _visualize_masks(self, image, longest_way):

        white_image = np.ones_like(image) * 255

        # 合并所有的 mask
        combined_mask = self.masks['mask1'] | self.masks['mask2'] | self.masks['mask3'] | self.masks['mask4'] | \
                        self.masks['mask5']

        # 将合并后的 mask 应用到白色图像上
        masked_white = cv2.bitwise_and(white_image, white_image, mask=combined_mask)
        # 将合并后的 mask 应用到原始图像上
        masked_original = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(combined_mask))
        # 将两部分合并
        result_image = cv2.add(masked_original, masked_white)
        height, width = result_image.shape
        disp_img = cv2.resize(result_image, (width//3, height//3))

        # 可视化图像
        cv2.imshow("Processed Image", disp_img)
        cv2.waitKey(1)
        cv2.moveWindow("Processed Image", 100, 100)
        # 设置窗口置顶
        cv2.setWindowProperty('Processed Image', cv2.WND_PROP_TOPMOST, 1)

        text_image = np.zeros((200, 600, 3), dtype=np.uint8)
        text = "Distance:{} ".format(longest_way)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7 
        font_color = (255, 255, 255)
        font_thickness = 1 
        
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        # 计算文本的中心位置
        text_x = (text_image.shape[1] - text_size[0]) // 2
        text_y = (text_image.shape[0] + text_size[1]) // 2

        cv2.putText(text_image, text, (text_x, text_y), font, font_scale, font_color, font_thickness)
        # 显示文本
        cv2.imshow("Text", text_image)
        cv2.waitKey(1)
        cv2.moveWindow("Text", 1250, 100)
        # 设置窗口置顶
        cv2.setWindowProperty("Text", cv2.WND_PROP_TOPMOST, 1)
        
    def determine_action(self, image, prev_action, vis = False):
        start_time = time.time()
        
        if not self.masks_computed:
            self._compute_masks(image)

        longest_way = []

        image[:50, :] = 0  # 去掉人
 
        # 找最长无障碍距离
        hit_image1 = cv2.bitwise_and(self.masks['mask1'], image)
        longest_way.append(np.argmax(hit_image1.any(axis=1)))

        hit_image2 = cv2.bitwise_and(self.masks['mask2'], image)
        longest_way.append(np.argmax(hit_image2.any(axis=1)))

        hit_image3 = cv2.bitwise_and(self.masks['mask3'], image)
        longest_way.append(np.argmax(hit_image3.any(axis=1)))

        hit_image4 = cv2.bitwise_and(self.masks['mask4'], image)
        longest_way.append(np.argmax(hit_image4.any(axis=1)))

        hit_image5 = cv2.bitwise_and(self.masks['mask5'], image)
        longest_way.append(np.argmax(hit_image5.any(axis=1)))
        
        if 0 in longest_way:
            longest_way = [700 if x == 0 else x for x in longest_way]

        if (longest_way[0] == 700):
            if (vis):
                self._visualize_masks(image, longest_way)
            return 0
        
        if 0 in longest_way:
            longest_way = [700 if x == 0 else x for x in longest_way]
        
        action = np.argmax(longest_way)

        if (longest_way[prev_action] + 50 > longest_way[action]) and (longest_way[action] > 300): # 新的路跟之前比的差距
            action = prev_action

        if (vis):
            self._visualize_masks(image, longest_way)

        return action
