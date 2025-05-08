import cv2
import numpy as np
import os

# 读取图像
image = cv2.imread('anchor.png')

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 二值化处理
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

# 查找轮廓
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 过滤出红色括号和水平线的轮廓
red_contours = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if 1 < w < 8 and 9 < h < 18:  # 假设水平线宽度大于高度
        red_contours.append((x, y, w, h))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 用绿色框标出中括号位置

# 按y坐标排序
red_contours.sort(key=lambda x: x[1])

# 确定每个小题的区域
question_regions = []
for i in range(len(red_contours) - 1):
    y1 = red_contours[i][1] + red_contours[i][3]
    y2 = red_contours[i + 1][1]
    question_regions.append((0, y1, image.shape[1], y2 - y1))

# 确保每个区域大小一致
max_height = max([region[3] for region in question_regions])
for i in range(len(question_regions)):
    x, y, w, h = question_regions[i]
    question_regions[i] = (x, y, w, max_height)

# 保存每个小题的图片
for i, region in enumerate(question_regions):
    x, y, w, h = region
    question_image = image[y:y+h, x:x+w]
    cv2.imwrite(f'question_{i+1}.png', question_image)

print("答题卡分区完成，图片已保存。")