import cv2
import numpy as np
import os

# 读取图像
image_path = "D:\_Code2024_\CV\final_pre\warped_image.png"  # 替换为实际图片路径
image = cv2.imread(image_path)
original = image.copy()

# 转换为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 二值化处理
_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# 检测水平和垂直线
kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_horizontal)

kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_vertical)

# 合并线条
lines = cv2.add(horizontal_lines, vertical_lines)

# 寻找轮廓
contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 保存小题分区的图片
if not os.path.exists("output"):
    os.makedirs("output")

regions = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    # 筛选合适大小的区域（避免非小题选框的干扰）
    if 50 < w < 200 and 50 < h < 200:  # 可根据实际调整
        regions.append((x, y, w, h))

# 按从上到下、从左到右排序
regions = sorted(regions, key=lambda r: (r[1], r[0]))

# 切割并保存每小题的图片
for i, (x, y, w, h) in enumerate(regions):
    question_image = original[y:y+h, x:x+w]
    output_path = os.path.join("output", f"{i+1}.png")
    cv2.imwrite(output_path, question_image)

print(f"分区完成，共保存 {len(regions)} 个小题图片。")