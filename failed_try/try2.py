import cv2
import numpy as np

# 读取图片
image = cv2.imread('D:/_Code2024_/CV/final_pre/warped_image.png')

# 转换为HSV颜色空间
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义红色的HSV阈值范围
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

# 创建掩膜
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = cv2.bitwise_or(mask1, mask2)

# 进行形态学操作，去除噪声
kernel = np.ones((3,3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# 寻找轮廓
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 遍历轮廓，找到红色中括号
for contour in contours:
    # 计算轮廓的边界框
    x, y, w, h = cv2.boundingRect(contour)
    
    # 可以根据轮廓的宽高比和面积进一步筛选出中括号
    # if 0.3 < w / float(h) < 0.8 and 100 < cv2.contourArea(contour) < 500:
    # if 9 < h < 13 and 3 < w < 6:
    if 1:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 用绿色框标出中括号位置

# 显示结果
cv2.imwrite('Red Brackets.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()