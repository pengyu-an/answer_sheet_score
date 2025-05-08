import cv2
import numpy as np

# 读取图像
image = cv2.imread("anchor.png")  # 替换为你的图片路径
if image is None:
    print("无法加载图片，请检查路径。")
    exit()

# 转换为HSV色彩空间
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义红色和黑色的HSV范围
red_lower1 = np.array([0, 120, 70])      # 红色范围1（低范围）
red_upper1 = np.array([10, 255, 255])   # 红色范围1（高范围）
red_lower2 = np.array([170, 120, 70])   # 红色范围2（高范围）
red_upper2 = np.array([180, 255, 255])  # 红色范围2（高范围）
black_lower = np.array([0, 0, 0])       # 黑色低范围
black_upper = np.array([180, 255, 50])  # 黑色高范围

# 创建掩膜 (mask)
red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
red_mask = cv2.bitwise_or(red_mask1, red_mask2)  # 合并红色掩膜
black_mask = cv2.inRange(hsv, black_lower, black_upper)

# 合并红色和黑色掩膜
combined_mask = cv2.bitwise_or(red_mask, black_mask)

# 形态学操作（闭运算）
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
closed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

# 边缘检测
edges = cv2.Canny(closed_mask, 50, 150)

# 霍夫变换检测线条
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=9, maxLineGap=100)

# 画出竖直线
output = image.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # 计算线条的角度
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
        if 85 <= angle <= 95:  # 竖直线角度范围
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 用绿色绘制竖直线

# 显示结果
cv2.imwrite("Detected Vertical Lines.png", output)
