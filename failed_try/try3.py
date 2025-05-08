import cv2
import numpy as np

def detect_red_to_black_vertical_lines(image_path):
    # 1. 读取图像
    image = cv2.imread(image_path)

    # 2. 转换到 HSV 颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 3. 定义红色和黑色的 HSV 范围
    # 红色范围（分为两个部分）
    lower_red1 = np.array([0, 120, 70])  # 红色低范围
    upper_red1 = np.array([10, 255, 255])  # 红色高范围
    lower_red2 = np.array([170, 120, 70])  # 红色低范围（另一侧）
    upper_red2 = np.array([180, 255, 255])  # 红色高范围（另一侧）

    # 黑色范围（亮度低，S 和 V 值很低）
    lower_black = np.array([0, 0, 0])  # 黑色低范围
    upper_black = np.array([180, 255, 50])  # 黑色高范围

    # 4. 创建红色和黑色的掩膜
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)  # 合并两个红色掩膜
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    # 5. 在掩膜中寻找垂直红到黑线条
    height, width = mask_red.shape
    red_to_black_lines = []  # 保存竖直线的 x 坐标

    # 遍历图像的每一列（竖直方向）
    for x in range(width):
        red_found = False
        black_found = False

        for y in range(height):
            # 如果在某列中先找到红色像素
            if mask_red[y, x] > 0:
                red_found = True

            # 然后在相邻的下方找到黑色像素
            if red_found and mask_black[y, x] > 0:
                black_found = True
                break  # 找到红到黑的渐变，跳出当前列的检测

        if red_found and black_found:
            red_to_black_lines.append(x)

    # 6. 绘制检测到的竖直线条
    output_image = image.copy()
    for x in red_to_black_lines:
        # 在图像上画竖直线
        cv2.line(output_image, (x, 0), (x, height), (0, 255, 0), 2)

    # 7. 显示结果
    cv2.imwrite('Red to Black Vertical Lines.png', output_image)

    # 返回竖直线的 x 坐标
    return red_to_black_lines

# 调用函数并传入图像路径
image_path = 'anchor.png'  # 替换为你的图像路径
red_to_black_lines = detect_red_to_black_vertical_lines(image_path)

print("Detected red-to-black vertical line positions (x-coordinates):", red_to_black_lines)
