import cv2

def get_answer_card_cnts(img):
    """ 获得答题卡的左右答题区域轮廓
    # findContours 函数详解：https://blog.csdn.net/laobai1015/article/details/76400725
    # approxPolyDP 多边形近似 https://blog.csdn.net/kakiebu/article/details/79824856
    
    Args:
        img ([type]): 图片
    Returns:
        [type]: 答题卡的左右答题区域轮廓
    """

    # 检测图片中的最外围轮廓
    cnts, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("原始图片检测的轮廓总数：", len(cnts))
    if len(cnts) == 0:
        return None

    # 提取的轮廓总数
    contour_size = 0
    # 存储每个小答题区域的轮廓
    answer_cnts = []

    # 将轮廓按大小, 降序排序
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        # arcLength 计算周长
        peri = cv2.arcLength(c, True)
        print("轮廓周长：", peri)

        # 之前寻找到的轮廓可能是多边形，现在通过寻找近似轮廓，得到期望的四边形
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        print('原始轮廓的边数:', len(c), ', 近似轮廓的边数:', len(approx))

        # 当近似轮廓为4时，代表是需要提取的矩形区域
        if len(approx) == 4:
            contour_size = contour_size + 1
            answer_cnts.append(approx)

        # 只提取答题卡中最大的 10 个轮廓
        if contour_size == 10:
            break

    # 返回每个小答题区域的轮廓
    return answer_cnts

image = cv2.imread('D:/_Code2024_/CV/final_pre/answer_sheet_score/source/warped_image.png')

# 假设已加载图像
answer_card_cnts = get_answer_card_cnts(image)

# 绘制并保存每个小答题区域的轮廓
for idx, contour in enumerate(answer_card_cnts):
    # 创建一张与原图相同大小的空白图像
    contour_image = image.copy()

    # 在空白图像上绘制轮廓
    cv2.drawContours(contour_image, [contour], -1, (0, 255, 0), 2)  # 绿色轮廓线，线宽为2

    # 保存绘制轮廓后的图像
    cv2.imwrite(f'answer_card_contour_{idx + 1}.jpg', contour_image)
