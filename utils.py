import cv2
import numpy as np
import os
from settings import *

def correction(path):
    """
    对输入图像进行几何校正

    Args:
        path (str): 输入图像的路径

    Returns:
        ret_path (str): 对输入图像进行校正后保存的路径

    """
    
    # 读取图片
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二值化
    _, threshold = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # 膨胀和腐蚀
    kernel = np.ones((5,5), np.uint8)
    dilated_image = cv2.dilate(threshold, kernel, iterations=1)
    eroded_image = cv2.erode(dilated_image, kernel, iterations=1)

    # Canny边缘检测
    edges = cv2.Canny(eroded_image, 50, 150)

    # 霍夫变换检测直线
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

    # 存储线的斜率和长度
    lines_info = []

    # 在原图上绘制直线并计算斜率和长度
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  # 计算线的长度
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')  # 计算线的斜率
            lines_info.append((slope, length, (x1, y1, x2, y2)))  # 存储斜率、长度和线的坐标

    # 根据线的长度降序排序
    lines_info.sort(key=lambda x: x[1], reverse=True)

    # 获取最长的4条线的斜率
    max_lines = lines_info[:4]

    # 在原图上绘制最长的4条直线
    if max_lines:
        for slope, length, line in max_lines:
            x1, y1, x2, y2 = line

    max_slopes = [slope for slope, _, _ in max_lines]
    average_slope = sum(max_slopes) / len(max_slopes)

    theta = np.arctan(average_slope)
    rotation_angle = np.degrees(theta)  # 将弧度转换为度数

    # 获取图像的中心点
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # 获取旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

    # 执行旋转
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), borderValue=(255, 255, 255))

    # 再次读取黑色区域
    gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    dilated_image = cv2.dilate(threshold, kernel, iterations=1)
    eroded_image = cv2.erode(dilated_image, kernel, iterations=2)

    edges = cv2.Canny(eroded_image, 50, 150)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(rotated_image, contours, -1, (0, 255, 0), 3)
    areas = [cv2.contourArea(cnt) for cnt in contours]

    largest_two_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    cv2.drawContours(rotated_image, largest_two_contours, -1, (0, 255, 0), 3)

    corners = []
    for contour in largest_two_contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        points = np.array(box)
        sorted_points = points[np.argsort(points[:, 0])]
        point1 = ((sorted_points[0] + sorted_points[1]) / 2).astype(int)
        point2 = ((sorted_points[2] + sorted_points[3]) / 2).astype(int)
        corners.append(point1)
        corners.append(point2)

    corners = np.array(corners, dtype=np.float32)
    print(corners)

    target_points = np.array([
        [76, 1339], [900, 1339], [76, 532], [900, 532]
    ], dtype=np.float32)

    # 计算错切变换矩阵
    mat_perspective = cv2.getPerspectiveTransform(corners, target_points)

    # 应用错切变换
    warped_corners = cv2.warpPerspective(rotated_image, mat_perspective, (1043, 1418))

    # 保存结果
    ret_path = 'source/warped_image.png'
    cv2.imwrite('source/warped_image.png', warped_corners)
    print("Correction completed ~")

    return ret_path
    

def get_init_process_img(img_path):
    """
    对图片进行初始化处理，包括灰度，高斯模糊，腐蚀，膨胀和边缘检测等

    Args:
        img_path (str): 待初始化的图像路径

    Returns:
        image (ndarray): 初始化后的图像

    """

    if not img_path:
        print("No images found in this path ~")
        exit()
    else:
        print('Correction image getted ~')
    
    image = cv2.imread(img_path)

    # 转灰度
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 滤波
    image = cv2.blur(image, (3, 3))
    # image = cv2.GaussianBlur(image, (5, 5), 0)  # 效果不是很好
    # image = cv2.medianBlur(image, 5)  # 效果不是很好

    # 高斯模糊
    # image = cv2.GaussianBlur(image, (5, 5), 0)

    # 腐蚀erode与膨胀dilate
    # kernel = np.ones((3, 3), np.uint8)
    # blurred = cv2.erode(blurred, kernel, iterations=1) # 腐蚀
    # blurred = cv2.dilate(blurred, kernel, iterations=2) # 膨胀
    # blurred = cv2.erode(blurred, kernel, iterations=1) # 腐蚀
    # blurred = cv2.dilate(blurred, kernel, iterations=2) # 膨胀

    # 图像增强
    transformed_image = np.zeros_like(image)  # 创建空白图像
    # 阈值区间 (这些值可以根据需要进行调整)
    r1, s1 = 70, 50   # 低灰度区间
    r2, s2 = 115, 200  # 中灰度区间
    r3, s3 = 255, 255  # 高灰度区间
    # 三段式灰度变换
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel = image[i, j]
            if pixel < r1:
                transformed_image[i, j] = (s1 / r1) * pixel
            elif pixel < r2:
                transformed_image[i, j] = s1 + ((s2 - s1) / (r2 - r1)) * (pixel - r1)
            else:
                transformed_image[i, j] = s2 + ((s3 - s2) / (r3 - r2)) * (pixel - r2)
    

    # 边缘检测
    image = cv2.Canny(image, 50, 100)
    # image = cv2.Canny(blurred, 75, 200)
    
    # 填补小空隙
    kernel = np.ones((5, 5), np.uint8)  # 定义一个结构元素，通常可以选择矩形或椭圆
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)  # 使用闭运算来填补边缘上的小空隙

    cv2.imwrite("output/initialed.jpg", image)
    print('Image initial process completed ~')

    return image



def sort_contours(cnts, method="left-to-right"):
    """
    轮廓排序

    Args:
        cnts (ndarray): 轮廓
        method (str, optional): 排序方式. Defaults to "left-to-right".

    Returns:
        cnts (ndarray): 排序好的轮廓
        boundingBoxes (ndarray): 排序好的边框列表

    """
    
    if cnts is None or len(cnts) == 0:
        return [], []
    
    # 初始化逆序标志和排序索引
    reverse = False
    i = 0

    # 是否需逆序处理
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # 是否需要按照y坐标函数
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # 构造包围框列表，并从上到下对它们进行排序
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    
    # print('Contours sort process completed ~')

    # 返回已排序的轮廓线和边框列表
    return cnts, boundingBoxes



def capture_img(origin_image_path, target_image_path, contour):
    """根据轮廓截取图片

    Args:
        origin_image_path (str): 原始图片路径
        target_image_path (str): 目标图片路径
        contour (ndarray): 要截取的轮廓

    Returns:
        [type]: [description]

    """
    image = cv2.imread(origin_image_path)
                       
    # 获取待截取轮廓的坐标
    x, y, w, h = cv2.boundingRect(contour)

    # 截图
    cv2.imwrite(target_image_path, image[y:y + h, x:x + w])

    # print('Image capture process completed ~')


def capture_img_position(origin_image_path, target_image_path, x, y, w, h):
    """
    根据 (x, y, w, h) 截取图片

    Args:
        origin_image_path (str): 原始图片路径
        target_image_path (str): 目标图片路径
        x (int): 截取轮廓位置的 x 坐标
        y (int): 截取轮廓位置的 y 坐标
        w (int): 截取轮廓的宽度
        h (int): 截取轮廓的高度

    Returns:
        [type]: [description]

    """
    image = cv2.imread(origin_image_path)

    # 截图
    cv2.imwrite(target_image_path, image[y:y + h, x:x + w])

    # print('Image capture process completed ~')



def get_outside(img):
    """ 获取答题卡的外边界内的图片

    # findContours 函数详解: https://blog.csdn.net/laobai1015/article/details/76400725
    # approxPolyDP 多边形近似: https://blog.csdn.net/kakiebu/article/details/79824856
    
    Args:
        img (ndarray): 图片

    Returns:
        borders_path (ndarray): 答题卡的外边界内的图片地址列表

    """

    # 检测图片中的最外围轮廓，轮廓存到列表 cnts 中
    cnts, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # print("原始图片检测的轮廓总数：", len(cnts))

    # 图片中没有轮廓，直接返回无，终止函数
    if len(cnts) == 0:
        return None

    # 检测到的矩形轮廓数目
    rec_cnts_amount = 0

    # 检测到的矩形轮廓存放列表
    rec_cnts = []

    # 将轮廓按面积大小, 降序排序
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # 遍历每个轮廓
    for i, c in enumerate(cnts):
        # arcLength 计算周长
        peri = cv2.arcLength(c, True)

        # print(f"第 {i+1}  个轮廓周长为：", peri)

        # 之前寻找到的轮廓可能是多边形，现在通过寻找近似轮廓，得到期望的四边形
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # print(f'第 {i+1}  个原始轮廓的边数:', len(c), ', 对应的近似轮廓的边数:', len(approx))

        # 当近似轮廓为4时，代表是需要提取的矩形区域
        if len(approx) == 4:
            # 提取到的矩形轮廓数多了1个
            rec_cnts_amount = rec_cnts_amount + 1
            # 将新增的矩形轮廓存下来
            rec_cnts.append(approx)

        # 提取到答题卡中最大的 ANSWER_CARD_SIZE 个矩形轮廓后就终止提取
        if rec_cnts_amount == ANSWER_CARD_SIZE:
        # if rec_cnts_amount == 10:  # 先提取10个矩形轮廓
            break
    
    # 将提取到的这些矩形轮廓排序，只取轮廓线([0])
    rec_cnts = sort_contours(rec_cnts, method="left-to-right")[0]
    # rec_cnts = sorted(rec_cnts, key=cv2.contourArea, reverse=True)

    # 提取到的这些矩形轮廓的保存路径存放列表
    outsides_path = []

    if len(rec_cnts) > 0:
        # 为每个答题卡分配一个编号
        rec_cnts_id = 0

        # 从左到右遍历每一个矩形轮廓
        for c in rec_cnts:
            # 编号从1开始依次递增
            rec_cnts_id = rec_cnts_id + 1

            if len(rec_cnts) == 1:
                # 第 rec_cnts_id 个矩形边框内的图片保存的地址
                outside_image_path = 'output/outside.jpg'
            else:
                # 第 rec_cnts_id 个矩形边框内的图片保存的地址
                outside_image_path = 'output/border_' + str(rec_cnts_id) + '.jpg'

            # 放到存放列表中
            outsides_path.append(outside_image_path)

            # 对目标图片按照第 rec_cnts_id 个矩形边框进行截取，存放到目标路径中
            capture_img('source/warped_image.png', outside_image_path, c)

    print('答题卡外边界内的图像存放地址：', outsides_path)
    print('Get outside border process completed ~')

    return outsides_path



def get_inside(path):
    """ 获取答题卡的内边界内的图片

    # findContours 函数详解: https://blog.csdn.net/laobai1015/article/details/76400725
    # approxPolyDP 多边形近似: https://blog.csdn.net/kakiebu/article/details/79824856
    
    Args:
        path (str): 输入图片的路径

    Returns:
        insides_path (ndarray): 答题卡的内边界内的图片地址列表

    """

    img = get_init_process_img(path)

    # 检测图片中的所有轮廓，轮廓存到列表 cnts 中
    cnts, _ = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # print("原始图片检测的轮廓总数：", len(cnts))

    # 图片中没有轮廓，直接返回无，终止函数
    if len(cnts) == 0:
        return None

    # 检测到的矩形轮廓数目
    rec_cnts_amount = 0

    # 检测到的矩形轮廓存放列表
    rec_cnts = []

    # 将轮廓按面积大小, 降序排序
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # 遍历每个轮廓
    for i, c in enumerate(cnts):
        # arcLength 计算周长
        peri = cv2.arcLength(c, True)
        # print(f"第 {i+1}  个轮廓周长为：", peri)

        # 之前寻找到的轮廓可能是多边形，现在通过寻找近似轮廓，得到期望的四边形
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # print(f'第 {i+1}  个原始轮廓的边数:', len(c), ', 对应的近似轮廓的边数:', len(approx))

        (x, y, w, h) = cv2.boundingRect(c)

        # 若满足答题卡子框（内边界）的尺寸要求
        if w > SUB_ANSWER_CARD_MIN_WIDTH and SUB_ANSWER_CARD_MIN_HEIGHT < h < SUB_ANSWER_CARD_MAX_HEIGHT:
            # print(f"(x, y, w, h) = {(x, y, w, h)}")

            # 提取到的矩形轮廓数多了1个
            rec_cnts_amount = rec_cnts_amount + 1
            # 将新增的矩形轮廓存下来
            rec_cnts.append(approx)
    
    # 将提取到的这些矩形轮廓排序，只取轮廓线
    rec_cnts = sort_contours(rec_cnts, method="top-to-bottom")[0]
    # rec_cnts = sorted(rec_cnts, key=cv2.contourArea, reverse=True)

    # 提取到的这些矩形轮廓的保存路径存放列表
    insides_path = []

    if len(rec_cnts) > 0:
        # 为每个答题卡分配一个编号
        rec_cnts_id = 0

        # 从左到右遍历每一个矩形轮廓
        for c in rec_cnts:
            # 编号从1开始依次递增
            rec_cnts_id = rec_cnts_id + 1

            # 第 rec_cnts_id 个矩形边框内的图片保存的地址
            inside_image_path = 'output/inside_' + str(rec_cnts_id) + '.jpg'

            # 放到存放列表中
            insides_path.append(inside_image_path)

            # 对目标图片按照第 rec_cnts_id 个矩形边框进行截取，存放到目标路径中
            capture_img(path, inside_image_path, c)

    print('答题卡内边界内的图像存放地址：', insides_path)
    print('Get insider border process completed ~')

    return insides_path




def get_each_question_ans(paths):
    """
    将答题卡的内边界内的每张图片切分成一道道试题
    
    Args:
        paths (ndarray): 输入内边界内的图片的路径列表

    Returns:
        [type]: [description]

    """
    # 提取到的这些矩形轮廓的保存路径存放列表
    each_paths = []

    # 为每个切好的试题分配一个编号
    each_id = 0

    for m, path in enumerate(paths):
        if m <= 1:  # 前两个子框有20道小题
            for i in range(20):
                x0, y0 = 5, 0  # 切割起始点
                w, h = 40, 150  # 每个切块的宽度和高度
                each_id = each_id + 1
                each_path = 'cut_result/' + str(each_id) + '.jpg'
                each_paths.append(each_path)
                if i < 5:  # 每个子框中的第1~5小题
                    capture_img_position(path, each_path, x0+i*w, y0, w, h)
                else:
                    if i < 10:  # 每个子框中的第6~10小题
                        capture_img_position(path, each_path, x0+i*w+41, y0, w, h)
                    else:
                        if i < 15:  # 每个子框中的第11~15小题
                            capture_img_position(path, each_path, x0+i*w+79, y0, w, h)
                        else:  # 每个子框中的第16~20小题
                            capture_img_position(path, each_path, x0+i*w+118, y0, w, h)
        else:  # 后三个子框有15道小题
            for i in range(15):
                x0, y0 = 19, 0  # 切割起始点
                w, h = 40, 150  # 每个切块的宽度和高度
                each_id = each_id + 1
                each_path = 'cut_result/' + str(each_id) + '.jpg'
                each_paths.append(each_path)
                if i < 5:  # 每个子框中的第1~5小题
                    capture_img_position(path, each_path, x0+i*w, y0, w, h)
                else: 
                    if i < 10:  # 每个子框中的第6~10小题
                        capture_img_position(path, each_path, x0+i*w+37, y0, w, h)
                    else:  # 每个子框中的第11~15小题
                        capture_img_position(path, each_path, x0+i*w+78, y0, w, h)
    
    print('每个切好的试题存放地址：', each_paths)
    print('Get answers of each question process completed ~')

#用于检测每个小图片的选项
def recognize_answers(image_path):
    # 读取图片
    img = cv2.imread(image_path)
    #cv2.imshow('Image',img)
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #经过观察，发现答题卡填涂区域的反光规律是中间反光四周黑，因此进行腐蚀操作，把中间反光显白的区域尽可能缩小
    #一定程度上可以弥补反光带来的影响。但由于腐蚀操作的引入，使得原填涂区域会增大，因此需要把ABCD选项的区域扩大
    #同时结合二值化的阈值、判断是否填涂的阈值调整
    #需要多次实验，使得ABCD四个区域内的黑色素百分比，高的低的差距尽可能大
    kernel = np.ones((2, 2), np.uint8)
    eroded_image = cv2.erode(gray, kernel, iterations=2)
    #cv2.imshow('膨胀后',eroded_image)
    _, thresh = cv2.threshold(eroded_image, 130, 255, cv2.THRESH_BINARY)
    #cv2.imshow('Thresholded Image',thresh)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows() 
    # 定义ABCD选项的区域
    options = {
        'A': (0, 41, 39, 64),
        'B': (0, 65, 39, 89),
        'C': (0, 90, 39, 115),
        'D': (0, 116, 39, 139)
    }
    
    # 初始化被填涂选项列表
    filled_options = []
    
    # 检查每个区域，确定哪个选项被填涂
    for option, (x1, y1, x2, y2) in options.items():
        # 提取选项区域
        option_region = thresh[y1:y2, x1:x2]
        
        # 计算区域中黑色像素的比例
        black_ratio = np.sum(option_region == 0) / (option_region.shape[0] * option_region.shape[1])
        # print(black_ratio)
        # 如果黑色像素比例超过一定阈值，认为该选项被填涂
        if black_ratio > 0.408:  # 阈值可以根据实际情况调整
            filled_options.append(option)
    
    # 返回所有被填涂的选项，如果没有选项被填涂，则返回'No Answer'
    return filled_options if filled_options else 'No Answer'

def process_all_images(directory):
    # 初始化答案字符串
    answers = ""
    
    # 遍历1到85的文件
    for i in range(1, 86):
        # 构建文件名，直接使用i作为文件名，因为图片命名是1.jpg, 2.jpg...
        filename = str(i) + '.jpg'
        image_path = os.path.join(directory, filename)
        
        # 检查文件是否存在
        if os.path.exists(image_path):
            # 处理图像并获取答案
            answer = recognize_answers(image_path)
            # 如果answer是一个列表，将其转换为字符串
            if isinstance(answer, list):
                answer_str = ''.join(answer)
            else:
                answer_str = answer  # 如果answer已经是字符串，直接使用
            answers += answer_str  # 将答案字符串追加到答案字符串中
            # 每处理5个答案，添加一个空格
            # if (i % 5) == 0:
            #     answers += " "
        else:
            print(f"File {filename} not found in {directory}.")
            answers += "Fail to found picfile"  # 如果文件不存在，添加 "Fail to found picfile" 或其他标记
            # 每处理5个答案，添加一个空格，包括"No Answer"
            # if ((i // 5) * 5) == i:
            #     answers += " "    
    # 输出所有答案
    #print(f"检测到的填涂结果是{answers}")
    return answers

def answersheet_score(image_path):
    """
    对图片进行几何校正、切割和分数统计的完整流程。
    输入答题卡图片路径（请输入本文件中相对路径下'source/1.jpg'、'source/2.jpg' 或 'source/3.jpg'三者之一）
    输出answers，检测到的填涂答案序列

    参数:
    image_path (str): 输入图片的路径，可以是 'source/1.jpg'、'source/2.jpg' 或 'source/3.jpg'。
    
    返回:
    list: 检测到的填涂结果。
    """
    # 对图片进行几何校正
    corrected_path = correction(image_path)
    
    # 对图片进行初始化处理，包括灰度，高斯模糊和边缘检测等
    image = get_init_process_img(corrected_path)
    
    # 获取答题卡的外边界内的图片地址列表
    outsides_path = get_outside(image)
    
    insides_path = []
    # 对于每个外边界内的图片
    for outside_path in outsides_path:
        # 获取该图片中的内边界内的每张图片
        insides_path.extend(get_inside(outside_path))
    
    # 将答题卡的内边界内的每张图片切分成一道道试题
    get_each_question_ans(insides_path)
    
    # 对图片进行分数统计
    answers = process_all_images('cut_result')
    #print(f'检测到的填涂结果是：{answers}，共有{len(answers)}个')
    return answers