# def get_inside(path):
#     """
#     获取答题卡的内边界内的每张图片

#     Args:
#         img ([type]): 图片
#     Returns:
#         insides_path ([type]): 答题卡的内边界内的图片地址列表
#     """
#     img = cv2.imread(path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # img = cv2.blur(img, (5, 5))
#     # img = cv2.Canny(img, 50, 100)
#     # warped_answer_image_1 = four_point_transform(gray, answer_contour_1.reshape(4, 2))

#     # 二值化
#     # 将灰度图像进行二值化处理，生成一个二值图像，其中像素值为 0 或 255。在这里，0 表示背景（黑色），255 表示前景（白色）
#     thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
#     cv2.imwrite('thresh.jpg', img)
    
#     # 在二值图像中查找轮廓，包括内围、外围轮廓，但是检测到的轮廓不建立等级关系，彼此之间独立
#     thresh_cnts, _ = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

#     cnt_size = 0
#     sub_answer_cnts = []
#     if len(thresh_cnts) > 0:
#         # 将轮廓按面积大小, 降序排序
#         thresh_cnts = sorted(thresh_cnts, key=cv2.contourArea, reverse=True)
#         for c in thresh_cnts:
#             # arcLength 计算轮廓的周长
#             peri = cv2.arcLength(c, True)

#             # 计算轮廓的边界框，即该轮廓外接的最小矩形框的坐标 (x, y) 和宽高 (w, h)
#             (x, y, w, h) = cv2.boundingRect(c)
#             cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             # if x>10 and x<30 and y>700 and y<730:
#                 # print(f"(x, y, w, h) = {(x, y, w, h)}")

#             # 之前寻找到的轮廓可能是多边形，现在通过寻找近似轮廓，得到期望的四边形
#             approx = cv2.approxPolyDP(c, 0.02 * peri, True)

#             # 只提取近似轮廓为四边形的区域, 且轮廓长度大于指定长度
#             # if len(approx) == 4 and h > 130 and h < 170:
#             # if h > 100 and h < 200:
#             # if w > 600 and h < 200:
#             if 1:
#                 print(f"(x, y, w, h) = {(x, y, w, h)}")
#                 # 计数器加一，表示处理了一个轮廓
#                 cnt_size = cnt_size + 1
#                 cv2.imwrite(f'D:/_Code2024_/CV/final_pre/answer_sheet_score/output/inside_{cnt_size}.png', img)
#                 sub_answer_cnts.append(approx)
#                 # capture_img('D:/_Code2024_/CV/final_pre/answer_sheet_score/source/warped_image.png', border_image_path, c)

#             # print("轮廓周长：", peri, '宽:', w)
#             # print('原始轮廓的边数:', len(c), ', 近似轮廓的边数:', len(approx))

#             # 只处理前几个最大轮廓
#             if cnt_size >= 5:
#                 break
#         cv2.imwrite(f'D:/_Code2024_/CV/final_pre/answer_sheet_score/output/inside_{cnt_size}.png', img)

#     # 从上到下，将轮廓排序
#     sub_answer_cnts = sort_contours(sub_answer_cnts, method="top-to-bottom")[0]
#     return sub_answer_cnts