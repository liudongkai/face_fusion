# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np
import time
import math

import core


def transformation_points(src_img, src_points, dst_img, dst_points):
    """
    运用普氏分析将要融合的人脸转换到模板图角度
    :param src_img: 模板人脸图片
    :param src_points: 模板人脸坐标点
    :param dst_img: 目标人脸图片
    :param dst_points: 目标人脸坐标点
    :return: 转换后的图像
    """
    src_points = src_points.astype(np.float64)  # 模板人脸坐标点转化为浮点型
    dst_points = dst_points.astype(np.float64)

    c1 = np.mean(src_points, axis=0)  #
    c2 = np.mean(dst_points, axis=0)

    src_points -= c1
    dst_points -= c2

    s1 = np.std(src_points)
    s2 = np.std(dst_points)

    src_points /= s1
    dst_points /= s2

    u, s, vt = np.linalg.svd(src_points.T * dst_points)
    r = (u * vt).T

    m = np.vstack([np.hstack(((s2 / s1) * r, c2.T - (s2 / s1) * r * c1.T)), np.matrix([0., 0., 1.])])

    output = cv2.warpAffine(dst_img, m[:2],
                            (src_img.shape[1], src_img.shape[0]),
                            borderMode=cv2.BORDER_TRANSPARENT,
                            flags=cv2.WARP_INVERSE_MAP)

    return output


def tran_similarity(src_img, src_points, dst_img, dst_points):  # 获取人脸平均脸
    s60 = math.sin(60 * math.pi / 180)
    c60 = math.cos(60 * math.pi / 180)

    in_pts = np.copy(dst_points).tolist()
    out_pts = np.copy(src_points).tolist()

    x_in = c60 * (in_pts[0][0] - in_pts[1][0]) - s60 * (in_pts[0][1] - in_pts[1][1]) + in_pts[1][0]
    y_in = s60 * (in_pts[0][0] - in_pts[1][0]) + c60 * (in_pts[0][1] - in_pts[1][1]) + in_pts[1][1]

    in_pts.append([np.int(x_in), np.int(y_in)])

    x_out = c60 * (out_pts[0][0] - out_pts[1][0]) - s60 * (out_pts[0][1] - out_pts[1][1]) + out_pts[1][0]
    y_out = s60 * (out_pts[0][0] - out_pts[1][0]) + c60 * (out_pts[0][1] - out_pts[1][1]) + out_pts[1][1]

    out_pts.append([np.int(x_out), np.int(y_out)])

    m = cv2.estimateRigidTransform(np.array([in_pts]), np.array([out_pts]), False)

    output = cv2.warpAffine(dst_img, m, (src_img.shape[1], src_img.shape[0]))

    return output


def correct_color(img1, img2, landmark):
    blur_amount = 0.4 * np.linalg.norm(
        np.mean(landmark[core.LEFT_EYE_POINTS], axis=0)
        - np.mean(landmark[core.RIGHT_EYE_POINTS], axis=0)
    )
    blur_amount = int(blur_amount)

    if blur_amount % 2 == 0:
        blur_amount += 1

    img1_blur = cv2.GaussianBlur(img1, (blur_amount, blur_amount), 0)
    img2_blur = cv2.GaussianBlur(img2, (blur_amount, blur_amount), 0)

    img2_blur += (128 * (img2_blur <= 1.0)).astype(img2_blur.dtype)

    return img2.astype(np.float64) * img1_blur.astype(np.float64) / img2_blur.astype(np.float64)


def tran_src(src_img, src_points, dst_points, face_area=None):
    """
    应用三角仿射转换将模板图人脸轮廓仿射成目标图像人脸轮廓
    :param src_img:
    :param src_points:
    :param dst_points:
    :param face_area:
    :return:
    """
    jaw = core.JAW_END
    #
    dst_list = dst_points \
               + core.matrix_rectangle(face_area[0], face_area[1], face_area[2], face_area[3]) \
               + core.matrix_rectangle(0, 0, src_img.shape[1], src_img.shape[0])

    src_list = src_points \
               + core.matrix_rectangle(face_area[0], face_area[1], face_area[2], face_area[3]) \
               + core.matrix_rectangle(0, 0, src_img.shape[1], src_img.shape[0])
    #
    jaw_points = []

    for i in range(0, jaw):
        jaw_points.append(dst_list[i])
        jaw_points.append(src_list[i])

    warp_jaw = cv2.convexHull(np.array(jaw_points), returnPoints=False)
    warp_jaw = warp_jaw.tolist()

    for i in range(0, len(warp_jaw)):
        warp_jaw[i] = warp_jaw[i][0]

    warp_jaw.sort()

    if len(warp_jaw) <= jaw:
        dst_list = dst_list[jaw - len(warp_jaw):]
        src_list = src_list[jaw - len(warp_jaw):]
        for i in range(0, len(warp_jaw)):
            dst_list[i] = jaw_points[int(warp_jaw[i])]
            src_list[i] = jaw_points[int(warp_jaw[i])]
    else:
        for i in range(0, jaw):
            if len(warp_jaw) > jaw and warp_jaw[i] == 2 * i and warp_jaw[i + 1] == 2 * i + 1:
                warp_jaw.remove(2 * i)

            dst_list[i] = jaw_points[int(warp_jaw[i])]

    dt = core.measure_triangle(src_img.shape, dst_list)

    res_img = np.zeros(src_img.shape, dtype=src_img.dtype)

    for i in range(0, len(dt)):
        t_src = []
        t_dst = []

        for j in range(0, 3):
            t_src.append(src_list[dt[i][j]])
            t_dst.append(dst_list[dt[i][j]])

        core.affine_triangle(src_img, res_img, t_src, t_dst)

    return res_img

    # 五、将融合后的脸部贴到模特图上
    # 最后一步是将融合后的新图片脸部区域用泊松融合算法贴到模特图上。泊松融合可直接使用opencv提供的函数


def merge_img(src_img, dst_img, dst_points, k_size=None, mat_multiple=None, is_peach=None):
    face_mask = np.zeros(src_img.shape[:2], dtype=src_img.dtype)

    overlay = core.PEACH_POINTS if is_peach else core.OVERLAY_POINTS

    for group in overlay:
        cv2.fillConvexPoly(face_mask, cv2.convexHull(np.array(dst_points)[group]), (255, 255, 255))

    r = cv2.boundingRect(np.float32([dst_points[:core.FACE_END]]))

    center = (r[0] + int(r[2] / 2 - 35), r[1] + int(r[3] / 2))

    if mat_multiple:
        mat = cv2.getRotationMatrix2D(center, 0,
                                      mat_multiple)  # Point2f center：表示旋转的中心点;double angle：表示旋转的角度;double scale：图像缩放因子 返回一个旋转矩阵
        face_mask = cv2.warpAffine(face_mask, mat, (face_mask.shape[1], face_mask.shape[0]))

    if k_size:
        face_mask = cv2.blur(face_mask, k_size, center)

    return cv2.seamlessClone(np.uint8(dst_img), src_img, face_mask, center, cv2.NORMAL_CLONE)


def merge_img1(src_img, dst_img, dst_points, k_size=None, mat_multiple=None, is_peach=None):
    face_mask = np.zeros(src_img.shape[:2], dtype=src_img.dtype)

    overlay = core.PEACH_POINTS if is_peach else core.OVERLAY_POINTS

    for group in overlay:
        cv2.fillConvexPoly(face_mask, cv2.convexHull(np.array(dst_points)[group]), (255, 255, 255))

    r = cv2.boundingRect(np.float32([dst_points[:core.FACE_END]]))

    center = (r[0] + int(r[2] / 2), r[1] + int(r[3] / 2) + 2)

    if mat_multiple:
        mat = cv2.getRotationMatrix2D(center, 0,
                                      mat_multiple)  # Point2f center：表示旋转的中心点;double angle：表示旋转的角度;double scale：图像缩放因子 返回一个旋转矩阵
        face_mask = cv2.warpAffine(face_mask, mat, (face_mask.shape[1], face_mask.shape[0]))

    if k_size:
        face_mask = cv2.blur(face_mask, k_size, center)

    return cv2.seamlessClone(np.uint8(dst_img), src_img, face_mask, center, cv2.NORMAL_CLONE)


def merge_img_nan_g(src_img, dst_img, dst_points, k_size=None, mat_multiple=None, is_peach=None):
    face_mask = np.zeros(src_img.shape[:2], dtype=src_img.dtype)

    overlay = core.PEACH_POINTS if is_peach else core.OVERLAY_POINTS

    for group in overlay:
        cv2.fillConvexPoly(face_mask, cv2.convexHull(np.array(dst_points)[group]), (255, 255, 255))

    r = cv2.boundingRect(np.float32([dst_points[:core.FACE_END]]))

    center = (r[0] + int(r[2] / 2 + 10), r[1] + int(r[3] / 2) - 20)

    if mat_multiple:
        mat = cv2.getRotationMatrix2D(center, 0,
                                      mat_multiple)  # Point2f center：表示旋转的中心点;double angle：表示旋转的角度;double scale：图像缩放因子 返回一个旋转矩阵
        face_mask = cv2.warpAffine(face_mask, mat, (face_mask.shape[1], face_mask.shape[0]))

    if k_size:
        face_mask = cv2.blur(face_mask, k_size, center)

    return cv2.seamlessClone(np.uint8(dst_img), src_img, face_mask, center, cv2.NORMAL_CLONE)


def morph_img(src_img, src_points, dst_img, dst_points, alpha=0.8, show_bg=None):
    src_img = src_img.astype(np.float32)

    dst_img = dst_img.astype(np.float32)

    if show_bg:
        res_img = src_img.copy()
    else:
        res_img = np.zeros(src_img.shape, dtype=src_img.dtype)

    morph_points = []

    for i in range(0, len(src_points)):
        x = (1 - alpha) * src_points[i][0] + alpha * dst_points[i][0]
        y = (1 - alpha) * src_points[i][1] + alpha * dst_points[i][1]
        morph_points.append((x, y))

    dt = core.measure_triangle(src_img.shape, src_points)

    for i in range(0, len(dt)):
        t1 = []
        t2 = []
        t = []

        for j in range(0, 3):
            t1.append(src_points[dt[i][j]])
            t2.append(dst_points[dt[i][j]])
            t.append(morph_points[dt[i][j]])

        core.morph_triangle(src_img, dst_img, res_img, t1, t2, t, alpha)

    return res_img, morph_points


def face_merge(dst_img, dst_points, src_img, src_points,
               out_img, face_area, alpha=0.75,
               k_size=None, mat_multiple=None, is_peach=None):
    src_img = cv2.imread(src_img, cv2.IMREAD_COLOR)  # opencv读取原模板图像-打开方式彩色

    dst_img = cv2.imread(dst_img, cv2.IMREAD_COLOR)  # opencv读取目标模板图像-打开方式彩色
    # 二、对齐人脸角度
    # 在待融合图人像不是侧脸的情况下，我们可以同过调整平面位置及角度让其与模特图的人脸重合
    dst_img = tran_similarity(src_img, [src_points[core.LEFT_EYE_CENTER],
                                        src_points[core.RIGHT_EYE_CENTER],
                                        src_points[core.NOSE_TOP]],
                              dst_img, [dst_points[core.LEFT_EYE_CENTER],
                                        dst_points[core.RIGHT_EYE_CENTER],
                                        dst_points[core.NOSE_TOP]])

    dst_img = merge_img(src_img, dst_img, src_points, k_size, mat_multiple, is_peach)
    cv2.imwrite(out_img, dst_img)  # 将结果输出


def face_merge_nan_g(dst_img, dst_points, src_img, src_points,
                     out_img, face_area, alpha=0.75,
                     k_size=None, mat_multiple=None, is_peach=None):
    src_img = cv2.imread(src_img, cv2.IMREAD_COLOR)  # opencv读取原模板图像-打开方式彩色

    dst_img = cv2.imread(dst_img, cv2.IMREAD_COLOR)  # opencv读取目标模板图像-打开方式彩色
    # 二、对齐人脸角度
    # 在待融合图人像不是侧脸的情况下，我们可以同过调整平面位置及角度让其与模特图的人脸重合
    dst_img = tran_similarity(src_img, [src_points[core.LEFT_EYE_CENTER],
                                        src_points[core.RIGHT_EYE_CENTER],
                                        src_points[core.NOSE_TOP]],
                              dst_img, [dst_points[core.LEFT_EYE_CENTER],
                                        dst_points[core.RIGHT_EYE_CENTER],
                                        dst_points[core.NOSE_TOP]])

    dst_img = merge_img_nan_g(src_img, dst_img, src_points, k_size, mat_multiple, is_peach)
    cv2.imwrite(out_img, dst_img)  # 将结果输出


def face_merge1(dst_img, dst_points, src_img, src_points,
                out_img, face_area, alpha=0.75,
                k_size=None, mat_multiple=None, is_peach=None):
    src_img = cv2.imread(src_img, cv2.IMREAD_COLOR)  # opencv读取原模板图像-打开方式彩色

    dst_img = cv2.imread(dst_img, cv2.IMREAD_COLOR)  # opencv读取目标模板图像-打开方式彩色
    # 二、对齐人脸角度
    # 在待融合图人像不是侧脸的情况下，我们可以同过调整平面位置及角度让其与模特图的人脸重合
    dst_img = tran_similarity(src_img, [src_points[core.LEFT_EYE_CENTER],
                                        src_points[core.RIGHT_EYE_CENTER],
                                        src_points[core.NOSE_TOP]],
                              dst_img, [dst_points[core.LEFT_EYE_CENTER],
                                        dst_points[core.RIGHT_EYE_CENTER],
                                        dst_points[core.NOSE_TOP]])

    dst_img = merge_img1(src_img, dst_img, src_points, k_size, mat_multiple, is_peach)

    cv2.imwrite(out_img, dst_img)  # 将结果输出
