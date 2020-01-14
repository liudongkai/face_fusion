# -*- coding: utf-8 -*-

"""
recognize face landmark
"""
import json
import os
import requests

FACE_POINTS = list(range(0, 106))
JAW_POINTS = list(range(34, 102)) + list(range(15, 18)) + list(range(34, 72)) + list(range(88, 97))
LEFT_EYE_POINTS = list(range(33, 43))
LEFT_BROW_POINTS = list(range(43, 52))
MOUTH_POINTS = list(range(52, 72))
NOSE_POINTS = list(range(72, 87))
RIGHT_EYE_POINTS = list(range(87, 97))
RIGHT_BROW_POINTS = list(range(97, 106))
nv = list(range(34, 106)) + list(range(16, 18))
UPPER_LIP = list(range(52, 64))
LOWER_LIP = list(range(62, 72))

LEFT_FACE = list(range(0, 17)) + list(range(44, 52))
RIGHT_FACE = list(range(17, 33)) + list(range(88, 104))

JAW_END = 33
FACE_START = 0
FACE_END = 106
LEFT_EYE_CENTER = 33
RIGHT_EYE_CENTER = 92
NOSE_TOP = 76
NOSE_BRIDGE = 82
MOUTH_TOP = 62
NOSE_TOP1 = 75

OVERLAY_POINTS = [
    JAW_POINTS,
    # NOSE_POINTS + MOUTH_POINTS,
    # RIGHT_EYE_POINTS + LEFT_EYE_POINTS + RIGHT_BROW_POINTS + LEFT_BROW_POINTS

]

PEACH_POINTS = [
    nv,
    NOSE_POINTS + MOUTH_POINTS,
    RIGHT_EYE_POINTS + LEFT_EYE_POINTS + RIGHT_BROW_POINTS + LEFT_BROW_POINTS
]


# 一、 检测及关键的定位
#
def face_points(image):
    """
    人脸识别及五官定位
    :param image: 待识别图片
    :return: point_list, err
    """
    points = landmarks_by_face__(image)
    # print(dir(points))
    # print(points)
    faces = json.loads(points)['faces']  # faces = json.loads(points)['faces']取出K为face的人脸106个关键点的字典结构体的数组。
    # print(faces)
    if len(faces) == 0:
        err = 404
        return None, None, err
    else:
        err = 0

    point_list = face_marks(faces[0]['landmark'])  # 取出k为landmark的人脸特征坐标数组返回
    # print(point_list)
    are = faces[0]['face_rectangle']

    return point_list, are


def face_marks(res):
    pointer = [
        (res['contour_left1']['x'] - 5, res['contour_left1']['y']),
        (res['contour_left2']['x'] - 5, res['contour_left2']['y']),
        (res['contour_left3']['x'], res['contour_left3']['y']),
        (res['contour_left4']['x'], res['contour_left4']['y']),
        (res['contour_left5']['x'], res['contour_left5']['y']),
        (res['contour_left6']['x'], res['contour_left6']['y']),
        (res['contour_left7']['x'], res['contour_left7']['y']),
        (res['contour_left8']['x'], res['contour_left8']['y']),
        (res['contour_left9']['x'], res['contour_left9']['y']),
        (res['contour_left10']['x'], res['contour_left10']['y']),
        (res['contour_left11']['x'], res['contour_left11']['y']),
        (res['contour_left12']['x'], res['contour_left12']['y']),
        (res['contour_left13']['x'], res['contour_left13']['y']),
        (res['contour_left14']['x'], res['contour_left14']['y'] - 50),
        (res['contour_left15']['x'], res['contour_left15']['y'] - 50),
        (res['contour_left16']['x'], res['contour_left16']['y'] - 50),
        (res['contour_chin']['x'], res['contour_chin']['y'] - 50),
        (res['contour_right16']['x'], res['contour_right16']['y'] - 50),
        (res['contour_right15']['x'] - 4, res['contour_right15']['y'] - 50),
        (res['contour_right14']['x'] - 8, res['contour_right14']['y'] - 50),
        (res['contour_right13']['x'] - 12, res['contour_right13']['y']),
        (res['contour_right12']['x'] - 12, res['contour_right12']['y']),
        (res['contour_right11']['x'] - 10, res['contour_right11']['y']),
        (res['contour_right10']['x'] - 10, res['contour_right10']['y']),
        (res['contour_right9']['x'] - 8, res['contour_right9']['y']),
        (res['contour_right8']['x'] - 8, res['contour_right8']['y']),
        (res['contour_right7']['x'] - 8, res['contour_right7']['y']),
        (res['contour_right6']['x'] - 8, res['contour_right6']['y']),
        (res['contour_right5']['x'] - 8, res['contour_right5']['y']),
        (res['contour_right4']['x'] - 8, res['contour_right4']['y']),
        (res['contour_right3']['x'] - 8, res['contour_right3']['y']),
        (res['contour_right2']['x'] - 10, res['contour_right2']['y']),
        (res['contour_right1']['x'] - 10, res['contour_right1']['y']),

        (res['left_eye_left_corner']['x'], res['left_eye_left_corner']['y'] + 8),
        (res['left_eye_upper_left_quarter']['x'], res['left_eye_upper_left_quarter']['y']),
        (res['left_eye_top']['x'], res['left_eye_top']['y']),
        (res['left_eye_upper_right_quarter']['x'], res['left_eye_upper_right_quarter']['y']),
        (res['left_eye_right_corner']['x'], res['left_eye_right_corner']['y']),
        (res['left_eye_lower_right_quarter']['x'], res['left_eye_lower_right_quarter']['y']),
        (res['left_eye_bottom']['x'], res['left_eye_bottom']['y']),
        (res['left_eye_lower_left_quarter']['x'], res['left_eye_lower_left_quarter']['y']),
        (res['left_eye_pupil']['x'] + 2, res['left_eye_pupil']['y'] + 2),
        (res['left_eye_center']['x'], res['left_eye_center']['y']),

        (res['left_eyebrow_left_corner']['x'] - 30, res['left_eyebrow_left_corner']['y'] - 100),
        (res['left_eyebrow_upper_left_quarter']['x'] - 20, res['left_eyebrow_upper_left_quarter']['y'] - 80),
        (res['left_eyebrow_upper_middle']['x'] - 20, res['left_eyebrow_upper_middle']['y'] - 50),
        (res['left_eyebrow_upper_right_quarter']['x'], res['left_eyebrow_upper_right_quarter']['y'] - 50),
        (res['left_eyebrow_upper_right_corner']['x'], res['left_eyebrow_upper_right_corner']['y'] - 50),
        (res['left_eyebrow_lower_left_quarter']['x'], res['left_eyebrow_lower_left_quarter']['y']),
        (res['left_eyebrow_lower_middle']['x'], res['left_eyebrow_lower_middle']['y']),
        (res['left_eyebrow_lower_right_quarter']['x'], res['left_eyebrow_lower_right_quarter']['y']),
        (res['left_eyebrow_lower_right_corner']['x'], res['left_eyebrow_lower_right_corner']['y']),

        (res['mouth_upper_lip_left_contour1']['x'], res['mouth_upper_lip_left_contour1']['y']),
        (res['mouth_upper_lip_left_contour2']['x'], res['mouth_upper_lip_left_contour2']['y']),
        (res['mouth_upper_lip_left_contour3']['x'], res['mouth_upper_lip_left_contour3']['y']),
        (res['mouth_upper_lip_left_contour4']['x'], res['mouth_upper_lip_left_contour4']['y']),
        (res['mouth_upper_lip_right_contour1']['x'], res['mouth_upper_lip_right_contour1']['y']),
        (res['mouth_upper_lip_right_contour2']['x'], res['mouth_upper_lip_right_contour2']['y']),
        (res['mouth_upper_lip_right_contour3']['x'], res['mouth_upper_lip_right_contour3']['y']),
        (res['mouth_upper_lip_right_contour4']['x'], res['mouth_upper_lip_right_contour4']['y']),
        (res['mouth_upper_lip_top']['x'], res['mouth_upper_lip_top']['y']),
        (res['mouth_upper_lip_bottom']['x'], res['mouth_upper_lip_bottom']['y']),
        (res['mouth_left_corner']['x'], res['mouth_left_corner']['y']),
        (res['mouth_right_corner']['x'], res['mouth_right_corner']['y']),
        (res['mouth_lower_lip_right_contour1']['x'], res['mouth_lower_lip_right_contour1']['y']),
        (res['mouth_lower_lip_right_contour2']['x'], res['mouth_lower_lip_right_contour2']['y']),
        (res['mouth_lower_lip_right_contour3']['x'], res['mouth_lower_lip_right_contour3']['y']),
        (res['mouth_lower_lip_left_contour1']['x'], res['mouth_lower_lip_left_contour1']['y']),
        (res['mouth_lower_lip_left_contour2']['x'], res['mouth_lower_lip_left_contour2']['y']),
        (res['mouth_lower_lip_left_contour3']['x'], res['mouth_lower_lip_left_contour3']['y']),
        (res['mouth_lower_lip_top']['x'], res['mouth_lower_lip_top']['y']),
        (res['mouth_lower_lip_bottom']['x'], res['mouth_lower_lip_bottom']['y']),

        (res['nose_bridge1']['x'], res['nose_bridge1']['y']),
        (res['nose_bridge2']['x'], res['nose_bridge2']['y']),
        (res['nose_bridge3']['x'], res['nose_bridge3']['y']),
        (res['nose_tip']['x'], res['nose_tip']['y']),
        (res['nose_left_contour1']['x'], res['nose_left_contour1']['y']),
        (res['nose_left_contour2']['x'], res['nose_left_contour2']['y']),
        (res['nose_left_contour3']['x'], res['nose_left_contour3']['y']),
        (res['nose_left_contour4']['x'], res['nose_left_contour4']['y']),
        (res['nose_left_contour5']['x'], res['nose_left_contour5']['y']),
        (res['nose_middle_contour']['x'], res['nose_middle_contour']['y']),
        (res['nose_right_contour1']['x'], res['nose_right_contour1']['y']),
        (res['nose_right_contour2']['x'], res['nose_right_contour2']['y']),
        (res['nose_right_contour3']['x'], res['nose_right_contour3']['y']),
        (res['nose_right_contour4']['x'], res['nose_right_contour4']['y']),
        (res['nose_right_contour5']['x'], res['nose_right_contour5']['y']),

        (res['right_eye_left_corner']['x'], res['right_eye_left_corner']['y'] + 8),
        (res['right_eye_upper_left_quarter']['x'], res['right_eye_upper_left_quarter']['y']),
        (res['right_eye_top']['x'], res['right_eye_top']['y']),
        (res['right_eye_upper_right_quarter']['x'], res['right_eye_upper_right_quarter']['y']),
        (res['right_eye_right_corner']['x'], res['right_eye_right_corner']['y']),
        (res['right_eye_lower_right_quarter']['x'], res['right_eye_lower_right_quarter']['y']),
        (res['right_eye_bottom']['x'], res['right_eye_bottom']['y']),
        (res['right_eye_lower_left_quarter']['x'], res['right_eye_lower_left_quarter']['y']),
        (res['right_eye_pupil']['x'] + 2, res['right_eye_pupil']['y'] + 2),
        (res['right_eye_center']['x'], res['right_eye_center']['y']),

        (res['right_eyebrow_upper_left_corner']['x'], res['right_eyebrow_upper_left_corner']['y'] - 50),
        (res['right_eyebrow_upper_left_quarter']['x'], res['right_eyebrow_upper_left_quarter']['y'] - 50),
        (res['right_eyebrow_upper_middle']['x'], res['right_eyebrow_upper_middle']['y'] - 50),
        (res['right_eyebrow_upper_right_quarter']['x'], res['right_eyebrow_upper_right_quarter']['y'] - 80),
        (res['right_eyebrow_right_corner']['x'], res['right_eyebrow_right_corner']['y'] - 100),
        (res['right_eyebrow_lower_left_corner']['x'], res['right_eyebrow_lower_left_corner']['y']),
        (res['right_eyebrow_lower_left_quarter']['x'], res['right_eyebrow_lower_left_quarter']['y']),
        (res['right_eyebrow_lower_middle']['x'], res['right_eyebrow_lower_middle']['y']),
        (res['right_eyebrow_lower_right_quarter']['x'], res['right_eyebrow_lower_right_quarter']['y']),
    ]

    return pointer


def face_points1(image):
    """
    人脸识别及五官定位
    :param image: 待识别图片
    :return: point_list, err
    """
    points = landmarks_by_face__(image)
    # print(dir(points))
    # print(points)
    faces = json.loads(points)['faces']  # faces = json.loads(points)['faces']取出K为face的人脸106个关键点的字典结构体的数组。
    # print(faces)
    if len(faces) == 0:
        err = 404
        return None, None, err
    else:
        err = 0

    point_list = face_marks1(faces[0]['landmark'])  # 取出k为landmark的人脸特征坐标数组返回
    # print(point_list)
    are = faces[0]['face_rectangle']

    return point_list, are


def face_marks1(res):
    pointer1 = [
        (res['contour_left1']['x'] - 2, res['contour_left1']['y']),
        (res['contour_left2']['x'] - 2, res['contour_left2']['y']),
        (res['contour_left3']['x'], res['contour_left3']['y']),
        (res['contour_left4']['x'], res['contour_left4']['y']),
        (res['contour_left5']['x'], res['contour_left5']['y']),
        (res['contour_left6']['x'], res['contour_left6']['y']),
        (res['contour_left7']['x'], res['contour_left7']['y']),
        (res['contour_left8']['x'], res['contour_left8']['y']),
        (res['contour_left9']['x'], res['contour_left9']['y']),
        (res['contour_left10']['x'], res['contour_left10']['y']),
        (res['contour_left11']['x'], res['contour_left11']['y']),
        (res['contour_left12']['x'], res['contour_left12']['y']),
        (res['contour_left13']['x'] + 10, res['contour_left13']['y']),
        (res['contour_left14']['x'] + 10, res['contour_left14']['y']),
        (res['contour_left15']['x'] + 10, res['contour_left15']['y']),
        (res['contour_left16']['x'] + 10, res['contour_left16']['y']),
        (res['contour_chin']['x'], res['contour_chin']['y']),
        (res['contour_right16']['x'] - 7, res['contour_right16']['y']),
        (res['contour_right15']['x'] - 7, res['contour_right15']['y']),
        (res['contour_right14']['x'] - 7, res['contour_right14']['y']),
        (res['contour_right13']['x'] - 7, res['contour_right13']['y']),
        (res['contour_right12']['x'], res['contour_right12']['y']),
        (res['contour_right11']['x'], res['contour_right11']['y']),
        (res['contour_right10']['x'], res['contour_right10']['y']),
        (res['contour_right9']['x'], res['contour_right9']['y']),
        (res['contour_right8']['x'], res['contour_right8']['y']),
        (res['contour_right7']['x'], res['contour_right7']['y']),
        (res['contour_right6']['x'], res['contour_right6']['y']),
        (res['contour_right5']['x'], res['contour_right5']['y']),
        (res['contour_right4']['x'], res['contour_right4']['y']),
        (res['contour_right3']['x'], res['contour_right3']['y']),
        (res['contour_right2']['x'], res['contour_right2']['y']),
        (res['contour_right1']['x'], res['contour_right1']['y']),

        (res['left_eye_left_corner']['x'], res['left_eye_left_corner']['y']),
        (res['left_eye_upper_left_quarter']['x'], res['left_eye_upper_left_quarter']['y']),
        (res['left_eye_top']['x'], res['left_eye_top']['y']),
        (res['left_eye_upper_right_quarter']['x'], res['left_eye_upper_right_quarter']['y']),
        (res['left_eye_right_corner']['x'], res['left_eye_right_corner']['y']),
        (res['left_eye_lower_right_quarter']['x'], res['left_eye_lower_right_quarter']['y']),
        (res['left_eye_bottom']['x'], res['left_eye_bottom']['y']),
        (res['left_eye_lower_left_quarter']['x'], res['left_eye_lower_left_quarter']['y']),
        (res['left_eye_pupil']['x'] + 2, res['left_eye_pupil']['y'] + 2),
        (res['left_eye_center']['x'], res['left_eye_center']['y']),

        (res['left_eyebrow_left_corner']['x'], res['left_eyebrow_left_corner']['y'] - 90),
        (res['left_eyebrow_upper_left_quarter']['x'], res['left_eyebrow_upper_left_quarter']['y'] - 40),
        (res['left_eyebrow_upper_middle']['x'], res['left_eyebrow_upper_middle']['y'] - 70),
        (res['left_eyebrow_upper_right_quarter']['x'], res['left_eyebrow_upper_right_quarter']['y'] - 70),
        (res['left_eyebrow_upper_right_corner']['x'], res['left_eyebrow_upper_right_corner']['y'] - 70),
        (res['left_eyebrow_lower_left_quarter']['x'], res['left_eyebrow_lower_left_quarter']['y']),
        (res['left_eyebrow_lower_middle']['x'], res['left_eyebrow_lower_middle']['y']),
        (res['left_eyebrow_lower_right_quarter']['x'], res['left_eyebrow_lower_right_quarter']['y']),
        (res['left_eyebrow_lower_right_corner']['x'], res['left_eyebrow_lower_right_corner']['y']),

        (res['mouth_upper_lip_left_contour1']['x'], res['mouth_upper_lip_left_contour1']['y']),
        (res['mouth_upper_lip_left_contour2']['x'], res['mouth_upper_lip_left_contour2']['y']),
        (res['mouth_upper_lip_left_contour3']['x'], res['mouth_upper_lip_left_contour3']['y']),
        (res['mouth_upper_lip_left_contour4']['x'], res['mouth_upper_lip_left_contour4']['y']),
        (res['mouth_upper_lip_right_contour1']['x'], res['mouth_upper_lip_right_contour1']['y']),
        (res['mouth_upper_lip_right_contour2']['x'], res['mouth_upper_lip_right_contour2']['y']),
        (res['mouth_upper_lip_right_contour3']['x'], res['mouth_upper_lip_right_contour3']['y']),
        (res['mouth_upper_lip_right_contour4']['x'], res['mouth_upper_lip_right_contour4']['y']),
        (res['mouth_upper_lip_top']['x'], res['mouth_upper_lip_top']['y']),
        (res['mouth_upper_lip_bottom']['x'], res['mouth_upper_lip_bottom']['y']),
        (res['mouth_left_corner']['x'], res['mouth_left_corner']['y']),
        (res['mouth_right_corner']['x'], res['mouth_right_corner']['y']),
        (res['mouth_lower_lip_right_contour1']['x'], res['mouth_lower_lip_right_contour1']['y']),
        (res['mouth_lower_lip_right_contour2']['x'], res['mouth_lower_lip_right_contour2']['y']),
        (res['mouth_lower_lip_right_contour3']['x'], res['mouth_lower_lip_right_contour3']['y']),
        (res['mouth_lower_lip_left_contour1']['x'], res['mouth_lower_lip_left_contour1']['y']),
        (res['mouth_lower_lip_left_contour2']['x'], res['mouth_lower_lip_left_contour2']['y']),
        (res['mouth_lower_lip_left_contour3']['x'], res['mouth_lower_lip_left_contour3']['y']),
        (res['mouth_lower_lip_top']['x'], res['mouth_lower_lip_top']['y']),
        (res['mouth_lower_lip_bottom']['x'], res['mouth_lower_lip_bottom']['y']),

        (res['nose_bridge1']['x'], res['nose_bridge1']['y']),
        (res['nose_bridge2']['x'], res['nose_bridge2']['y']),
        (res['nose_bridge3']['x'], res['nose_bridge3']['y']),
        (res['nose_tip']['x'], res['nose_tip']['y']),
        (res['nose_left_contour1']['x'], res['nose_left_contour1']['y']),
        (res['nose_left_contour2']['x'], res['nose_left_contour2']['y']),
        (res['nose_left_contour3']['x'], res['nose_left_contour3']['y']),
        (res['nose_left_contour4']['x'], res['nose_left_contour4']['y']),
        (res['nose_left_contour5']['x'], res['nose_left_contour5']['y']),
        (res['nose_middle_contour']['x'], res['nose_middle_contour']['y']),
        (res['nose_right_contour1']['x'], res['nose_right_contour1']['y']),
        (res['nose_right_contour2']['x'], res['nose_right_contour2']['y']),
        (res['nose_right_contour3']['x'], res['nose_right_contour3']['y']),
        (res['nose_right_contour4']['x'], res['nose_right_contour4']['y']),
        (res['nose_right_contour5']['x'], res['nose_right_contour5']['y']),

        (res['right_eye_left_corner']['x'], res['right_eye_left_corner']['y']),
        (res['right_eye_upper_left_quarter']['x'], res['right_eye_upper_left_quarter']['y']),
        (res['right_eye_top']['x'], res['right_eye_top']['y']),
        (res['right_eye_upper_right_quarter']['x'], res['right_eye_upper_right_quarter']['y']),
        (res['right_eye_right_corner']['x'], res['right_eye_right_corner']['y']),
        (res['right_eye_lower_right_quarter']['x'], res['right_eye_lower_right_quarter']['y']),
        (res['right_eye_bottom']['x'], res['right_eye_bottom']['y']),
        (res['right_eye_lower_left_quarter']['x'], res['right_eye_lower_left_quarter']['y']),
        (res['right_eye_pupil']['x'] + 2, res['right_eye_pupil']['y'] + 2),
        (res['right_eye_center']['x'], res['right_eye_center']['y']),

        (res['right_eyebrow_upper_left_corner']['x'], res['right_eyebrow_upper_left_corner']['y'] - 70),
        (res['right_eyebrow_upper_left_quarter']['x'], res['right_eyebrow_upper_left_quarter']['y'] - 70),
        (res['right_eyebrow_upper_middle']['x'], res['right_eyebrow_upper_middle']['y'] - 70),
        (res['right_eyebrow_upper_right_quarter']['x'], res['right_eyebrow_upper_right_quarter']['y'] - 40),
        (res['right_eyebrow_right_corner']['x'], res['right_eyebrow_right_corner']['y'] - 90),
        (res['right_eyebrow_lower_left_corner']['x'], res['right_eyebrow_lower_left_corner']['y']),
        (res['right_eyebrow_lower_left_quarter']['x'], res['right_eyebrow_lower_left_quarter']['y']),
        (res['right_eyebrow_lower_middle']['x'], res['right_eyebrow_lower_middle']['y']),
        (res['right_eyebrow_lower_right_quarter']['x'], res['right_eyebrow_lower_right_quarter']['y']),
    ]

    return pointer1


def face_points2(image):
    """
    人脸识别及五官定位
    :param image: 待识别图片
    :return: point_list, err
    """
    points = landmarks_by_face__(image)
    # print(dir(points))
    # print(points)
    faces = json.loads(points)['faces']  # faces = json.loads(points)['faces']取出K为face的人脸106个关键点的字典结构体的数组。
    # print(faces)
    if len(faces) == 0:
        err = 404
        return None, None, err
    else:
        err = 0

    point_list = face_marks2(faces[0]['landmark'])  # 取出k为landmark的人脸特征坐标数组返回
    # print(point_list)
    return point_list


def face_marks2(res):
    pointer2 = [
        (res['contour_left1']['x'], res['contour_left1']['y']),
        (res['contour_left2']['x'], res['contour_left2']['y']),
        (res['contour_left3']['x'], res['contour_left3']['y']),
        (res['contour_left4']['x'], res['contour_left4']['y']),
        (res['contour_left5']['x'], res['contour_left5']['y']),
        (res['contour_left6']['x'], res['contour_left6']['y']),
        (res['contour_left7']['x'], res['contour_left7']['y']),
        (res['contour_left8']['x'], res['contour_left8']['y']),
        (res['contour_left9']['x'], res['contour_left9']['y']),
        (res['contour_left10']['x'], res['contour_left10']['y']),
        (res['contour_left11']['x'] + 8, res['contour_left11']['y']),
        (res['contour_left12']['x'] + 8, res['contour_left12']['y']),
        (res['contour_left13']['x'] + 8, res['contour_left13']['y'] - 70),
        (res['contour_left14']['x'] + 8, res['contour_left14']['y'] - 70),
        (res['contour_left15']['x'], res['contour_left15']['y'] - 70),
        (res['contour_left16']['x'], res['contour_left16']['y'] - 70),
        (res['contour_chin']['x'], res['contour_chin']['y'] - 70),
        (res['contour_right16']['x'] - 8, res['contour_right16']['y'] - 70),
        (res['contour_right15']['x'] - 8, res['contour_right15']['y'] - 70),
        (res['contour_right14']['x'] - 12, res['contour_right14']['y'] - 70),
        (res['contour_right13']['x'] - 12, res['contour_right13']['y'] - 70),
        (res['contour_right12']['x'] - 10, res['contour_right12']['y']),
        (res['contour_right11']['x'] - 10, res['contour_right11']['y']),
        (res['contour_right10']['x'] - 7, res['contour_right10']['y']),
        (res['contour_right9']['x'] - 7, res['contour_right9']['y']),
        (res['contour_right8']['x'] - 7, res['contour_right8']['y']),
        (res['contour_right7']['x'] - 7, res['contour_right7']['y']),
        (res['contour_right6']['x'] - 7, res['contour_right6']['y']),
        (res['contour_right5']['x'] - 7, res['contour_right5']['y']),
        (res['contour_right4']['x'] - 7, res['contour_right4']['y']),
        (res['contour_right3']['x'] - 5, res['contour_right3']['y']),
        (res['contour_right2']['x'] - 5, res['contour_right2']['y']),
        (res['contour_right1']['x'] - 5, res['contour_right1']['y']),

        (res['left_eye_left_corner']['x'], res['left_eye_left_corner']['y']),
        (res['left_eye_upper_left_quarter']['x'], res['left_eye_upper_left_quarter']['y']),
        (res['left_eye_top']['x'], res['left_eye_top']['y']),
        (res['left_eye_upper_right_quarter']['x'], res['left_eye_upper_right_quarter']['y']),
        (res['left_eye_right_corner']['x'], res['left_eye_right_corner']['y']),
        (res['left_eye_lower_right_quarter']['x'], res['left_eye_lower_right_quarter']['y']),
        (res['left_eye_bottom']['x'], res['left_eye_bottom']['y']),
        (res['left_eye_lower_left_quarter']['x'], res['left_eye_lower_left_quarter']['y']),
        (res['left_eye_pupil']['x'] + 2, res['left_eye_pupil']['y'] + 2),
        (res['left_eye_center']['x'], res['left_eye_center']['y']),

        (res['left_eyebrow_left_corner']['x'], res['left_eyebrow_left_corner']['y']),
        (res['left_eyebrow_upper_left_quarter']['x'], res['left_eyebrow_upper_left_quarter']['y']),
        (res['left_eyebrow_upper_middle']['x'], res['left_eyebrow_upper_middle']['y']),
        (res['left_eyebrow_upper_right_quarter']['x'], res['left_eyebrow_upper_right_quarter']['y']),
        (res['left_eyebrow_upper_right_corner']['x'], res['left_eyebrow_upper_right_corner']['y']),
        (res['left_eyebrow_lower_left_quarter']['x'], res['left_eyebrow_lower_left_quarter']['y']),
        (res['left_eyebrow_lower_middle']['x'], res['left_eyebrow_lower_middle']['y']),
        (res['left_eyebrow_lower_right_quarter']['x'], res['left_eyebrow_lower_right_quarter']['y']),
        (res['left_eyebrow_lower_right_corner']['x'], res['left_eyebrow_lower_right_corner']['y']),

        (res['mouth_upper_lip_left_contour1']['x'], res['mouth_upper_lip_left_contour1']['y']),
        (res['mouth_upper_lip_left_contour2']['x'], res['mouth_upper_lip_left_contour2']['y']),
        (res['mouth_upper_lip_left_contour3']['x'], res['mouth_upper_lip_left_contour3']['y']),
        (res['mouth_upper_lip_left_contour4']['x'], res['mouth_upper_lip_left_contour4']['y']),
        (res['mouth_upper_lip_right_contour1']['x'], res['mouth_upper_lip_right_contour1']['y']),
        (res['mouth_upper_lip_right_contour2']['x'], res['mouth_upper_lip_right_contour2']['y']),
        (res['mouth_upper_lip_right_contour3']['x'], res['mouth_upper_lip_right_contour3']['y']),
        (res['mouth_upper_lip_right_contour4']['x'], res['mouth_upper_lip_right_contour4']['y']),
        (res['mouth_upper_lip_top']['x'], res['mouth_upper_lip_top']['y']),
        (res['mouth_upper_lip_bottom']['x'], res['mouth_upper_lip_bottom']['y']),
        (res['mouth_left_corner']['x'], res['mouth_left_corner']['y']),
        (res['mouth_right_corner']['x'], res['mouth_right_corner']['y']),
        (res['mouth_lower_lip_right_contour1']['x'], res['mouth_lower_lip_right_contour1']['y']),
        (res['mouth_lower_lip_right_contour2']['x'], res['mouth_lower_lip_right_contour2']['y']),
        (res['mouth_lower_lip_right_contour3']['x'], res['mouth_lower_lip_right_contour3']['y']),
        (res['mouth_lower_lip_left_contour1']['x'], res['mouth_lower_lip_left_contour1']['y']),
        (res['mouth_lower_lip_left_contour2']['x'], res['mouth_lower_lip_left_contour2']['y']),
        (res['mouth_lower_lip_left_contour3']['x'], res['mouth_lower_lip_left_contour3']['y']),
        (res['mouth_lower_lip_top']['x'], res['mouth_lower_lip_top']['y']),
        (res['mouth_lower_lip_bottom']['x'], res['mouth_lower_lip_bottom']['y']),

        (res['nose_bridge1']['x'], res['nose_bridge1']['y']),
        (res['nose_bridge2']['x'], res['nose_bridge2']['y']),
        (res['nose_bridge3']['x'], res['nose_bridge3']['y']),
        (res['nose_tip']['x'], res['nose_tip']['y']),
        (res['nose_left_contour1']['x'], res['nose_left_contour1']['y']),
        (res['nose_left_contour2']['x'], res['nose_left_contour2']['y']),
        (res['nose_left_contour3']['x'], res['nose_left_contour3']['y']),
        (res['nose_left_contour4']['x'], res['nose_left_contour4']['y']),
        (res['nose_left_contour5']['x'], res['nose_left_contour5']['y']),
        (res['nose_middle_contour']['x'], res['nose_middle_contour']['y']),
        (res['nose_right_contour1']['x'], res['nose_right_contour1']['y']),
        (res['nose_right_contour2']['x'], res['nose_right_contour2']['y']),
        (res['nose_right_contour3']['x'], res['nose_right_contour3']['y']),
        (res['nose_right_contour4']['x'], res['nose_right_contour4']['y']),
        (res['nose_right_contour5']['x'], res['nose_right_contour5']['y']),

        (res['right_eye_left_corner']['x'], res['right_eye_left_corner']['y']),
        (res['right_eye_upper_left_quarter']['x'], res['right_eye_upper_left_quarter']['y']),
        (res['right_eye_top']['x'], res['right_eye_top']['y']),
        (res['right_eye_upper_right_quarter']['x'], res['right_eye_upper_right_quarter']['y']),
        (res['right_eye_right_corner']['x'], res['right_eye_right_corner']['y']),
        (res['right_eye_lower_right_quarter']['x'], res['right_eye_lower_right_quarter']['y']),
        (res['right_eye_bottom']['x'], res['right_eye_bottom']['y']),
        (res['right_eye_lower_left_quarter']['x'], res['right_eye_lower_left_quarter']['y']),
        (res['right_eye_pupil']['x'] + 2, res['right_eye_pupil']['y'] + 2),
        (res['right_eye_center']['x'], res['right_eye_center']['y']),

        (res['right_eyebrow_upper_left_corner']['x'], res['right_eyebrow_upper_left_corner']['y']),
        (res['right_eyebrow_upper_left_quarter']['x'], res['right_eyebrow_upper_left_quarter']['y']),
        (res['right_eyebrow_upper_middle']['x'], res['right_eyebrow_upper_middle']['y']),
        (res['right_eyebrow_upper_right_quarter']['x'], res['right_eyebrow_upper_right_quarter']['y']),
        (res['right_eyebrow_right_corner']['x'], res['right_eyebrow_right_corner']['y']),
        (res['right_eyebrow_lower_left_corner']['x'], res['right_eyebrow_lower_left_corner']['y']),
        (res['right_eyebrow_lower_left_quarter']['x'], res['right_eyebrow_lower_left_quarter']['y']),
        (res['right_eyebrow_lower_middle']['x'], res['right_eyebrow_lower_middle']['y']),
        (res['right_eyebrow_lower_right_quarter']['x'], res['right_eyebrow_lower_right_quarter']['y']),
    ]

    return pointer2


def face_points3(image):
    """
    人脸识别及五官定位
    :param image: 待识别图片
    :return: point_list, err
    """
    points = landmarks_by_face__(image)
    # print(dir(points))
    # print(points)
    faces = json.loads(points)['faces']  # faces = json.loads(points)['faces']取出K为face的人脸106个关键点的字典结构体的数组。
    # print(faces)
    if len(faces) == 0:
        err = 404
        return None, None, err
    else:
        err = 0

    point_list = face_marks3(faces[0]['landmark'])  # 取出k为landmark的人脸特征坐标数组返回

    return point_list


def face_marks3(res):
    pointer2 = [
        (res['contour_left1']['x'], res['contour_left1']['y'] + 50),
        (res['contour_left2']['x'], res['contour_left2']['y'] + 50),
        (res['contour_left3']['x'], res['contour_left3']['y'] + 50),
        (res['contour_left4']['x'], res['contour_left4']['y'] + 50),
        (res['contour_left5']['x'], res['contour_left5']['y'] + 50),
        (res['contour_left6']['x'], res['contour_left6']['y'] + 50),
        (res['contour_left7']['x'], res['contour_left7']['y'] + 50),
        (res['contour_left8']['x'], res['contour_left8']['y'] + 50),
        (res['contour_left9']['x'], res['contour_left9']['y'] + 50),
        (res['contour_left10']['x'], res['contour_left10']['y'] + 50),
        (res['contour_left11']['x'] + 8, res['contour_left11']['y'] + 50),
        (res['contour_left12']['x'] + 8, res['contour_left12']['y'] + 50),
        (res['contour_left13']['x'] + 8, res['contour_left13']['y'] + 50),
        (res['contour_left14']['x'] + 8, res['contour_left14']['y'] + 50),
        (res['contour_left15']['x'], res['contour_left15']['y'] + 50),
        (res['contour_left16']['x'], res['contour_left16']['y'] + 50),
        (res['contour_chin']['x'], res['contour_chin']['y'] + 50),
        (res['contour_right16']['x'] - 8, res['contour_right16']['y'] + 50),
        (res['contour_right15']['x'] - 8, res['contour_right15']['y'] + 50),
        (res['contour_right14']['x'] - 12, res['contour_right14']['y'] + 50),
        (res['contour_right13']['x'] - 12, res['contour_right13']['y'] + 50),
        (res['contour_right12']['x'] - 10, res['contour_right12']['y'] + 50),
        (res['contour_right11']['x'] - 10, res['contour_right11']['y'] + 50),
        (res['contour_right10']['x'] - 7, res['contour_right10']['y'] + 50),
        (res['contour_right9']['x'] - 7, res['contour_right9']['y'] + 50),
        (res['contour_right8']['x'] - 7, res['contour_right8']['y'] + 50),
        (res['contour_right7']['x'] - 7, res['contour_right7']['y'] + 50),
        (res['contour_right6']['x'] - 7, res['contour_right6']['y'] + 50),
        (res['contour_right5']['x'] - 7, res['contour_right5']['y'] + 50),
        (res['contour_right4']['x'] - 7, res['contour_right4']['y'] + 50),
        (res['contour_right3']['x'] - 5, res['contour_right3']['y'] + 50),
        (res['contour_right2']['x'] - 5, res['contour_right2']['y'] + 50),
        (res['contour_right1']['x'] - 5, res['contour_right1']['y'] + 50),

        (res['left_eye_left_corner']['x'], res['left_eye_left_corner']['y']),
        (res['left_eye_upper_left_quarter']['x'], res['left_eye_upper_left_quarter']['y']),
        (res['left_eye_top']['x'], res['left_eye_top']['y']),
        (res['left_eye_upper_right_quarter']['x'], res['left_eye_upper_right_quarter']['y']),
        (res['left_eye_right_corner']['x'], res['left_eye_right_corner']['y']),
        (res['left_eye_lower_right_quarter']['x'], res['left_eye_lower_right_quarter']['y']),
        (res['left_eye_bottom']['x'], res['left_eye_bottom']['y']),
        (res['left_eye_lower_left_quarter']['x'], res['left_eye_lower_left_quarter']['y']),
        (res['left_eye_pupil']['x'] + 2, res['left_eye_pupil']['y'] + 2),
        (res['left_eye_center']['x'], res['left_eye_center']['y']),

        (res['left_eyebrow_left_corner']['x'], res['left_eyebrow_left_corner']['y']),
        (res['left_eyebrow_upper_left_quarter']['x'], res['left_eyebrow_upper_left_quarter']['y']),
        (res['left_eyebrow_upper_middle']['x'], res['left_eyebrow_upper_middle']['y']),
        (res['left_eyebrow_upper_right_quarter']['x'], res['left_eyebrow_upper_right_quarter']['y']),
        (res['left_eyebrow_upper_right_corner']['x'], res['left_eyebrow_upper_right_corner']['y']),
        (res['left_eyebrow_lower_left_quarter']['x'], res['left_eyebrow_lower_left_quarter']['y']),
        (res['left_eyebrow_lower_middle']['x'], res['left_eyebrow_lower_middle']['y']),
        (res['left_eyebrow_lower_right_quarter']['x'], res['left_eyebrow_lower_right_quarter']['y']),
        (res['left_eyebrow_lower_right_corner']['x'], res['left_eyebrow_lower_right_corner']['y']),

        (res['mouth_upper_lip_left_contour1']['x'], res['mouth_upper_lip_left_contour1']['y'] - 50),
        (res['mouth_upper_lip_left_contour2']['x'], res['mouth_upper_lip_left_contour2']['y'] - 50),
        (res['mouth_upper_lip_left_contour3']['x'], res['mouth_upper_lip_left_contour3']['y'] - 50),
        (res['mouth_upper_lip_left_contour4']['x'], res['mouth_upper_lip_left_contour4']['y'] - 50),
        (res['mouth_upper_lip_right_contour1']['x'], res['mouth_upper_lip_right_contour1']['y'] - 50),
        (res['mouth_upper_lip_right_contour2']['x'], res['mouth_upper_lip_right_contour2']['y'] - 50),
        (res['mouth_upper_lip_right_contour3']['x'], res['mouth_upper_lip_right_contour3']['y'] - 50),
        (res['mouth_upper_lip_right_contour4']['x'], res['mouth_upper_lip_right_contour4']['y'] - 50),
        (res['mouth_upper_lip_top']['x'], res['mouth_upper_lip_top']['y'] - 50),
        (res['mouth_upper_lip_bottom']['x'], res['mouth_upper_lip_bottom']['y'] - 50),
        (res['mouth_left_corner']['x'], res['mouth_left_corner']['y'] - 50),
        (res['mouth_right_corner']['x'], res['mouth_right_corner']['y'] - 50),
        (res['mouth_lower_lip_right_contour1']['x'], res['mouth_lower_lip_right_contour1']['y'] - 50),
        (res['mouth_lower_lip_right_contour2']['x'], res['mouth_lower_lip_right_contour2']['y'] - 50),
        (res['mouth_lower_lip_right_contour3']['x'], res['mouth_lower_lip_right_contour3']['y'] - 50),
        (res['mouth_lower_lip_left_contour1']['x'], res['mouth_lower_lip_left_contour1']['y'] - 50),
        (res['mouth_lower_lip_left_contour2']['x'], res['mouth_lower_lip_left_contour2']['y'] - 50),
        (res['mouth_lower_lip_left_contour3']['x'], res['mouth_lower_lip_left_contour3']['y'] - 50),
        (res['mouth_lower_lip_top']['x'], res['mouth_lower_lip_top']['y'] - 50),
        (res['mouth_lower_lip_bottom']['x'], res['mouth_lower_lip_bottom']['y'] - 50),

        (res['nose_bridge1']['x'], res['nose_bridge1']['y']),
        (res['nose_bridge2']['x'], res['nose_bridge2']['y']),
        (res['nose_bridge3']['x'], res['nose_bridge3']['y']),
        (res['nose_tip']['x'], res['nose_tip']['y']),
        (res['nose_left_contour1']['x'], res['nose_left_contour1']['y']),
        (res['nose_left_contour2']['x'], res['nose_left_contour2']['y']),
        (res['nose_left_contour3']['x'], res['nose_left_contour3']['y']),
        (res['nose_left_contour4']['x'], res['nose_left_contour4']['y']),
        (res['nose_left_contour5']['x'], res['nose_left_contour5']['y']),
        (res['nose_middle_contour']['x'], res['nose_middle_contour']['y']),
        (res['nose_right_contour1']['x'], res['nose_right_contour1']['y']),
        (res['nose_right_contour2']['x'], res['nose_right_contour2']['y']),
        (res['nose_right_contour3']['x'], res['nose_right_contour3']['y']),
        (res['nose_right_contour4']['x'], res['nose_right_contour4']['y']),
        (res['nose_right_contour5']['x'], res['nose_right_contour5']['y']),

        (res['right_eye_left_corner']['x'], res['right_eye_left_corner']['y']),
        (res['right_eye_upper_left_quarter']['x'], res['right_eye_upper_left_quarter']['y']),
        (res['right_eye_top']['x'], res['right_eye_top']['y']),
        (res['right_eye_upper_right_quarter']['x'], res['right_eye_upper_right_quarter']['y']),
        (res['right_eye_right_corner']['x'], res['right_eye_right_corner']['y']),
        (res['right_eye_lower_right_quarter']['x'], res['right_eye_lower_right_quarter']['y']),
        (res['right_eye_bottom']['x'], res['right_eye_bottom']['y']),
        (res['right_eye_lower_left_quarter']['x'], res['right_eye_lower_left_quarter']['y']),
        (res['right_eye_pupil']['x'] + 2, res['right_eye_pupil']['y'] + 2),
        (res['right_eye_center']['x'], res['right_eye_center']['y']),

        (res['right_eyebrow_upper_left_corner']['x'], res['right_eyebrow_upper_left_corner']['y']),
        (res['right_eyebrow_upper_left_quarter']['x'], res['right_eyebrow_upper_left_quarter']['y']),
        (res['right_eyebrow_upper_middle']['x'], res['right_eyebrow_upper_middle']['y']),
        (res['right_eyebrow_upper_right_quarter']['x'], res['right_eyebrow_upper_right_quarter']['y']),
        (res['right_eyebrow_right_corner']['x'], res['right_eyebrow_right_corner']['y']),
        (res['right_eyebrow_lower_left_corner']['x'], res['right_eyebrow_lower_left_corner']['y']),
        (res['right_eyebrow_lower_left_quarter']['x'], res['right_eyebrow_lower_left_quarter']['y']),
        (res['right_eyebrow_lower_middle']['x'], res['right_eyebrow_lower_middle']['y']),
        (res['right_eyebrow_lower_right_quarter']['x'], res['right_eyebrow_lower_right_quarter']['y']),
    ]

    return pointer2


def face_points4(image):
    """
    人脸识别及五官定位
    :param image: 待识别图片
    :return: point_list, err
    """
    points = landmarks_by_face__(image)
    faces = json.loads(points)['faces']
    if len(faces) == 0:
        err = 404
        return None, None, err
    else:
        err = 0

    point_list = face_marks4(faces[0]['landmark'])  # 取出k为landmark的人脸特征坐标数组返回

    return point_list


def face_marks4(res):
    pointer4 = [
        (res['contour_left1']['x'], res['contour_left1']['y']),
        (res['contour_left2']['x'], res['contour_left2']['y']),
        (res['contour_left3']['x'], res['contour_left3']['y']),
        (res['contour_left4']['x'], res['contour_left4']['y']),
        (res['contour_left5']['x'], res['contour_left5']['y']),
        (res['contour_left6']['x'], res['contour_left6']['y']),
        (res['contour_left7']['x'], res['contour_left7']['y']),
        (res['contour_left8']['x'], res['contour_left8']['y']),
        (res['contour_left9']['x'], res['contour_left9']['y']),
        (res['contour_left10']['x'], res['contour_left10']['y']),
        (res['contour_left11']['x'], res['contour_left11']['y']),
        (res['contour_left12']['x'], res['contour_left12']['y']),
        (res['contour_left13']['x'], res['contour_left13']['y']),
        (res['contour_left14']['x'], res['contour_left14']['y']),
        (res['contour_left15']['x'], res['contour_left15']['y']),
        (res['contour_left16']['x'], res['contour_left16']['y']),
        (res['contour_chin']['x'], res['contour_chin']['y']),
        (res['contour_right16']['x'] - 8, res['contour_right16']['y']),
        (res['contour_right15']['x'] - 4, res['contour_right15']['y']),
        (res['contour_right14']['x'] - 8, res['contour_right14']['y']),
        (res['contour_right13']['x'] - 12, res['contour_right13']['y']),
        (res['contour_right12']['x'] - 12, res['contour_right12']['y']),
        (res['contour_right11']['x'] - 10, res['contour_right11']['y']),
        (res['contour_right10']['x'] - 10, res['contour_right10']['y']),
        (res['contour_right9']['x'] - 8, res['contour_right9']['y']),
        (res['contour_right8']['x'] - 8, res['contour_right8']['y']),
        (res['contour_right7']['x'] - 8, res['contour_right7']['y']),
        (res['contour_right6']['x'] - 8, res['contour_right6']['y']),
        (res['contour_right5']['x'] - 8, res['contour_right5']['y']),
        (res['contour_right4']['x'] - 8, res['contour_right4']['y']),
        (res['contour_right3']['x'] - 8, res['contour_right3']['y']),
        (res['contour_right2']['x'] - 10, res['contour_right2']['y']),
        (res['contour_right1']['x'] - 10, res['contour_right1']['y']),

        (res['left_eye_left_corner']['x'], res['left_eye_left_corner']['y']),
        (res['left_eye_upper_left_quarter']['x'], res['left_eye_upper_left_quarter']['y']),
        (res['left_eye_top']['x'], res['left_eye_top']['y']),
        (res['left_eye_upper_right_quarter']['x'], res['left_eye_upper_right_quarter']['y']),
        (res['left_eye_right_corner']['x'], res['left_eye_right_corner']['y']),
        (res['left_eye_lower_right_quarter']['x'], res['left_eye_lower_right_quarter']['y']),
        (res['left_eye_bottom']['x'], res['left_eye_bottom']['y']),
        (res['left_eye_lower_left_quarter']['x'], res['left_eye_lower_left_quarter']['y']),
        (res['left_eye_pupil']['x'] + 2, res['left_eye_pupil']['y'] + 2),
        (res['left_eye_center']['x'], res['left_eye_center']['y']),

        (res['left_eyebrow_left_corner']['x'], res['left_eyebrow_left_corner']['y']),
        (res['left_eyebrow_upper_left_quarter']['x'], res['left_eyebrow_upper_left_quarter']['y']),
        (res['left_eyebrow_upper_middle']['x'], res['left_eyebrow_upper_middle']['y']),
        (res['left_eyebrow_upper_right_quarter']['x'], res['left_eyebrow_upper_right_quarter']['y']),
        (res['left_eyebrow_upper_right_corner']['x'], res['left_eyebrow_upper_right_corner']['y']),
        (res['left_eyebrow_lower_left_quarter']['x'], res['left_eyebrow_lower_left_quarter']['y']),
        (res['left_eyebrow_lower_middle']['x'], res['left_eyebrow_lower_middle']['y']),
        (res['left_eyebrow_lower_right_quarter']['x'], res['left_eyebrow_lower_right_quarter']['y']),
        (res['left_eyebrow_lower_right_corner']['x'], res['left_eyebrow_lower_right_corner']['y']),

        (res['mouth_upper_lip_left_contour1']['x'], res['mouth_upper_lip_left_contour1']['y']),
        (res['mouth_upper_lip_left_contour2']['x'], res['mouth_upper_lip_left_contour2']['y']),
        (res['mouth_upper_lip_left_contour3']['x'], res['mouth_upper_lip_left_contour3']['y']),
        (res['mouth_upper_lip_left_contour4']['x'], res['mouth_upper_lip_left_contour4']['y']),
        (res['mouth_upper_lip_right_contour1']['x'], res['mouth_upper_lip_right_contour1']['y']),
        (res['mouth_upper_lip_right_contour2']['x'], res['mouth_upper_lip_right_contour2']['y']),
        (res['mouth_upper_lip_right_contour3']['x'], res['mouth_upper_lip_right_contour3']['y']),
        (res['mouth_upper_lip_right_contour4']['x'], res['mouth_upper_lip_right_contour4']['y']),
        (res['mouth_upper_lip_top']['x'], res['mouth_upper_lip_top']['y']),
        (res['mouth_upper_lip_bottom']['x'], res['mouth_upper_lip_bottom']['y']),
        (res['mouth_left_corner']['x'], res['mouth_left_corner']['y']),
        (res['mouth_right_corner']['x'], res['mouth_right_corner']['y']),
        (res['mouth_lower_lip_right_contour1']['x'], res['mouth_lower_lip_right_contour1']['y']),
        (res['mouth_lower_lip_right_contour2']['x'], res['mouth_lower_lip_right_contour2']['y']),
        (res['mouth_lower_lip_right_contour3']['x'], res['mouth_lower_lip_right_contour3']['y']),
        (res['mouth_lower_lip_left_contour1']['x'], res['mouth_lower_lip_left_contour1']['y']),
        (res['mouth_lower_lip_left_contour2']['x'], res['mouth_lower_lip_left_contour2']['y']),
        (res['mouth_lower_lip_left_contour3']['x'], res['mouth_lower_lip_left_contour3']['y']),
        (res['mouth_lower_lip_top']['x'], res['mouth_lower_lip_top']['y']),
        (res['mouth_lower_lip_bottom']['x'], res['mouth_lower_lip_bottom']['y']),

        (res['nose_bridge1']['x'], res['nose_bridge1']['y']),
        (res['nose_bridge2']['x'], res['nose_bridge2']['y']),
        (res['nose_bridge3']['x'], res['nose_bridge3']['y']),
        (res['nose_tip']['x'], res['nose_tip']['y']),
        (res['nose_left_contour1']['x'], res['nose_left_contour1']['y']),
        (res['nose_left_contour2']['x'], res['nose_left_contour2']['y']),
        (res['nose_left_contour3']['x'], res['nose_left_contour3']['y']),
        (res['nose_left_contour4']['x'], res['nose_left_contour4']['y']),
        (res['nose_left_contour5']['x'], res['nose_left_contour5']['y']),
        (res['nose_middle_contour']['x'], res['nose_middle_contour']['y']),
        (res['nose_right_contour1']['x'], res['nose_right_contour1']['y']),
        (res['nose_right_contour2']['x'], res['nose_right_contour2']['y']),
        (res['nose_right_contour3']['x'], res['nose_right_contour3']['y']),
        (res['nose_right_contour4']['x'], res['nose_right_contour4']['y']),
        (res['nose_right_contour5']['x'], res['nose_right_contour5']['y']),

        (res['right_eye_left_corner']['x'], res['right_eye_left_corner']['y']),
        (res['right_eye_upper_left_quarter']['x'], res['right_eye_upper_left_quarter']['y']),
        (res['right_eye_top']['x'], res['right_eye_top']['y']),
        (res['right_eye_upper_right_quarter']['x'], res['right_eye_upper_right_quarter']['y']),
        (res['right_eye_right_corner']['x'], res['right_eye_right_corner']['y']),
        (res['right_eye_lower_right_quarter']['x'], res['right_eye_lower_right_quarter']['y']),
        (res['right_eye_bottom']['x'], res['right_eye_bottom']['y']),
        (res['right_eye_lower_left_quarter']['x'], res['right_eye_lower_left_quarter']['y']),
        (res['right_eye_pupil']['x'] + 2, res['right_eye_pupil']['y'] + 2),
        (res['right_eye_center']['x'], res['right_eye_center']['y']),

        (res['right_eyebrow_upper_left_corner']['x'], res['right_eyebrow_upper_left_corner']['y']),
        (res['right_eyebrow_upper_left_quarter']['x'], res['right_eyebrow_upper_left_quarter']['y']),
        (res['right_eyebrow_upper_middle']['x'], res['right_eyebrow_upper_middle']['y']),
        (res['right_eyebrow_upper_right_quarter']['x'], res['right_eyebrow_upper_right_quarter']['y']),
        (res['right_eyebrow_right_corner']['x'], res['right_eyebrow_right_corner']['y']),
        (res['right_eyebrow_lower_left_corner']['x'], res['right_eyebrow_lower_left_corner']['y']),
        (res['right_eyebrow_lower_left_quarter']['x'], res['right_eyebrow_lower_left_quarter']['y']),
        (res['right_eyebrow_lower_middle']['x'], res['right_eyebrow_lower_middle']['y']),
        (res['right_eyebrow_lower_right_quarter']['x'], res['right_eyebrow_lower_right_quarter']['y']),
    ]

    return pointer4


def landmarks_by_face__(image):
    url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'
    params = {
        'api_key': 'AbnawESWT1tNA6mI9PfQWNAF4iTeiza-',
        'api_secret': 'n4GQyk1XMjcdN_F3Nq3dwbXR6qonbodH',
        'return_landmark': 2
    }
    file = {'image_file': open(image, 'rb')}
    r = requests.post(url=url, files=file, data=params)

    if r.status_code == requests.codes.ok:
        return r.content.decode('utf-8')
    else:
        return r.content


def matrix_rectangle(left, top, width, height):
    pointer = [
        (left, top),
        (left + width / 2, top),
        (left + width - 1, top),
        (left + width - 1, top + height / 2),
        (left, top + height / 2),
        (left, top + height - 1),
        (left + width / 2, top + height - 1),
        (left + width - 1, top + height - 1)
    ]

    return pointer


def face_mark(face_detect):
    faces = json.loads(face_detect)['faces']

    point_list = face_marks(faces[0]['landmark'])
    return point_list


def face_mark1(face_detect):
    faces = json.loads(face_detect)['faces']

    point_list = face_marks1(faces[0]['landmark'])
    return point_list


def face_mark2(face_detect):
    faces = json.loads(face_detect)['faces']

    point_list = face_marks2(faces[0]['landmark'])
    return point_list


def face_mark3(face_detect):
    faces = json.loads(face_detect)['faces']

    point_list = face_marks3(faces[0]['landmark'])
    return point_list
