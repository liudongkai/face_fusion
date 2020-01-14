# -*- coding: utf-8 -*-
import flask  # 导入模块
import cv2
import json
from flask import make_response
import base64
import core  # 引入这个包下的所有方法
from flask import request  # 导入request方法，用来接收传参
from flask_cors import CORS
from gevent import monkey
import pymysql
import os
from gevent.pywsgi import WSGIServer

monkey.patch_all()
import time

app = flask.Flask(__name__)  # 创建接口服务，格式：flask.Flask(__name__),类似于redis.Redis,其中(__name__)是指当前的python文件，也就是说接口是基于此文档在运行
app.config.update(DEBUG=True)

# resources: 全局配置允许跨域的API接口; r'/*': 所有的API接口都允许跨域请求
# origins: 配置允许跨域访问的源; max_age: 预检请求的有效时长 1800s即30分钟
CORS(app, resources={r'/*': {"origins": "*", "max_age": 1800}})


@app.route('/face_point_p', methods=['post'])  # 创建好接口服务后，需要指定路径，路径包含后缀以及请求方法，必须要是@装饰，下面的函数才能正常运行
def face_point():  # 接口主体部分，运行内容
    w = request.form.get("is_exp")
    is_exp = int(w)
    face_data = request.form.get("face_data")
    account_body_id = request.form.get("account_body_id")
    token = request.form.get("token")
    a = ''.join(token)
    xuan_zhuan = request.form.get("xuan_zhuan")
    face = ':/' + face_data.strip('data:image/jpeg;base64,')
    imgdata = base64.b64decode(face)
    source_image1 = "/home/scy/face_fusion/pugongyin/" + a + ".jpg"
    file = open("/home/scy/face_fusion/pugongyin/" + a + ".jpg", "wb+")
    file.write(imgdata)
    file.close()
    image = cv2.imread(source_image1)
    value = 20
    image_dst = cv2.bilateralFilter(image, value, value * 2, value / 2)
    source_image1_beauty = "/home/scy/face_fusion/pugongyin/" + a + "a1.jpg"
    source_image2_beauty = "/home/scy/face_fusion/pugongyin/" + a + "a.jpg"
    # cv2.imwrite(source_image1_beauty, image_dst)
    # if int(xuan_zhuan) == 1:
    #	image_xuanzhuan = cv2.imread(source_image1_beauty)
    #	(h, w) = image_xuanzhuan.shape[:2]
    #	center = (w // 2, h // 2)
    #	M = cv2.getRotationMatrix2D(center, -90, 1.0)
    #	rotated = cv2.warpAffine(image_xuanzhuan, M, (w, h))
    #	cv2.imwrite(source_image2_beauty,rotated)
    #	with open(source_image2_beauty,"rb") as f:
    #		base64_data = base64.b64encode(f.read())
    # else:
    cv2.imwrite(source_image2_beauty, image_dst)
    des_points = core.face_points4(source_image2_beauty)

    result = {}
    if is_exp == 0:
        conn = pymysql.connect(host='192.168.1.228', port=3306, user='text', password='123456', db='pgy')
        cursor = conn.cursor()
        cursor.execute("update md_account_body set status_model='1'  where account_body_id=%s", [account_body_id, ])
        conn.commit()
        cursor.close()
        conn.close()
        result["result"] = "successful"
        result["status"] = "200"
        result["data"] = des_points
        json_str = json.dumps(result)
        rst = make_response(json_str)
        return rst
    elif is_exp == 1:
        result["result"] = "successful"
        result["status"] = "200"
        result["data"] = des_points
        json_str = json.dumps(result)
        rst = make_response(json_str)
        return rst
    else:
        result["result"] = "failed"
        result["status"] = "400"
        json_str = json.dumps(result)
        rst = make_response(json_str)
        return rst


@app.route('/face_fusion_p', methods=['post'])
def reg():
    token = request.form.get("token")
    is_exp = request.form.get("is_exp")
    is_exp = int(is_exp)
    account_body_id = request.form.get("account_body_id")
    a = ''.join(token)
    source_image1_beauty = "/home/scy/face_fusion/pugongyin/" + a + "a.jpg"
    source_image1_draw = "/home/scy/face_fusion/pugongyin/" + a + "aceshi.jpg"
    model_image_nv = "/home/scy/face_fusion/yry-master/yry-master/images/nvgai1.png"
    face_image_output = "/home/scy/face_fusion/pugongyin/" + a + "bceshi.jpg"
    model_image_nan = "/home/scy/face_fusion/yry-master/yry-master/images/nan.png"
    if is_exp == 1:
        conn = pymysql.connect(host='192.168.1.228', port=3306, user='text', password='123456', db='pgy')
        cursor = conn.cursor()
        cursor.execute("select sex from md_exp_body where account_body_id=%s", [account_body_id, ])
        reslut = cursor.fetchone()
        sex = reslut[0]
        if account_body_id:
            if sex == 2:
                img_400x320 = cv2.imread(source_image1_beauty)
                bt = cv2.resize(img_400x320, None, fx=1.1, fy=1, interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(source_image1_draw, bt)
                src_points = [(701, 276), (701, 316), (707, 355), (710, 393), (714, 431), (720, 469), (726, 508),
                              (736, 546), (749, 582), (768, 616), (794, 646), (826, 669), (862, 688), (900, 653),
                              (941, 663), (983, 670), (1028, 674), (1072, 674), (1111, 668), (1147, 657), (1181, 691),
                              (1214, 672), (1245, 646), (1270, 617), (1292, 583), (1306, 547), (1314, 509), (1320, 470),
                              (1324, 431), (1326, 393), (1327, 354), (1325, 315), (1324, 276), (840, 308), (865, 284),
                              (897, 277), (930, 286), (953, 311), (926, 316), (894, 319), (864, 313), (907, 297),
                              (905, 295), (769, 155), (818, 147), (865, 171), (930, 177), (973, 191), (841, 248),
                              (884, 247), (927, 251), (970, 260), (1000, 540), (961, 551), (937, 565), (975, 569),
                              (1050, 541), (1087, 554), (1108, 569), (1072, 571), (1025, 547), (1025, 573), (922, 564),
                              (1123, 568), (1073, 571), (1096, 590), (1064, 605), (976, 567), (951, 586), (983, 603),
                              (1025, 571), (1024, 610), (1023, 292), (1024, 340), (1024, 388), (1024, 436), (983, 312),
                              (961, 412), (944, 456), (969, 470), (997, 475), (1027, 485), (1066, 312), (1088, 412),
                              (1107, 456), (1082, 470), (1055, 475), (1097, 319), (1121, 287), (1155, 278), (1188, 284),
                              (1216, 301), (1189, 312), (1158, 318), (1126, 316), (1151, 297), (1149, 295), (1085, 185),
                              (1127, 175), (1170, 173), (1213, 148), (1248, 154), (1087, 253), (1128, 247), (1169, 245),
                              (1209, 248)]
                dst_points = core.face_points3(source_image1_draw)
                core.face_merge1(src_img=model_image_nv,
                                 src_points=src_points,
                                 dst_img=source_image1_draw,
                                 out_img=face_image_output,
                                 dst_points=dst_points,
                                 face_area=[705, 157, 629, 629],
                                 alpha=0.9,
                                 k_size=(1, 240),
                                 mat_multiple=1.3, )
                with open(face_image_output, "rb") as f:
                    base64_data1 = base64.b64encode(f.read())
                face_output = 'data:image/jpeg;base64,' + (str(base64_data1, 'utf-8'))
                cursor.execute("update md_exp_body set model_img=%s  where  account_body_id=%s",
                               [face_output, account_body_id])
                conn.commit()
                cursor.close()
                conn.close()
                reslut = {}
                reslut["status"] = "200"
                reslut["result"] = "successful"
                json_str = json.dumps(reslut)
                rst = make_response(json_str)
                return rst
            else:
                img_400x320 = cv2.imread(source_image1_beauty)
                bt = cv2.resize(img_400x320, None, fx=1.3, fy=1, interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(source_image1_draw, bt)
                src_points = [(735, 281), (731, 319), (736, 355), (738, 391), (741, 426), (745, 461), (751, 498),
                              (758, 533), (768, 569), (782, 602), (803, 631), (832, 654), (867, 671), (904, 633),
                              (944, 640), (985, 644), (1027, 645), (1068, 645), (1104, 641), (1137, 632), (1168, 669),
                              (1199, 651), (1226, 626), (1246, 597), (1263, 565), (1274, 530), (1280, 495), (1284, 459),
                              (1287, 424), (1288, 388), (1287, 353), (1283, 318), (1279, 283), (846, 314), (867, 289),
                              (896, 282), (927, 289), (951, 309), (925, 315), (896, 318), (868, 315), (898, 300),
                              (896, 298), (782, 158), (831, 152), (876, 176), (940, 180), (980, 192), (854, 253),
                              (896, 251), (937, 254), (977, 261), (1005, 506), (970, 521), (951, 540), (983, 536),
                              (1051, 506), (1086, 522), (1105, 541), (1072, 537), (1028, 512), (1028, 536), (938, 541),
                              (1119, 542), (1072, 536), (1091, 555), (1061, 563), (985, 535), (964, 554), (993, 563),
                              (1028, 534), (1027, 566), (1028, 294), (1028, 335), (1028, 376), (1027, 417), (988, 312),
                              (968, 404), (953, 440), (977, 450), (1002, 454), (1028, 461), (1070, 312), (1088, 405),
                              (1102, 441), (1078, 451), (1053, 455), (1106, 317), (1128, 289), (1157, 282), (1186, 289),
                              (1208, 306), (1185, 315), (1159, 318), (1131, 314), (1159, 301), (1157, 299), (1075, 192),
                              (1112, 181), (1155, 177), (1199, 152), (1238, 155), (1078, 260), (1116, 253), (1156, 250),
                              (1196, 252)]
                dst_points = core.face_points2(source_image1_draw)
                core.face_merge(src_img=model_image_nan,
                                src_points=src_points,
                                dst_img=source_image1_draw,
                                out_img=face_image_output,
                                dst_points=dst_points,
                                face_area=[736, 559, 181, 559],
                                alpha=0.9,
                                k_size=(1, 260),
                                mat_multiple=1.3)
                with open(face_image_output, "rb") as f:
                    base64_data1 = base64.b64encode(f.read())
                face_output = 'data:image/jpeg;base64,' + (str(base64_data1, 'utf-8'))
                cursor.execute("update md_exp_body set model_img=%s  where  account_body_id=%s",
                               [face_output, account_body_id])
                conn.commit()
                cursor.close()
                conn.close()
                reslut = {}
                reslut["status"] = "200"
                reslut["result"] = "successful"
                json_str = json.dumps(reslut)
                rst = make_response(json_str)
                return rst
        else:
            reslut = {}
            reslut["status"] = "400"
            reslut["result"] = "failed"
            json_str = json.dumps(reslut)
            rst = make_response(json_str)
            return rst
    elif is_exp == 0:
        conn = pymysql.connect(host='192.168.1.228', port=3306, user='text', password='123456', db='pgy')
        cursor = conn.cursor()
        cursor.execute("select sex from md_account_body where account_body_id=%s", [account_body_id, ])
        reslut = cursor.fetchone()
        sex = reslut[0]
        if account_body_id:
            if sex == 2:
                img_400x320 = cv2.imread(source_image1_beauty)
                bt = cv2.resize(img_400x320, None, fx=1.1, fy=1, interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(source_image1_draw, bt)
                src_points = [(701, 276), (701, 316), (707, 355), (710, 393), (714, 431), (720, 469), (726, 508),
                              (736, 546), (749, 582), (768, 616), (794, 646), (826, 669), (862, 688), (900, 653),
                              (941, 663), (983, 670), (1028, 674), (1072, 674), (1111, 668), (1147, 657), (1181, 691),
                              (1214, 672), (1245, 646), (1270, 617), (1292, 583), (1306, 547), (1314, 509), (1320, 470),
                              (1324, 431), (1326, 393), (1327, 354), (1325, 315), (1324, 276), (840, 308), (865, 284),
                              (897, 277), (930, 286), (953, 311), (926, 316), (894, 319), (864, 313), (907, 297),
                              (905, 295), (769, 155), (818, 147), (865, 171), (930, 177), (973, 191), (841, 248),
                              (884, 247), (927, 251), (970, 260), (1000, 540), (961, 551), (937, 565), (975, 569),
                              (1050, 541), (1087, 554), (1108, 569), (1072, 571), (1025, 547), (1025, 573), (922, 564),
                              (1123, 568), (1073, 571), (1096, 590), (1064, 605), (976, 567), (951, 586), (983, 603),
                              (1025, 571), (1024, 610), (1023, 292), (1024, 340), (1024, 388), (1024, 436), (983, 312),
                              (961, 412), (944, 456), (969, 470), (997, 475), (1027, 485), (1066, 312), (1088, 412),
                              (1107, 456), (1082, 470), (1055, 475), (1097, 319), (1121, 287), (1155, 278), (1188, 284),
                              (1216, 301), (1189, 312), (1158, 318), (1126, 316), (1151, 297), (1149, 295), (1085, 185),
                              (1127, 175), (1170, 173), (1213, 148), (1248, 154), (1087, 253), (1128, 247), (1169, 245),
                              (1209, 248)]
                dst_points = core.face_points3(source_image1_draw)
                core.face_merge1(src_img=model_image_nv,
                                 src_points=src_points,
                                 dst_img=source_image1_draw,
                                 out_img=face_image_output,
                                 dst_points=dst_points,
                                 face_area=[705, 157, 629, 629],
                                 alpha=0.9,
                                 k_size=(1, 240),
                                 mat_multiple=1.3, )
                with open(face_image_output, "rb") as f:
                    base64_data1 = base64.b64encode(f.read())
                face_output = 'data:image/jpeg;base64,' + (str(base64_data1, 'utf-8'))
                cursor.execute("update md_account_body set model_img=%s  where  account_body_id=%s",
                               [face_output, account_body_id])
                conn.commit()
                cursor.close()
                conn.close()
                reslut = {}
                reslut["status"] = "200"
                reslut["result"] = "successful"
                json_str = json.dumps(reslut)
                rst = make_response(json_str)
                return rst
            else:
                img_400x320 = cv2.imread(source_image1_beauty)
                bt = cv2.resize(img_400x320, None, fx=1.3, fy=1, interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(source_image1_draw, bt)
                src_points = [(735, 281), (731, 319), (736, 355), (738, 391), (741, 426), (745, 461), (751, 498),
                              (758, 533), (768, 569), (782, 602), (803, 631), (832, 654), (867, 671), (904, 633),
                              (944, 640), (985, 644), (1027, 645), (1068, 645), (1104, 641), (1137, 632), (1168, 669),
                              (1199, 651), (1226, 626), (1246, 597), (1263, 565), (1274, 530), (1280, 495), (1284, 459),
                              (1287, 424), (1288, 388), (1287, 353), (1283, 318), (1279, 283), (846, 314), (867, 289),
                              (896, 282), (927, 289), (951, 309), (925, 315), (896, 318), (868, 315), (898, 300),
                              (896, 298), (782, 158), (831, 152), (876, 176), (940, 180), (980, 192), (854, 253),
                              (896, 251), (937, 254), (977, 261), (1005, 506), (970, 521), (951, 540), (983, 536),
                              (1051, 506), (1086, 522), (1105, 541), (1072, 537), (1028, 512), (1028, 536), (938, 541),
                              (1119, 542), (1072, 536), (1091, 555), (1061, 563), (985, 535), (964, 554), (993, 563),
                              (1028, 534), (1027, 566), (1028, 294), (1028, 335), (1028, 376), (1027, 417), (988, 312),
                              (968, 404), (953, 440), (977, 450), (1002, 454), (1028, 461), (1070, 312), (1088, 405),
                              (1102, 441), (1078, 451), (1053, 455), (1106, 317), (1128, 289), (1157, 282), (1186, 289),
                              (1208, 306), (1185, 315), (1159, 318), (1131, 314), (1159, 301), (1157, 299), (1075, 192),
                              (1112, 181), (1155, 177), (1199, 152), (1238, 155), (1078, 260), (1116, 253), (1156, 250),
                              (1196, 252)]
                dst_points = core.face_points2(source_image1_draw)
                core.face_merge(src_img=model_image_nan,
                                src_points=src_points,
                                dst_img=source_image1_draw,
                                out_img=face_image_output,
                                dst_points=dst_points,
                                face_area=[736, 559, 181, 559],
                                alpha=0.9,
                                k_size=(1, 260),
                                mat_multiple=1.3)
                with open(face_image_output, "rb") as f:
                    base64_data1 = base64.b64encode(f.read())
                face_output = 'data:image/jpeg;base64,' + (str(base64_data1, 'utf-8'))
                cursor.execute("update md_account_body set model_img=%s  where  account_body_id=%s",
                               [face_output, account_body_id])
                conn.commit()
                cursor.close()
                conn.close()
                reslut = {}
                reslut["status"] = "200"
                reslut["result"] = "successful"
                json_str = json.dumps(reslut)
                rst = make_response(json_str)
                return rst
        else:
            reslut = {}
            reslut["status"] = "400"
            reslut["result"] = "failed"
            json_str = json.dumps(reslut)
            rst = make_response(json_str)
            return rst
    else:
        reslut = {}
        reslut["status"] = "400"
        reslut["result"] = "failed"
        json_str = json.dumps(reslut)
        rst = make_response(json_str)
        return rst


@app.route('/face_type_p', methods=['post'])
def face_point1():  # 接口主体部分，运行内容
    face_type = request.form.get("face_type")
    is_exp = request.form.get("is_exp")
    type(is_exp)
    is_exp = int(is_exp)
    token = request.form.get("token")
    account_body_id = request.form.get("account_body_id")
    if is_exp == 1:
        try:
            conn = pymysql.connect(host='192.168.1.228', port=3306, user='text', password='123456', db='pgy')
            cursor = conn.cursor()
            cursor.execute("update md_exp_body set model_face=%s  where account_body_id=%s",
                           [face_type, account_body_id])
            conn.commit()
            cursor.close()
            conn.close()
            result = {}
            result["result"] = "successful"
            result["status"] = "200"
            json_str = json.dumps(result)
            rst = make_response(json_str)
            return rst
        except:
            result = {}
            result["result"] = "failed"
            result["status"] = "400"
            json_str = json.dumps(result)
            rst = make_response(json_str)
            return rst
    else:
        try:
            conn = pymysql.connect(host='192.168.1.228', port=3306, user='text', password='123456', db='pgy')
            cursor = conn.cursor()
            cursor.execute("select account_id from md_account_token where token=%s", [token, ])
            reslut = cursor.fetchone()
            id = reslut[0]
            cursor.execute("update md_account_body set model_face=%s  where account_id=%s", [face_type, id])
            conn.commit()
            cursor.close()
            conn.close()
            result = {}
            result["result"] = "successful"
            result["status"] = "200"
            json_str = json.dumps(result)
            rst = make_response(json_str)
            return rst
        except:
            result = {}
            result["result"] = "failed"
            result["status"] = "400"
            json_str = json.dumps(result)
            rst = make_response(json_str)
            return rst


@app.route('/check', methods=['get'])
def check():
    return "pass"

# if __name__ == '__main__':
# WSGIServer(('0.0.0.0', 8913), app,keyfile='/home/scy/face_fusion/yry-master/https/server.key',certfile='/home/scy/face_fusion/yry-master/https/server.crt').serve_forever()
# app.run(port=8913, host='0.0.0.0', debug=True,threaded=True,ssl_context=('/home/scy/face_fusion/yry-master/https/server.crt','/home/scy/face_fusion/yry-master/https/server.key') )
