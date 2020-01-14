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
    exp_id = request.form.get("exp_id")
    exp_id = int(exp_id)
    face_data = request.form.get("face_data")
    sex = request.form.get("sex")
    face = ':/' + face_data.strip('data:image/jpeg;base64,')
    imgdata = base64.b64decode(face)
    source_image1 = "/home/scy/face_fusion/pugongyin/" + str(exp_id) + ".jpg"
    source_image2_beauty = "/home/scy/face_fusion/pugongyin/" + str(exp_id) + "a.jpg"
    with open(source_image1, "wb+") as f:
        f.write(imgdata)
    image110 = core.beauty_image(source_image1)
    faces = json.loads(image110)['result']
    faces_2 = ':' + faces
    imagedata = base64.b64decode(faces_2)
    with open(source_image2_beauty, 'wb+') as  f:
        f.write(imagedata)
    des_points = core.face_points4(source_image2_beauty)
    if des_points:
        result = {}
        result["result"] = "successful"
        result["status"] = "200"
        result["data"] = des_points
        json_str = json.dumps(result)
        rst = make_response(json_str)
        return rst
    else:
        result = {}
        result["result"] = "failed"
        result["status"] = "400"
        json_str = json.dumps(result)
        rst = make_response(json_str)
        return rst


@app.route('/face_fusion_p', methods=['post'])
def reg():
    exp_id = request.form.get("exp_id")
    exp_id = int(exp_id)
    face_data = request.form.get("face_data")
    sex = request.form.get("sex")
    sex = int(sex)
    source_image1_beauty = "/home/scy/face_fusion/pugongyin/" + str(exp_id) + "a.jpg"
    source_image1_draw = "/home/scy/face_fusion/pugongyin/" + str(exp_id) + "aceshi.jpg"
    model_image_nv = "/home/scy/face_fusion/yry-master/yry-master/images/nvgai1.png"
    face_image_output = "/home/scy/face_fusion/pugongyin/" + str(exp_id) + "bceshi.jpg"
    model_image_nan = "/home/scy/face_fusion/yry-master/yry-master/images/fenglei.jpg"
    if sex == 2:
        img_400x320 = cv2.imread(source_image1_beauty)
        bt = cv2.resize(img_400x320, None, fx=1.1, fy=1, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(source_image1_draw, bt)
        src_points = [(701, 276), (701, 316), (707, 355), (710, 393), (714, 431), (720, 469), (726, 508), (736, 546),
                      (749, 582), (768, 616), (794, 646), (826, 669), (862, 688), (900, 653), (941, 663), (983, 670),
                      (1028, 674), (1072, 674), (1111, 668), (1147, 657), (1181, 691), (1214, 672), (1245, 646),
                      (1270, 617), (1292, 583), (1306, 547), (1314, 509), (1320, 470), (1324, 431), (1326, 393),
                      (1327, 354), (1325, 315), (1324, 276), (840, 308), (865, 284), (897, 277), (930, 286), (953, 311),
                      (926, 316), (894, 319), (864, 313), (907, 297), (905, 295), (769, 155), (818, 147), (865, 171),
                      (930, 177), (973, 191), (841, 248), (884, 247), (927, 251), (970, 260), (1000, 540), (961, 551),
                      (937, 565), (975, 569), (1050, 541), (1087, 554), (1108, 569), (1072, 571), (1025, 547),
                      (1025, 573), (922, 564), (1123, 568), (1073, 571), (1096, 590), (1064, 605), (976, 567),
                      (951, 586), (983, 603), (1025, 571), (1024, 610), (1023, 292), (1024, 340), (1024, 388),
                      (1024, 436), (983, 312), (961, 412), (944, 456), (969, 470), (997, 475), (1027, 485), (1066, 312),
                      (1088, 412), (1107, 456), (1082, 470), (1055, 475), (1097, 319), (1121, 287), (1155, 278),
                      (1188, 284), (1216, 301), (1189, 312), (1158, 318), (1126, 316), (1151, 297), (1149, 295),
                      (1085, 185), (1127, 175), (1170, 173), (1213, 148), (1248, 154), (1087, 253), (1128, 247),
                      (1169, 245), (1209, 248)]
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
        conn = pymysql.connect(host='192.168.1.228', port=3306, user='root', password='123456', db='fenglei')
        cursor = conn.cursor()
        cursor.execute("update fl_exp set faceTexture=%s  where  exp_id=%s", [face_output, exp_id])
        conn.commit()
        cursor.close()
        conn.close()
        reslut = {}
        reslut["status"] = "200"
        reslut["result"] = "successful"
        json_str = json.dumps(reslut)
        rst = make_response(json_str)
        return rst
    elif sex == 1:
        img_400x320 = cv2.imread(source_image1_beauty)
        bt = cv2.resize(img_400x320, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(source_image1_draw, bt)
        src_points = [(762, 305), (762, 344), (767, 381), (769, 417), (772, 453), (775, 490), (780, 527), (786, 564),
                      (796, 600), (811, 634), (831, 664), (857, 690), (885, 712), (916, 683), (951, 699), (988, 711),
                      (1031, 715), (1077, 712), (1116, 702), (1153, 687), (1187, 717), (1222, 696), (1255, 670),
                      (1281, 640), (1304, 606), (1318, 569), (1324, 529), (1328, 488), (1331, 448), (1332, 409),
                      (1333, 369), (1330, 329), (1327, 287), (867, 335), (887, 312), (914, 306), (944, 314), (966, 335),
                      (941, 340), (913, 344), (887, 339), (917, 322), (915, 320), (776, 194), (821, 176), (871, 194),
                      (940, 202), (981, 222), (848, 279), (891, 272), (934, 279), (976, 293), (1003, 559), (972, 573),
                      (959, 593), (987, 591), (1047, 558), (1079, 571), (1095, 590), (1066, 589), (1025, 564),
                      (1026, 592), (947, 593), (1108, 589), (1066, 592), (1087, 611), (1061, 627), (987, 594),
                      (967, 614), (992, 629), (1026, 595), (1027, 635), (1021, 316), (1020, 363), (1019, 410),
                      (1019, 457), (987, 335), (973, 442), (959, 488), (978, 497), (999, 501), (1021, 505), (1056, 335),
                      (1070, 441), (1086, 486), (1065, 496), (1044, 500), (1080, 343), (1105, 313), (1135, 306),
                      (1164, 311), (1187, 325), (1165, 335), (1137, 341), (1108, 339), (1136, 322), (1134, 320),
                      (1069, 227), (1113, 205), (1164, 196), (1210, 182), (1237, 204), (1073, 295), (1118, 279),
                      (1161, 272), (1201, 283)]
        dst_points = core.face_points4(source_image1_draw)
        core.face_merge(src_img=model_image_nan,
                        src_points=src_points,
                        dst_img=source_image1_draw,
                        out_img=face_image_output,
                        dst_points=dst_points,
                        face_area=[767, 217, 573, 573],
                        alpha=0.9,
                        k_size=(1, 260),
                        mat_multiple=1.3)
        with open(face_image_output, "rb") as f:
            base64_data1 = base64.b64encode(f.read())
        face_output = 'data:image/jpeg;base64,' + (str(base64_data1, 'utf-8'))
        conn = pymysql.connect(host='192.168.1.228', port=3306, user='root', password='123456', db='fenglei')
        cursor = conn.cursor()
        cursor.execute("update fl_exp set faceTexture=%s  where  exp_id=%s", [face_output, exp_id])
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


@app.route('/face_type_p', methods=['post'])
def face_point1():  # 接口主体部分，运行内容
    face_type = request.form.get("face_type")
    exp_id = request.form.get("exp_id")
    exp_id = int(exp_id)
    try:
        conn = pymysql.connect(host='192.168.1.228', port=3306, user='root', password='123456', db='fenglei')
        cursor = conn.cursor()
        cursor.execute("update fl_exp set face_types=%s  where exp_id=%s", [face_type, exp_id])
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
# WSGIServer(('0.0.0.0', 8913), app,).serve_forever()
# app.run(port=8913, host='0.0.0.0', debug=True,threaded=True )
