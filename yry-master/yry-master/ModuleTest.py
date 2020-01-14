# -*- coding: utf-8 -*-


import core  # 引入这个包下的所有方法

if __name__ == '__main__':
    src = 'images/zhang.jpg'
    dst = 'core/huge_newsize.jpg'
    output = 'D:\\WWW\\bihuang\\Public\\upload\\face_ouput\\huge_output.jpg'
    src_points, _ = core.face_points(src)
    dst_points, _ = core.face_points(dst)

    core.face_merge(src_img='images/zhang.jpg',  # 源图像.jpg
                    src_points=src_points,  # 源图像人脸关键点数组坐标
                    dst_img='core/huge_newsize.jpg',  # 输入带融合的目标图像
                    out_img='D:\\WWW\\bihuang\\Public\\upload\\face_ouput\\huge_output.jpg',  # 输出到指定的路径
                    dst_points=dst_points,  # 目标图像人脸关键点数组坐标
                    face_area=[100, 100, 100, 100],  # 模板图中人脸融合的位置左上角横坐标（left），左上角纵坐标（top），人脸框宽度（width），人脸框高度（height）
                    alpha=0.65,  # [0~1]融合比，比例越大目标图像的特征就越多
                    k_size=(300, 250),  # 滤波窗口尺寸-图像均值平滑滤波模板
                    mat_multiple=1.2)  # 缩放获取到的人脸心型区域-图像缩放因子
