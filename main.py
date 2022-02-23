import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from moviepy.editor import VideoFileClip
from pylab import mpl

# 设置plt显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]

# 参数设置
nx = 9
ny = 6
file_paths = glob.glob("./camera_cal/calibration*.jpg")


# 绘制对比图
def plot_contrast_image(origin_img, converted_img, origin_img_title="origin_img", converted_img_title="converted_img",
                        converted_img_gray=False):
    fig, axes = plt.subplots(1, 2, figsize=(15, 20))
    axes[0].set_title(origin_img_title)
    axes[0].imshow(origin_img[:, :, ::-1])
    axes[1].set_title(converted_img_title)
    if converted_img_gray:
        axes[1].imshow(converted_img, cmap='gray')
    else:
        axes[1].imshow(converted_img[:, :, ::-1])
    plt.show()


# 相机校正：得到内参、外参、畸变系数
def cal_calibrate_params(file_paths):
    # 存储角点数据的坐标
    object_points = []  # 三维空间中的点：3D
    image_points = []  # 图像空间中的点：2d
    # 生成角点在真实世界中的位置，类似(0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)的三维点
    obj_p = np.zeros((nx * ny, 3), np.float32)
    obj_p[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    # 角点检测
    for file_path in file_paths:
        # 读取图片
        img = cv2.imread(file_path)
        # 灰度化
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 角点检测
        # rect：是否找到角点，找到角点返回1，否则返回0
        # corners：检测到的角点
        rect, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # 绘制对比图
        # imgcopy = img.copy()
        # cv2.drawChessboardCorners(imgcopy, (nx, ny), corners, rect)
        # plot_contrast_image(img, imgcopy)

        # 若检测到角点，进行保存
        if rect:
            object_points.append(obj_p)
            image_points.append(corners)
    # 相机校正，获取相机参数
    # ret: 返回值    mtx: 相机的内参矩阵，大小为3*3的矩阵
    # dist: 畸变系数，为5*1大小的矢量    rvecs: 旋转变量    tvecs: 平移变量
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)

    return ret, mtx, dist, rvecs, tvecs


# 图像去畸变
def img_undistort(img, mtx, dist):
    res = cv2.undistort(img, mtx, dist, None, mtx)
    return res


# 车道线提取
# 颜色空间转换 -> 边缘检测 -> 颜色阈值 -> 合并，并使用L通道进行白色区域过滤
def pipeline(img, s_thresh=(170, 255), sx_thresh=(40, 200)):
    # 复制原图像
    img = np.copy(img)
    # 颜色空间转换
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # sobel边缘检测
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
    # 求绝对值
    abs_sobelx = np.absolute(sobelx)
    # 将其转换为8bit整数
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    # 对边缘提取的结果进行二值化
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # sxbinary可视化
    # plt.figure()
    # plt.imshow(sxbinary, cmap='gray')
    # plt.title("sobel")
    # plt.show()

    # s通道阈值处理
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # s_binary可视化
    # plt.figure()
    # plt.imshow(s_binary, cmap='gray')
    # plt.title("schanel")
    # plt.show()

    # 结合边缘提取结果和颜色过滤的结果
    color_binary = np.zeros_like(sxbinary)
    color_binary[((sxbinary == 1) | (s_binary == 1)) & (l_channel > 100)] = 1

    return color_binary


# 透视变换
# 获取透视变换的参数矩阵
def cal_perspective_params(img, points):
    # 设置偏移
    offset_x = 330
    offset_y = 0
    img_size = (img.shape[1], img.shape[0])  # 获取图像长宽
    src = np.float32(points)

    # 设置俯视图中的对应的四个点，左上、右上、左下、右下
    dst = np.float32([[offset_x, offset_y],
                      [img_size[0] - offset_x, offset_y],
                      [offset_x, img_size[1] - offset_y],
                      [img_size[0] - offset_x, img_size[1] - offset_y]])

    # 从原始图像转换为俯视图的透视变换的参数矩阵
    M = cv2.getPerspectiveTransform(src, dst)
    # 从俯视图转换为原始图像的透视变换参数矩阵
    M_inverse = cv2.getPerspectiveTransform(dst, src)

    return M, M_inverse


# 根据参数矩阵完成透视变换
def img_perspect_transform(img, M):
    img_size = (img.shape[1], img.shape[0])
    return cv2.warpPerspective(img, M, img_size)


# 精确定位车道线
def cal_lane_param(binary_wraped):
    # 确定左右车道线的位置
    # 统计直方图，axis=0为y
    histogram = np.sum(binary_wraped[:, :], axis=0)
    # 图像中点，划分为左右两部分，分别定位峰值（车道线）位置
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # 设置滑动窗口的数量，并计算窗口高度（ 列/数量 ）
    nwindows = 9
    window_height = np.int(binary_wraped.shape[0] / nwindows)

    # 获取图像中不为零的点
    nonzero = binary_wraped.nonzero()
    x_nonzero = np.array(nonzero[1])
    y_nonzero = np.array(nonzero[0])

    # 车道线检测的当前位置
    leftx_current = leftx_base
    rightx_current = rightx_base

    # 指定x的检测范围（滑动窗口的一般）
    margin = 100
    # 设置最小像素点，阈值用于统计滑动窗口区域内的非零像素个数，小于50的窗口不对x的中心值进行更新
    minpix = 50

    # 用来记录搜索窗口非零点在x_nonzero和y_nonzero的索引
    left_lane_inds = []
    right_lane_inds = []

    # 遍历窗口
    for i in range(nwindows):
        # 设置y的检测范围
        win_y_low = binary_wraped.shape[0] - window_height * (i + 1)
        win_y_high = binary_wraped.shape[0] - window_height * i
        # 左车道线x的范围
        win_x_left_low = leftx_current - margin
        win_x_left_high = leftx_current + margin
        # 右车道线x的范围
        win_x_right_low = rightx_current - margin
        win_x_right_high = rightx_current + margin

        # 确定非零点的位置（x，y）是否在搜索窗口中，保存在的
        good_left_inds = ((y_nonzero >= win_y_low) & (y_nonzero <= win_y_high) &
                          (x_nonzero >= win_x_left_low) & (x_nonzero <= win_x_left_high)).nonzero()[0]
        good_right_inds = ((y_nonzero >= win_y_low) & (y_nonzero <= win_y_high) &
                           (x_nonzero >= win_x_right_low) & (x_nonzero <= win_x_right_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # 如果获取的点的个数大于最小个数，则利用其更新滑动窗口在x轴的位置
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(x_nonzero[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(x_nonzero[good_right_inds]))

    # 将检测出的左右车道点转换为array
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # 获取检测出的左右车道点在图像中的位置
    leftx = x_nonzero[left_lane_inds]
    lefty = y_nonzero[left_lane_inds]
    rightx = x_nonzero[right_lane_inds]
    righty = y_nonzero[right_lane_inds]

    # 拟合车道线，返回的结果是系数
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit


# 填充车道线之间的多边形
def fill_lane_poly(img, left_fit, right_fit):
    # 获取图像的行
    y_max = img.shape[0]
    # 设置输出图像的大小，并将1的位置设为255白色
    out_img = np.dstack((img, img, img)) * 255
    # 从拟合曲线中获取左右车道线位置
    left_points = [[left_fit[0] * y ** 2 + left_fit[1] * y + left_fit[2], y] for y in range(y_max)]
    right_points = [[right_fit[0] * y ** 2 + right_fit[1] * y + right_fit[2], y] for y in range(y_max - 1, -1, -1)]
    # 左右车道像素点进行合并
    line_points = np.vstack((left_points, right_points))
    # 绘制多边形
    cv2.fillPoly(out_img, np.int_([line_points]), (0, 255, 0))
    return out_img


# 计算车道线曲率
def cal_radius(img, left_fit, right_fit):
    # 图像中像素个数与实际中距离的比率（经验值）
    ym_per_pix = 30 / 720
    xm_per_pix = 3.7 / 700

    # 计算得到曲线上的每个点
    left_y_axis = np.linspace(0, img.shape[0], img.shape[0] - 1)
    left_x_axis = left_fit[0] * left_y_axis ** 2 + left_fit[1] * left_y_axis + left_fit[2]
    right_y_axis = np.linspace(0, img.shape[0], img.shape[0] - 1)
    right_x_axis = right_fit[0] * right_y_axis ** 2 + right_fit[1] * right_y_axis + right_fit[2]

    # 获取真实环境中的曲线
    left_fit_cr = np.polyfit(left_y_axis * ym_per_pix, left_x_axis * xm_per_pix, 2)
    right_fit_cr = np.polyfit(right_y_axis * ym_per_pix, right_x_axis * xm_per_pix, 2)

    # 获取真实环境中的曲率半径
    left_roc = ((1 + (2 * left_fit_cr[0] * left_y_axis * ym_per_pix +
                      left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    right_roc = ((1 + (2 * right_fit_cr[0] * right_y_axis * ym_per_pix +
                       right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

    # 将曲率半径绘制在图像上
    cv2.putText(img, 'Radius of Curvature = {}(m)'.format(np.mean(left_roc)),
                (20, 50), cv2.FONT_ITALIC, 1, (255, 255, 255), 5)
    return img


# 计算车道线中心位置
def cal_lane_center(img, mtx, dist, M):
    undistort_img = img_undistort(img, mtx, dist)  # 去畸变
    pipeline_img = pipeline(undistort_img)  # 车道线提取
    trasform_img = img_perspect_transform(pipeline_img, M)  # 透视变换
    left_fit, right_fit = cal_lane_param(trasform_img)  # 车道线定位
    y_max = img.shape[0]
    left_x = left_fit[0] * y_max ** 2 + left_fit[1] * y_max + left_fit[2]
    right_x = right_fit[0] * y_max ** 2 + right_fit[1] * y_max + right_fit[2]
    return (left_x + right_x) / 2


# 计算车辆偏离中心点的距离
def cal_center_departure(img, left_fit, right_fit, lane_center):
    # 计算中心点
    y_max = img.shape[0]
    left_x = left_fit[0] * y_max ** 2 + left_fit[1] * y_max + left_fit[2]
    right_x = right_fit[0] * y_max ** 2 + right_fit[1] * y_max + right_fit[2]
    xm_per_pix = 3.7 / 700
    center_depart = ((left_x + right_x) / 2 - lane_center) * xm_per_pix
    # 在图像上显示偏移
    if center_depart > 0:
        cv2.putText(img, 'Vehicle is {}m right of center'.format(center_depart),
                    (20, 100), cv2.FONT_ITALIC, 1, (255, 255, 255), 5)
    elif center_depart < 0:
        cv2.putText(img, 'Vehicle is {}m left of center'.format(-center_depart),
                    (20, 100), cv2.FONT_ITALIC, 1, (255, 255, 255), 5)
    else:
        cv2.putText(img, 'Vehicle is in the center',
                    (20, 100), cv2.FONT_ITALIC, 1, (255, 255, 255), 5)
    return img


if __name__ == '__main__':
    ret, mtx, dist, rvecs, tvecs = cal_calibrate_params(file_paths)

    # 1.机校正与图像去畸测试
    # img = cv2.imread("./test/test4.jpg")
    # undistort_img = img_undistort(img, mtx, dist)
    # plot_contrast_image(img, undistort_img)
    # print('done')

    # 2.车道线提取测试
    # img = cv2.imread("./test/frame45.jpg")
    # result = pipeline(img)
    # plot_contrast_image(img, result, converted_img_gray=True)

    # 3.测试透视变换
    img = cv2.imread("./test/straight_lines2.jpg")
    points = [[601, 448], [683, 448], [230, 717], [1097, 717]]
    # img = cv2.line(img, (601, 448), (683, 448), (0, 0, 255), 3)
    # img = cv2.line(img, (683, 448), (1097, 717), (0, 0, 255), 3)
    # img = cv2.line(img, (1097, 717), (230, 717), (0, 0, 255), 3)
    # img = cv2.line(img, (230, 717), (601, 448), (0, 0, 255), 3)
    # M, M_inverse = cal_perspective_params(img, points)
    # transform_img = img_perspect_transform(img, M)
    # plot_contrast_image(img, transform_img, origin_img_title="原图", converted_img_title="俯视图")

    # 4.测试车道线定位
    # img = cv2.imread("./test/straight_lines2.jpg")
    # points = [[601, 448], [683, 448], [230, 717], [1097, 717]]
    M, M_inverse = cal_perspective_params(img, points)
    # undistort_img = img_undistort(img, mtx, dist)
    # pipeline_img = pipeline(undistort_img)
    # transform_img = img_perspect_transform(pipeline_img, M)
    # left_fit, right_fit = cal_lane_param(transform_img)
    # result = fill_lane_poly(transform_img, left_fit, right_fit)
    # trasform_img_inv = img_perspect_transform(result, M_inverse)
    # res = cv2.addWeighted(img, 1, trasform_img_inv, 0.5, 0)
    # plot_contrast_image(trasform_img_inv, res, origin_img_title="填充结果", converted_img_title="安全区域")

    # 5.测试车道线中心偏离
    # img = cv2.imread("./test/straight_lines2.jpg")
    # points = [[601, 448], [683, 448], [230, 717], [1097, 717]]
    # M, M_inverse = cal_perspective_params(img, points)
    # undistort_img = img_undistort(img, mtx, dist)
    # pipeline_img = pipeline(undistort_img)
    # transform_img = img_perspect_transform(pipeline_img, M)
    # left_fit, right_fit = cal_lane_param(transform_img)
    # result = fill_lane_poly(transform_img, left_fit, right_fit)
    # trasform_img_inv = img_perspect_transform(result, M_inverse)
    # res = cv2.addWeighted(img, 1, trasform_img_inv, 0.5, 0)
    # roc = cal_radius(res, left_fit, right_fit)
    lane_center = cal_lane_center(img, mtx, dist, M)
    # center = cal_center_departure(roc, left_fit, right_fit, lane_center)
    # plt.imshow(center[:, :, ::-1])
    # plt.show()

    # 车道线检测方法汇总，用于之后视频检测
    def process_image(img):
        # 图像去畸变
        undistort_img = img_undistort(img, mtx, dist)
        # 车道线检测
        pipline_img = pipeline(undistort_img)
        # 透视变换
        transform_img = img_perspect_transform(pipline_img, M)
        # 拟合车道线
        left_fit, right_fit = cal_lane_param(transform_img)
        # 绘制安全区域
        result = fill_lane_poly(transform_img, left_fit, right_fit)
        # 反透视变换
        transform_img_inverse = img_perspect_transform(result, M_inverse)
        # 计算曲率半径和偏离中心的距离
        transform_img_inverse = cal_radius(transform_img_inverse, left_fit, right_fit)
        transform_img_inverse = cal_center_departure(transform_img_inverse, left_fit, right_fit, lane_center)
        # 将检测结果与原始图像叠加
        transform_img_inverse = cv2.addWeighted(undistort_img, 1, transform_img_inverse, 0.5, 0)

        return transform_img_inverse


    # 视频检测
    clip = VideoFileClip("./video/project_video.mp4")
    clip = clip.fl_image(process_image)
    clip.write_videofile("./video/output.mp4", audio=False)
