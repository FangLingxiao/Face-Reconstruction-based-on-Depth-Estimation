import numpy as np

def depth_to_pointcloud(depth_img, color_img, depth_unit='mm', fx=500, fy=500, cx=None, cy=None):
    """
    将深度图和彩色图转换为带颜色的点云
    :param depth_img: 深度图,16位原始格式
    :param color_img: 彩色图,BGR格式
    :param depth_scale: 深度缩放因子,将原始深度值转换为以米为单位的深度
    :param fx: 相机焦距(x轴)
    :param fy: 相机焦距(y轴)
    :param cx: 相机光心(x轴),默认为图像中心
    :param cy: 相机光心(y轴),默认为图像中心
    :return: 点云,N*6的矩阵,每一行表示一个点的x,y,z,r,g,b
    """

    if depth_unit == 'mm':
        depth_scale = 1000.0
    elif depth_unit == 'cm':
        depth_scale = 100.0
    else:  # assume depth is already in meters
        depth_scale = 1.0

    H, W = depth_img.shape
    if cx is None:
        cx = W // 2
    if cy is None:
        cy = H // 2
        
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.astype(np.float32) - cx
    v = v.astype(np.float32) - cy
    
    Z = depth_img.astype(np.float32) / depth_scale  # 将原始深度值转换为以米为单位的深度
    X = (u * Z) / fx
    Y = (v * Z) / fy
    
    pointcloud = np.stack((X, Y, Z), axis=2)
    pointcloud = pointcloud.reshape((-1, 3))
    
    color = color_img.reshape((-1, 3))
    pointcloud_color = np.concatenate((pointcloud, color), axis=1)
    
    return pointcloud_color


def depth_to_pointcloud_reverse(depth_img, color_img, depth_scale=1000, depth_trunc=5, stride=1, fx=500, fy=500, cx=None, cy=None):
    H, W = depth_img.shape
    if cx is None:
        cx = W // 2
    if cy is None:
        cy = H // 2
    
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.astype(np.float32) - cx
    v = v.astype(np.float32) - cy
    
    max_depth = depth_scale * depth_trunc
    depth_img = max_depth - depth_img  # Invert depth values
    depth_img = depth_img.astype(np.float32) / depth_scale
    
    Z = depth_img[::stride, ::stride]
    X = (u[::stride, ::stride] * Z) / fx
    Y = (v[::stride, ::stride] * Z) / fy
    
    pointcloud = np.stack((X, Y, Z), axis=2)
    pointcloud = pointcloud.reshape((-1, 3))
    
    color = color_img[::stride, ::stride].reshape((-1, 3))
    pointcloud_color = np.concatenate((pointcloud, color), axis=1)
    
    return pointcloud_color

import numpy as np
import cv2

def depth_to_pointcloud_colour(depth_img, color_img, depth_scale=1, fx=500, fy=500, cx=None, cy=None):
    """
    将深度图和彩色图转换为带颜色的点云
    :param depth_img: 深度图,彩色格式(红色表示近,蓝色表示远)
    :param color_img: 彩色图,BGR格式
    :param fx: 相机焦距(x轴)
    :param fy: 相机焦距(y轴)
    :param cx: 相机光心(x轴),默认为图像中心
    :param cy: 相机光心(y轴),默认为图像中心
    :return: 点云,N*6的矩阵,每一行表示一个点的x,y,z,r,g,b
    """
    H, W, _ = depth_img.shape
    if cx is None:
        cx = W // 2
    if cy is None:
        cy = H // 2
        
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.astype(np.float32) - cx
    v = v.astype(np.float32) - cy
    
    depth = depth_img.astype(np.float32) / 255.0 
    depth = (depth[:, :, 2] + depth[:, :, 1], depth[:, :, 0]) / (255.0 * 3) # Average all channels

    # Scale depth
    depth = depth * depth_scale

    # Z = 1.0 - depth  # 反转深度值
    Z = depth
    X = (u * Z) / fx
    Y = (v * Z) / fy
    
    pointcloud = np.stack((X, Y, Z), axis=2)
    pointcloud = pointcloud.reshape((-1, 3))
    
    color = color_img.reshape((-1, 3))
    pointcloud_color = np.concatenate((pointcloud, color), axis=1)
    
    return pointcloud_color