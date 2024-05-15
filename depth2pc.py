import cv2
import numpy as np

def face_segmentation(depth_img, color_img):
    face_cascade = cv2.CascadeClassifier('util/haarcascade_frontalface_default.xml')
    
    # Detect faces in color image
    faces = face_cascade.detectMultiScale(color_img, 1.3, 5)
    
    # Create mask image
    mask = np.zeros(depth_img.shape[:2], dtype=np.uint8)
    
    # Set face regions in mask to white
    for (x,y,w,h) in faces:
        mask[y:y+h, x:x+w] = 255
    
    # Apply mask to depth and color images
    depth_face = cv2.bitwise_and(depth_img, depth_img, mask=mask)
    color_face = cv2.bitwise_and(color_img, color_img, mask=mask)
    
    return depth_face, color_face


def depth_to_pointcloud_colour(depth_img, color_img, depth_scale=1.0, depth_trunc=5.0, max_depth=10.0, fx=500, fy=500, cx=None, cy=None):
    H, W, _ = depth_img.shape
    if cx is None:
        cx = W // 2
    if cy is None:
        cy = H // 2
    
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.astype(np.float32) - cx
    v = v.astype(np.float32) - cy
    
    depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2RGB)  # Convert depth image from BGR to RGB
    depth = depth_img.astype(np.float32)
    
    # Calculate depth with different weights for each color channel
    red_weight = 0.6
    green_weight = 0.2
    blue_weight = 0.2
    depth = (red_weight * depth[:, :, 2] + green_weight * depth[:, :, 1] + blue_weight * depth[:, :, 0]) / 255.0
    
    # Apply median filtering to depth image
    depth = cv2.medianBlur(depth, 5)
    
    # Adjust depth range
    min_depth = 0.5  # Minimum depth value (in meters)
    max_depth = 5.0  # Maximum depth value (in meters)
    #depth = min_depth + (max_depth - min_depth) * depth
    
    depth *= depth_scale
    depth = np.minimum(depth, depth_trunc)  # Truncate depth values
    
    Z = depth
    X = (u * Z) / fx
    Y = (v * Z) / fy
    
    pointcloud = np.stack((X, Y, Z), axis=2)
    pointcloud = pointcloud.reshape((-1, 3))
    
    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)  # Convert color from BGR to RGB
    color = color_img.reshape((-1, 3))
    
    pointcloud_color = np.concatenate((pointcloud, color), axis=1)

    # Rotate point cloud by 180 degrees
    pointcloud_color[:, 1] = -pointcloud_color[:, 1]  # Flip Y axis
    pointcloud_color[:, 2] = -pointcloud_color[:, 2]  # Flip Z axis
    
    return pointcloud_color