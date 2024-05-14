import numpy as np
import cv2
import math
import os


def depth_to_point_cloud(depth_map, depth_scale, color_map, fov):
    h, w = depth_map.shape[:2]
    focal_length = (w / 2) / math.tan(fov / 2)
    
    point_cloud = []
    for v in range(h):
        for u in range(w):
            d = depth_map[v, u] * depth_scale
            if d == 0:
                continue

            x = (u - w / 2) * d / focal_length
            y = (v - h / 2) * d / focal_length
            z = -d

            color = color_map[v, u]
            r, g, b = color

            point_cloud.append((x, y, z, r, g, b))            

    return np.array(point_cloud)


def create_point_cloud(depth_path, depth_scale, color_path, fov):
    #try:
    depth_map = cv2.imread(depth_path, -1).astype(np.float32) / 1000.0
    color_map = cv2.imread(color_path)
    #color_map = cv2.cvtColor(color_map, cv2.COLOR_BAYER_BG2BGR)
    
    if len(depth_map.shape) > 2 and depth_map.shape[2] > 1:
        print('Expecting a 1D map, but depth map at path %s has shape %r' % (depth_path, depth_map.shape))
        return

    point_cloud = depth_to_point_cloud(depth_map, depth_scale, color_map, fov)
    return point_cloud


    """
        # save as .ply
        header = "ply\nformat ascii 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n".format(len(point_cloud))
        with open(output_path, 'w') as f:
            f.write(header)
            for point in point_cloud:
                f.write("{:.6f} {:.6f} {:.6f} {:d} {:d} {:d}\n".format(*point[:3], int(point[3]), int(point[4]), int(point[5])))

        print("Point cloud generated and saved as {}".format(output_path))
    except IOError as e:
        print("An error occurred while writing to the output file:")
        print(str(e))
    except Exception as e:
        print("An unexpected error occurred:")
        print(str(e))

        """
