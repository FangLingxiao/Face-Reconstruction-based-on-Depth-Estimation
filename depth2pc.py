import numpy as np
import cv2
import math
import os


def depth_to_point_cloud(depth_map, invert_depth, fov):
    h, w = depth_map.shape[:2]
    focal_length = (w / 2) / math.tan(fov / 2)
    point_cloud = []

    for v in range(h):
        for u in range(w):
            d = depth_map[v, u]
            if d == 0:
                continue

            if invert_depth:
                d = 1.0 - d

            x = (u - w / 2) * d / focal_length
            y = (v - h / 2) * d / focal_length
            z = -d

            point_cloud.append((x, y, z))

    return np.array(point_cloud)


def create_point_cloud(depth_path, invert_depth, output_path, fov):
    try:
        img = cv2.imread(depth_path, -1).astype(np.float32) / 1000.0
        if len(img.shape) > 2 and img.shape[2] > 1:
            print('Expecting a 1D map, but depth map at path %s has shape %r' % (depth_path, img.shape))
            return

        point_cloud = depth_to_point_cloud(img, invert_depth, fov)

        # save as .ply
        header = "ply\nformat ascii 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nend_header\n".format(len(point_cloud))
        with open(output_path, 'w') as f:
            f.write(header)
            for point in point_cloud:
                f.write("{:.6f} {:.6f} {:.6f}\n".format(*point))

        print("Point cloud generated and saved as {}".format(output_path))
    except IOError as e:
        print("An error occurred while writing to the output file:")
        print(str(e))
    except Exception as e:
        print("An unexpected error occurred:")
        print(str(e))