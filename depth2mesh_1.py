import argparse
import numpy as np
import cv2
import math
import os
from loop_subdivision import loop_subdiv

def parse_arguments():
    parser = argparse.ArgumentParser(description='Convert depth map to 3D model with adjusted camera parameters and texture.')
    parser.add_argument('--depth_path', dest='depth_path', default='depth.png',
                        help='Path to depth map image.')
    parser.add_argument('--texture_path', dest='texture_path', default='texture.jpg',
                        help='Path to texture image.')
    parser.add_argument('--output_obj', dest='output_obj', default='model.obj',
                        help='Output path of the .obj file.')
    parser.add_argument('--output_mtl', dest='output_mtl', default='model.mtl',
                        help='Output path of the .mtl file.')
    parser.add_argument('--fov', dest='fov', type=float, default=60.0,
                        help='Field of view in degrees.')
    parser.add_argument('--depth_scale', dest='depth_scale', type=float, default=1.0,
                        help='Depth scale factor.')
    parser.add_argument('--scale_factor', dest='scale_factor', type=float, default=1.0,
                        help='Depth map interpolation scale factor.')
    parser.add_argument('--subdivisions', dest='subdivisions', type=int, default=0,
                        help='Number of mesh subdivisions.')
    return parser.parse_args()

def create_mtl(mtl_path, texture_path):
    with open(mtl_path, "w") as f:
        f.write("newmtl material0\n")
        f.write("Ka 1.000000 1.000000 1.000000\n")
        f.write("Kd 1.000000 1.000000 1.000000\n")
        f.write("Ks 0.000000 0.000000 0.000000\n")
        f.write("Tr 1.000000\n")
        f.write("illum 1\n")
        f.write("Ns 0.000000\n")
        f.write("map_Kd {}\n".format(texture_path))

def create_obj(depth_path, obj_path, mtl_path, fov, depth_scale, scale_factor, subdivisions):
    img = cv2.imread(depth_path, -1).astype(np.float32) / 1000.0
    img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC) # Resize
    h, w = img.shape[:2]

    fov_rad = fov * math.pi / 180.0
    fx = w / (2 * math.tan(fov_rad / 2))
    fy = h / (2 * math.tan(fov_rad / 2))
    cx = w / 2
    cy = h / 2

    #with open(obj_path, "w") as f:
    #    f.write("mtllib {}\n".format(os.path.basename(mtl_path)))
    #    f.write("usemtl material0\n")

    vertices = []
    uvs = []
    faces = []

    ids = np.zeros((w, h), int)
    vid = 1

    for u in range(w):
        for v in range(h - 1, -1, -1):
            d = img[v, u] * depth_scale
            ids[u, v] = vid if d != 0.0 else 0
            vid += 1

            x = (u - cx) * d / fx
            y = (v - cy) * d / fy
            z = -d

            vertices.append([x, y, z])
            uvs.append([u / w, v / h])

    for u in range(w - 1):
        for v in range(h - 1):
            v1, v2, v3, v4 = ids[u, v], ids[u + 1, v], ids[u, v + 1], ids[u + 1, v + 1]
            if v1 == 0 or v2 == 0 or v3 == 0 or v4 == 0:
                continue
            faces.append([v1 - 1, v2 - 1, v3 - 1])
            faces.append([v3 - 1, v2 - 1, v4 - 1])

    for _ in range(subdivisions):
        vertices, faces, uvs = loop_subdiv(vertices, faces, uvs)

    with open(obj_path, "w") as f:
        f.write("mtllib {}\n".format(os.path.basename(mtl_path)))
        f.write("usemtl material0\n")

        for v in vertices:
            f.write("v {:.6f} {:.6f} {:.6f}\n".format(*v))

        for uv in uvs:
            f.write("vt {:.6f} {:.6f}\n".format(*uv))

        for face in faces:
            f.write(f"f {' '.join(f'{vt[0]}/{vt[1]}' for vt in zip(face, face))}\n")

def vete(v, vt):
    return "{}/{}".format(v + 1, vt + 1)

if __name__ == '__main__':
    args = parse_arguments()
    output_dir = os.path.dirname(args.output_obj)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.dirname(args.output_mtl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    create_mtl(args.output_mtl, args.texture_path)
    create_obj(args.depth_path, args.output_obj, args.output_mtl, args.fov, args.depth_scale, args.scale_factor, args.subdivisions)