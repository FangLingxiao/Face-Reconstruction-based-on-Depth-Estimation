import argparse
import numpy as np
import cv2
import math
import os

# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Convert depth map to 3D model.')

    parser.add_argument('--depth_path', dest='depth_path', default='depth.png',
                        help='Path to depth map image.')
    parser.add_argument('--invert_depth', dest='invert_depth', action='store_true',
                        help='Invert the depth map.')
    parser.add_argument('--texture_path', dest='texture_path', default='',
                        help='Path to corresponding texture image.')
    parser.add_argument('--output_obj', dest='output_obj', default='model.obj',
                        help='Output path of the .obj file.')
    parser.add_argument('--output_mtl', dest='output_mtl', default='model.mtl',
                        help='Output path of the .mtl file.')
    parser.add_argument('--material_name', dest='material_name', default='colored',
                        help='Name of the material to create.')

    return parser.parse_args()

# 计算法向量
def calculate_normals(vertices, faces):
    normals = np.zeros_like(vertices)
    face_normals = np.zeros((len(faces), 3))

    for i, face in enumerate(faces):
        v1, v2, v3 = [vertices[j - 1] for j in face]
        edge1 = v2 - v1
        edge2 = v3 - v1
        face_normal = np.cross(edge1, edge2)
        face_normal /= np.linalg.norm(face_normal)
        face_normals[i] = face_normal

        for j in face:
            normals[j - 1] += face_normal

    for i in range(len(normals)):
        normals[i] /= np.linalg.norm(normals[i])

    return normals

# 平滑法向量
def smooth_normals(normals, faces, iterations=1):
    smoothed_normals = normals.copy()

    for _ in range(iterations):
        new_normals = smoothed_normals.copy()

        for i, face in enumerate(faces):
            face_normal = np.mean([smoothed_normals[j - 1] for j in face], axis=0)
            face_normal /= np.linalg.norm(face_normal)

            for j in face:
                new_normals[j - 1] += face_normal

        for i in range(len(new_normals)):
            new_normals[i] /= np.linalg.norm(new_normals[i])

        smoothed_normals = new_normals

    return smoothed_normals

# Create .mtl material file
def create_mtl(mtl_path, material_name, texture_path):
    os.makedirs(os.path.dirname(mtl_path), exist_ok=True)
    with open(mtl_path, "w") as f:
        f.write("newmtl {}\n".format(material_name))  # Write material name
        f.write("Ns 10.0000\n")  # Set shininess
        f.write("d 1.0000\n")  # Set dissolve
        f.write("Tr 0.0000\n")  # Set transparency
        f.write("illum 2\n")  # Set illumination model
        f.write("Ka 1.000 1.000 1.000\n")  # Set ambient reflectivity
        f.write("Kd 1.000 1.000 1.000\n")  # Set diffuse reflectivity
        f.write("Ks 0.000 0.000 0.000\n")  # Set specular reflectivity
        f.write("map_Ka {}\n".format(texture_path))  # Set ambient texture map
        f.write("map_Kd {}\n".format(texture_path))  # Set diffuse texture map

# Compute texture coordinates for vertices
def vertex_tex_coord(u, v, w, h):
    return u / w, 1 - v / h

# Create .obj model file
def create_obj(depth_path, invert_depth, obj_path, mtl_path, material_name, use_material=True):
    # Read depth map
    img = cv2.imread(depth_path, -1).astype(np.float32) / 1000.0

    # Check depth map dimensions
    if len(img.shape) > 2 and img.shape[2] > 1:
        print('Expecting a 1D map, but depth map at path %s has shape %r' % (depth_path, img.shape))
        return

    # Invert depth map pixel values if needed
    if invert_depth:
        img = 1.0 - img

    w, h = img.shape[1], img.shape[0]
    fov = math.pi / 4
    D = (h / 2) / math.tan(fov / 2)

    vertices = []
    uvs = []
    faces = []

    ids = np.zeros((w, h), int)
    vid = 1

    # Iterate over depth map pixels to generate vertices
    for u in range(w):
        for v in range(h - 1, -1, -1):
            d = img[v, u]
            ids[u, v] = vid if d != 0.0 else 0
            vid += 1

            x = u - w / 2
            y = v - h / 2
            z = -D

            norm = 1 / math.sqrt(x * x + y * y + z * z)
            t = d / (z * norm)

            x, y, z = -t * x * norm, t * y * norm, -t * z * norm
            vertices.append((x, y, z))
            uvs.append(vertex_tex_coord(u, v, w, h))  # Write vertex coordinates

    # Generate texture coordinates
    #for u in range(w):
    #    for v in range(h):
    #        u_tex, v_tex = vertex_tex_coord(u, v, w, h)
    #        f.write("vt {:.6f} {:.6f}\n".format(u_tex, v_tex))  # Write texture coordinates

    # Generate faces based on vertex indices
    for u in range(w - 1):
        for v in range(h - 1):
            v1, v2, v3, v4 = ids[u, v], ids[u + 1, v], ids[u, v + 1], ids[u + 1, v + 1]
            if v1 == 0 or v2 == 0 or v3 == 0 or v4 == 0:
                continue
            faces.append((v1, v2, v3))
            faces.append((v3, v2, v4))

    # 计算法向量
    normals = calculate_normals(np.array(vertices), np.array(faces))

    # 平滑法向量
    smoothed_normals = smooth_normals(normals, np.array(faces), iterations=5)
    
    # Create .obj file
    os.makedirs(os.path.dirname(obj_path), exist_ok=True)
    with open(obj_path, "w") as f:
        if use_material:
            f.write("mtllib {}\n".format(mtl_path))
            f.write("usemtl {}\n".format(material_name))

        for vertex in vertices:
            f.write("v {:.6f} {:.6f} {:.6f}\n".format(*vertex))

        for normal in smoothed_normals:
            f.write("vn {:.6f} {:.6f} {:.6f}\n".format(*normal))

        for uv in uvs:
            f.write("vt {:.6f} {:.6f}\n".format(*uv))

        for face in faces:
            f.write("f")
            for j in face:
                f.write(" {}/{}/{} ".format(j, j, j))
            f.write("\n")

# Combine vertex indices and texture coordinates into a string
#def vete(v, vt):
#    return "{}/{}".format(v, vt)

if __name__ == '__main__':
    print("STARTED")
    args = parse_arguments()
    use_mat = args.texture_path != ''
    if use_mat:
        create_mtl(args.output_mtl, args.material_name, args.texture_path)
    create_obj(args.depth_path, args.invert_depth, args.output_obj, args.output_mtl, args.material_name, use_mat)
    print("FINISHED")
