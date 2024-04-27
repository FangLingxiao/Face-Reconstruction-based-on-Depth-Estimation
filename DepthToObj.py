import argparse
import numpy as np
import cv2
import os
from skimage import measure

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--depthPath', dest='depthPath',
                        help='depth map path',
                        default='depth.png', type=str)
    parser.add_argument('--depthInvert', dest='depthInvert',
                        help='Invert depth map',
                        default=False, action='store_true')
    parser.add_argument('--texturePath', dest='texturePath',
                        help='corresponding image path',
                        default='', type=str)
    parser.add_argument('--objPath', dest='objPath',
                        help='output path of .obj file',
                        default='model.obj', type=str)
    parser.add_argument('--mtlPath', dest='mtlPath',
                        help='output path of .mtl file',
                        default='model.mtl', type=str)
    parser.add_argument('--matName', dest='matName',
                        help='name of material to create',
                        default='colored', type=str)
    parser.add_argument('--smooth', dest='smooth', action='store_true',
                        help='generate smooth mesh')
    parser.add_argument('--level', type=float,
                        help='threshold for marching cubes',
                        default=0.5)
    parser.add_argument('--step', type=int,
                        help='downsampling step size',
                        default=1)
    parser.add_argument('--block_size', type=int, 
                        default=256, 
                        help='block size for mesh generation')
    parser.add_argument('--downsample', type=int, 
                        default=1, 
                        help='downsampling factor for smooth mesh generation')

    args = parser.parse_args()
    return args

def smooth_mesh(depth_map, level=0.5, step_size=1, downsample_factor=1):
    '''
    use Marching Cubes algorithm to generate mesh from depth map

    Args:
        depth_map: depth map
        level(float): threshold for marching cubes from 0 to 1
        step_size(int): step size for marching cubes
        downsample_factor(int): factor for downsampling depth map

    return:
        verts(numpy.ndarray): vertex coordinates
        faces(numpy.ndarray): face indices
    '''

    # downsamples depth map
    depth_map = depth_map.astype(np.float64)
    depth_map = np.expand_dims(depth_map, axis=2)
    depth_map = depth_map[::step_size, ::step_size, :]
    depth_map = depth_map[::downsample_factor, ::downsample_factor, :]

    # use Marching Cubes algorithm to generate mesh
    verts, faces, _, _ = measure.marching_cubes(depth_map, level)

    return verts, faces

def create_mtl(mtlPath, matName, texturePath):
    if max(mtlPath.find('\\'), mtlPath.find('/')) > -1:
        os.makedirs(os.path.dirname(mtlPath), exist_ok=True)
    with open(mtlPath, "w") as f:
        f.write("newmtl " + matName + "\n"      )
        f.write("Ns 10.0000\n"                  )
        f.write("d 1.0000\n"                    )
        f.write("Tr 0.0000\n"                   )
        f.write("illum 2\n"                     )
        f.write("Ka 1.000 1.000 1.000\n"        )
        f.write("Kd 1.000 1.000 1.000\n"        )
        f.write("Ks 0.000 0.000 0.000\n"        )
        f.write("map_Ka " + texturePath + "\n"  )
        f.write("map_Kd " + texturePath + "\n"  )

#def vete(v, vt):
#    return str(v)+"/"+str(vt)

def create_obj(depthPath, depthInvert, objPath, mtlPath, matName, useMaterial=True, smooth=False, level=0.5, step_size=1, block_size=256):
    img = cv2.imread(depthPath, -1).astype(np.float64) / 1000.0
    img = np.expand_dims(img, axis=2)  # add a new axis

    if len(img.shape) > 2 and img.shape[2] > 1:
        print('Expecting a 1D map, but depth map at path %s has shape %r' % (depthPath, img.shape))
        return

    if depthInvert:
        img = 1.0 - img

    h, w = img.shape[:2]

    with open(objPath, "w") as f:
        if useMaterial:
            f.write("mtllib " + mtlPath + "\n")
            f.write("usemtl " + matName + "\n")

        vertex_id = 1

        for x_start in range(0, w, block_size):
            for y_start in range(0, h, block_size):
                x_end = min(x_start + block_size, w)
                y_end = min(y_start + block_size, h)
                block = img[y_start:y_end, x_start:x_end]

                verts, faces = smooth_mesh(block, level, step_size)

                # Write texture coordinates
                for v in verts:
                    x, y, z = v
                    f.write(f"v {x + x_start} {y + y_start} {z}\n")

                for u in range(x_start, x_end):
                    for v in range(y_start, y_end):
                        f.write(f"vt {(u - x_start) / block_size} {(v - y_start) / block_size}\n")

                for face in faces:
                    f.write(f"f {face[0] + vertex_id} {face[1] + vertex_id} {face[2] + vertex_id}\n")

                vertex_id += len(verts)

    if smooth:
        verts, faces = smooth_mesh(img, level, step_size, args.downsample)
        with open(objPath, "a") as f:  # Open file in append mode
            for v in verts:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for face in faces:
                f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

    #else:
    #    create_obj(depthPath, depthInvert, objPath, mtlPath, matName, useMaterial)

if __name__ == '__main__':
    print("STARTED")
    args = parse_args()
    useMat = args.texturePath != ''
    if useMat:
        create_mtl(args.mtlPath, args.matName, args.texturePath)
    create_obj(args.depthPath, args.depthInvert, args.objPath, args.mtlPath, args.matName, useMat, args.smooth, args.level, args.step, args.block_size)
    print("FINISHED")
