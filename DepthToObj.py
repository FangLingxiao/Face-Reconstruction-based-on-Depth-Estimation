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

    args = parser.parse_args()
    return args

def smooth_mesh(depth_map, level=0.5, step_size=1):
    '''
    use Marching Cubes algorithm to generate mesh from depth map

    Args:
        depth_map: depth map
        level(float): threshold for marching cubes from 0 to 1
        step_size(int): step size for marching cubes

    return:
        verts(numpy.ndarray): vertex coordinates
        faces(numpy.ndarray): face indices
    '''
    # downsamples depth map
    depth_map = depth_map[::step_size, ::step_size]

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

def vete(v, vt):
    return str(v)+"/"+str(vt)

def create_obj(depthPath, depthInvert, objPath, mtlPath, matName, useMaterial = True, smooth=False, level=0.5, step_size=1):
    
    img = cv2.imread(depthPath, -1).astype(np.float32) / 1000.0

    if len(img.shape) > 2 and img.shape[2] > 1:
       print('Expecting a 1D map, but depth map at path %s has shape %r'% (depthPath, img.shape))
       return

    if depthInvert == True:
        img = 1.0 - img

    h, w = img.shape[:2]
    cx, cy = w //2, h // 2 # Assume that the main point is at the center of the image

    #FOV = math.pi/4
    #D = (img.shape[0]/2)/math.tan(FOV/2)

    #if max(objPath.find('\\'), objPath.find('/')) > -1:
    #    os.makedirs(os.path.dirname(mtlPath), exist_ok=True)
    
    with open(objPath,"w") as f:    
        if useMaterial:
            f.write("mtllib " + mtlPath + "\n")
            f.write("usemtl " + matName + "\n")

        #ids = np.zeros((img.shape[1], img.shape[0]), int)
        #vid = 1
            
        vertex_id = 1

        for v in range(h):
            for u in range(w):
                depth = img[v, u]
                if depth > 0:
                    x = (u - cx) * depth
                    y = (v - cy) * depth
                    z = depth
                    f.write(f"v{x} {y} {z}\n")
                    vertex_id += 1

        for u in range(w):  #The range here is [0, w-1) and [0, h-1). If the width or height of the image is 1, the loop condition will never be satisfied, resulting in an infinite loop.
            for v in range(h):
                f.write(f"vt {u/w} {v/h}\n")

        for u in range(w-1):
            for v in range(h-1):
                v1 = v * w + u + 1
                v2 = v1 + 1
                v3 = v1 + w
                v4 = v3 + 1

                if u+1 < w and v+1 < h:
                    if img[v,u] > 0 and img[v,u+1] > 0 and img[v+1,u] > 0 and img[v+1,u+1] > 0:
                        f.write(f"f {v1}/{v1} {v2}/{v2} {v3}/{v3}\n")
                        f.write(f"f {v3}/{v3} {v2}/{v2} {v4}/{v4}\n")

    if smooth:
        verts, faces = smooth_mesh(img, level, step_size)

        with open(objPath, "w") as f:
            if useMaterial:
                f.write("mtllib" + mtlPath + "\n")
                f.write("usemtl" + matName + "\n")

            for v in verts:
                f.write("v {} {} {}\n".format(v[0], v[1], v[2]))

            for face in faces:
                f.write("f {} {} {}\n".format(face[0]+1, face[1]+1, face[2]+1))

    #else:
    #    create_obj(depthPath, depthInvert, objPath, mtlPath, matName, useMaterial)

if __name__ == '__main__':
    print("STARTED")
    args = parse_args()
    useMat = args.texturePath != ''
    if useMat:
        create_mtl(args.mtlPath, args.matName, args.texturePath)
    create_obj(args.depthPath, args.depthInvert, args.objPath, args.mtlPath, args.matName, useMat, args.smooth, args.level, args.step)
    print("FINISHED")
