import math

def center_point(p1, p2):
    return [(p1[i] + p2[i]) / 2 for i in range(3)]

def sum_point(p1, p2):
    return [p1[i] + p2[i] for i in range(3)]

def sub_point(p1, p2):
    return [p1[i] - p2[i] for i in range(3)]

def div_point(p, d):
    return [p[i] / d for i in range(3)]

def mul_point(p, m):
    return [p[i] * m for i in range(3)]

def get_edges_faces(input_points, input_faces):
    edges = []
    for facenum, face in enumerate(input_faces):
        num_points = len(face)
        for pointindex in range(num_points):
            pointnum_1 = face[pointindex]
            pointnum_2 = face[(pointindex + 1) % num_points]
            if pointnum_1 > pointnum_2:
                pointnum_1, pointnum_2 = pointnum_2, pointnum_1
            edges.append([pointnum_1, pointnum_2, facenum])

    edges.sort()

    merged_edges = []
    num_edges = len(edges)
    eindex = 0
    while eindex < num_edges:
        e1 = edges[eindex]
        if eindex < num_edges - 1:
            e2 = edges[eindex + 1]
            if e1[0] == e2[0] and e1[1] == e2[1]:
                merged_edges.append([e1[0], e1[1], e1[2], e2[2]])
                eindex += 2
            else:
                merged_edges.append([e1[0], e1[1], e1[2], None])
                eindex += 1
        else:
            merged_edges.append([e1[0], e1[1], e1[2], None])
            eindex += 1

    edges_centers = []
    for me in merged_edges:
        p1 = input_points[me[0]]
        p2 = input_points[me[1]]
        cp = center_point(p1, p2)
        edges_centers.append(me + [cp])

    return edges_centers

def get_faces_points(input_points, input_faces, edges_faces):
    faces_points = []
    for edge in edges_faces:
        if edge[3] is not None:
            faces_point = [0.0, 0.0, 0.0]
            for face_num in [edge[2], edge[3]]:
                for point_num in input_faces[face_num]:
                    if point_num != edge[0] and point_num != edge[1]:
                        faces_point = sum_point(faces_point, mul_point(input_points[point_num], 0.25))
                    else:
                        faces_point = sum_point(faces_point, mul_point(input_points[point_num], 0.125))
            faces_points.append(faces_point)
        else:
            faces_points.append([0.0, 0.0, 0.0])

    return faces_points

def get_edge_points(edges_faces, faces_points):
    edge_points = []
    for i, edge in enumerate(edges_faces):
        if edge[3] is not None:
            edge_points.append(center_point(edge[4], faces_points[i]))
        else:
            edge_points.append(edge[4])

    return edge_points

def get_points_faces(input_points, input_faces):
    num_points = len(input_points)
    points_faces = [[0, []] for _ in range(num_points)]

    for facenum, face in enumerate(input_faces):
        for pointnum in face:
            points_faces[pointnum][0] += 1
            points_faces[pointnum][1].append(facenum)

    return points_faces

def get_face_sum(input_points, input_faces):
    return [sum_point(input_points[face[2]], sum_point(input_points[face[0]], input_points[face[1]])) for face in input_faces]

def get_new_points(input_points, points_faces, face_sum):
    new_points = []

    for i, point in enumerate(input_points):
        q = [0.0, 0.0, 0.0]
        beta = 5.0 / 8 - pow(3.0 / 8 + 1.0 / 4 * math.cos(2 * math.pi / points_faces[i][0]), 2)
        for point_face_num in points_faces[i][1]:
            q = sum_point(q, face_sum[point_face_num])
        q = sub_point(q, mul_point(point, points_faces[i][0]))
        q = div_point(q, 2 * points_faces[i][0])
        v = mul_point(point, 1 - beta)
        new_points.append(sum_point(v, mul_point(q, beta)))

    return new_points

def switch_nums(point_nums):
    if point_nums[0] < point_nums[1]:
        return point_nums
    else:
        return (point_nums[1], point_nums[0])

def loop_subdiv(input_points, input_faces, input_uvs):
    edges_faces = get_edges_faces(input_points, input_faces)
    faces_points = get_faces_points(input_points, input_faces, edges_faces)
    edge_points = get_edge_points(edges_faces, faces_points)
    points_faces = get_points_faces(input_points, input_faces)
    face_sum = get_face_sum(input_points, input_faces)
    new_points = get_new_points(input_points, points_faces, face_sum)

    next_pointnum = len(new_points)
    edge_point_nums = {}

    for edgenum, edge in enumerate(edges_faces):
        pointnum_1 = edge[0]
        pointnum_2 = edge[1]
        edge_point = edge_points[edgenum]
        new_points.append(edge_point)
        edge_point_nums[(pointnum_1, pointnum_2)] = next_pointnum
        next_pointnum += 1

    new_faces = []
    new_uvs = []

    for oldfacenum, oldface in enumerate(input_faces):
        if len(oldface) == 3:
            a, b, c = oldface
            uv_a, uv_b, uv_c = input_uvs[a], input_uvs[b], input_uvs[c]

            edge_point_ab = edge_point_nums[switch_nums((a, b))]
            edge_point_ca = edge_point_nums[switch_nums((c, a))]
            edge_point_bc = edge_point_nums[switch_nums((b, c))]

            uv_ab = [(uv_a[0] + uv_b[0]) / 2, (uv_a[1] + uv_b[1]) / 2]
            uv_ca = [(uv_c[0] + uv_a[0]) / 2, (uv_c[1] + uv_a[1]) / 2]
            uv_bc = [(uv_b[0] + uv_c[0]) / 2, (uv_b[1] + uv_c[1]) / 2]

            new_faces.append((a, edge_point_ab, edge_point_ca))
            new_faces.append((b, edge_point_bc, edge_point_ab))
            new_faces.append((c, edge_point_ca, edge_point_bc))
            new_faces.append((edge_point_ca, edge_point_ab, edge_point_bc))

            new_uvs.append(uv_a)
            new_uvs.append(uv_ab)
            new_uvs.append(uv_ca)
            new_uvs.append(uv_b)
            new_uvs.append(uv_bc)
            new_uvs.append(uv_ab)
            new_uvs.append(uv_c)
            new_uvs.append(uv_ca)
            new_uvs.append(uv_bc)
            new_uvs.append(uv_ca)
            new_uvs.append(uv_ab)
            new_uvs.append(uv_bc)

    return new_points, new_faces, new_uvs