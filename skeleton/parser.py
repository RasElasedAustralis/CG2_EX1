import numpy as np

def get_vertecies_faces(path):
     with open(path, 'r') as off:
        lines = off.readlines()
        meta = lines[1].rstrip().split(' ')
        n_vertices = int(meta[0])
        n_faces = int(meta[1])
        n_edges = int(meta[2])
        v_start = 2
        v_end = n_vertices + 2
        f_start = v_end
        f_end = f_start + n_faces
        vertices = lines[v_start:v_end]
        faces = lines[f_start:f_end]
        vertex_array = np.zeros((n_vertices, 3))
        face_array = np.zeros((n_faces, 3))
        for i, vert in enumerate(vertices):
            str_arr = vert.rstrip().lstrip().split(' ')
            doub_arr = np.zeros((3,), dtype=float)
            for j, str in enumerate(str_arr):
                doub_arr[j] = float(str)
            vertex_array[i] = doub_arr

        for i, face in enumerate(faces):
            str_arr = face.rstrip().lstrip().split(' ')[3:]
            doub_arr = np.zeros((3,), dtype=float)
            for j, str in enumerate(str_arr):
                doub_arr[j] = int(str)
            face_array[i] = doub_arr

        return (vertex_array, face_array)