import polyscope as ps
import polyscope.imgui as psim
import numpy as np

def get_bounding_lines(center, half_size):
    corners = []
    for dx in [-half_size, half_size]:
        for dy in [-half_size, half_size]:
            for dz in [-half_size, half_size]:
                corners.append(center + np.array([dx, dy, dz]))

    corners = np.array(corners)
    
    edges = [
        [0,1], [0,2], [1,3], [2,3],
        [4,5], [4,6], [5,7], [6,7],
        [0,4], [1,5], [2,6], [3,7]
    ]

    return corners, edges

def draw_box_lines(all_nodes):
    for i, node in enumerate(all_nodes):
        corners, edges = get_bounding_lines(node.center, node.half_size)
        curve_network = ps.register_curve_network(f'node_{i}', corners, edges)
        curve_network.set_color((1.0, 0.0, 0.0))
        curve_network.set_radius(0.001)

#def draw_planes():
    

def show_pointcloud(verts):
    pointcloud = ps.register_point_cloud("my points", verts)
    return pointcloud

def show_mesh(verts, faces):
    ps.register_surface_mesh("my mesh", verts, faces, smooth_shade=True)