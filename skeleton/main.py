import polyscope as ps
import polyscope.imgui as psim
import numpy as np

import octree as oc
import kdtree as kd
import parser as parse
import visualize as vs

selection_index = None

file = "cube2"
extension = "off"
folder = "off_files"
path = f'{folder}/{file}.{extension}'


verts, faces = parse.get_vertecies_faces(path)
octree = oc.Octree(6, 50, verts)
kdtree = kd.KDTree(6, 50, verts)

def set_point_color(pointcloud: ps.PointCloud, indices, color):
    curr_color = pointcloud.get_color()
    amount = pointcloud.n_points()
    colors = np.full((amount, 3), curr_color)
    for index in indices:
        colors[index] = color
    pointcloud.add_color_quantity("selection_coloring", colors, enabled=True)

def callback():
    selection = None
    pointcloud = None

    io = psim.GetIO()
    if io.MouseClicked[0]:
        screen_coords = io.MousePos
        pick_result = ps.pick(screen_coords=screen_coords)
        if pick_result.is_hit:
            selection = pick_result.local_index
            pointcloud = ps.get_point_cloud("my points")
            node = octree.get_node_by_point(octree.root, octree.verts[selection])
            set_point_color(pointcloud, node.indices, (1.0, 0.0, 0.0))
            #set_point_color(pointcloud, [selection], (0.0, 1.0, 0.0))


ps.init()
vs.show_pointcloud(verts)
ps.set_user_callback(callback)

#vs.draw_box_lines(octree.all_nodes)

ps.show()