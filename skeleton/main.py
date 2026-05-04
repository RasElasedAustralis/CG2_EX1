import polyscope as ps
import octree as oc
import parser as parse
import visualize as vs

file = "cube2"
extension = "off"
folder = "off_files"
path = f'{folder}/{file}.{extension}'


verts, faces = parse.get_vertecies_faces(path)
octree = oc.Octree(6, 50, verts)
all_nodes = []
octree.collect_nodes(octree.root, all_nodes)
ps.init()
vs.show_pointcloud(verts)
vs.draw_box_lines(all_nodes)
ps.show()