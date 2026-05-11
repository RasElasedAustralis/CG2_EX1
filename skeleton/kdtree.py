import numpy as np

def get_pivot(verts):
        max_axis = np.max(verts, axis=0)
        min_axis = np.min(verts, axis=0)
        max_dist = np.argmax(max_axis - min_axis)
        index = int(np.trunc(len(verts) / 2))

        axis_array = verts[:,max_dist]
        sorted_along_axis = np.argsort(axis_array)
        sorted_verts = verts.take(sorted_along_axis, axis=0)
        pivot = sorted_verts[index]

        return max_dist, pivot, index, sorted_verts


class KDNode:
    def __init__(self, pivot: np.ndarray, axis: int, depth=0):
        self.pivot = pivot
        self.axis = axis
        self.depth = depth
        self.left_child = None
        self.right_child = None
        self.is_leaf = True


class KDTree:
    def __init__(self, max_depth, max_points, verts):
        self.max_depth = max_depth
        self.max_points = max_points
        self.verts = verts
        self.root = None

        self.create_kdtree()

    def build_kdtree(self, kd_node: KDNode, verts: np.ndarray):
        if kd_node.depth >= self.max_depth:
            return

        if len(verts) > self.max_points:
            axis, pivot, pivot_index, sorted_verts = get_pivot(verts)
            new_node = KDNode(pivot, axis, kd_node.depth + 1)
            left_half = sorted_verts[0:pivot_index]
            right_half = sorted_verts[pivot_index+1:]
            
            new_node.left_child = self.build_kdtree(new_node, left_half)
            new_node.right_child = self.build_kdtree(new_node, right_half)
            new_node.is_leaf = False

    def create_kdtree(self):
        axis, pivot, pivot_index, sorted_verts = get_pivot(self.verts)

        root_node = KDNode(pivot, axis, 0)
        self.root = root_node
        self.root.left_child = self.build_kdtree(root_node, sorted_verts[0:pivot_index])
        self.root.right_child = self.build_kdtree(root_node, sorted_verts[pivot_index+1:])

    
        