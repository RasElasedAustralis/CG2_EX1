import numpy as np

def get_pivot(verts):
        max_axis = np.max(verts, axis=0)
        min_axis = np.min(verts, axis=0)
        max_dist = np.argmax(max_axis - min_axis)
        index = int(np.floor(len(verts) / 2))

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
        self.left_half = None
        self.right_half = None
        self.left_child = None
        self.right_child = None
        self.is_leaf = True


class KDTree:
    def __init__(self, max_depth, max_points, verts):
        self.max_depth = max_depth
        self.max_points = max_points
        self.verts = verts
        self.root = self.build_kdtree(self.verts, 0)

        self.create_kdtree()

    def build_kdtree(self, verts: np.ndarray, depth):
        axis, pivot, pivot_index, sorted_verts = get_pivot(verts)
        new_node = KDNode(pivot, axis, depth)
        left_half = sorted_verts[0:pivot_index]
        right_half = sorted_verts[pivot_index+1:]
        
        if depth <= self.max_depth and len(left_half) >= self.max_points:
            new_node.left_child = self.build_kdtree(left_half, depth + 1)
            new_node.right_child = self.build_kdtree(right_half, depth + 1)
            new_node.is_leaf = False
        return new_node

    def create_kdtree(self):
        axis, pivot, pivot_index, sorted_verts = get_pivot(self.verts)

        root_node = KDNode(pivot, axis, 0)
        self.root = root_node
        self.root.left_child = self.build_kdtree(root_node, sorted_verts[0:pivot_index], root_node.depth + 1)
        self.root.right_child = self.build_kdtree(root_node, sorted_verts[pivot_index+1:], root_node.depth + 1)

    
        