import numpy as np
import heapq


class OCNode:
    def __init__(self, center, half_size, points, indices, depth=0):
        self.center = center
        self.half_size = half_size
        self.points = points
        self.indices = indices
        self.depth = depth
        self.children = []
        self.is_leaf = True


class Octree:
    def __init__(self, max_depth, max_points, verts):
        self.max_depth = max_depth
        self.max_points = max_points
        self.verts = verts
        self.root = None
        self.all_nodes = []

        self.create_octree()
        self.collect_nodes(self.root, self.all_nodes)

    def get_node_for_point(self, oc_node: OCNode, p):
        idx = 0
        if p[0] > oc_node.center[0]: idx |= 1
        if p[1] > oc_node.center[1]: idx |= 2
        if p[2] > oc_node.center[2]: idx |= 4

        return idx

    def tree_division(self, oc_node: OCNode):
        half = oc_node.half_size / 2.0
        offsets = np.array([
            [-1,-1,-1], [1,-1,-1], [-1, 1,-1], [1, 1,-1],
            [-1,-1, 1], [1,-1, 1], [-1, 1, 1], [1, 1, 1], 
        ]) * half

        children_points = [[] for _ in range(8)]
        children_indices = [[] for _ in range(8)]

        for p in oc_node.points:
            idx = 0
            if p[0] > oc_node.center[0]: idx |= 1
            if p[1] > oc_node.center[1]: idx |= 2
            if p[2] > oc_node.center[2]: idx |= 4

            children_points[idx].append(p)
            
        for i in oc_node.indices:
            p = self.verts[i]
            idx = 0
            if p[0] > oc_node.center[0]: idx |= 1
            if p[1] > oc_node.center[1]: idx |= 2
            if p[2] > oc_node.center[2]: idx |= 4

            children_indices[idx].append(i)
        
        for i in range(8):
            child_center = oc_node.center + offsets[i]
            child_points = np.array(children_points[i])
            child_indices = np.array(children_indices[i])
            #if(child_points.size == 0):
            #    continue
            child = OCNode(child_center, half, child_points, child_indices, oc_node.depth + 1)
            oc_node.children.append(child)

        oc_node.points = None
        oc_node.indices = None
        oc_node.is_leaf = False

    def build_octree(self, oc_node: OCNode):
        if oc_node.depth >= self.max_depth:
            return

        if len(oc_node.points) > self.max_points:
            self.tree_division(oc_node)
            for child in oc_node.children:
                if len(child.points) > 0:
                    self.build_octree(child)

    def create_octree(self):
        min_corner = self.verts.min(axis=0)
        max_corner = self.verts.max(axis=0)

        center = (min_corner + max_corner) / 2
        half_size = np.max(max_corner - min_corner) / 2

        indices = range(len(self.verts))
        root_node = OCNode(center, half_size, self.verts, indices)
        self.build_octree(root_node)
        self.root = root_node

    def collect_nodes(self, node: OCNode, all_nodes):
        if node.is_leaf:
            all_nodes.append(node)
        else:
            for child in node.children:
                self.collect_nodes(child, all_nodes)

    def get_node_by_point(self, oc_node: OCNode, coords):
        if oc_node.is_leaf:
            return oc_node
        
        child_idx = self.get_node_for_point(oc_node, coords)
        return self.get_node_by_point(oc_node.children[child_idx], coords)
    
    def aabb_distance_squared(self, point, center, half_size):
        min = center - half_size
        max = center + half_size

        delta = np.maximum(0, np.maximum(min - point, point - max))
        return np.dot(delta, delta)

    def knn_search(self, node: OCNode, query, k, heap):
        if node.is_leaf:
            for i in node.indices:
                p = self.verts[i]
                dist_2 = np.sum((p - query) ** 2)
                if len(heap) < k:
                    heapq.heappush(heap, (-dist_2, i))
                else:
                    worst_dist_2 = -heap[0][0]
                    if dist_2 < worst_dist_2:
                        heapq.heapreplace(heap, (-dist_2, i))
            return
        
        child_distances = []
        for child in node.children:
            aabb_dist_2 = self.aabb_distance_squared(query, child.center, child.half_size)
            child_distances.append((aabb_dist_2, child))

        child_distances.sort(key=lambda x: x[0])
        for d2, child in child_distances:
            if len(heap) == k:
                current_worst = -heap[0][0]
                if d2 > current_worst:
                    continue

            self.knn_search(child, query, k, heap)

    def knn_wrapper(self, selection_index, k=8):
        heap = []
        query = self.verts[selection_index]
        self.knn_search(self.root, query, k, heap)

        results = [(np.sqrt(-dist_2), i) for dist_2, i in heap]
        #results.sort(key=lambda x: x[0])

        return results
    
    def radius_search(self, node: OCNode, query, radius_squared, results):
        aabb_dist_2 = self.aabb_distance_squared(query, node.center, node.half_size)

        if aabb_dist_2 >  radius_squared:
            return
        
        if node.is_leaf:
            for i in node.indices:
                p = self.verts[i]
                dist_2 = np.sum((p - query) ** 2)
                
                if dist_2 <= radius_squared:
                    results.append(i)
            return
        
        for child in node.children:
            self.radius_search(child, query, radius_squared, results)

    def radius_wrapper(self, selection_index, radius=0.01):
        results = []
        radius_squared = radius ** 2

        query = self.verts[selection_index]
        self.radius_search(self.root, query, radius_squared, results)

        return results


        