import numpy as np
import time
import matplotlib.pyplot as plt
import octree as oc

sizes = [
    1_000,
    5_000,
    10_000,
    50_000,
    100_000,
    500_000,
]
depths = [1, 5, 10, 50, 100, 500, 1000]
max_points = [1000, 500, 100, 50, 10, 5, 1]

NUM_QUERIES = 100
SAMPLE_SIZE = 10_000
K = 12

def construct_random_octrees(tree_size, num_queries, depth=10, max_points=50):
    print(f"Testing N = {tree_size}, Queries = {num_queries}, Depth = {depth}, max Points = {max_points}")
    pts = np.random.rand(tree_size, 3)
    octree = oc.Octree(depth, max_points, pts)
    query_indices = np.random.randint(0, tree_size, size=num_queries)

    return pts, octree, query_indices

def perform_size_benchmark(sizes, num_queries, k, radius):
    octree_times = []
    radius_search_times = []
    bruteforce_times = []
    for n in sizes:
        pts, octree, query_indices = construct_random_octrees(n, num_queries)
        start = time.perf_counter()
        for idx in query_indices:
            octree.knn_wrapper(idx, k)
        end = time.perf_counter()
        octree_avg = (end - start) / num_queries
        octree_times.append(octree_avg)

        start = time.perf_counter()
        for idx in query_indices:
            octree.radius_wrapper(idx, radius)
        end = time.perf_counter()
        radius_avg = (end- start) / num_queries
        radius_search_times.append(radius_avg)

        start = time.perf_counter()
        for idx in query_indices:
            p = pts[idx]
            dists = np.linalg.norm(pts - p, axis=1)
            nearest = np.argsort(dists)[:k]
        end = time.perf_counter()
        brute_avg = (end - start) / num_queries
        bruteforce_times.append(brute_avg)

        print(f"octree:      {octree_avg:.6e}")
        print(f"bruteforce:  {brute_avg:.6e}")

    return octree_times, radius_search_times, bruteforce_times

def perform_depth_benchmark(depths, num_queries, k, radius):
    octree_times = []
    radius_search_times = []
    bruteforce_times = []
    for n in depths:
        pts, octree, query_indices = construct_random_octrees(SAMPLE_SIZE, num_queries, n, SAMPLE_SIZE)
        start = time.perf_counter()
        for idx in query_indices:
            octree.knn_wrapper(idx, k)
        end = time.perf_counter()
        octree_avg = (end - start) / num_queries
        octree_times.append(octree_avg)

        start = time.perf_counter()
        for idx in query_indices:
            octree.radius_wrapper(idx, radius)
        end = time.perf_counter()
        radius_avg = (end- start) / num_queries
        radius_search_times.append(radius_avg)

        start = time.perf_counter()
        for idx in query_indices:
            p = pts[idx]
            dists = np.linalg.norm(pts - p, axis=1)
            nearest = np.argsort(dists)[:k]
        end = time.perf_counter()
        brute_avg = (end - start) / num_queries
        bruteforce_times.append(brute_avg)

        print(f"octree:      {octree_avg:.6e}")
        print(f"bruteforce:  {brute_avg:.6e}")

    return octree_times, radius_search_times, bruteforce_times

def perform_bucket_benchmark(max_points, num_queries, k, radius):
    octree_times = []
    radius_search_times = []
    bruteforce_times = []
    for n in max_points:
        pts, octree, query_indices = construct_random_octrees(SAMPLE_SIZE, num_queries, SAMPLE_SIZE, n)
        start = time.perf_counter()
        for idx in query_indices:
            octree.knn_wrapper(idx, k)
        end = time.perf_counter()
        octree_avg = (end - start) / num_queries
        octree_times.append(octree_avg)

        start = time.perf_counter()
        for idx in query_indices:
            octree.radius_wrapper(idx, radius)
        end = time.perf_counter()
        radius_avg = (end- start) / num_queries
        radius_search_times.append(radius_avg)

        start = time.perf_counter()
        for idx in query_indices:
            p = pts[idx]
            dists = np.linalg.norm(pts - p, axis=1)
            nearest = np.argsort(dists)[:k]
        end = time.perf_counter()
        brute_avg = (end - start) / num_queries
        bruteforce_times.append(brute_avg)

        print(f"octree:      {octree_avg:.6e}")
        print(f"bruteforce:  {brute_avg:.6e}")

    return octree_times, radius_search_times, bruteforce_times

def plot_size_benchmark(octree_times, radius_times, bruteforce_times):
    plt.figure(figsize=(8,5))

    plt.plot(sizes, octree_times, marker="o", label="knn")
    plt.plot(sizes, radius_times, marker="o", label="Radius")
    plt.plot(sizes, bruteforce_times, marker="o", label="Brute Force")

    plt.xlabel("Number of points")
    plt.ylabel("Average query time (s)")
    plt.xscale("log")
    plt.yscale("log")

    plt.title(f'Query time for a varying pointcloud size\nand a constant tree-depth and bucket-size of depth = 10, bucket_s = 50')
    plt.legend()
    plt.grid(True)

    plt.show()

def plot_depth_benchmark(octree_times, radius_times, bruteforce_times):
    plt.figure(figsize=(8,5))

    plt.plot(depths, octree_times, marker="o", label="knn")
    plt.plot(depths, radius_times, marker="o", label="Radius")
    plt.plot(depths, bruteforce_times, marker="o", label="Brute Force")

    plt.xlabel("Tree depth")
    plt.ylabel("Average query time (s)")
    plt.xscale("log")
    plt.yscale("log")

    plt.title(f'Query time for a constant pointcloud size of N = {SAMPLE_SIZE},\na varying tree-depth and constant bucket-size of bucket_s = 50')
    plt.legend()
    plt.grid(True)

    plt.show()

def plot_bucket_benchmark(octree_times, radius_times, bruteforce_times):
    plt.figure(figsize=(8,5))

    plt.plot(max_points, octree_times, marker="o", label="knn")
    plt.plot(max_points, radius_times, marker="o", label="Radius")
    plt.plot(max_points, bruteforce_times, marker="o", label="Brute Force")

    plt.xlabel("Bucket Size")
    plt.ylabel("Average query time (s)")
    plt.xscale("log")
    plt.yscale("log")

    plt.title(f'Query time for a constant pointcloud size of N = {SAMPLE_SIZE},\na constant tree-depth of depth = 10 and a varying bucket-size')
    plt.legend()
    plt.grid(True)

    plt.show()