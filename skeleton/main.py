import polyscope as ps
import polyscope.imgui as psim
import numpy as np

import octree as oc
import kdtree as kd
import parser as parse
import visualize as vs
import benchmark as bm

file = "bunny"
extension = "off"
folder = "off_files"
path = f'{folder}/{file}.{extension}'
sizes = [1000, 5000, 10000, 50000, 100000, 500000]
depths = [1, 5, 10, 50, 100, 500, 1000]
max_points = [1000, 500, 100, 50, 10, 5, 1]

search_states = ["None", "knn", "radius"]
data_structures = ["None", "octree", "kdtree"]
benchmark_types = ["cloudsize", "depth", "bucket_size"]
ui_state = {
    "mouse_selection": None,
    "pointcloud": None,
    "off_loaded": False,
    "verts": None,
    "faces": None,
    "octree": None,
    "kdtree": None,
    "tree_visible": False,
    "tree_visualization": None,
    "k": 5,
    "radius": 0.01,
    "current_search_method": "None",
    "current_data_structure": "None",
    "current_benchmark_type": "cloudsize",
    "benchmarks_done": False,
    "oct_benchmark": None,
    "rad_benchmark": None,
    "brute_benchmark": None
}


def load_off(path):
    verts, faces = parse.get_vertecies_faces(path)
    ui_state["verts"] = verts
    ui_state["faces"] = faces
    ui_state["off_loaded"] = True
    vs.show_pointcloud(verts)

def load_octree():
    if not ui_state["off_loaded"] or ui_state["octree"] != None: return
    ui_state["octree"] = oc.Octree(6, 50, ui_state["verts"])

def load_kdtree():
    if not ui_state["off_loaded"]or ui_state["kdtree"] != None: return
    ui_state["kdtree"] = kd.KDTree(6, 50, ui_state["verts"])


def set_point_color(pointcloud: ps.PointCloud, indices, color, selection, s_color=(1.0, 0.0, 0.0)):
    curr_color = pointcloud.get_color()
    amount = pointcloud.n_points()
    colors = np.full((amount, 3), curr_color)
    for index in indices:
        colors[index] = color
    colors[selection] = s_color
    pointcloud.add_color_quantity("selection_coloring", colors, enabled=True)

def handle_trees():
    if not ui_state["off_loaded"]: return
    match ui_state["current_data_structure"]:
        case "octree":
            load_octree()

        case "kdtree":
            load_kdtree()

def handle_search():
    if ui_state["mouse_selection"] is None: return
    match ui_state["current_search_method"]:
        case "knn":
            if ui_state["octree"] is None: return
            knn = ui_state["octree"].knn_wrapper(ui_state["mouse_selection"], 12)
            knn_indices = [i for _, i in knn]
            set_point_color(ui_state["pointcloud"], knn_indices, (0.0, 1.0, 0.0), ui_state["mouse_selection"], (1.0, 0.0, 0.0))

        case "radius":
            if ui_state["octree"] is None: return
            radius_indices = ui_state["octree"].radius_wrapper(ui_state["mouse_selection"], 0.01)
            set_point_color(ui_state["pointcloud"], radius_indices, (0.0, 0.0, 1.0), ui_state["mouse_selection"], (1.0, 0.0, 0.0))

        case "None":
            set_point_color(ui_state["pointcloud"], [ui_state["mouse_selection"]], (1.0, 0.0, 0.0), ui_state["mouse_selection"])

def handle_benchmarks():
    match ui_state["current_benchmark_type"]:
        case "cloudsize":
            knn, rad, brute = bm.perform_size_benchmark(sizes, 100, 12, 0.01)
            ui_state["oct_benchmark"] = knn
            ui_state["brute_benchmark"] = brute
            ui_state["rad_benchmark"] = rad
            ui_state["benchmarks_done"] = True
            #return knn, rad, brute
        case "depth":
            knn, rad, brute = bm.perform_depth_benchmark(depths, 100, 12, 0.01)
            ui_state["oct_benchmark"] = knn
            ui_state["brute_benchmark"] = brute
            ui_state["rad_benchmark"] = rad
            ui_state["benchmarks_done"] = True
            #return knn, rad, brute
        case "bucket_size":
            knn, rad, brute = bm.perform_bucket_benchmark(max_points, 100, 12, 0.01)
            ui_state["oct_benchmark"] = knn
            ui_state["brute_benchmark"] = brute
            ui_state["rad_benchmark"] = rad
            ui_state["benchmarks_done"] = True


def callback():
    io = psim.GetIO()
    handle_trees()
    handle_search()

    if psim.Button("Load Off-file"):
        load_off(path)

    if io.MouseClicked[0]:
        screen_coords = io.MousePos
        pick_result = ps.pick(screen_coords=screen_coords)
        if pick_result.is_hit and pick_result.structure_type_name == "Point Cloud":
            selection = pick_result.local_index
            pointcloud = ps.get_point_cloud("my points")
            ui_state["mouse_selection"] = selection
            ui_state["pointcloud"] = pointcloud
            #set_point_color(pointcloud, [selection], (0.0, 1.0, 0.0))

    if psim.BeginCombo("Datastructure", ui_state["current_data_structure"]):
        for structure in data_structures:
            selected = (structure == ui_state["current_data_structure"])
            clicked = psim.Selectable(structure, selected)

            if clicked:
                ui_state["current_data_structure"] = structure

            if selected:
                psim.SetItemDefaultFocus()

        psim.EndCombo()

    #depth_changed, ui_state["tree_depth"] = psim.SliderInt("Depth", ui_state["tree_depth"], v_min=1, v_max=20)
    #bucket_size_changed, ui_state["bucket_size"] = psim.SliderInt("Bucket Size", ui_state["bucket_size"], v_min=1, v_max=100)


    changed, ui_state["tree_visible"] = psim.Checkbox("Show Spatial Datastructure", ui_state["tree_visible"])
    if changed and ui_state["tree_visible"] and ui_state["octree"] != None:
        ui_state["tree_visualization"] = vs.draw_box_lines(ui_state["octree"].all_nodes)
    
    if changed and not ui_state["tree_visible"] and ui_state["tree_visualization"] != None:
        for i in ui_state["tree_visualization"]:
            i.remove()

    if psim.BeginCombo("Search Method", ui_state["current_search_method"]):
        for state in search_states:
            selected = (state == ui_state["current_search_method"])
            clicked = psim.Selectable(state, selected)

            if clicked:
                ui_state["current_search_method"] = state

            if selected:
                psim.SetItemDefaultFocus()

        psim.EndCombo()

    if psim.BeginCombo("Benchmark type", ui_state["current_benchmark_type"]):
        for type in benchmark_types:
            selected = (type == ui_state["current_benchmark_type"])
            clicked = psim.Selectable(type, selected)

            if clicked:
                ui_state["current_benchmark_type"] = type

            if selected:
                psim.SetItemDefaultFocus()

        psim.EndCombo()

    if psim.Button("Run Benchmark"):
        #oc_runtime, brute_runtime =  #np.random.random()
        #oct, rad, brute = bm.perform_size_benchmark(sizes, 100, 12, 0.01)
        #ui_state["oct_benchmark"] = oct
        #ui_state["brute_benchmark"] = brute
        #ui_state["rad_benchmark"] = rad
        #ui_state["benchmarks_done"] = True
        handle_benchmarks()

    if ui_state["benchmarks_done"]:
        text = ""
        match ui_state["current_benchmark_type"]:
            case "cloudsize":
                text = f'{sizes} points'
            case "depth":
                text = f'{depths} levels'
            case "bucket_size":
                text = f'{max_points} bucket size'
        oct = ui_state["oct_benchmark"]
        plot_oc = np.array(oct, dtype=np.float32)
        psim.TextUnformatted(f"knn Benchmark")
        psim.PlotLines("##knn Runtime", plot_oc, scale_min=0.0, scale_max=0.01, graph_size=(400,200))
        psim.TextUnformatted(text)

        rad = ui_state["rad_benchmark"]
        plot_rad = np.array(rad, dtype=np.float32)
        psim.TextUnformatted(f"Radius Benchmark")
        psim.PlotLines("##Radius Runtime", plot_rad, scale_min=0.0, scale_max=0.01, graph_size=(400,200))
        psim.TextUnformatted(text)

        brute = ui_state["brute_benchmark"]
        plot_brute = np.array(brute, dtype=np.float32)
        psim.TextUnformatted("Bruteforce Benchmark")
        psim.PlotLines("##Bruteforce Runtime", plot_brute, scale_min=0.0, scale_max=0.01, graph_size=(400,200))
        psim.TextUnformatted(text)

        if psim.Button("Get detailed Plot"):
            match ui_state["current_benchmark_type"]:
                case "cloudsize":
                    bm.plot_size_benchmark(ui_state["oct_benchmark"], ui_state["rad_benchmark"], ui_state["brute_benchmark"])
                case "depth":
                    bm.plot_depth_benchmark(ui_state["oct_benchmark"], ui_state["rad_benchmark"], ui_state["brute_benchmark"])
                case "bucket_size":
                    bm.plot_bucket_benchmark(ui_state["oct_benchmark"], ui_state["rad_benchmark"], ui_state["brute_benchmark"])

    
ps.init()
ps.set_user_callback(callback)
ps.show()