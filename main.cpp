#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "args/args.hxx"
#include "polyscope/combining_hash_functions.h"
#include "polyscope/file_helpers.h"
#include "polyscope/messages.h"
#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include "portable-file-dialogs.h"

#include <fstream>
#include <algorithm>
#include <queue>

using Point = std::array<float, 3>;
using Normal = std::array<float, 3>;
using Face = std::array<unsigned int, 3>;

static int selectedIdx = 0;
static float radius = 0.05f;
static std::vector<glm::vec3> colors;

static int k = 5;

void readOff(const std::string& filename, std::vector<Point>& points, std::vector<Normal>& normals) {

    std::vector<Face> faces;

    points.clear();
    normals.clear();
    faces.clear();

    std::default_random_engine engine(std::random_device{}());
    std::uniform_real_distribution<float> distribution(-1.f, 1.f);

    std::ifstream file(filename);
    std::string header;
    file >> header;

    if (header == "OFF") {
        unsigned int numVertices, numFaces, numEdges;
        
        file >> numVertices >> numFaces >> numEdges;
        std::cout << "vertices: " << numVertices << std::endl;
        std::cout << "faces: " << numFaces << std::endl;
        std::cout << "edges: " << numEdges << std::endl;

        points.resize(numVertices);
        for (unsigned int i = 0; i < numVertices; ++i) {
            Point p;
            file >> p[0] >> p[1] >> p[2];
            points[i] = p;
        }

        for (unsigned int i = 0; i < numFaces; ++i) {
            unsigned int numVerticesInFace;
            file >> numVerticesInFace;

            Face face;
            for (unsigned int j = 0; j < numVerticesInFace; ++j) {
                file >> face[j];
            }      
            faces.push_back(face);
        }
    }
}

void readOff(const std::string& filename, std::vector<Point>& points) {
    std::vector<Normal> dummy_normals;
    readOff(filename, points, dummy_normals);  // reuse the full version
}

struct EuclideanDistance {
    static float measure(Point const& p1, Point const& p2) {
        float dx = p1[0] - p2[0];
        float dy = p1[1] - p2[1];
        float dz = p1[2] - p2[2];
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }
};

/*
 * This is not yet a spatial data structure :)
 */
class SpatialDataStructure {
public:
    using Neighbor = std::pair<float, std::size_t>;

    struct KDTreeNode {
        Point point;
        unsigned int index;

        KDTreeNode* left = nullptr;
        KDTreeNode* right = nullptr;

        int axis; // 0 = x, 1 = y, 2 = z
    };

    SpatialDataStructure(std::vector<Point> const& points) : m_points(points) {
        std::vector<std::pair<Point, std::size_t>> pointIndexPairs;
        for (std::size_t i = 0; i < points.size(); ++i) {
            pointIndexPairs.push_back({points[i], i});
        }
        root = buildKDTree(pointIndexPairs, 0);
    }

    virtual ~SpatialDataStructure() = default;

    std::vector<Point> const& getPoints() const { 
        return m_points; 
    }

    virtual std::vector<std::size_t> collectInRadius(Point const& p, float radius) const {
        std::vector<std::size_t> result;

        std::vector<std::size_t> resultIndices;

        collectInRadiusRecursive(root, p, radius, resultIndices);
        
        return resultIndices;
    }

    void collectInRadiusRecursive(KDTreeNode* node, Point const& p, float radius, std::vector<std::size_t>& resultIndices) const {
        if (node == nullptr) {
            return;
        }

        float distance = EuclideanDistance::measure(p, node->point);

        if (distance <= radius) {
            resultIndices.push_back(node->index);
        }

        KDTreeNode* closerChild ;
        KDTreeNode* otherChild;

        float axisDistance = p[node->axis] - node->point[node->axis];

        if (axisDistance < 0){
            closerChild = node->left;
            otherChild = node->right;
        } else {
            closerChild = node->right;
            otherChild = node->left;
        }

        collectInRadiusRecursive(closerChild, p, radius, resultIndices);

        if (std::abs(axisDistance) <= radius) {
            collectInRadiusRecursive(otherChild, p, radius, resultIndices);
        }
    }

    std::vector<std::size_t> bruteForceRadiusSearch(Point const& p, float radius) const {
        std::vector<std::size_t> result;
        for (std::size_t i = 0; i < m_points.size(); ++i) {
            float distance = EuclideanDistance::measure(p, m_points[i]);
            if (distance <= radius) {
                result.push_back(i);
            }
        }

        return result;
    }

    virtual std::vector<std::size_t> collectKNearest(Point const& p, unsigned int k) const {
        std::priority_queue<Neighbor> heap; 

        collectKNearestRecursive(root, p, k, heap);

        std::vector<std::size_t> resultIndices;
        while (!heap.empty()) {
            resultIndices.push_back(heap.top().second);
            heap.pop();
        }

        return resultIndices;
    }

    void collectKNearestRecursive(KDTreeNode* node, Point const& p, unsigned int k, std::priority_queue<Neighbor>& heap) const {
        if (node == nullptr) {
            return;
        }

        float distance = EuclideanDistance::measure(p, node->point);

        if (heap.size() < k) {
            heap.push({distance, node->index});
        } else if (distance < heap.top().first) {
            heap.pop();
            heap.push({distance, node->index});
        }

        float distAxis = p[node->axis] - node->point[node->axis];

        KDTreeNode* closerChild;
        KDTreeNode* otherChild;

        if (distAxis < 0) {
            closerChild = node->left;
            otherChild = node->right;
        } else {
            closerChild = node->right;
            otherChild = node->left;
        }

        collectKNearestRecursive(closerChild, p, k, heap);

        if (heap.size() < k || std::abs(distAxis) < heap.top().first) {
            collectKNearestRecursive(otherChild, p, k, heap);
        }
    }

private:
    std::vector<Point> m_points;
    KDTreeNode* root = nullptr;

    KDTreeNode* buildKDTree(std::vector<std::pair<Point, std::size_t>> points, int depth) {
        if (points.empty()) {
            return nullptr;
        }

        int axis = depth % 3;

        std::size_t median = points.size() / 2;

        std::nth_element(points.begin(), points.begin() + median, points.end(), [axis](const std::pair<Point, std::size_t>& a, const std::pair<Point, std::size_t>& b){
            return a.first[axis] < b.first[axis];
        }); //if i got it right this is like quicksort so O(n) on average

        KDTreeNode* node = new KDTreeNode();
        node->point = points[median].first;
        node->index = points[median].second;
        node->axis = axis;

        std::vector<std::pair<Point, std::size_t>> leftPoints(points.begin(), points.begin() + median);
        std::vector<std::pair<Point, std::size_t>> rightPoints(points.begin() + median + 1, points.end());

        node->left = buildKDTree(leftPoints, depth + 1);
        node->right = buildKDTree(rightPoints, depth + 1);

        return node;
    }
};

// Application variables
polyscope::PointCloud* pc = nullptr;
std::unique_ptr<SpatialDataStructure> sds;

void callback() {
    if (ImGui::Button("Load Off")) {
        auto paths =
            pfd::open_file("Load Off", "", std::vector<std::string>{"point data (*.off)", "*.off"}, pfd::opt::none)
                .result();
        if (!paths.empty()) {
            std::filesystem::path path(paths[0]);

            if (path.extension() == ".off") {
                // Read the point cloud
                std::vector<Point> points;
                readOff(path.string(), points);

                // Create the polyscope geometry
                pc = polyscope::registerPointCloud("Points", points);

                // Build spatial data structure
                sds = std::make_unique<SpatialDataStructure>(points);

                // Initialize colors
                colors.resize(points.size(), {28.f/255.f, 99.f/255.f, 227.f/255.f}); // ich will diese farbe: #1C63E3
            }
        }
    }
    if (pc != nullptr) {

        ImGui::InputInt("Point Index", &selectedIdx);
        ImGui::InputFloat("Radius", &radius);
        selectedIdx = std::clamp(selectedIdx, 0, (int)colors.size() - 1);

        pc->addColorQuantity("Colors", colors)->setEnabled(true);

        if (ImGui::Button("Collect in Radius")) {

            if (sds && !sds->getPoints().empty()) {

                const auto& points = sds->getPoints();

                Point pivot = points[selectedIdx];

                auto resultIndices = sds->collectInRadius(pivot, radius);

                printf("Found %zu neighbors within radius %.2f of point %d\n",
                    resultIndices.size(), radius, selectedIdx);

                std::fill(colors.begin(), colors.end(), glm::vec3(28.f/255.f, 99.f/255.f, 227.f/255.f));

                colors[selectedIdx] = glm::vec3(1.f, 0.f, 0.f);

                for (std::size_t idx : resultIndices) {
                    colors[idx] = glm::vec3(0.f, 1.f, 0.f);
                }

                pc->addColorQuantity("Colors", colors)->setEnabled(true);
            }
        }

        ImGui::InputInt("K-nearest", &k);
        if (ImGui::Button("Collect K-Nearest")) {
            if (sds && !sds->getPoints().empty()) {

                const auto& points = sds->getPoints();

                Point pivot = points[selectedIdx];

                auto resultIndices = sds->collectKNearest(pivot, k);

                printf("Found %zu nearest neighbors of point %d\n", resultIndices.size(), selectedIdx);

                std::fill(colors.begin(), colors.end(), glm::vec3(28.f/255.f, 99.f/255.f, 227.f/255.f));

                colors[selectedIdx] = glm::vec3(1.f, 0.f, 0.f);

                for (std::size_t idx : resultIndices) {
                    colors[idx] = glm::vec3(0.f, 1.f, 0.f);
                }
                pc->addColorQuantity("Colors", colors)->setEnabled(true);
            }
        }
    }
}

int main(int argc, char** argv) {
  // Configure the argument parser
  args::ArgumentParser parser("Computer Graphics 2 Sample Code.");

  // Parse args
  try {
    parser.ParseCLI(argc, argv);
  } catch (const args::Help&) {
    std::cout << parser;
    return 0;
  } catch (const args::ParseError& e) {
    std::cerr << e.what() << std::endl;

    std::cerr << parser;
    return 1;
  }

  // Options
  polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::ShadowOnly;
  polyscope::options::shadowBlurIters = 6;

  // Initialize polyscope
  polyscope::init();

  // Add a few gui elements
  polyscope::state::userCallback = callback;

  // Show the gui
  polyscope::show();

  return 0;
}
