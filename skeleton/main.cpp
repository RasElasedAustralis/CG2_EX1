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

using Point = std::array<float, 3>;
using Normal = std::array<float, 3>;

void readOff(const std::string& filename, std::vector<Point>& points, std::vector<Normal>& normals) {

    points.clear();
    normals.clear();

    std::default_random_engine engine(std::random_device{}());
    std::uniform_real_distribution<float> distribution(-1.f, 1.f);

    for (unsigned int i = 0; i < 100; ++i) {
        points.emplace_back(Point{
            distribution(engine),
            distribution(engine),
            distribution(engine)
        });
    }

    polyscope::warning("Reading *.off files is not implemented. Generated a dummy point cloud.");
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
    SpatialDataStructure(std::vector<Point> const& points) : m_points(points) {}

    virtual ~SpatialDataStructure() = default;

    std::vector<Point> const& getPoints() const { return m_points; }

    virtual std::vector<std::size_t> collectInRadius(Point const& p, float radius) const {
        std::vector<std::size_t> result;

        // Dummy brute-force implementation
        // TODO: Use spatial data structure for sub-linear search
        for (std::size_t i = 0; i < m_points.size(); ++i) {
            float distance = EuclideanDistance::measure(p, m_points[i]);
            if (distance <= radius) result.push_back(i);
        }

        return result;
    }

    virtual std::vector<std::size_t> collectKNearest(Point const& p, unsigned int k) const {
        std::vector<std::size_t> result;

        // Bogus knn implementation, giving you the first k points!
        // TODO: Use spatial data structure for sub-linear search
        for (std::size_t i = 0; (i < k) && (i < m_points.size()); ++i) {
            result.push_back(i);
        }

        return result;
    }

   private:
    std::vector<Point> m_points;
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
            }
        }
    }

    // TODO: Implement radius search
    // TODO: Implement visualizations
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
