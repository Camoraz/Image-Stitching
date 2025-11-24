#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>

namespace py = pybind11;

// Compute Euclidean distance between two descriptor rows
double euclidean_distance(const py::array_t<float> &d1,
                          const py::array_t<float> &d2,
                          int idx1, int idx2)
{
    auto r1 = d1.unchecked<2>();
    auto r2 = d2.unchecked<2>();

    int length = (int)r1.shape(1);
    double dist = 0.0;

    for (int i = 0; i < length; i++) {
        double diff = r1(idx1, i) - r2(idx2, i);
        dist += diff * diff;
    }

    return std::sqrt(dist);
}

// Basic brute-force k-NN matcher
std::vector<std::vector<std::tuple<int, int, double>>> knn_match(
        py::array_t<float> des1,
        py::array_t<float> des2,
        int k = 2)
{
    int n1 = (int)des1.shape(0);
    int n2 = (int)des2.shape(0);

    std::vector<std::vector<std::tuple<int, int, double>>> matches;
    matches.reserve(n1);

    for (int i = 0; i < n1; i++) {
        std::vector<std::tuple<int, int, double>> dists;
        dists.reserve(n2);

        // Compute distances from descriptor i to all descriptors in des2
        for (int j = 0; j < n2; j++) {
            double dist = euclidean_distance(des1, des2, i, j);
            dists.push_back(std::make_tuple(i, j, dist));
        }

        // Sort by distance
        std::sort(dists.begin(), dists.end(),
                  [](const std::tuple<int, int, double> &a,
                     const std::tuple<int, int, double> &b)
                  {
                      return std::get<2>(a) < std::get<2>(b);
                  });

        // Take the top k matches
        int num = std::min(k, (int)dists.size());
        std::vector<std::tuple<int, int, double>> topk(dists.begin(),
                                                       dists.begin() + num);

        matches.push_back(topk);
    }

    return matches;
}

// Lowe's ratio test
std::vector<std::tuple<int, int, double>> ratio_test(
        const std::vector<std::vector<std::tuple<int, int, double>>> &matches,
        double ratio = 0.75)
{
    std::vector<std::tuple<int, int, double>> good_matches;

    for (const auto &m_n : matches) {
        if (m_n.size() < 2)
            continue;

        const auto &m = m_n[0];
        const auto &n = m_n[1];

        if (std::get<2>(m) < ratio * std::get<2>(n)) {
            good_matches.push_back(m);
        }
    }

    return good_matches;
}

PYBIND11_MODULE(bfmatcher_cpp, m) {
    m.def("knn_match", &knn_match, py::arg("des1"), py::arg("des2"), py::arg("k") = 2);
    m.def("ratio_test", &ratio_test, py::arg("matches"), py::arg("ratio") = 0.75);
}
