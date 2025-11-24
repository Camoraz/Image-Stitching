#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>

namespace py = pybind11;

// Euclidean distance between two descriptor vectors
double euclidean_distance(const py::array_t<float>& d1, const py::array_t<float>& d2, ssize_t idx1, ssize_t idx2) {
    auto r1 = d1.unchecked<2>();
    auto r2 = d2.unchecked<2>();
    ssize_t length = r1.shape(1);
    double dist = 0.0;
    for (ssize_t i = 0; i < length; ++i) {
        double diff = r1(idx1, i) - r2(idx2, i);
        dist += diff * diff;
    }
    return std::sqrt(dist);
}

// k-NN matching (k=2)
std::vector<std::vector<std::tuple<ssize_t, ssize_t, double>>> knn_match(
        py::array_t<float> des1,
        py::array_t<float> des2,
        ssize_t k = 2) {

    auto n1 = des1.shape(0);
    auto n2 = des2.shape(0);

    std::vector<std::vector<std::tuple<ssize_t, ssize_t, double>>> matches;

    for (ssize_t i = 0; i < n1; ++i) {
        // Compute distances to all descriptors in des2
        std::vector<std::tuple<ssize_t, ssize_t, double>> dists; // (queryIdx, trainIdx, distance)
        for (ssize_t j = 0; j < n2; ++j) {
            double dist = euclidean_distance(des1, des2, i, j);
            dists.emplace_back(i, j, dist);
        }
        // Sort distances and keep top k
        std::sort(dists.begin(), dists.end(), [](auto &a, auto &b) { return std::get<2>(a) < std::get<2>(b); });
        std::vector<std::tuple<ssize_t, ssize_t, double>> topk(dists.begin(), dists.begin() + std::min(k, (ssize_t)dists.size()));
        matches.push_back(topk);
    }
    return matches;
}

// Ratio test
std::vector<std::tuple<ssize_t, ssize_t, double>> ratio_test(
        const std::vector<std::vector<std::tuple<ssize_t, ssize_t, double>>>& matches,
        double ratio = 0.75) {

    std::vector<std::tuple<ssize_t, ssize_t, double>> good_matches;

    for (auto &m_n : matches) {
        if (m_n.size() < 2) continue;
        auto m = m_n[0];
        auto n = m_n[1];
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
