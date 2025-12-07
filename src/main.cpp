#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace py = pybind11;

using Matrix = std::vector<std::vector<double>>;

Matrix create_identity(size_t n) {
    Matrix I(n, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i) {
        I[i][i] = 1.0;
    }
    return I;
}

Matrix gauss_jordan_inverse(const Matrix& input) {
    size_t n = input.size();
    Matrix A = input;
    Matrix I = create_identity(n);

    for (size_t i = 0; i < n; ++i) {
        size_t pivot_row = i;
        double max_val = std::abs(A[i][i]);

        for (size_t k = i + 1; k < n; ++k) {
            if (std::abs(A[k][i]) > max_val) {
                max_val = std::abs(A[k][i]);
                pivot_row = k;
            }
        }

        if (max_val < 1e-12) {
            throw std::runtime_error("Matrix is singular, cannot invert.");
        }

        if (pivot_row != i) {
            std::swap(A[i], A[pivot_row]);
            std::swap(I[i], I[pivot_row]);
        }

        double pivot = A[i][i];
        for (size_t j = 0; j < n; ++j) {
            A[i][j] /= pivot;
            I[i][j] /= pivot;
        }

        for (size_t k = 0; k < n; ++k) {
            if (k != i) {
                double factor = A[k][i];
                for (size_t j = 0; j < n; ++j) {
                    A[k][j] -= factor * A[i][j];
                    I[k][j] -= factor * I[i][j];
                }
            }
        }
    }

    return I;
}

PYBIND11_MODULE(mygauss, m) {
    m.def("inverse", &gauss_jordan_inverse);
}