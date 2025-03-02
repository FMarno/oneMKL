/*******************************************************************************
* Copyright 2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions
* and limitations under the License.
*
*
* SPDX-License-Identifier: Apache-2.0
*******************************************************************************/

#include <algorithm>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/mkl.hpp"
#include "oneapi/mkl/detail/config.hpp"
#include "sparse_reference.hpp"
#include "test_common.hpp"
#include "test_helper.hpp"

#include <gtest/gtest.h>

extern std::vector<sycl::device *> devices;

namespace {

template <typename fpType, typename intType>
int test(sycl::device *dev, intType nrows_A, intType ncols_A, intType ncols_C,
         double density_A_matrix, oneapi::mkl::index_base index,
         oneapi::mkl::layout dense_matrix_layout, oneapi::mkl::transpose transpose_A,
         oneapi::mkl::transpose transpose_B, fpType alpha, fpType beta, intType ldb, intType ldc,
         bool opt_1_input, bool opt_2_inputs) {
    sycl::queue main_queue(*dev, exception_handler_t());

    intType int_index = (index == oneapi::mkl::index_base::zero) ? 0 : 1;
    std::size_t opa_nrows = static_cast<std::size_t>(
        transpose_A == oneapi::mkl::transpose::nontrans ? nrows_A : ncols_A);
    std::size_t opa_ncols = static_cast<std::size_t>(
        transpose_A == oneapi::mkl::transpose::nontrans ? ncols_A : nrows_A);

    // Input matrix in CSR format
    std::vector<intType> ia_host, ja_host;
    std::vector<fpType> a_host;
    intType nnz = generate_random_matrix<fpType, intType>(nrows_A, ncols_A, density_A_matrix,
                                                          int_index, ia_host, ja_host, a_host);

    // Input and output dense vectors
    std::vector<fpType> b_host, c_host;
    rand_matrix(b_host, dense_matrix_layout, opa_ncols, static_cast<std::size_t>(ncols_C),
                static_cast<std::size_t>(ldb));
    rand_matrix(c_host, dense_matrix_layout, opa_nrows, static_cast<std::size_t>(ncols_C),
                static_cast<std::size_t>(ldc));
    std::vector<fpType> c_ref_host(c_host);

    // Shuffle ordering of column indices/values to test sortedness
    shuffle_data(ia_host.data(), ja_host.data(), a_host.data(), static_cast<std::size_t>(nrows_A));

    auto ia_usm_uptr = malloc_device_uptr<intType>(main_queue, ia_host.size());
    auto ja_usm_uptr = malloc_device_uptr<intType>(main_queue, ja_host.size());
    auto a_usm_uptr = malloc_device_uptr<fpType>(main_queue, a_host.size());
    auto b_usm_uptr = malloc_device_uptr<fpType>(main_queue, b_host.size());
    auto c_usm_uptr = malloc_device_uptr<fpType>(main_queue, c_host.size());

    intType *ia_usm = ia_usm_uptr.get();
    intType *ja_usm = ja_usm_uptr.get();
    fpType *a_usm = a_usm_uptr.get();
    fpType *b_usm = b_usm_uptr.get();
    fpType *c_usm = c_usm_uptr.get();

    std::vector<sycl::event> mat_dependencies;
    std::vector<sycl::event> gemm_dependencies;
    // Copy host to device
    mat_dependencies.push_back(
        main_queue.memcpy(ia_usm, ia_host.data(), ia_host.size() * sizeof(intType)));
    mat_dependencies.push_back(
        main_queue.memcpy(ja_usm, ja_host.data(), ja_host.size() * sizeof(intType)));
    mat_dependencies.push_back(
        main_queue.memcpy(a_usm, a_host.data(), a_host.size() * sizeof(fpType)));
    gemm_dependencies.push_back(
        main_queue.memcpy(b_usm, b_host.data(), b_host.size() * sizeof(fpType)));
    gemm_dependencies.push_back(
        main_queue.memcpy(c_usm, c_host.data(), c_host.size() * sizeof(fpType)));

    sycl::event ev_copy, ev_release;
    oneapi::mkl::sparse::matrix_handle_t handle = nullptr;
    try {
        sycl::event event;
        CALL_RT_OR_CT(oneapi::mkl::sparse::init_matrix_handle, main_queue, &handle);

        CALL_RT_OR_CT(event = oneapi::mkl::sparse::set_csr_data, main_queue, handle, nrows_A,
                      ncols_A, nnz, index, ia_usm, ja_usm, a_usm, mat_dependencies);

        if (opt_1_input) {
            CALL_RT_OR_CT(event = oneapi::mkl::sparse::optimize_gemm, main_queue, transpose_A,
                          handle, { event });
        }

        if (opt_2_inputs) {
            CALL_RT_OR_CT(event = oneapi::mkl::sparse::optimize_gemm, main_queue, transpose_A,
                          transpose_B, dense_matrix_layout, static_cast<std::int64_t>(ncols_C),
                          handle, { event });
        }

        gemm_dependencies.push_back(event);
        CALL_RT_OR_CT(event = oneapi::mkl::sparse::gemm, main_queue, dense_matrix_layout,
                      transpose_A, transpose_B, alpha, handle, b_usm, ncols_C, ldb, beta, c_usm,
                      ldc, gemm_dependencies);

        CALL_RT_OR_CT(ev_release = oneapi::mkl::sparse::release_matrix_handle, main_queue, &handle,
                      { event });

        ev_copy = main_queue.memcpy(c_host.data(), c_usm, c_host.size() * sizeof(fpType), event);
    }
    catch (const sycl::exception &e) {
        std::cout << "Caught synchronous SYCL exception during sparse GEMV:\n"
                  << e.what() << std::endl;
        print_error_code(e);
        return 0;
    }
    catch (const oneapi::mkl::unimplemented &e) {
        wait_and_free(main_queue, &handle);
        return test_skipped;
    }
    catch (const std::runtime_error &error) {
        std::cout << "Error raised during execution of sparse GEMV:\n" << error.what() << std::endl;
        return 0;
    }

    // Compute reference.
    prepare_reference_gemm_data(ia_host.data(), ja_host.data(), a_host.data(), nrows_A, ncols_A,
                                ncols_C, nnz, int_index, dense_matrix_layout, transpose_A,
                                transpose_B, alpha, beta, ldb, ldc, b_host.data(),
                                c_ref_host.data());

    // Compare the results of reference implementation and DPC++ implementation.
    ev_copy.wait_and_throw();
    bool valid = check_equal_vector(c_host, c_ref_host);

    ev_release.wait_and_throw();
    return static_cast<int>(valid);
}

class SparseGemmUsmTests : public ::testing::TestWithParam<sycl::device *> {};

/**
 * Helper function to run tests in different configuration.
 * 
 * @tparam fpType Complex or scalar, single or double precision type
 * @param dev Device to test
 * @param transpose_A Transpose value for the A matrix
 * @param transpose_B Transpose value for the B matrix
 */
template <typename fpType>
void test_helper(sycl::device *dev, oneapi::mkl::transpose transpose_A,
                 oneapi::mkl::transpose transpose_B) {
    double density_A_matrix = 0.8;
    fpType fp_zero = set_fp_value<fpType>()(0.f, 0.f);
    fpType fp_one = set_fp_value<fpType>()(1.f, 0.f);
    oneapi::mkl::index_base index_zero = oneapi::mkl::index_base::zero;
    oneapi::mkl::layout col_major = oneapi::mkl::layout::col_major;
    int nrows_A = 4, ncols_A = 6, ncols_C = 5;
    int ldb = transpose_A == oneapi::mkl::transpose::nontrans ? ncols_A : nrows_A;
    int ldc = transpose_A == oneapi::mkl::transpose::nontrans ? nrows_A : ncols_A;
    bool no_opt_1_input = false;
    bool opt_2_inputs = true;

    // Basic test
    EXPECT_TRUEORSKIP(test(dev, nrows_A, ncols_A, ncols_C, density_A_matrix, index_zero, col_major,
                           transpose_A, transpose_B, fp_one, fp_zero, ldb, ldc, no_opt_1_input,
                           opt_2_inputs));
    // Test index_base 1
    EXPECT_TRUEORSKIP(test(dev, nrows_A, ncols_A, ncols_C, density_A_matrix,
                           oneapi::mkl::index_base::one, col_major, transpose_A, transpose_B,
                           fp_one, fp_zero, ldb, ldc, no_opt_1_input, opt_2_inputs));
    // Test non-default alpha
    EXPECT_TRUEORSKIP(test(dev, nrows_A, ncols_A, ncols_C, density_A_matrix, index_zero, col_major,
                           transpose_A, transpose_B, set_fp_value<fpType>()(2.f, 1.5f), fp_zero,
                           ldb, ldc, no_opt_1_input, opt_2_inputs));
    // Test non-default beta
    EXPECT_TRUEORSKIP(test(dev, nrows_A, ncols_A, ncols_C, density_A_matrix, index_zero, col_major,
                           transpose_A, transpose_B, fp_one, set_fp_value<fpType>()(3.2f, 1.f), ldb,
                           ldc, no_opt_1_input, opt_2_inputs));
    // Test 0 alpha
    EXPECT_TRUEORSKIP(test(dev, nrows_A, ncols_A, ncols_C, density_A_matrix, index_zero, col_major,
                           transpose_A, transpose_B, fp_zero, fp_one, ldb, ldc, no_opt_1_input,
                           opt_2_inputs));
    // Test 0 alpha and beta
    EXPECT_TRUEORSKIP(test(dev, nrows_A, ncols_A, ncols_C, density_A_matrix, index_zero, col_major,
                           transpose_A, transpose_B, fp_zero, fp_zero, ldb, ldc, no_opt_1_input,
                           opt_2_inputs));
    // Test non-default ldb
    EXPECT_TRUEORSKIP(test(dev, nrows_A, ncols_A, ncols_C, density_A_matrix, index_zero, col_major,
                           transpose_A, transpose_B, fp_one, fp_zero, ldb + 5, ldc, no_opt_1_input,
                           opt_2_inputs));
    // Test non-default ldc
    EXPECT_TRUEORSKIP(test(dev, nrows_A, ncols_A, ncols_C, density_A_matrix, index_zero, col_major,
                           transpose_A, transpose_B, fp_one, fp_zero, ldb, ldc + 6, no_opt_1_input,
                           opt_2_inputs));
    // Test row major layout
    EXPECT_TRUEORSKIP(test(dev, nrows_A, ncols_A, ncols_C, density_A_matrix, index_zero,
                           oneapi::mkl::layout::row_major, transpose_A, transpose_B, fp_one,
                           fp_zero, ncols_C, ncols_C, no_opt_1_input, opt_2_inputs));
    // Test int64 indices
    long long_nrows_A = 27, long_ncols_A = 13, long_ncols_C = 6;
    long long_ldb = transpose_A == oneapi::mkl::transpose::nontrans ? long_ncols_A : long_nrows_A;
    long long_ldc = transpose_A == oneapi::mkl::transpose::nontrans ? long_nrows_A : long_ncols_A;
    EXPECT_TRUEORSKIP(test(dev, long_nrows_A, long_ncols_A, long_ncols_C, density_A_matrix,
                           index_zero, col_major, transpose_A, transpose_B, fp_one, fp_zero,
                           long_ldb, long_ldc, no_opt_1_input, opt_2_inputs));
    // Use optimize_gemm with only the sparse gemm input
    EXPECT_TRUEORSKIP(test(dev, nrows_A, ncols_A, ncols_C, density_A_matrix, index_zero, col_major,
                           transpose_A, transpose_B, fp_one, fp_zero, ldb, ldc, true, false));
    // Use the 2 optimize_gemm versions
    EXPECT_TRUEORSKIP(test(dev, nrows_A, ncols_A, ncols_C, density_A_matrix, index_zero, col_major,
                           transpose_A, transpose_B, fp_one, fp_zero, ldb, ldc, true, true));
    // Do not use optimize_gemm
    EXPECT_TRUEORSKIP(test(dev, nrows_A, ncols_A, ncols_C, density_A_matrix, index_zero, col_major,
                           transpose_A, transpose_B, fp_one, fp_zero, ldb, ldc, false, false));
}

/**
 * Helper function to test combination of transpose vals.
 * Only test \p conjtrans if \p fpType is complex.
 * 
 * @tparam fpType Complex or scalar, single or double precision type
 * @param dev Device to test
 */
template <typename fpType>
void test_helper_transpose(sycl::device *dev) {
    std::vector<oneapi::mkl::transpose> transpose_vals{ oneapi::mkl::transpose::nontrans,
                                                        oneapi::mkl::transpose::trans };
    if (complex_info<fpType>::is_complex) {
        transpose_vals.push_back(oneapi::mkl::transpose::conjtrans);
    }
    for (auto transpose_A : transpose_vals) {
        for (auto transpose_B : transpose_vals) {
            test_helper<fpType>(dev, transpose_A, transpose_B);
        }
    }
}

TEST_P(SparseGemmUsmTests, RealSinglePrecision) {
    using fpType = float;
    test_helper_transpose<fpType>(GetParam());
}

TEST_P(SparseGemmUsmTests, RealDoublePrecision) {
    using fpType = double;
    CHECK_DOUBLE_ON_DEVICE(GetParam());
    test_helper_transpose<fpType>(GetParam());
}

TEST_P(SparseGemmUsmTests, ComplexSinglePrecision) {
    using fpType = std::complex<float>;
    test_helper_transpose<fpType>(GetParam());
}

TEST_P(SparseGemmUsmTests, ComplexDoublePrecision) {
    using fpType = std::complex<double>;
    CHECK_DOUBLE_ON_DEVICE(GetParam());
    test_helper_transpose<fpType>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(SparseGemmUsmTestSuite, SparseGemmUsmTests, testing::ValuesIn(devices),
                         ::DeviceNamePrint());

} // anonymous namespace
