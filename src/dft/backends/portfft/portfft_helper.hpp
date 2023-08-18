/*******************************************************************************
* Copyright Codeplay Software Ltd.
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

#include <type_traits>

#include "oneapi/mkl/dft/detail/commit_impl.hpp"
#include "oneapi/mkl/dft/detail/descriptor_impl.hpp"
#include <portfft.hpp>

namespace pfft = portfft;

namespace oneapi::mkl::dft::portfft::detail {
template <dft::precision prec, dft::domain dom>
inline dft::detail::commit_impl<prec, dom> *checked_get_commit(
    dft::detail::descriptor<prec, dom> &desc) {
    auto commit_handle = dft::detail::get_commit(desc);
    if (commit_handle == nullptr || commit_handle->get_backend() != backend::portfft) {
        throw mkl::invalid_argument("dft/backends/portfft", "get_commit",
                                    "DFT descriptor has not been commited for portFFT");
    }
    return commit_handle;
}

template <typename descriptor_type>
constexpr pfft::domain to_pfft_domain() {
    if constexpr (std::is_floating_point_v<fwd<descriptor_type>>) {
        return pfft::domain::REAL;
    }
    else {
        return pfft::domain::COMPLEX;
    }
}

template <typename descriptor_type>
auto get_descriptors(descriptor_type &desc) {
    constexpr auto domain = detail::to_pfft_domain<descriptor_type>();
    auto commit = detail::checked_get_commit(desc);
    return reinterpret_cast<pfft::committed_descriptor<scalar<descriptor_type>, domain> *>(
        commit->get_handle());
}
} // namespace oneapi::mkl::dft::portfft::detail
