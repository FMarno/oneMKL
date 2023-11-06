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
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/mkl/exceptions.hpp"

#include "oneapi/mkl/dft/detail/commit_impl.hpp"
#include "oneapi/mkl/dft/detail/portfft/onemkl_dft_portfft.hpp"
#include "oneapi/mkl/dft/types.hpp"

#include "portfft_helper.hpp"

#include <portfft/portfft.hpp>

namespace oneapi::mkl::dft::portfft {

// BUFFER version

//In-place transform
template <typename descriptor_type>
ONEMKL_EXPORT void compute_forward(descriptor_type &desc,
                                   sycl::buffer<fwd<descriptor_type>, 1> &inout) {
    constexpr auto domain = detail::to_pfft_domain<descriptor_type>();

    if constexpr (domain == pfft::domain::COMPLEX) {
        detail::get_descriptors(desc)[0]->compute_forward(inout);
    }
}

//In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type>
ONEMKL_EXPORT void compute_forward(descriptor_type &, sycl::buffer<scalar<descriptor_type>, 1> &,
                                   sycl::buffer<scalar<descriptor_type>, 1> &) {
    throw oneapi::mkl::unimplemented("DFT", "compute_forward(desc, inout_re, inout_im)",
                                     "portFFT does not support real-real complex storage.");
}

//Out-of-place transform
template <typename descriptor_type>
ONEMKL_EXPORT void compute_forward(descriptor_type &desc, sycl::buffer<fwd<descriptor_type>, 1> &in,
                                   sycl::buffer<bwd<descriptor_type>, 1> &out) {
    constexpr auto domain = detail::to_pfft_domain<descriptor_type>();

    if constexpr (domain == pfft::domain::COMPLEX) {
        detail::get_descriptors(desc)[0]->compute_forward(in, out);
    }
}

//Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type>
ONEMKL_EXPORT void compute_forward(descriptor_type &, sycl::buffer<scalar<descriptor_type>, 1> &,
                                   sycl::buffer<scalar<descriptor_type>, 1> &,
                                   sycl::buffer<scalar<descriptor_type>, 1> &,
                                   sycl::buffer<scalar<descriptor_type>, 1> &) {
    throw oneapi::mkl::unimplemented("DFT", "compute_forward(desc, in_re, in_im, out_re, out_im)",
                                     "portFFT does not support real-real complex storage.");
}

//USM version

//In-place transform
template <typename descriptor_type>
ONEMKL_EXPORT sycl::event compute_forward(descriptor_type &desc, fwd<descriptor_type> *inout,
                                          const std::vector<sycl::event> &dependencies) {
    constexpr auto domain = detail::to_pfft_domain<descriptor_type>();

    if constexpr (domain == pfft::domain::COMPLEX) {
        return detail::get_descriptors(desc)[0]->compute_forward(inout);
    }
    else {
        return {};
    }
}

//In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type>
ONEMKL_EXPORT sycl::event compute_forward(descriptor_type &, scalar<descriptor_type> *,
                                          scalar<descriptor_type> *,
                                          const std::vector<sycl::event> &) {
    throw oneapi::mkl::unimplemented("DFT",
                                     "compute_forward(desc, inout_re, inout_im, dependencies)",
                                     "portFFT does not support real-real complex storage.");
}

//Out-of-place transform
template <typename descriptor_type>
ONEMKL_EXPORT sycl::event compute_forward(descriptor_type &desc, fwd<descriptor_type> *in,
                                          bwd<descriptor_type> *out,
                                          const std::vector<sycl::event> &dependencies) {
    constexpr auto domain = detail::to_pfft_domain<descriptor_type>();

    if constexpr (domain == pfft::domain::COMPLEX) {
        return detail::get_descriptors(desc)[0]->compute_forward(in, out);
    }
    else {
        return {};
    }
}

//Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type>
ONEMKL_EXPORT sycl::event compute_forward(descriptor_type &, scalar<descriptor_type> *,
                                          scalar<descriptor_type> *, scalar<descriptor_type> *,
                                          scalar<descriptor_type> *,
                                          const std::vector<sycl::event> &) {
    throw oneapi::mkl::unimplemented(
        "DFT", "compute_forward(desc, in_re, in_im, out_re, out_im, dependencies)",
        "portFFT does not support real-real complex storage.");
}

// Template function instantiations
#include "dft/backends/backend_forward_instantiations.cxx"

} // namespace oneapi::mkl::dft::portfft
