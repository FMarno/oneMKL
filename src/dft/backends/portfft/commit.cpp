/*******************************************************************************
* Copyright Codeplay Software Ltd
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

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include <array>
#include <algorithm>
#include <optional>

#include "oneapi/mkl/exceptions.hpp"

#include "oneapi/mkl/dft/detail/commit_impl.hpp"
#include "oneapi/mkl/dft/detail/descriptor_impl.hpp"
#include "oneapi/mkl/dft/detail/portfft/onemkl_dft_portfft.hpp"
#include "oneapi/mkl/dft/types.hpp"

#include <portfft.hpp>

// alias to avoid ambiguity
namespace pfft = portfft;

namespace oneapi::mkl::dft::portfft {
namespace detail {

/// Commit impl class specialization for cuFFT.
template <dft::precision prec, dft::domain dom>
class portfft_commit final : public dft::detail::commit_impl<prec, dom> {
private:
    using scalar = std::conditional_t<prec == dft::precision::SINGLE, float, double>;
    static constexpr pfft::domain domain =
        std::conditional_t<dom == dft::domain::REAL,
                           std::integral_constant<pfft::domain, pfft::domain::REAL>,
                           std::integral_constant<pfft::domain, pfft::domain::COMPLEX>>::value;
    using committed_desc = pfft::committed_descriptor<scalar, domain>;
    // since only complex-to-complex transforms are supported, we expect both directions to be valid or neither.
    std::array<std::unique_ptr<committed_desc>, 2> committed_descriptors;

public:
    portfft_commit(sycl::queue& queue, const dft::detail::dft_values<prec, dom>& config_values)
            : oneapi::mkl::dft::detail::commit_impl<prec, dom>(queue, backend::portfft) {
        if constexpr (prec == dft::detail::precision::DOUBLE) {
            if (!queue.get_device().has(sycl::aspect::fp64)) {
                throw mkl::exception("DFT", "commit", "Device does not support double precision.");
            }
        }
    }

    void commit(const dft::detail::dft_values<prec, dom>& config_values) override {
        // not available in portFFT
        // real_storage, conj_even_storage, and packed_format don't apply since portFFT only does complex-to-complex transforms.
        if (config_values.workspace != config_value::ALLOW) {
            throw mkl::unimplemented("dft/backends/portfft", __FUNCTION__,
                                     "portFFT only supports ALLOW for the WORKSPACE parameter");
        }
        if (config_values.ordering != config_value::ORDERED) {
            throw mkl::unimplemented("dft/backends/portfft", __FUNCTION__,
                                     "portFFT only supports ORDERED for the ORDERING parameter");
        }
        if (config_values.transpose) {
            throw mkl::unimplemented("dft/backends/portfft", __FUNCTION__,
                                     "portFFT does not supported transposed output");
        }

        // forward descriptor
        pfft::descriptor<scalar, domain> fwd_desc(
            { config_values.dimensions.cbegin(), config_values.dimensions.cend() });
        fwd_desc.forward_scale = config_values.fwd_scale;
        fwd_desc.backward_scale = config_values.bwd_scale;
        fwd_desc.number_of_transforms =
            static_cast<std::size_t>(config_values.number_of_transforms);
        fwd_desc.complex_storage = config_values.complex_storage == config_value::COMPLEX_COMPLEX
                                       ? pfft::complex_storage::COMPLEX
                                       : pfft::complex_storage::REAL_REAL;
        fwd_desc.placement = config_values.placement == config_value::INPLACE
                                 ? pfft::placement::IN_PLACE
                                 : pfft::placement::OUT_OF_PLACE;
        fwd_desc.forward_strides = { config_values.input_strides.cbegin(),
                                     config_values.input_strides.cend() };
        fwd_desc.backward_strides = { config_values.output_strides.cbegin(),
                                      config_values.output_strides.cend() };
        fwd_desc.forward_distance = static_cast<std::size_t>(config_values.fwd_dist);
        fwd_desc.backward_distance = static_cast<std::size_t>(config_values.bwd_dist);

        // backward descriptor
        pfft::descriptor<scalar, domain> bwd_desc(
            { config_values.dimensions.cbegin(), config_values.dimensions.cend() });
        bwd_desc.forward_scale = config_values.fwd_scale;
        bwd_desc.backward_scale = config_values.bwd_scale;
        bwd_desc.number_of_transforms =
            static_cast<std::size_t>(config_values.number_of_transforms);
        bwd_desc.complex_storage = config_values.complex_storage == config_value::COMPLEX_COMPLEX
                                       ? pfft::complex_storage::COMPLEX
                                       : pfft::complex_storage::REAL_REAL;
        bwd_desc.placement = config_values.placement == config_value::INPLACE
                                 ? pfft::placement::IN_PLACE
                                 : pfft::placement::OUT_OF_PLACE;
        bwd_desc.forward_strides = { config_values.output_strides.cbegin(),
                                     config_values.output_strides.cend() };
        bwd_desc.backward_strides = { config_values.input_strides.cbegin(),
                                      config_values.input_strides.cend() };
        bwd_desc.forward_distance = static_cast<std::size_t>(config_values.fwd_dist);
        bwd_desc.backward_distance = static_cast<std::size_t>(config_values.bwd_dist);

        try {
            auto q = this->get_queue();
            committed_descriptors = { std::make_unique<committed_desc>(fwd_desc, q),
                                      std::make_unique<committed_desc>(bwd_desc, q) };
        }
        catch (const pfft::unsupported_configuration& e) {
            throw oneapi::mkl::unimplemented("dft/backends/portfft", __FUNCTION__, e.what());
        }
        {
            auto q = this->get_queue();
            sycl::buffer<std::complex<scalar>, 1> inout_buf{ sycl::range<1>(8) };
            auto usm_ptr = sycl::malloc_device<std::complex<scalar>>(8, q);

      //      committed_descriptors[0]->compute_forward(inout_buf);
      //      committed_descriptors[1]->compute_backward(inout_buf);
            committed_descriptors[0]->compute_forward(usm_ptr);
            committed_descriptors[1]->compute_backward(usm_ptr);
            sycl::free(usm_ptr, q);
            q.wait_and_throw();
        }
    }

    ~portfft_commit() override = default;

    void* get_handle() noexcept override {
        return committed_descriptors.data();
    }

#define BACKEND portfft
#include "../backend_compute_signature.cxx"
#undef BACKEND
};
} // namespace detail

template <dft::precision prec, dft::domain dom>
dft::detail::commit_impl<prec, dom>* create_commit(const dft::detail::descriptor<prec, dom>& desc,
                                                   sycl::queue& sycl_queue) {
    return new detail::portfft_commit<prec, dom>(sycl_queue, desc.get_values());
}

template dft::detail::commit_impl<dft::detail::precision::SINGLE, dft::detail::domain::REAL>*
create_commit(
    const dft::detail::descriptor<dft::detail::precision::SINGLE, dft::detail::domain::REAL>&,
    sycl::queue&);
template dft::detail::commit_impl<dft::detail::precision::SINGLE, dft::detail::domain::COMPLEX>*
create_commit(
    const dft::detail::descriptor<dft::detail::precision::SINGLE, dft::detail::domain::COMPLEX>&,
    sycl::queue&);
template dft::detail::commit_impl<dft::detail::precision::DOUBLE, dft::detail::domain::REAL>*
create_commit(
    const dft::detail::descriptor<dft::detail::precision::DOUBLE, dft::detail::domain::REAL>&,
    sycl::queue&);
template dft::detail::commit_impl<dft::detail::precision::DOUBLE, dft::detail::domain::COMPLEX>*
create_commit(
    const dft::detail::descriptor<dft::detail::precision::DOUBLE, dft::detail::domain::COMPLEX>&,
    sycl::queue&);

} // namespace oneapi::mkl::dft::portfft
