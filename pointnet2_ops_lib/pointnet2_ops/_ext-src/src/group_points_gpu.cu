#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"

// input: points(b, c, n) idx(b, npoints, nsample)
// output: out(b, c, npoints, nsample)
template <typename scalar_t>
__global__ void group_points_kernel(int b, int c, int n, int npoints,
                                    int nsample,
                                    const scalar_t *__restrict__ points,
                                    const int *__restrict__ idx,
                                    scalar_t *__restrict__ out) {
  int batch_index = blockIdx.x;
  points += batch_index * n * c;
  idx += batch_index * npoints * nsample;
  out += batch_index * npoints * nsample * c;

  const int index = threadIdx.y * blockDim.x + threadIdx.x;
  const int stride = blockDim.y * blockDim.x;
  for (int i = index; i < c * npoints; i += stride) {
    const int l = i / npoints;
    const int j = i % npoints;
    for (int k = 0; k < nsample; ++k) {
      int ii = idx[j * nsample + k];
      out[(l * npoints + j) * nsample + k] = points[l * n + ii];
    }
  }
}

void group_points_kernel_wrapper(int b, int c, int n, int npoints, int nsample,
                                 at::Tensor points, const int *idx,
                                 at::Tensor &out) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // group_points_kernel<<<b, opt_block_config(npoints, c), 0, stream>>>( #TODO remove this
  //     b, c, n, npoints, nsample, points, idx, out);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(points.type(), "group_points", ([&] {
    group_points_kernel<scalar_t><<<b, opt_block_config(npoints, c), 0, stream>>>(
        b, c, n, npoints, nsample,
        points.data_ptr<scalar_t>(),
        idx,
        out.data_ptr<scalar_t>());
  }));

  CUDA_CHECK_ERRORS();
}

// input: grad_out(b, c, npoints, nsample), idx(b, npoints, nsample)
// output: grad_points(b, c, n)
template <typename scalar_t>
__global__ void group_points_grad_kernel(int b, int c, int n, int npoints,
                                         int nsample,
                                         const scalar_t *__restrict__ grad_out,
                                         const int *__restrict__ idx,
                                         scalar_t *__restrict__ grad_points) {
  int batch_index = blockIdx.x;
  grad_out += batch_index * npoints * nsample * c;
  idx += batch_index * npoints * nsample;
  grad_points += batch_index * n * c;

  const int index = threadIdx.y * blockDim.x + threadIdx.x;
  const int stride = blockDim.y * blockDim.x;
  for (int i = index; i < c * npoints; i += stride) {
    const int l = i / npoints;
    const int j = i % npoints;
    for (int k = 0; k < nsample; ++k) {
      int ii = idx[j * nsample + k];
      atomicAdd(grad_points + l * n + ii,
                grad_out[(l * npoints + j) * nsample + k]);
    }
  }
}

void group_points_grad_kernel_wrapper(int b, int c, int n, int npoints,
                                      int nsample, at::Tensor grad_out,
                                      const int *idx, at::Tensor &grad_points) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // group_points_grad_kernel<<<b, opt_block_config(npoints, c), 0, stream>>>(
  //     b, c, n, npoints, nsample, grad_out, idx, grad_points);


  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_out.type(), "group_points_grad", ([&] {
    group_points_grad_kernel<scalar_t><<<b, opt_block_config(npoints, c), 0, stream>>>(
        b, c, n, npoints, nsample,
        grad_out.data_ptr<scalar_t>(),
        idx,
        grad_points.data_ptr<scalar_t>());
  }));

  CUDA_CHECK_ERRORS();
}
