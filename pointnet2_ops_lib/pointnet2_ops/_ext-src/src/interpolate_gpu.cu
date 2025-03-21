#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"

// input: unknown(b, n, 3) known(b, m, 3)
// output: dist2(b, n, 3), idx(b, n, 3)
template <typename scalar_t>
__global__ void three_nn_kernel(int b, int n, int m,
                                const scalar_t *__restrict__ unknown,
                                const scalar_t *__restrict__ known,
                                scalar_t *__restrict__ dist2,
                                int *__restrict__ idx) {
  int batch_index = blockIdx.x;
  unknown += batch_index * n * 3;
  known += batch_index * m * 3;
  dist2 += batch_index * n * 3;
  idx += batch_index * n * 3;

  int index = threadIdx.x;
  int stride = blockDim.x;
  for (int j = index; j < n; j += stride) {
    scalar_t ux = unknown[j * 3 + 0];
    scalar_t uy = unknown[j * 3 + 1];
    scalar_t uz = unknown[j * 3 + 2];

    scalar_t best1 = 6e4, best2 = 6e4, best3 = 6e4;
    int besti1 = 0, besti2 = 0, besti3 = 0;
    for (int k = 0; k < m; ++k) {
      scalar_t x = known[k * 3 + 0];
      scalar_t y = known[k * 3 + 1];
      scalar_t z = known[k * 3 + 2];
      scalar_t d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);
      if (d < best1) {
        best3 = best2;
        besti3 = besti2;
        best2 = best1;
        besti2 = besti1;
        best1 = d;
        besti1 = k;
      } else if (d < best2) {
        best3 = best2;
        besti3 = besti2;
        best2 = d;
        besti2 = k;
      } else if (d < best3) {
        best3 = d;
        besti3 = k;
      }
    }
    dist2[j * 3 + 0] = best1;
    dist2[j * 3 + 1] = best2;
    dist2[j * 3 + 2] = best3;

    idx[j * 3 + 0] = besti1;
    idx[j * 3 + 1] = besti2;
    idx[j * 3 + 2] = besti3;
  }
}

void three_nn_kernel_wrapper(int b, int n, int m, const at::Tensor &unknown,
                             const at::Tensor &known, at::Tensor &dist2, int *idx) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // three_nn_kernel<<<b, opt_n_threads(n), 0, stream>>>(b, n, m, unknown, known,
  //                                                     dist2, idx);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(unknown.type(), "three_nn", ([&] {
    three_nn_kernel<scalar_t><<<b, opt_n_threads(n), 0, stream>>>(
        b, n, m,
        unknown.data_ptr<scalar_t>(),
        known.data_ptr<scalar_t>(),
        dist2.data_ptr<scalar_t>(),
        idx);
  }));

  CUDA_CHECK_ERRORS();
}

// input: points(b, c, m), idx(b, n, 3), weight(b, n, 3)
// output: out(b, c, n)
template <typename scalar_t>
__global__ void three_interpolate_kernel(int b, int c, int m, int n,
                                         const scalar_t *__restrict__ points,
                                         const int *__restrict__ idx,
                                         const scalar_t *__restrict__ weight,
                                         scalar_t *__restrict__ out) {
  int batch_index = blockIdx.x;
  points += batch_index * m * c;

  idx += batch_index * n * 3;
  weight += batch_index * n * 3;

  out += batch_index * n * c;

  const int index = threadIdx.y * blockDim.x + threadIdx.x;
  const int stride = blockDim.y * blockDim.x;
  for (int i = index; i < c * n; i += stride) {
    const int l = i / n;
    const int j = i % n;
    scalar_t w1 = weight[j * 3 + 0];
    scalar_t w2 = weight[j * 3 + 1];
    scalar_t w3 = weight[j * 3 + 2];

    int i1 = idx[j * 3 + 0];
    int i2 = idx[j * 3 + 1];
    int i3 = idx[j * 3 + 2];

    out[i] = points[l * m + i1] * w1 + points[l * m + i2] * w2 +
             points[l * m + i3] * w3;
  }
}

void three_interpolate_kernel_wrapper(int b, int c, int m, int n,
                                      const at::Tensor &points, const int *idx,
                                      const at::Tensor &weight, at::Tensor &out) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // three_interpolate_kernel<<<b, opt_block_config(n, c), 0, stream>>>(
  //     b, c, m, n, points, idx, weight, out);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(points.type(), "three_interpolate", ([&] {
    three_interpolate_kernel<scalar_t><<<b, opt_block_config(n, c), 0, stream>>>(
        b, c, m, n,
        points.data_ptr<scalar_t>(),
        idx,
        weight.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>());
  }));

  CUDA_CHECK_ERRORS();
}

// input: grad_out(b, c, n), idx(b, n, 3), weight(b, n, 3)
// output: grad_points(b, c, m)
template <typename scalar_t>
__global__ void three_interpolate_grad_kernel(
    int b, int c, int n, int m, const scalar_t *__restrict__ grad_out,
    const int *__restrict__ idx, const scalar_t *__restrict__ weight,
    scalar_t *__restrict__ grad_points) {
  int batch_index = blockIdx.x;
  grad_out += batch_index * n * c;
  idx += batch_index * n * 3;
  weight += batch_index * n * 3;
  grad_points += batch_index * m * c;

  const int index = threadIdx.y * blockDim.x + threadIdx.x;
  const int stride = blockDim.y * blockDim.x;
  for (int i = index; i < c * n; i += stride) {
    const int l = i / n;
    const int j = i % n;
    scalar_t w1 = weight[j * 3 + 0];
    scalar_t w2 = weight[j * 3 + 1];
    scalar_t w3 = weight[j * 3 + 2];

    int i1 = idx[j * 3 + 0];
    int i2 = idx[j * 3 + 1];
    int i3 = idx[j * 3 + 2];

    gpuAtomicAdd(grad_points + l * m + i1, grad_out[i] * w1);
    gpuAtomicAdd(grad_points + l * m + i2, grad_out[i] * w2);
    gpuAtomicAdd(grad_points + l * m + i3, grad_out[i] * w3);
  }
}

void three_interpolate_grad_kernel_wrapper(int b, int c, int n, int m,
                                           const at::Tensor &grad_out,
                                           const int *idx, const at::Tensor &weight,
                                           at::Tensor &grad_points) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // three_interpolate_grad_kernel<<<b, opt_block_config(n, c), 0, stream>>>(
  //     b, c, n, m, grad_out, idx, weight, grad_points);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_out.type(), "three_interpolate_grad", ([&] {
    three_interpolate_grad_kernel<scalar_t><<<b, opt_block_config(n, c), 0, stream>>>(
        b, c, n, m,
        grad_out.data_ptr<scalar_t>(),
        idx,
        weight.data_ptr<scalar_t>(),
        grad_points.data_ptr<scalar_t>());
  }));

  CUDA_CHECK_ERRORS();
}
