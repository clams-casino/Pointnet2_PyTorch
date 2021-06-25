#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"

// input: new_xyz(b, m, 3) xyz(b, n, 3)
// output: idx(b, m, nsample)
template <typename scalar_t>
__global__ void query_ball_point_kernel(int b, int n, int m, float radius,
                                        int nsample,
                                        const scalar_t *__restrict__ new_xyz,
                                        const scalar_t *__restrict__ xyz,
                                        int *__restrict__ idx) {
  int batch_index = blockIdx.x;
  xyz += batch_index * n * 3;
  new_xyz += batch_index * m * 3;
  idx += m * nsample * batch_index;

  int index = threadIdx.x;
  int stride = blockDim.x;

  float radius2 = radius * radius;
  for (int j = index; j < m; j += stride) {
    scalar_t new_x = new_xyz[j * 3 + 0];
    scalar_t new_y = new_xyz[j * 3 + 1];
    scalar_t new_z = new_xyz[j * 3 + 2];
    for (int k = 0, cnt = 0; k < n && cnt < nsample; ++k) {
      scalar_t x = xyz[k * 3 + 0];
      scalar_t y = xyz[k * 3 + 1];
      scalar_t z = xyz[k * 3 + 2];
      scalar_t d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
                 (new_z - z) * (new_z - z);
      if (d2 < radius2) {
        if (cnt == 0) {
          for (int l = 0; l < nsample; ++l) {
            idx[j * nsample + l] = k;
          }
        }
        idx[j * nsample + cnt] = k;
        ++cnt;
      }
    }
  }
}

void query_ball_point_kernel_wrapper(int b, int n, int m, float radius,
                                     int nsample, at::Tensor new_xyz,
                                     at::Tensor xyz, int *idx); {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(xyz.type(), "query_ball_point_kernel", ([&] {
    query_ball_point_kernel<scalar_t><<<b, opt_n_threads(m), 0, stream>>>(
        b, n, m, radius, nsample,
        new_xyz.data_ptr<scalar_t>(),
        xyz.data_ptr<scalar_t>(),
        idx);
  }));


  CUDA_CHECK_ERRORS();
}
