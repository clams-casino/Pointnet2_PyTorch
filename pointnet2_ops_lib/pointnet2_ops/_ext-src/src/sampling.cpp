#include "sampling.h"
#include "utils.h"

void gather_points_kernel_wrapper(int b, int c, int n, int npoints,
                                  const at::Tensor &points, const int *idx,
                                  at::Tensor &out);
void gather_points_grad_kernel_wrapper(int b, int c, int n, int npoints,
                                       const at::Tensor &grad_out, const int *idx,
                                       at::Tensor grad_points);

void furthest_point_sampling_kernel_wrapper(int b, int n, int m,
                                            const at::Tensor &dataset, at::Tensor &temp,
                                            int *idxs);

at::Tensor gather_points(at::Tensor points, at::Tensor idx) {
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(points);
  CHECK_IS_INT(idx);

  if (points.is_cuda()) {
    CHECK_CUDA(idx);
  }

  at::Tensor output =
      torch::zeros({points.size(0), points.size(1), idx.size(1)},
                   at::device(points.device()).dtype(points.scalar_type()));

  if (points.is_cuda()) {
    gather_points_kernel_wrapper(points.size(0), points.size(1), points.size(2),
                                 idx.size(1), points,
                                 idx.data_ptr<int>(), output);
  } else {
    AT_ASSERT(false, "CPU not supported");
  }

  return output;
}

at::Tensor gather_points_grad(at::Tensor grad_out, at::Tensor idx,
                              const int n) {
  CHECK_CONTIGUOUS(grad_out);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(grad_out);
  CHECK_IS_INT(idx);

  if (grad_out.is_cuda()) {
    CHECK_CUDA(idx);
  }

  at::Tensor output =
      torch::zeros({grad_out.size(0), grad_out.size(1), n},
                   at::device(grad_out.device()).dtype(grad_out.scalar_type()));

  if (grad_out.is_cuda()) {
    gather_points_grad_kernel_wrapper(grad_out.size(0), grad_out.size(1), n,
                                      idx.size(1), grad_out,
                                      idx.data_ptr<int>(),
                                      output);
  } else {
    AT_ASSERT(false, "CPU not supported");
  }

  return output;
}
at::Tensor furthest_point_sampling(at::Tensor points, const int nsamples) {
  CHECK_CONTIGUOUS(points);
  CHECK_IS_FLOAT(points);

  at::Tensor output =
      torch::zeros({points.size(0), nsamples},
                   at::device(points.device()).dtype(at::ScalarType::Int));

  at::Tensor tmp =
      torch::full({points.size(0), points.size(1)}, 1e10,
                  at::device(points.device()).dtype(points.scalar_type()));

  if (points.is_cuda()) {
    furthest_point_sampling_kernel_wrapper(
        points.size(0), points.size(1), nsamples, points,
        tmp, output.data_ptr<int>());
  } else {
    AT_ASSERT(false, "CPU not supported");
  }

  return output;
}
