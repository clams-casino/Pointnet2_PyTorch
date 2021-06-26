#include "group_points.h"
#include "utils.h"

void group_points_kernel_wrapper(int b, int c, int n, int npoints, int nsample,
                                 const at::Tensor &points, const int *idx,
                                 at::Tensor &out);

void group_points_grad_kernel_wrapper(int b, int c, int n, int npoints,
                                      int nsample, const at::Tensor &grad_out,
                                      const int *idx, at::Tensor &grad_points);

at::Tensor group_points(at::Tensor points, at::Tensor idx) {
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(points);
  CHECK_IS_INT(idx);

  if (points.is_cuda()) {
    CHECK_CUDA(idx);
  }

  at::Tensor output =
      torch::zeros({points.size(0), points.size(1), idx.size(1), idx.size(2)},
                   at::device(points.device()).dtype(points.scalar_type()));

  if (points.is_cuda()) {
    group_points_kernel_wrapper(points.size(0), points.size(1), points.size(2),
                                idx.size(1), idx.size(2),
                                points, idx.data_ptr<int>(),
                                output);
  } else {
    AT_ASSERT(false, "CPU not supported");
  }

  return output;
}

at::Tensor group_points_grad(at::Tensor grad_out, at::Tensor idx, const int n) {
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
    group_points_grad_kernel_wrapper(
        grad_out.size(0), grad_out.size(1), n, idx.size(1), idx.size(2),
        grad_out, idx.data_ptr<int>(),
        output);
  } else {
    AT_ASSERT(false, "CPU not supported");
  }

  return output;
}
