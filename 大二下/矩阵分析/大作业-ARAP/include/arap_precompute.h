#ifndef ARAP_PRECOMPUTE_H
#define ARAP_PRECOMPUTE_H
#include <Eigen/Core>
#include <Eigen/Sparse>

namespace igl
{
  template <typename T> struct min_quad_with_fixed_data;
}

// Precompute data needed to efficiently conduct local-global iterations for an
// arap deformation. This includes the `data` struct employed by
// `igl::min_quad_with_fixed` to solve the global step" and constructing the
// bi-linear form `K` that mixes rotation degrees of freedom with unknown
// positions for preparing the covariance matrices of the local step and the
// linear term of the global step.
//
// Inputs:
//   V  #V by dim vertex positions
//   F  #F by simplex-size list of element indices
//   b  #b indices into V of handle vertices
// Outputs:
//   data  pre-factorized system matrix etc. (see `igl::min_quad_with_fixed`)
//   K  #R*dim by #V
void arap_precompute(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & F,
  const Eigen::VectorXi & b,
  igl::min_quad_with_fixed_data<double> & data,
  Eigen::SparseMatrix<double> & K);

#endif

