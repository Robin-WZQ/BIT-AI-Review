#include "arap_single_iteration.h"
#include "biharmonic_precompute.h"
#include <igl/min_quad_with_fixed.h>
#include <iostream>
#include <igl/polar_svd3x3.h>

void arap_single_iteration(
  const igl::min_quad_with_fixed_data<double> & data,
  const Eigen::SparseMatrix<double> & K,
  const Eigen::MatrixXd & bc,
  Eigen::MatrixXd & U)
{
  int N = U.rows();

  // 构建 C:
  Eigen::MatrixXd C(3*N, 3);
  C = (U.transpose() * K).transpose();

  // 构建 R:
  Eigen::MatrixXd R(3*N, 3);
  for(int i = 0; i < N; i ++) {
    Eigen::Matrix3d ck, rk;
    ck << C.row(i*3),
          C.row(i*3 + 1),
          C.row(i*3 + 2);
    igl::polar_svd3x3(ck, rk);
    R.row(i*3) = rk.row(0);
    R.row(i*3 + 1) = rk.row(1);
    R.row(i*3 + 2) = rk.row(2);
  }
  igl::min_quad_with_fixed_solve(data, K*R, bc, 
                      Eigen::MatrixXd(), U);

}
