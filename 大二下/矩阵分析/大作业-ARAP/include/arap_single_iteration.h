#ifndef ARAP_SINGLE_ITERATION_H
#define ARAP_SINGLE_ITERATION_H
#include <Eigen/Core>
#include <Eigen/Sparse>

namespace igl
{
  template <typename T> struct min_quad_with_fixed_data;
}

// Given precomputed data (`data` and `K`), handle _positions_ `bc` and current
// positions of all vertices `U`, conduct a _single_ iteration of the
// local-global solver for minimizing the as-rigid-as-possible energy. Output
// the _positions_ of all vertices of the mesh (by overwriting `U`).
//
// Inputs:
//   data  pre-factorized system matrix etc. (see `arap_precompute`
//   K  pre-constructed bi-linear term of energy combining rotations and
//     positions
//   U  #V by dim list of current positions (see output)
// Outputs:
//   U  #V by dim list of new positions (see input)
void arap_single_iteration(
  const igl::min_quad_with_fixed_data<double> & data,
  const Eigen::SparseMatrix<double> & K,
  const Eigen::MatrixXd & bc,
  Eigen::MatrixXd & U);

#endif
