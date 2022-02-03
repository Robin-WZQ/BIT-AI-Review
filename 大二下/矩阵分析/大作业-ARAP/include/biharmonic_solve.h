#ifndef BIHARMONIC_SOLVE_H
#define BIHARMONIC_SOLVE_H
#include <Eigen/Core>

namespace igl
{
  template <typename T> struct min_quad_with_fixed_data;
}

// Given precomputation data and a list of handle _displacements_ determine
// _displacements_ for all vertices in the mesh.
//
// Inputs:
//   data  pre-factorized system matrix etc. (see `biharmonic_precompute`
//   bc  #b by 3 list of displacements for each handle vertex
// Outputs:
//   D  #V by 3 list of displacements
void biharmonic_solve(
  const igl::min_quad_with_fixed_data<double> & data,
  const Eigen::MatrixXd & bc,
  Eigen::MatrixXd & D);


#endif

