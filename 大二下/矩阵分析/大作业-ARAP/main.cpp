#include "biharmonic_precompute.h"
#include "biharmonic_solve.h"
#include "arap_precompute.h"
#include "arap_single_iteration.h"
#include <igl/min_quad_with_fixed.h>
#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject.h>
#include <igl/snap_points.h>
#include <igl/unproject_onto_mesh.h>
#include <Eigen/Core>
#include <iostream>
#include <stack>


struct State
{
  // 控制部分
  Eigen::MatrixXd CV, CU;
  bool placing_handles = true;
} s;

int main(int argc, char *argv[])
{
  
  std::stack<State> undo_stack,redo_stack;
  const auto push_undo = [&](State & _s=s)
  {
    undo_stack.push(_s);
    // 清楚操作
    redo_stack = std::stack<State>();
  };
  const auto undo = [&]()
  {
    if(!undo_stack.empty())
    {
      redo_stack.push(s);
      s = undo_stack.top();
      undo_stack.pop();
    }
  };
  const auto redo = [&]()
  {
    if(!redo_stack.empty())
    {
      undo_stack.push(s);
      s = redo_stack.top();
      redo_stack.pop();
    }
  };

  Eigen::MatrixXd V,U;
  Eigen::MatrixXi F;
  long sel = -1;
  Eigen::RowVector3f last_mouse;
  igl::min_quad_with_fixed_data<double> biharmonic_data, arap_data;
  Eigen::SparseMatrix<double> arap_K;

  // 导入模型
  igl::read_triangle_mesh(
    (argc>1?argv[1]:"../shared/data/decimated-knight.off"),V,F);
  U = V;
  igl::opengl::glfw::Viewer viewer;
  std::cout<<R"(
[点击]   选点
[拖动]   拖动模型空间状态
[空格]   切换成变形状态
M,m      切换arap or asap
)";
  enum Method
  {
    BIHARMONIC = 0,
    ARAP = 1,
    NUM_METHODS = 2,
  } method = BIHARMONIC;

  const auto & update = [&]()
  {
    // 预定义颜色
    const Eigen::RowVector3d orange(1.0,0.7,0.2);
    const Eigen::RowVector3d yellow(1.0,0.9,0.2);
    const Eigen::RowVector3d blue(0.2,0.3,0.8);
    const Eigen::RowVector3d green(0.2,0.6,0.3);
    if(s.placing_handles)
    {
      viewer.data().set_vertices(V);
      viewer.data().set_colors(blue);
      viewer.data().set_points(s.CV,orange);
    }else
    {
      // 变形算法
      switch(method)
      {
        default:
        case BIHARMONIC:
        {
          Eigen::MatrixXd D;
          biharmonic_solve(biharmonic_data,s.CU-s.CV,D);
          U = V+D;
          break;
        }
        case ARAP:
        {
          arap_single_iteration(arap_data,arap_K,s.CU,U);
          break;
        }
      }
      viewer.data().set_vertices(U);
      viewer.data().set_colors(method==BIHARMONIC?orange:yellow);
      viewer.data().set_points(s.CU,method==BIHARMONIC?blue:green);
    }
    viewer.data().compute_normals();
  };
  viewer.callback_mouse_down = 
    [&](igl::opengl::glfw::Viewer&, int, int)->bool
  {
    last_mouse = Eigen::RowVector3f(
      viewer.current_mouse_x,viewer.core.viewport(3)-viewer.current_mouse_y,0);
    if(s.placing_handles)
    {
      // 选择格点上最近的位置去做形变
      int fid;
      Eigen::Vector3f bary;
      if(igl::unproject_onto_mesh(
        last_mouse.head(2),
        viewer.core.view,
        viewer.core.proj, 
        viewer.core.viewport, 
        V, F, 
        fid, bary))
      {
        long c;
        bary.maxCoeff(&c);
        Eigen::RowVector3d new_c = V.row(F(fid,c));
        if(s.CV.size()==0 || (s.CV.rowwise()-new_c).rowwise().norm().minCoeff() > 0)
        {
          push_undo();
          s.CV.conservativeResize(s.CV.rows()+1,3);
          s.CV.row(s.CV.rows()-1) = new_c;
          update();
          return true;
        }
      }
    }else
    {
      // 移动最近的所选取的点
      Eigen::MatrixXf CP;
      igl::project(
        Eigen::MatrixXf(s.CU.cast<float>()),
        viewer.core.view,
        viewer.core.proj, viewer.core.viewport, CP);
      Eigen::VectorXf D = (CP.rowwise()-last_mouse).rowwise().norm();
      sel = (D.minCoeff(&sel) < 30)?sel:-1;
      if(sel != -1)
      {
        last_mouse(2) = CP(sel,2);
        push_undo();
        update();
        return true;
      }
    }
    return false;
  };

  viewer.callback_mouse_move = [&](igl::opengl::glfw::Viewer &, int,int)->bool
  {
    if(sel!=-1)
    {
      Eigen::RowVector3f drag_mouse(
        viewer.current_mouse_x,
        viewer.core.viewport(3) - viewer.current_mouse_y,
        last_mouse(2));
      Eigen::RowVector3f drag_scene,last_scene;
      igl::unproject(
        drag_mouse,
        viewer.core.view,
        viewer.core.proj,
        viewer.core.viewport,
        drag_scene);
      igl::unproject(
        last_mouse,
        viewer.core.view,
        viewer.core.proj,
        viewer.core.viewport,
        last_scene);
      s.CU.row(sel) += (drag_scene-last_scene).cast<double>();
      last_mouse = drag_mouse;
      update();
      return true;
    }
    return false;
  };
  viewer.callback_mouse_up = [&](igl::opengl::glfw::Viewer&, int, int)->bool
  {
    sel = -1;
    return false;
  };
  viewer.callback_key_pressed = 
    [&](igl::opengl::glfw::Viewer &, unsigned int key, int mod)
  {
    switch(key)
    {
      case 'M':
      case 'm':
      {
        method = (Method)(((int)(method)+1)%((int)(NUM_METHODS)));
        break;
      }
      case 'R':
      case 'r':
      {
        push_undo();
        s.CU = s.CV;
        break;
      }
      case ' ':
        push_undo();
        s.placing_handles ^= 1;
        if(!s.placing_handles && s.CV.rows()>0)
        {
          // 切换形变模式
          s.CU = s.CV;
          Eigen::VectorXi b;
          igl::snap_points(s.CV,V,b);
          // 预计算
          biharmonic_precompute(V,F,b,biharmonic_data);
          arap_precompute(V,F,b,arap_data,arap_K);
        }
        break;
      default:
        return false;
    }
    update();
    return true;
  };

  
  viewer.callback_key_down = 
    [&](igl::opengl::glfw::Viewer &, unsigned char key, int mod)->bool
  {
    if(key == 'Z' && (mod & GLFW_MOD_SUPER))
    {
      (mod & GLFW_MOD_SHIFT) ? redo() : undo();
      update();
      return true;
    }
    return false;
  };
  viewer.callback_pre_draw = 
    [&](igl::opengl::glfw::Viewer &)->bool
  {
    if(viewer.core.is_animating && !s.placing_handles && method == ARAP)
    {
      arap_single_iteration(arap_data,arap_K,s.CU,U);
      update();
    }
    return false;
  };
  viewer.data().set_mesh(V,F);
  viewer.data().show_lines = false;
  viewer.core.is_animating = true;
  viewer.data().face_based = true;
  update();
  viewer.launch();
  return EXIT_SUCCESS;
}
