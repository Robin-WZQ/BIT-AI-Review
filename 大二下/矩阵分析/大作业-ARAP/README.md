### 矩阵分析大作业四——ARAP deformation

#### 项目

运行代码放到网盘上了

链接：https://pan.baidu.com/s/1KsUws7LlURdUSHxvOaA8Hw 

提取码：g77a 


#### 日期

2021/4/22

#### 文件内容

./shared 下是第三方库依赖以及数据(由于文件太大所以放到百度网盘上了)

./include 下是定义好的头文件，课程自带，本组未作修改

./src 下是可执行文件

./CMakeLists 是cmake编译需要的配置文件

#### 项目环境

cmake 3.18.4，win10，VS2019

#### 执行操作

cd ./arap

mkdir build

cd build

cmake ..

make

./deformation

#### 项目基本信息

我们学习了多伦多大学开源在GitHub上的课程，网址为https://github.com/alecjacobson/geometry-processing-introduction
课程只提供思路和方法，代码部分为“paste your code here”，直接运行仅有选点交互操作。
我们完成了arap部分的代码，详见./src目录下

#### 改进方向

1、本次项目并未实现导入不同模型的功能，仅仅编译生成固定模型的工程文件，然而这是libigl教程中可实现的。
2、在未使用gpu加速下延迟性比较大，考虑适当增加帧数处理速度，或进一步优化算法。


注：
1、这里仅引用到libigl和Eigen等第三方库，其余如test、image、tutorial部分由于git链接到网址自动下载，实际编译中并不需要。
2、./libigl/externel中文件由于网络原因，部分需要手动下载对应版本，下载内容均在对应的Cmake文件中提到，为了保证项目完整性这里保留了第三方库
