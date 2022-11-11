---
title: 程序设计期末大作业报告
subtitle: 基于神经网络的手写数字识别
abstract_zh: 用 C++ 实现用于识别手写数字的神经网络，采用 MNIST 作为训练集和测试集进行训练，并使用 OpenCV 进行图像支持。
abstract_en: Using C++ to implement a Neural Network which classifies handwritten digits. This program takes MNIST as training dataset and testing dataset, using OpenCV for graphic support.
---

# 需求分析
## 任务

## 输入数据

## 输出数据


# 概要设计
## 神经网络部分
### 数据
数据在 `data.h` 中声明，并用结构体 `Data` 表示，每个 `Data` 中包含变量：

* `float *in`：指向 784 大小的一维数组，表示输入向量
* `int out`：输出值

```cpp
struct Data {
    float *in;
    int out;
};
```

### 层
层在 `layer.h` 中声明，并用类 `Layer` 表示，每个 `Layer` 中包含变量：

* `ACTIVATION activate`：激活函数的类型，可取 `SIGMOID` 或 `ReLU`
* `int in_n`：输入向量的大小
* `int n`：输出向量的大小
* `float **W`：指向 in_n * n 大小的二维数组，表示转移系数
* `float *b`：指向 n 大小的一维数组，表示阈值
* `float *y`：指向 n 大小的一维数组，`y[i]` 表示第 `i` 个神经元的输出
* `float *dy`：指向 n 大小的一维数组，`dy[i]` 表示网络损失函数对 `y[i]` 的偏导数

包含公共函数：

* `Layer(int _in_n, int _n, ACTIVATION _activate)`：构造函数
* `~Layer()`：析构函数
* `void forward_propagation(float *in)`：将 `*in` 作为输入进行前向传播，即计算出输出 `y[i]` 的值
* `void backward_propagation(float *prev_y, float *prev_dy, float eta)`：利用前一层的输出 `*prev_y` 进行反向传播，`eta` 作为学习率（Learning Rate），改变 `**W` 和 `*b`，并计算前一层的输出的偏导数 `*prev_dy`

包含私有函数：

* `inline float f(float x, ACTIVATION activate)`：将 `x` 作为输入，返回 `activate` 类型激活函数的值
* `inline float gradient(float x, ACTIVATION activate)`：将 `x` 作为输入，返回 `activate` 类型激活函数的导数的值

```cpp
class Layer {
public:
    Layer(int _in_n, int _n, ACTIVATION _activate);
    ~Layer();

    void forward_propagation(float *in);
    void backward_propagation(float *prev_y, float *prev_dy, float eta);

private:
    inline float f(float x, ACTIVATION activate);
    inline float gradient(float x, ACTIVATION activate);

public:
    ACTIVATION activate;
    int in_n, n;
    float **W;
    float *b;
    float *y;
    float *dy;
};
```

### 网络
### 网络集合

## MNIST 数据读取器

## 训练程序

## 识别程序

# 成果
## 源代码编译
首先，安装 OpenCV（C++）。

然后，从 Github 下载代码：

```bash
git clone https://github.com/cutekibry/solosis.git
```

进入目录并编译：

```bash
cd solosis
make
```

在本目录下会产生可执行文件 `solosis`。