


## 卷积层
$$
\begin{aligned}
    \frac {\partial E}{\partial w_{x, y}} &= \sum_{i, j} \frac {\partial E}{\partial out_{i, j}} \frac {\partial out_{i, j}}{\partial w_{x, y}} \\
    &= \sum_{i, j} \frac {\partial E}{\partial out_{i, j}} \frac {\partial out_{i, j}}{\partial w_{x, y}} \\
    &= \sum_{i, j} \frac {\partial E}{\partial out_{i, j}} \sigma'(out_{i, j}) in_{i + x, j + y} \\
\end{aligned}
$$

$$
\begin{aligned}
    \frac {\partial E}{\partial in_{i + x, j + y}} &= \sum_{x, y} \frac {\partial E}{\partial out_{i, j}} \frac {\partial out_{i, j}}{\partial in_{i + x, j + y}} \\
    &= \sum_{x, y} \frac {\partial E}{\partial out_{i, j}} \sigma'(\partial out_{i, j})w_{x, y} \\
\end{aligned}
$$

## 池化层
$$
\begin{aligned}
    \frac {\partial E}{\partial in_{i, j}} = \frac {\partial E}{\partial out_{[\frac it], [\frac jt]}} \frac 1{t^2}
\end{aligned}
$$

## 全连接层
$$\frac {\partial E}{\partial w_{i, j}} = \frac {\partial E}{\partial out_j} \sigma'(out_j) in_i$$

$$\frac {\partial E}{\partial b_j} = \frac {\partial E}{\partial out_j} \sigma'(out_j)$$

$$\frac {\partial E}{\partial in_i} = \sum_j \frac {\partial E}{\partial out_j} \sigma'(out_j) w_{i, j}$$

## 初始
$$
\begin{aligned}
E &= -\ln \frac {e^{\hat y_{out}}}{\sum_{i = 1}^m e^{\hat y_i}} \\
&= \ln \left(\sum_{i = 1}^m e^{\hat y_i}\right) - \hat y_{out} \\
\end{aligned}
$$

$$
\begin{aligned}
\frac {\partial E}{\partial \hat y_i} &= \frac {\partial \ln \sum_{j = 1}^m e^{\hat y_j}}{\partial \hat y_i} - \frac {\partial \hat y_{out}}{\partial \hat y_i} \\
&= \frac {\partial \ln \sum_{j = 1}^m e^{\hat y_j}}{\partial \sum_{j = 1}^m e^{\hat y_j}} \frac {\partial \sum_{j = 1}^m e^{\hat y_j}}{\partial \hat y_i} - [i = out] \\
&= \frac 1{\sum_{j = 1}^m e^{\hat y_j}} e^{\partial y_i} - [i = out] \\
&= \frac {e^{\partial y_i}}{\sum_{j = 1}^m e^{\hat y_j}} - [i = out] \\
\end{aligned}
$$