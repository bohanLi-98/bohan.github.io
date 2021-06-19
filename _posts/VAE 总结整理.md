---
layout: default
　　title: VAE 总结整理

---

# VAE 总结整理

## 原始VAE介绍

VAE是一种生成模型，通过隐变量z生成目标数据X。核心是希望训练一种模型，可以将某个概率分布映射到训练集的概率分布。

所以核心在于：

1）**隐变量**的概率分布如何选择

2）如何衡量生成的概率分布与已有的概率分布之间的**差异**

3）如何将这种差异用于**调整**网络参数

### 变量列表

|          变量名          |                           含义                            |
| :----------------------: | :-------------------------------------------------------: |
|     $X, \mathcal X$      |                  数据点，数据点所属空间                   |
|          $Date$          |             数据集，$X$是从$Data$中采样得到的             |
|          $P(*)$          |                       变量$*$的概率                       |
|          $p(*)$          |                   变量$*$的概率密度函数                   |
|      $z,\mathcal Z$      |                  隐变量，隐变量所属空间                   |
|          $f(*)$          |                         神经网络                          |
|          $q(z)$          |           更容易产生X的z空间对应的概率密度函数            |
| $\theta,\mathcal \Theta$ |                    网络参数，参数空间                     |
|          $\mu$           |                           均值                            |
|         $\sigma$         |                          标准差                           |
|        $\epsilon$        | 参数重整化技巧用到的变量，$\epsilon \sim \mathcal N(0,I)$ |
|       $\mathcal D$       |                          KL散度                           |

### 网络思想

在生成模型中，需要一个隐变量指导网络生成对象。比如，在生成0~9的手写字符时，先决定生成哪一个数字，再进行生成，也就是有一个映射$f:\mathcal Z × \mathcal \Theta \rightarrow \mathcal X$。如果在$\mathcal Z$上随机采样，随机变量是$z$，概率密度函数是$p(z)$，根据全概率公式：
$$
p(X)=\int p(X|z;\theta)p(z)dz
$$
也就是希望生成$X$的概率最大。在VAE中，一般假定输出满足正态分布，即$X=f(z;\theta) \sim \mathcal N(f(z;\theta),\sigma^2I)$

#### 编码器

为了求解（1）中的积分，需要处理两个问题：如何定义$z$，如何在$z$上积分。

**问题1：如何定义z**

假定$z$并不能直接解释，但是可以从一个简单的分布中提取，比如正态分布$\mathcal N(0,I)$，因为只要有一个足够复杂的映射，比如神经网络$f(z;\theta)$，d维的正态分布可以变为d维中的任意分布。

**问题2：如何求解积分(1)**

首先从因空间中提取足够多的$z=\{z_1,z_2,...z_n\}$，再计算$p(X) \approx \frac {1}{n}\sum\limits_{i=1}^{n}p(X|z_i)$，

但是这带来了新的问题：

1）为了成功估计p(x)​，n值要取很大，很多的z对于X的生成不起作用；

2）由于假定X满足高斯分布，其实也就是$Aexp\{-\frac {||X-f(z;\theta)||^2}{2\sigma^2}\}$，核心是一个平方距离$||X-f(z;\theta)||^2$，也就导致不那么像X的生成结果反而得分比较高：

![QQ截图20200903160245](E:\学术\组会\2020.9.4组会\VAE 总结整理.assets\QQ截图20200903160245.png)

为了避免以上的问题，VAE将重心放在对X生成贡献大的z上。

采用一个新的分布$q(z|X)∈Q$，显然$Q\subsetneqq\mathcal Z$，计算$E_{z\sim Q}p(X|z)$就很容易，$q$不是标准正态分布。首先将$E_{z\sim Q}p(X|z)$与$p(X)$联系起来。从KL散度入手：
$$
\begin{align}
\mathcal D[q(z|X)||p(z|X)] &= E_{z\sim Q}[logq(z|X)-logp(z|X)]\\
&=E_{z\sim Q}[logq(z|X)-log[p(X|z)p(z)/(p(X))]\\
&=E_{z\sim Q}[logq(z|X)-logp(X|z)-logp(z)+logp(X)]\\
&=E_{z\sim Q}[logq(z|X)-logp(X|z)-logp(z)]+logp(X)\
\end{align}
$$
移项，得
$$
\begin{align}
logp(X)-\mathcal D[q(z|X)||p(z|X)]&=-E_{z\sim Q}[logq(z|X)-logp(X|z)-logp(z)]\\
&=E_{z\sim Q}[logp(X|z)-(logq(z|X)-logp(z))]\\
&=E_{z\sim Q}[logp(X|z)]-\mathcal D[q(z|X)||p(z)]
\end{align}
$$
观察这个等式，左边是我们的优化目标：

1）$logp(X)$要尽可能大，越大说明我们从z生成X的效果越好；

2）$\mathcal D[q(z)||p(z|X)]$要尽可能小，越小说明$q$与$p$越接近，也就是说明我们$Q$空间接近可以产生X的$\mathcal Z$空间部分；

3）$p(X|z)]$充当了解码器，将z映射到X；

4）$q(z|X)$充当了编码器，将X映射到z；

#### 损失函数

##### 第一项：$logp(X|z)$

由于我们假设$X=f(z;\theta) \sim \mathcal N(f(z;\theta),\sigma^2I)$，所以取了对数以后，直接看分子的平方项即可，其余都是常数。那么也就是$E_{z\sim Q}[\frac{1}{n\sigma^2}||X-f(z;\theta)||^2]$

##### 第二项：$\mathcal D[q(z|X)||p(z)]$

假设$q(z|X)= \mathcal N(\mu(X),\Sigma(X))$，那么计算两个正态分布的KL散度即可。

首先看一般的：
$$
\begin{align}
\mathcal D[\mathcal N(\mu_0,\Sigma_0)||\mathcal N(\mu_1,\Sigma_1)]&=E_{X\sim N_0}[log\mathcal N_0-log\mathcal N_1]\\
&=\frac{1}{2} E_{X\sim N_0}[-log|\Sigma_0|+log|\Sigma_1|\\
&-(X-\mu_0)^T\Sigma_0^{-1}(X-\mu_0)+(X-\mu_1)^T\Sigma_1^{-1}(X-\mu_1)]\\
&=\frac{1}{2}log\frac{|\Sigma_1|}{|\Sigma_0|}\\
&-\frac{1}{2}E_{X\sim N_0}[(X-\mu_0)^T\Sigma_0^{-1}(X-\mu_0)+(X-\mu_0-\mu_1+\mu_0)^T\Sigma_1^{-1}(X-\mu_0-\mu_1+\mu_0)]\\
&=\frac{1}{2}(log\frac{|\Sigma_1|}{|\Sigma_0|}+(\mu_1-\mu_0)^T\Sigma_1^{-1}(\mu_1-\mu_0))\\
&+E_{X\sim N_0}[(X-\mu_0)^T(\Sigma_1^{-1}-\Sigma_0^{-1})(X-\mu_0)]\\
&=\frac{1}{2}(log\frac{|\Sigma_1|}{|\Sigma_0|}+(\mu_1-\mu_0)^T\Sigma_1^{-1}(\mu_1-\mu_0))-\frac{1}{2}E_{X\sim N_0}[X^T(\Sigma_1^{-1}-\Sigma_0^{-1})X]\\

&=\frac{1}{2}(log\frac{|\Sigma_1|}{|\Sigma_0|}+(\mu_1-\mu_0)^T\Sigma_1^{-1}(\mu_1-\mu_0)+tr(\Sigma_0\Sigma_1^{-1})-k)
\end{align}
$$

再看特殊的，我们假设$p(z)$是标准正态分布，所以$\mu_1=0,\Sigma_1=I$，

上式化为：
$$
\mathcal D[q(z|X)||p(z)]=\frac{1}{2}(-log{|\Sigma(X)|}+\mu(X)^T\mu(X)+tr(|\Sigma(X)|)-k)
$$


这样，在从数据集$Data$中对$X$进行采样，并进行优化，所以损失函数为：
$$
\mathcal L=E_{X\sim Data}[E_{z\sim Q}[\frac{1}{\sigma^2}MSE(X,f(z;\theta))]-\frac{1}{2}(-log{|\Sigma(X)|}+\mu(X)^T\mu(X)+tr(|\Sigma(X)|)-k)]
$$

##### 参数重整化

反向传播过程中，因为是“采样”操作，z无法传播到Q，所以需要**参数重整化技巧**：
$$
z=\mu(X)+\Sigma^{1/2}(X)*\epsilon\\
\epsilon\sim\mathcal N(0,I)
$$
所以实际上，损失函数为：
$$
\mathcal L=E_{X\sim Data}[E_{\epsilon\sim\mathcal N(o,I)}[\frac{1}{\sigma^2}MSE(X,f(z=\mu(X)+\Sigma^{1/2}(X)*\epsilon;\theta))]-\frac{1}{2}(-log{|\Sigma(X)|}+\mu(X)^T\mu(X)+tr(|\Sigma(X)|)-k)]
$$

### 网络结构

![QQ图片20200903144356](E:\学术\组会\92020..4组会\VAE 总结整理.assets\QQ图片20200903144356.png)

### 生成过程

![QQ截图20200903205542](E:\学术\组会\2020.9.4组会\VAE 总结整理.assets\QQ截图20200903205542.png)

### 关于多个隐变量

#### eg.1  Deep generative modeling for single-cell transcriptomics

![QQ截图20200903210432](E:\学术\组会\2020.9.4组会\VAE 总结整理.assets\QQ截图20200903210432.png)

本文想要生成一种模型，通过每个细胞n的每个基因g作为样本，在有批次注释$s_n$下生成ZINB模型中的参数。这些参数由隐变量$z_n$与$l_n$确定，所以这个模型中，输入多了一个条件$s_n$，并且隐变量有两个，即$z_n$与$\mathcal l_n$。

![QQ截图20200903210340](E:\学术\组会\2020.9.4组会\VAE 总结整理.assets\QQ截图20200903210340.png)

并且假设这两个隐变量分别满足以下分布,且相互独立：

![QQ截图20200903211418](E:\学术\组会\2020.9.4组会\VAE 总结整理.assets\QQ截图20200903211418.png)

由于隐变量$z_n$与$l_n$互相独立，所以
$$
q(z_n,l_n|s_n,x_n)=q(z_n|x_n,s_n)q(l_n|x_n,s_n)
$$
那么进行同样的分析，则由于
$$
p(z,l|x,s)p(x|s)p(s)=p(x|z,l,s)p(z,l)p(s)\\
p(z,l|x,s)=\frac {p(x|z,l,s)p(z,l|s)}{p(x|s)}\\
\rightarrow logp(z,l|x,s)=logp(x|z,l,s)+logp(z,l)-logp(x|s)
$$
所以
$$
\begin{align}
\mathcal D[q(z,l|x,s)||p(z,l|x,s)] &= E_{z,l\sim Q}[logq(z,l|x,s)-logp(z,l|x,s)]\\
&=E_{z,l\sim Q}[logq(z|x,s)+logq(l|x,s)-log[p(x|z,l,s)-logp(z,l|s)+logp(x|s)]\\
&=E_{z,l\sim Q}[logq(z|x,s)-logp(z|s)+logq(l|x,s)-logp(l|s)-logp(x|z,l,s)]+logp(x|s)\\
&=E_{z,l\sim Q}[-logp(x|z,l,s)]+\mathcal D[q(z|x,s)||p(z)]+\mathcal D[q(l|x,s)||p(l)]+logp(x|s)\\
\end{align}
$$
移项，有
$$
logp(x|s)-\mathcal D[q(z,l|x,s)||p(z,l|x,s)]=E_{z,l\sim Q}[logp(x|z,l,s)]-\mathcal D[q(z|x,s)||p(z)]-\mathcal D[q(l|x,s)||p(l)]
$$
则有以下不等式标定了变分下界
$$
logp(x|s)≥E_{z,l\sim Q}[logp(x|z,l,s)]-\mathcal D[q(z|x,s)||p(z)]-\mathcal D[q(l|x,s)||p(l)]
$$

### 关于隐变量的分布P(z|X)

高斯分布过于简单，高维情况下分布不够集中，出现边缘效应，为了解决这些问题，标准化流被提出。而IAF是一种特殊的标准化流。

#### IAF 可逆自回归流

首先从正态分布中采样$\epsilon\sim\mathcal N(0,I)$
$$
z_0=\mu_0+\sigma_0*\epsilon\\
z_t=\mu_t+\sigma_t*z_{t-1}
$$
这样的迭代会进行$T$次，最后$z_T$的分布为：

![QQ截图20200903221029](E:\学术\组会\2020.9.4组会\VAE 总结整理.assets\QQ截图20200903221029.png)

证明：
$$
\begin{align}
q(z_t|x)&=q(\mu_t+\sigma_t*z_{t-1}|x)=q(z_{t-1})\frac{1}{\frac{dz_t}{dz_{t-1}}}\\
&=q(z_{t-1})\frac{1}{\sigma_t}
\end{align}
$$
额外引入一个输入h，与z共同决定下一次迭代所需的$\mu,\sigma$：

![QQ截图20200903221709](E:\学术\组会\2020.9.4组会\VAE 总结整理.assets\QQ截图20200903221709.png)

另外还有一个数值稳定的版本：

![QQ截图20200903221902](E:\学术\组会\2020.9.4组会\VAE 总结整理.assets\QQ截图20200903221902.png)

## Sptial-VAE

包含空间信息的VAE叫做Spatial-VAE，对于空间信息的处理，有不同的方法和观点。

### Spatial-VAE实例

#### eg.1 SPATIAL VARIATIONAL AUTO-ENCODING VIA MATRIX-VARIATE NORMAL DISTRIBUTIONS

VAE中的隐变量是向量，可以解释为大小为1x1的多个特征映射。这种表示只有在与强大的解码器结合时才能隐式地传递空间信息。在这项工作中，提出了使用较大尺寸的特征地图作为潜在变量来显式地捕捉空间信息的spatial-VAE。这是通过允许潜在变量从矩阵变量正态（MVN）分布中采样来实现的，MVN分布的参数由编码器网络计算。为了增加潜在特征图上位置间的依赖性和减少参数的数量，进一步提出了基于低秩MVN分布的空间VAE。实验结果表明，所提出的空间VAEs在捕获丰富的结构和空间信息方面优于原VAEs。

![QQ截图20200903222504](E:\学术\组会\2020.9.4组会\VAE 总结整理.assets\QQ截图20200903222504.png)

若一个矩阵满足矩阵正态分布，我们记作
$$
A\sim\mathcal N_{m,n}(M,\Omega\otimes\Psi)
$$
在VAE的特征图$F$中，每个变量都是相互独立的，所以互相关矩阵是一个对角阵，每一个位置的变量都采样自单变量高斯分布：
$$
F_{i,j}\sim\mathcal N(M(i,j),diag(\Omega\otimes\Psi)_{i,j})
$$
如果有N个独立的特征图，那么

![QQ截图20200903223649](E:\学术\组会\2020.9.4组会\VAE 总结整理.assets\QQ截图20200903223649.png)

所有带有角标$\phi$的都是从编码器中计算得来的。

同样的进行参数重整化的操作：

![QQ截图20200903223904](E:\学术\组会\2020.9.4组会\VAE 总结整理.assets\QQ截图20200903223904.png)

注意到，M是一个d×d的矩阵，共有参数$d^2N$个，$\Omega,\Psi$都是对角阵，共有参数$2dN$个，所以从解码器得到的参数共有$(d^2+2d)N$个。

那么low-rank从哪里来呢？

进行以下的处理：
$$
F_{k}\sim\mathcal N_{d,d}(M(x)_{k\phi},\Omega(x)_{k\phi}\otimes\Psi(x)_{k\phi})\rightarrow \mathcal N_{d,d}(\mu(x)_{k\phi}\nu(x)_{k\phi}^{T},\Omega(x)_{k\phi}\otimes\Psi(x)_{k\phi})
$$
平均矩阵M由外积$\mu(x)_{k\phi}\nu(x)_{k\phi}^{T}$计算。这里，µ和ν分别是m维和n维向量。与通过两个独立矩阵的Kronecker积计算协方差矩阵类似，它显式地强制平均矩阵行列之间的结构相互作用。应用这个低秩公式得到我们的最终模型，通过低秩MVN分布的空间VAEs。

![image-20200903225015281](E:\学术\组会\2020.9.4组会\VAE 总结整理.assets\image-20200903225015281.png)

![image-20200903225043202](E:\学术\组会\2020.9.4组会\VAE 总结整理.assets\image-20200903225043202.png)

#### eg.2  Explicitly disentangling image content from translation and rotation with spatial-VAE

给定一个图像数据集，我们通常感兴趣的是寻找数据生成因素，这些因素独立于姿势变量（如旋转和平移）来编码语义内容。然而，目前的解纠缠方法并没有将任何特定的结构强加给所学的潜在表征。本文提出了一种在变分自动编码器（VAE）框架中明确分离图像旋转和平移与其他非结构化潜在因素的方法。通过将生成模型表示为空间坐标的函数，使重建误差相对于潜在的平移和旋转参数是可微的。这个公式允许我们训练一个神经网络来对这些潜在变量进行近似推理，同时显式地约束它们只表示旋转和平移。spatial-VAE的框架，有效地学习了潜在的表示，将图像旋转和翻译从内容中分离出来。

![image-20200903225511205](E:\学术\组会\2020.9.4组会\VAE 总结整理.assets\image-20200903225511205.png)

在编码的时候，将旋转$\theta$与位移变量$\Delta x$分离出来，保留结构变量z用于后续生成。

我们设$y$是在坐标位置$x$出的输出。

旋转矩阵$R$：
$$
R(\theta)=\begin{pmatrix}cos\theta&sin\theta\\-sin\theta&cos\theta\end{pmatrix}\\
logp(y|z,\theta,\Delta x)=\sum\limits^{n}_{i=1}logp(y^i|x^iR(\theta)+\Delta x,z)
$$
分布假设：
$$
logq(z,\Delta x,\theta)=log \mathcal N(\mu(y),\sigma^2(y)I)\\
z\sim\mathcal N(0,I)\\
\Delta x\sim\mathcal N(0,\sigma^2)\\
\theta\sim\mathcal N(0,s_{\theta}^2),s_{\theta}\rightarrow \infty
$$
所以，有
$$
\mathcal D[q(\theta|y)||p(\theta)]=\frac{1}{2}(log\frac{s_{\theta}^2}{\sigma^2_{\theta}}+(\sigma_{\theta}^2s_{\theta}^{-2})-1)
$$
参考：

> ![image-20200903231733373](E:\学术\组会\2020.9.4组会\VAE 总结整理.assets\image-20200903231733373.png)

对比原文：

> ![image-20200903231752961](E:\学术\组会\2020.9.4组会\VAE 总结整理.assets\image-20200903231752961.png)

损失函数的计算：

对比:

> ![image-20200903232053840](E:\学术\组会\2020.9.4组会\VAE 总结整理.assets\image-20200903232053840.png)

本文的损失函数下界（将$z,\Delta x,\theta$用$\phi$表示）：

> ![image-20200903232145127](E:\学术\组会\2020.9.4组会\VAE 总结整理.assets\image-20200903232145127.png)

若不想将$\Delta x,\theta$的影响分离出来，应他们恒等于0即可。

![image-20200903232548477](E:\学术\组会\2020.9.4组会\VAE 总结整理.assets\image-20200903232548477.png)

![image-20200903232620647](E:\学术\组会\2020.9.4组会\VAE 总结整理.assets\image-20200903232620647.png)

## 文献目录

[1]Welling, M. (n.d.). Auto-Encoding Variational Bayes arXiv : 1312 . 6114v10 [ stat . ML ] 1 May 2014. Ml, 1–14.

[2]Kingma, D. P., Salimans, T., Jozefowicz, R., Chen, X., Sutskever, I., & Welling, M. (2016). Improved variational inference with inverse autoregressive flow. Advances in Neural Information Processing Systems, Nips, 4743–4751.

[3]Bepler, T., Zhong, E. D., Kelley, K., Brignole, E., & Berger, B. (2019). Explicitly disentangling image content from translation and rotation with spatial-VAE. NeurIPS 2019. http://arxiv.org/abs/1909.11663

[4]Doersch, C. (2016). Tutorial on Variational Autoencoders. 1–23. http://arxiv.org/abs/1606.05908

[5]It, B., Learning, M., Autoen-, V., Autoencoder, M. V., Vae, E., Vae, E., & Generative, N. (2020). Spatial Variational Auto-Encoding Via Matrix-Variate Normal Distributions. i, 1–44.

[6]Lopez, R., Regier, J., Cole, M. B., Jordan, M. I., & Yosef, N. (2018). Deep generative modeling for single-cell transcriptomics. Nature Methods, 15(12), 1053–1058. https://doi.org/10.1038/s41592-018-0229-2