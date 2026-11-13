# Homework 1: Uniform Source Mixing Geometry and Multivariable Optimization Methods

**Course:** Blind Source Separation  
**Instructor:** Dr. Saeed Akhavan  
**Author:** Arman Majidi  
**Student ID:** 810100205  
**University:** University of Tehran  
**Semester:** Spring 2023-2024

## 1. Overview & Assignment Structure

This folder contains the **English problem statements, analytical derivations, MATLAB implementations, and numerical results for Homework 1**, together with the original Persian assignment sheet and the submitted Persian report. The assignment covers geometric and statistical properties of instantaneous linear mixtures of uniform sources alongside unconstrained and constrained multivariable optimization methods.

1. **Section 1 (Blind Source Separation of Uniform Mixtures):** Parts `1` through `6` covering observation scatter-plot geometry, automated mixing-matrix estimation from parallelogram boundaries (both extreme-vertex and `polyfit` nearest-neighbor boundary regression), robustness under $15\text{ dB}$ AWGN, analytical marginal PDFs versus $20$-bin histograms, and identifiability from marginal distributions alone.
2. **Section 2 (Multivariable Quadratic Optimization):** Parts `1` through `9` covering 3D `mesh` and 2D `contour` visualization in linear and decibel ($\text{dB}$) scales, gradient and Hessian derivation with a positive-definiteness convexity proof, closed-form unconstrained minimization, Steepest Descent ($\mu=0.01$ and $\mu=0.1$), Newton's method, closed-form Alternating Minimization, unit-circle Gradient Projection ($\|x\|_2=1$), and analytical verification via Lagrange multipliers.

Although the uploaded assignment sheet (`hw1.pdf`) and submitted report (`BSS_HW#1_810100205_Arman_Majidi.pdf`) are in Persian, all problem statements, mathematical derivations, implementation details, and experimental discussions are fully documented below in English so that this folder is completely self-contained.

## 2. Directory Contents

| File | Description |
| --- | --- |
| [codes/section_1.m](./codes/section_1.m) | Noiseless uniform source generation, linear mixing, vertex-based and `polyfit` estimation of $A$, and 20-bin histograms of $x_1$ and $x_2$. |
| [codes/section_1_noiseTest.m](./codes/section_1_noiseTest.m) | Mixing-matrix estimation under $15\text{ dB}$ AWGN using both the extreme-vertex and `polyfit` boundary estimators. |
| [codes/section_2.m](./codes/section_2.m) | 3D mesh and 2D contour plots, Steepest Descent, Newton's method, Alternating Minimization, and Gradient Projection on the unit circle. |
| [hw1.pdf](./hw1.pdf) | Original Persian assignment sheet. |
| [BSS_HW#1_810100205_Arman_Majidi.pdf](./BSS_HW%231_810100205_Arman_Majidi.pdf) | Submitted Persian report containing analytical derivations, figures, and numerical results. |
| `README.md` | English problem statements, mathematical derivations, results summary, and usage guide. |

Keep the existing `codes` subfolder inside `HW#1`.

## 3. Assignment Problem Statement

### 3.1 Section 1: Geometric Blind Source Separation and Marginal Distributions

Generate $T=1000$ samples of a random process $s_1$ uniformly distributed on $[-3,3]$ and $T=1000$ samples of an independent random process $s_2$ uniformly distributed on $[-2,2]$. Combine these two sources via a linear instantaneous mixing model $X=AS$ using the mixing matrix

```math
A=egin{bmatrix}0.6&0.8\0.8&-0.6\end{bmatrix}
```

to generate the observed signals $x_1$ and $x_2$.

#### Part 1 - Observation Scatter Plot and Mixing Geometry

Plot the scatter diagram of $x_2$ versus $x_1$ (`scatter` plot). What is the geometric relationship between this shape and the mixing matrix $A$?

#### Part 2 - Automated Estimation of the Mixing Matrix

Propose a mathematical method, directly related to your answer in Part `1`, to automatically estimate the mixing matrix $A$ in this problem, and implement your method in MATLAB.

#### Part 3 - Estimation in the Presence of Observation Noise

Add low-power noise to the observation matrix $X$ and apply the method proposed in Part `2`. Is your method still applicable in the noisy scenario? If not, modify your method accordingly, report its performance in the presence of noise, and propose a metric demonstrating that the method performs well under noise.

*(Solve Parts `4` through `6` without considering noise.)*

#### Part 4 - Empirical Histogram and Analytical PDF of $x_1$

Plot the histogram of $x_1$ using $20$ bins, and derive the analytical probability density function (PDF) of $x_1$ using statistical and mathematical relations. The derived analytical distribution must match the empirical histogram.

#### Part 5 - Empirical Histogram and Analytical PDF of $x_2$

Repeat Part `4` for the second observed signal $x_2$.

#### Part 6 - Identifiability from Marginal Histograms Alone

Suppose $T$ is sufficiently large that the empirical histograms in Parts `4` and `5` match the exact analytical distributions derived from statistical theory, and suppose we also know the exact source distributions (both their uniformity and their support intervals $[-3,3]$ and $[-2,2]$). Can the BSS problem be solved in this case using the marginal histograms? Explain.

### 3.2 Section 2: Multivariable Quadratic Optimization

Consider the two-variable objective function

```math
f(x)=x_1^2+x_2^2-4x_1-6x_2+13+x_1x_2,
\qquad
x=egin{bmatrix}x_1\x_2\end{bmatrix}\in\mathbb{R}^2.
```

#### Part 1 - 3D Surface Plot in Linear and Decibel Scales

Using the `mesh` command, plot $f(x)$ as a 3D surface over $-10<x_1,x_2<10$. Plot the objective function in decibels ($	ext{dB}$) as well so that variations of the objective function are more clearly visible.

#### Part 2 - 2D Equipotential Contour Plot

Using the `contour` command, plot the equipotential level curves of $f(x)$ in 2D space, and display the numerical values of the level curves directly on the plot (consult `help contour`).

#### Part 3 - Gradient, Hessian, and Convexity Proof

Compute the gradient vector $g(x)$ and the Hessian matrix $H$. Using the resulting Hessian matrix, prove that the given objective function is strictly convex and therefore has a unique minimum (as discussed in lectures, show that the inequality $x^THx>0$ holds for all nonzero vectors $x$).

#### Part 4 - Closed-Form Unconstrained Minimizer

Find the point at which the minimum of $f(x)$ occurs by setting the gradient vector to zero.

#### Part 5 - Steepest Descent Method

Starting from the initial point $x_1=x_2=6$, implement the **Steepest Descent** algorithm for step sizes $\mu=0.01$ and $\mu=0.1$. Does the algorithm converge in both cases, and which one converges faster? Plot the convergence curves (objective value versus iteration number) for both values of $\mu$ on a single figure.

#### Part 6 - Newton's Method

Starting from the initial point $x_1=x_2=6$, implement **Newton's method**. In how many iterations does the algorithm converge? Explain the mathematical reason for this behavior.

#### Part 7 - Alternating Minimization (Coordinate Descent)

Starting from the initial point $x_1=x_2=6$, solve the optimization problem using **Alternating Minimization** between the two variables $x_1$ and $x_2$. At each alternation, a univariate optimization subproblem arises; because this subproblem is simple, use its closed-form solution rather than Steepest Descent or another iterative solver. Plot the convergence trajectory from the initial point to the optimal point in the 2D $(x_1,x_2)$-plane overlaid on the contour plot.

#### Part 8 - Constrained Minimization via Gradient Projection

Add the unit-norm constraint $x_1^2+x_2^2=1$ (or equivalently $\|x\|_2=1$) to the problem. Starting from $x_1=x_2=6$, minimize the objective function using the **Gradient Projection** approach combined with Steepest Descent at $\mu=0.1$, and report the resulting optimal point.

#### Part 9 - Analytical Constrained Solution via Lagrange Multipliers

Using the method of **Lagrange multipliers**, solve Part `8` analytically and show that the numerical point obtained in Part `8` is the true constrained optimal solution.

## 4. Mathematical Formulation & Solutions

### 4.1 Section 1: Geometric Blind Source Separation and Marginal Distributions

#### Part 1 - Observation Scatter Plot and Mixing Geometry

Let $S=[s_1,s_2]^T$ with independent components $s_1\sim\mathcal{U}(-3,3)$ and $s_2\sim\mathcal{U}(-2,2)$. In the $(s_1,s_2)$-plane, the joint support of $S$ is the rectangle $[-3,3]	imes[-2,2]$. Under the linear instantaneous mixing transformation

```math
X=AS
\quad\Longrightarrow\quad
egin{bmatrix}x_1\x_2\end{bmatrix}
=egin{bmatrix}a_{11}&a_{12}\a_{21}&a_{22}\end{bmatrix}
egin{bmatrix}s_1\s_2\end{bmatrix}
=egin{bmatrix}a_{11}\a_{21}\end{bmatrix}s_1
+egin{bmatrix}a_{12}\a_{22}\end{bmatrix}s_2
=egin{bmatrix}a_{11}s_1+a_{12}s_2\a_{21}s_1+a_{22}s_2\end{bmatrix},
```

the rectangular source support is mapped onto a **parallelogram** in the $(x_1,x_2)$-plane whose boundary slopes and scale are governed by the columns of $A$:

- Holding $s_2$ constant and varying $s_1\in[-3,3]$ traces a line parallel to the first mixing vector $a_1=[a_{11},a_{21}]^T=[0.6,0.8]^T$ with slope

```math
m_1=rac{a_{21}}{a_{11}}=rac{0.8}{0.6}=rac{4}{3}pprox1.3333.
```

- Holding $s_1$ constant and varying $s_2\in[-2,2]$ traces a line parallel to the second mixing vector $a_2=[a_{12},a_{22}]^T=[0.8,-0.6]^T$ with slope

```math
m_2=rac{a_{22}}{a_{12}}=rac{-0.6}{0.8}=-rac{3}{4}=-0.75.
```

Evaluating $X=AS$ at the four source corners $(s_1,s_2)\in\{(\pm3,\pm2)\}$ yields the four exact vertices of the observation parallelogram:

```math
egin{aligned}
V_{\max x_1}&=Aegin{bmatrix}3\2\end{bmatrix}=egin{bmatrix}3.4\1.2\end{bmatrix},
&\qquad
V_{\min x_1}&=Aegin{bmatrix}-3\-2\end{bmatrix}=egin{bmatrix}-3.4\-1.2\end{bmatrix},\
V_{\max x_2}&=Aegin{bmatrix}3\-2\end{bmatrix}=egin{bmatrix}0.2\3.6\end{bmatrix},
&\qquad
V_{\min x_2}&=Aegin{bmatrix}-3\2\end{bmatrix}=egin{bmatrix}-0.2\-3.6\end{bmatrix}.
\end{aligned}
```

#### Part 2 - Automated Estimation of the Mixing Matrix

To recover $A$ up to the inherent scaling and permutation ambiguities of BSS, we assume that the spatial mixing vectors have **unit Euclidean norm** ($\|a_k\|_2=1$) and positive first entries. Given an estimated edge slope $\hat{m}_k$, the corresponding unit mixing vector is constructed by normalizing $[1,\hat{m}_k]^T$:

```math
\hat{a}_k=rac{1}{\sqrt{1+\hat{m}_k^2}}egin{bmatrix}1\\hat{m}_k\end{bmatrix}.
```

Two automated algorithms are implemented in `codes/section_1.m`:

**Algorithm 1 (Direct Extreme-Vertex Estimator):**

1. Identify the four empirical extreme observations attaining $\min(x_1)$, $\max(x_1)$, $\min(x_2)$, and $\max(x_2)$:

```math
egin{aligned}
P_{\min x_1}&=ig(x_{1,\min},\ x_2(\mathrm{ind}_{\min x_1})ig),
&\qquad
P_{\max x_1}&=ig(x_{1,\max},\ x_2(\mathrm{ind}_{\max x_1})ig),\
P_{\min x_2}&=ig(x_1(\mathrm{ind}_{\min x_2}),\ x_{2,\min}ig),
&\qquad
P_{\max x_2}&=ig(x_1(\mathrm{ind}_{\max x_2}),\ x_{2,\max}ig).
\end{aligned}
```

2. Compute the slopes of the four parallelogram sides connecting these vertices and average each pair of parallel sides:

```math
egin{aligned}
\hat{m}_1&=rac{1}{2}\left(
rac{x_2(\mathrm{ind}_{\max x_1})-x_{2,\min}}{x_{1,\max}-x_1(\mathrm{ind}_{\min x_2})}
+rac{x_{2,\max}-x_2(\mathrm{ind}_{\min x_1})}{x_1(\mathrm{ind}_{\max x_2})-x_{1,\min}}
ight),\
\hat{m}_2&=rac{1}{2}\left(
rac{x_2(\mathrm{ind}_{\min x_1})-x_{2,\min}}{x_{1,\min}-x_1(\mathrm{ind}_{\min x_2})}
+rac{x_{2,\max}-x_2(\mathrm{ind}_{\max x_1})}{x_1(\mathrm{ind}_{\max x_2})-x_{1,\max}}
ight).
\end{aligned}
```

3. Normalize each vector $[1,\hat{m}_k]$ by its Euclidean norm $\sqrt{1+\hat{m}_k^2}$. A representative single-run output reported in the submission is

```math
oxed{
\hat{A}_{\mathrm{alg1}}pprox
egin{bmatrix}
0.5986&0.8011\
0.8154&-0.5788
\end{bmatrix}.
}
```

**Algorithm 2 (Refined Boundary Regression via `linspace`, `unique`, and `polyfit`):**

1. Locate the four initial extreme points $P_{\min x_1},P_{\max x_1},P_{\min x_2},P_{\max x_2}$ as in Algorithm 1.
2. Discretize each of the four candidate boundary segments connecting these vertices using `linspace` with `line_len` points (`line_len = 100000` in `section_1.m`).
3. For every point on a discretized segment, compute its Euclidean distance to all $T=1000$ observation samples and record the index of the closest observation within `dist_threshold = 0.8`.
4. Deduplicate the matched sample indices along each edge using `unique()`.
5. Fit a first-order line (`polyfit(x_1(ind_line), x_2(ind_line), 1)`) to the unique boundary observations on each of the four sides, average the parallel slopes, and normalize to unit length. A representative single-run output reported in the submission is

```math
oxed{
\hat{A}_{\mathrm{alg2}}pprox
egin{bmatrix}
0.6004&0.7997\
0.8086&-0.5883
\end{bmatrix}.
}
```

#### Part 3 - Estimation in the Presence of Observation Noise

In `codes/section_1_noiseTest.m`, Additive White Gaussian Noise (AWGN) is added to the observations at $\mathrm{SNR}=15	ext{ dB}$ via `X = awgn(X, 15)`.

- **Sensitivity of Algorithm 1:** Because Gaussian noise disturbs the four extreme sample vertices, relying on four single points degrades the slope estimates:

```math
\hat{A}_{\mathrm{alg1,noisy}}pprox
egin{bmatrix}
0.5784&0.8158\
0.7839&-0.6209
\end{bmatrix}.
```

- **Robustness of Algorithm 2:** Using `line_len = 1000` and `dist_threshold = 0.3`, Algorithm 2 fits each boundary line across multiple near-boundary samples via `polyfit`, averaging out individual noise perturbations:

```math
oxed{
\hat{A}_{\mathrm{alg2,noisy}}pprox
egin{bmatrix}
0.5909&0.8068\
0.8086&-0.6120
\end{bmatrix}.
}
```

- **Evaluation metrics:** Using the entrywise absolute error $|\hat{a}_{ij}-a_{ij}|$ or the relative Frobenius error $\|\hat{A}-A\|_F/\|A\|_F$, Algorithm 2 keeps every entry within $0.012$ of its true value under $15	ext{ dB}$ AWGN and reduces the relative Frobenius error from $3.25\%$ (Algorithm 1) to $1.23\%$.

#### Part 4 - Empirical Histogram and Analytical PDF of $x_1$

Because $s_1\sim\mathcal{U}(-3,3)$ and $s_2\sim\mathcal{U}(-2,2)$ are independent, their joint density on $[-3,3]	imes[-2,2]$ is

```math
f_{S_1,S_2}(s_1,s_2)=f_{S_1}(s_1)f_{S_2}(s_2)
=rac{1}{3-(-3)}	imesrac{1}{2-(-2)}
=rac{1}{6}	imesrac{1}{4}=rac{1}{24}.
```

The Jacobian determinant of $X=AS$ satisfies

```math
J=\det(A)=egin{vmatrix}0.6&0.8\0.8&-0.6\end{vmatrix}=-0.36-0.64=-1
\quad\Longrightarrow\quad
|J|=1,
```

and the inverse matrix is $A^{-1}=A$. Therefore, the joint PDF of $(X_1,X_2)$ is uniform on the parallelogram $\mathcal{D}_X$:

```math
f_{X_1,X_2}(x_1,x_2)=rac{f_{S_1,S_2}(s_1,s_2)}{|J|}=rac{1}{24},
\qquad
egin{cases}
s_1=0.6x_1+0.8x_2\in[-3,3],\
s_2=0.8x_1-0.6x_2\in[-2,2].
\end{cases}
```

Writing the support constraints as bounds on $x_2$ for a fixed $x_1$ gives

```math
egin{cases}
-3\leq0.6x_1+0.8x_2\leq3,\
-2\leq0.8x_1-0.6x_2\leq2
\end{cases}
\quad\Longrightarrow\quad
egin{cases}
-3.75-0.75x_1\leq x_2\leq3.75-0.75x_1,\
-\dfrac{10}{3}+\dfrac{4}{3}x_1\leq x_2\leq\dfrac{10}{3}+\dfrac{4}{3}x_1.
\end{cases}
```

Integrating $f_{X_1,X_2}(x_1,x_2)=1/24$ with respect to $x_2$ across the three intervals partitioned by the vertex abscissas $\{-3.4,-0.2,0.2,3.4\}$ yields the trapezoidal marginal PDF:

```math
oxed{
f_{X_1}(x_1)=\int_{-\infty}^{\infty}f_{X_1,X_2}(x_1,x_2)\,dx_2=
egin{cases}
\dfrac{1}{24}\left(\dfrac{85}{12}+\dfrac{25}{12}x_1ight)pprox\dfrac{1}{24}(7.08+2.08x_1),&-3.4<x_1<-0.2,\[8pt]
\dfrac{1}{24}\left(\dfrac{20}{3}ight)=\dfrac{5}{18}pprox0.2778,&-0.2\leq x_1\leq0.2,\[8pt]
\dfrac{1}{24}\left(\dfrac{85}{12}-\dfrac{25}{12}x_1ight)pprox\dfrac{1}{24}(7.08-2.08x_1),&0.2<x_1<3.4,\[6pt]
0,&	ext{otherwise.}
\end{cases}
}
```

#### Part 5 - Empirical Histogram and Analytical PDF of $x_2$

Expressing the support constraints as bounds on $x_1$ for a fixed $x_2$ gives

```math
egin{cases}
-3\leq0.6x_1+0.8x_2\leq3,\
-2\leq0.8x_1-0.6x_2\leq2
\end{cases}
\quad\Longrightarrow\quad
egin{cases}
-5-\dfrac{4}{3}x_2\leq x_1\leq5-\dfrac{4}{3}x_2,\[6pt]
-2.5+0.75x_2\leq x_1\leq2.5+0.75x_2.
\end{cases}
```

Integrating $f_{X_1,X_2}(x_1,x_2)=1/24$ with respect to $x_1$ across the intervals partitioned by the vertex ordinates $\{-3.6,-1.2,1.2,3.6\}$ yields the trapezoidal marginal PDF:

```math
oxed{
f_{X_2}(x_2)=\int_{-\infty}^{\infty}f_{X_1,X_2}(x_1,x_2)\,dx_1=
egin{cases}
\dfrac{1}{24}\left(7.5+\dfrac{25}{12}x_2ight)pprox\dfrac{1}{24}(7.5+2.08x_2),&-3.6<x_2<-1.2,\[8pt]
\dfrac{5}{24}pprox0.2083,&-1.2\leq x_2\leq1.2,\[8pt]
\dfrac{1}{24}\left(7.5-\dfrac{25}{12}x_2ight)pprox\dfrac{1}{24}(7.5-2.08x_2),&1.2<x_2<3.6,\[6pt]
0,&	ext{otherwise.}
\end{cases}
}
```

#### Part 6 - Identifiability from Marginal Histograms Alone

For each channel $i\in\{1,2\}$, define the scaled variables $s_3=a_{i1}s_1$ and $s_4=a_{i2}s_2$. By independence of $s_1$ and $s_2$, the cumulative distribution function (CDF) of $X_i=s_3+s_4$ is

```math
egin{aligned}
F_{X_i}(x_i)&=\Pr\{a_{i1}s_1+a_{i2}s_2<x_i\}\
&=\int_{-\infty}^{\infty}f_{S_4}(s_4)\left(\int_{-\infty}^{x_i-s_4}f_{S_3}(s_3)\,ds_3ight)ds_4,
\end{aligned}
```

so the marginal PDF is the **convolution** of two zero-mean uniform densities:

```math
oxed{
f_{X_i}(x_i)=ig(f_{S_3}*f_{S_4}ig)(x_i).
}
```

Although the corner breakpoints of each trapezoidal histogram determine the magnitudes $|a_{i1}|$ and $|a_{i2}|$ up to intra-row ordering, **the BSS problem cannot be uniquely solved from the marginal histograms alone**:

1. **Sign ambiguity:** Because $s_1$ and $s_2$ are symmetric about zero, the marginal PDFs depend only on $|a_{ij}|$. The signs of the entries of $A$ (for example, distinguishing $a_{22}=-0.6$ from $+0.6$) cannot be recovered without joint cross-channel information.
2. **Permutation ambiguity:** Convolution is commutative ($f_{S_3}*f_{S_4}=f_{S_4}*f_{S_3}$), so the marginal distributions alone do not fix the pairing of coefficients across the two rows.

### 4.2 Section 2: Multivariable Quadratic Optimization

#### Parts 1 and 2 - 3D Mesh and 2D Contour Visualizations

In `codes/section_2.m`, $f(x_1,x_2)$ is evaluated over a uniform grid on $[-10,10]^2$ with step size $0.02$ (`meshgrid(-10:0.02:10)`). Because the minimum of $f(x)$ is strictly positive ($f(x^*)=11/3>0$), its decibel representation

```math
f_{\mathrm{dB}}(x_1,x_2)=10\log_{10}f(x_1,x_2)
```

is real and finite everywhere, compressing large values near the boundary while sharpening the elliptic level curves near the minimizer.

#### Part 3 - Gradient, Hessian, and Convexity Proof

Differentiating $f(x)=x_1^2+x_2^2-4x_1-6x_2+13+x_1x_2$ gives the gradient vector and Hessian matrix:

```math
oxed{
g(x)=
abla f(x)=
egin{bmatrix}
\dfrac{\partial f}{\partial x_1}\[8pt]
\dfrac{\partial f}{\partial x_2}
\end{bmatrix}
=egin{bmatrix}
2x_1+x_2-4\
x_1+2x_2-6
\end{bmatrix},
\qquad
H=
abla^2f(x)=
egin{bmatrix}
2&1\
1&2
\end{bmatrix}.
}
```

For any nonzero vector $x=[x_1,x_2]^T\in\mathbb{R}^2$,

```math
x^THx
=egin{bmatrix}x_1&x_2\end{bmatrix}
egin{bmatrix}2&1\1&2\end{bmatrix}
egin{bmatrix}x_1\x_2\end{bmatrix}
=2(x_1^2+x_1x_2+x_2^2).
```

Viewing $x_1^2+x_1x_2+x_2^2=0$ as a quadratic equation in $x_1$, its discriminant is

```math
\Delta=x_2^2-4x_2^2=-3x_2^2<0\qquad (x_2
eq0),
```

and completing the square gives $2ig[(x_1+x_2/2)^2+rac{3}{4}x_2^2ig]>0$ for all $x
eq0$. Thus $x^THx>0$ for all $x
eq0$, proving that $H\succ0$ and that $f(x)$ is **strictly convex** with a unique global minimizer.

#### Part 4 - Closed-Form Unconstrained Minimizer

Setting $
abla f(x)=0$ yields

```math
egin{cases}
2x_1+x_2-4=0,\
x_1+2x_2-6=0
\end{cases}
\quad\Longrightarrow\quad
oxed{
x^*=egin{bmatrix}2/3\8/3\end{bmatrix}
pproxegin{bmatrix}0.6667\2.6667\end{bmatrix},
\qquad
f(x^*)=rac{11}{3}pprox3.6667.
}
```

#### Part 5 - Steepest Descent Method

Starting from $x_0=[6,6]^T$, the Steepest Descent recursion is

```math
x_{k+1}=x_k-\mu
abla f(x_k),
```

with stopping condition $|f(x_{k+1})-f(x_k)|<10^{-6}$. Because the eigenvalues of $H$ are $\lambda_{\min}=1$ and $\lambda_{\max}=3$, constant-step Steepest Descent converges whenever $0<\mu<2/\lambda_{\max}=2/3$. Both $\mu=0.1$ and $\mu=0.01$ lie inside this range:

- For $\mu=0.1$, the algorithm converges in **$59$ iterations** to $xpprox[0.6687,2.6647]^T$.
- For $\mu=0.01$, the algorithm converges in **$494$ iterations** to $xpprox[0.6736,2.6597]^T$.

#### Part 6 - Newton's Method

The Newton update rule is

```math
x_{k+1}=x_k-H^{-1}
abla f(x_k).
```

Because $f(x)$ is an exact second-degree quadratic polynomial with constant positive definite Hessian $H$, its gradient satisfies $
abla f(x_0)=H(x_0-x^*)$. Consequently,

```math
x_1=x_0-H^{-1}H(x_0-x^*)=x^*=egin{bmatrix}2/3\8/3\end{bmatrix},
```

and the algorithm reaches the exact global minimizer in **$1$ iteration**.

#### Part 7 - Alternating Minimization (Coordinate Descent)

Setting each partial derivative to zero while holding the other coordinate fixed gives the closed-form coordinate updates

```math
egin{aligned}
x_1^{\mathrm{new}}&=2-rac{1}{2}x_2\qquad (	ext{minimizing over }x_1	ext{ for fixed }x_2),\
x_2^{\mathrm{new}}&=3-rac{1}{2}x_1\qquad (	ext{minimizing over }x_2	ext{ for fixed }x_1).
\end{aligned}
```

Alternating between these two updates from $x_0=[6,6]^T$ generates an orthogonal staircase trajectory on the contour plot and satisfies $|f(x_{k+1})-f(x_k)|<10^{-6}$ in **$14$ iterations** at

```math
oxed{
x_{\mathrm{AM}}pproxegin{bmatrix}0.6663\2.6669\end{bmatrix}.
}
```

#### Part 8 - Constrained Minimization via Gradient Projection

To enforce $x_1^2+x_2^2=1$ ($\|x\|_2=1$), each Steepest Descent step ($\mu=0.1$) is followed by Euclidean normalization onto the unit circle:

```math
	ilde{x}_{k+1}=x_k-\mu
abla f(x_k),
\qquad
x_{k+1}=rac{	ilde{x}_{k+1}}{\|	ilde{x}_{k+1}\|_2}.
```

Under $|f(x_{k+1})-f(x_k)|<10^{-6}$, the algorithm converges in **$15$ iterations** to

```math
oxed{
x_{\mathrm{GP}}^*pproxegin{bmatrix}0.4946\0.8691\end{bmatrix},
\qquad
f(x_{\mathrm{GP}}^*)pprox7.2375.
}
```

#### Part 9 - Analytical Constrained Solution via Lagrange Multipliers

With the constraint $g(x)=x_1^2+x_2^2=1$, the Lagrange multiplier condition $
abla f(x)=\gamma
abla g(x)$ gives

```math
egin{cases}
2x_1-4+x_2=\gamma(2x_1),\
2x_2-6+x_1=\gamma(2x_2),\
x_1^2+x_2^2=1.
\end{cases}
```

Assuming $x_1
eq0$ and $x_2
eq0$, dividing the two stationarity equations by $2x_1$ and $2x_2$, respectively, eliminates $\gamma$:

```math
egin{cases}
1-\dfrac{2}{x_1}+\dfrac{x_2}{2x_1}=\gamma,\[8pt]
1-\dfrac{3}{x_2}+\dfrac{x_1}{2x_2}=\gamma
\end{cases}
\quad\Longrightarrow\quad
rac{2}{x_1}-rac{x_2}{2x_1}=rac{3}{x_2}-rac{x_1}{2x_2}.
```

Solving this equation simultaneously with $x_1^2+x_2^2=1$ yields two real stationary points on the unit circle and a complex conjugate pair ($x_1pprox3.047\pm j1.893$, which is rejected):

```math
egin{cases}
x_1=0.494,\ x_2=0.869\quad\Longrightarrow\quad f=7.237\quad (	ext{global minimizer}),\
x_1=-0.589,\ x_2=-0.807\quad\Longrightarrow\quad f=21.68\quad (	ext{global maximizer}).
\end{cases}
```

Thus the analytical constrained minimizer is

```math
oxed{
x^*=egin{bmatrix}0.494\0.869\end{bmatrix},
\qquad
f(x^*)=7.237,
}
```

which agrees with the Gradient Projection result from Part `8`.

## 5. Results & Discussion

### 5.1 Summary of Results

| Section & Part | Method / Problem | Main Result |
| --- | --- | --- |
| Section 1, Part 1 | Scatter Plot Geometry | Parallelogram with edge slopes $m_1=4/3pprox1.3333$ and $m_2=-3/4=-0.75$. |
| Section 1, Part 2 | Algorithm 1 (Extreme Vertices, Noiseless) | $\hat{A}pproxegin{bmatrix}0.5986&0.8011\0.8154&-0.5788\end{bmatrix}$ ($\|\hat{A}-A\|_F/\|A\|_F=1.87\%$). |
| Section 1, Part 2 | Algorithm 2 (`polyfit` Boundary, Noiseless) | $\hat{A}pproxegin{bmatrix}0.6004&0.7997\0.8086&-0.5883\end{bmatrix}$ ($\|\hat{A}-A\|_F/\|A\|_F=1.03\%$). |
| Section 1, Part 3 | Algorithm 1 ($15	ext{ dB}$ AWGN) | $\hat{A}pproxegin{bmatrix}0.5784&0.8158\0.7839&-0.6209\end{bmatrix}$ ($\|\hat{A}-A\|_F/\|A\|_F=3.25\%$). |
| Section 1, Part 3 | Algorithm 2 ($15	ext{ dB}$ AWGN) | $\hat{A}pproxegin{bmatrix}0.5909&0.8068\0.8086&-0.6120\end{bmatrix}$ ($\|\hat{A}-A\|_F/\|A\|_F=1.23\%$). |
| Section 1, Part 4 | Marginal PDF $f_{X_1}(x_1)$ | Symmetric trapezoid on $[-3.4,3.4]$ with plateau $[-0.2,0.2]$ at density $5/18pprox0.2778$. |
| Section 1, Part 5 | Marginal PDF $f_{X_2}(x_2)$ | Symmetric trapezoid on $[-3.6,3.6]$ with plateau $[-1.2,1.2]$ at density $5/24pprox0.2083$. |
| Section 1, Part 6 | BSS from Histograms Alone | Not uniquely solvable; $f_{X_i}=f_{S_3}*f_{S_4}$ leaves sign and permutation ambiguities. |
| Section 2, Parts 3–4 | Convexity & Unconstrained Optimum | $H=egin{bmatrix}2&1\1&2\end{bmatrix}\succ0$; unique minimizer $x^*=[2/3,8/3]^Tpprox[0.6667,2.6667]^T$, $f(x^*)=11/3$. |
| Section 2, Part 5 | Steepest Descent ($\mu=0.1$ vs. $\mu=0.01$) | $\mu=0.1$: $59$ iterations ($[0.6687,2.6647]^T$); $\mu=0.01$: $494$ iterations ($[0.6736,2.6597]^T$). |
| Section 2, Part 6 | Newton's Method | Converges in $1$ iteration to $[0.6667,2.6667]^T$ because $f(x)$ is quadratic. |
| Section 2, Part 7 | Alternating Minimization | Converges in $14$ coordinate updates to $[0.6663,2.6669]^T$. |
| Section 2, Parts 8–9 | Unit-Circle Constrained Minimum ($\|x\|_2=1$) | Gradient Projection ($15$ iterations): $[0.4946,0.8691]^T$; Lagrange multipliers: $[0.494,0.869]^T$, $fpprox7.237$. |

### 5.2 Notes on the Submitted Solutions

- **Decimal rounding in Section 1, Parts 3–5:** On page 5 of the Persian report, the entrywise deviation of $\hat{A}_{\mathrm{alg2,noisy}}$ is described in text as `0.1`, which is a typographical slip for **`0.01`** (the actual entrywise errors are between $0.0068$ and $0.0120$). In Parts `4` and `5`, rounding $10/3pprox3.33$ and $4/3pprox1.33$ produces $6.66/24=0.2775$ instead of $5/18pprox0.2778$, and on page 9 the lower bound $-5-1.33x_2$ is printed without its minus sign in the boxed display, while the plotted trapezoids and exact breakpoints $\{-3.4,-0.2,0.2,3.4\}$ and $\{-3.6,-1.2,1.2,3.6\}$ are exact.
- **CDF vs. PDF notation in Section 1, Part 6:** On page 10 of the report, the convolution is written as $F_{X_i}(x_i)=f_{s_4}(x_i)*f_{s_3}(x_i)$; strictly speaking, differentiating the CDF integral gives the marginal density $f_{X_i}(x_i)=(f_{s_4}*f_{s_3})(x_i)$.

## 6. How to Use

### 6.1 Run the Supplied Scripts

Open MATLAB, set the current folder to `HW#1/codes`, and run any of the three driver scripts:

```matlab
section_1
section_1_noiseTest
section_2
```

### 6.2 Expected Console Output

Running `section_2.m` produces the following deterministic console output:

```text
S.D: final value of x for mu = 0.10 is [0.668660, 2.664674] in iteration: 59
S.D: final value of x for mu = 0.01 is [0.673646, 2.659688] in iteration: 494

Newton: final value of x for is [0.666667, 2.666667] in iteration: 1

Alternation Minimization: final value of x for is [0.666260, 2.666870] in iteration: 14

G.P: final value of x for mu = 0.10 is [0.494565, 0.869141] in iteration: 15
```

Because `section_1.m` and `section_1_noiseTest.m` generate fresh uniform samples via `unifrnd` without a fixed random seed, their printed estimates `A_pred` fluctuate slightly across runs around the true matrix $A=egin{bmatrix}0.6&0.8\0.8&-0.6\end{bmatrix}$.

## 7. References

1. Akhavan, S. *Blind Source Separation: Homework 1*. Course assignment sheet, University of Tehran (`hw1.pdf`).
2. Comon, P., & Jutten, C. (2010). *Handbook of Blind Source Separation: Independent Component Analysis and Applications*. Academic Press.
3. Chong, E. K. P., & Zak, S. H. (2001). *An Introduction to Optimization* (2nd ed.). Wiley.
4. [GitHub Docs: Writing mathematical expressions](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions).
