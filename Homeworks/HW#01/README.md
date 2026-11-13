# Homework 1: Geometric Blind Source Separation of Uniform Sources and Multivariable Optimization Methods

**Course:** Blind Source Separation

**Instructor:** Dr. Saeed Akhavan

**Author:** Arman Majidi

**Student ID:** 810100205

**University:** University of Tehran

**Semester:** Spring 2023-2024

## 1. Overview & Assignment Structure

This folder contains the **complete English problem statements, mathematical derivations, MATLAB implementations, and numerical results for Homework 1**, together with the original Persian assignment sheet (`hw1.pdf`) and the submitted Persian report (`BSS_HW#1_810100205_Arman_Majidi.pdf`). The assignment is organized into two main parts covering foundational geometric/statistical concepts in blind source separation (BSS) and core multivariable optimization methods:

1. **Section 1 — Geometric Blind Source Separation and Marginal Distributions of Uniform Mixtures:** Generates two independent uniformly distributed random processes $s_1(t)$ and $s_2(t)$ with $T = 1000$ samples, mixes them instantaneously via a $2 \times 2$ mixing matrix $A$, analyzes the parallelogram geometry of the observation scatter plot, implements two automated boundary-slope algorithms (an extreme-vertex estimator and a `polyfit` nearest-neighbor boundary regression estimator) to recover $A$ in both noiseless and noisy ($\mathrm{SNR} = 15\text{ dB}$ AWGN) settings, derives the exact piecewise-trapezoidal marginal probability density functions (PDFs) $f_{X_1}(x_1)$ and $f_{X_2}(x_2)$ to compare against 20-bin empirical histograms, and analyzes whether marginal histograms alone suffice to uniquely solve the BSS problem.

2. **Section 2 — Unconstrained and Unit-Circle Constrained Quadratic Optimization:** Visualizes a two-variable quadratic objective function $f(x_1, x_2)$ via 3D `mesh` and 2D `contour` plots in both linear and decibel ($\text{dB}$) scales, computes its gradient vector $g(x)$ and Hessian matrix $H$, proves strict convexity via positive definiteness ($x^T H x > 0$), finds the closed-form stationary point, and compares iterative minimization starting from $x_0 = [6, 6]^T$ using **Steepest Descent** ($\mu = 0.01$ and $\mu = 0.1$), **Newton's method**, and closed-form **Alternating Minimization** (coordinate descent). Finally, imposes the unit-circle constraint $\Vert{}x\Vert{}_2 = 1$ ($x_1^2 + x_2^2 = 1$), solves the constrained problem numerically via **Gradient Projection**, and verifies the result analytically using **Lagrange multipliers**.

Because both `hw1.pdf` and `BSS_HW#1_810100205_Arman_Majidi.pdf` are written in Persian, this `README.md` provides a self-contained English reference covering every problem statement, analytical proof, algorithmic step, and experimental finding in the submission.

## 2. Directory Contents

| File | Description | 
| ----- | ----- | 
| [hw1.pdf](https://www.google.com/search?q=./hw1.pdf) | Original Persian assignment sheet specifying Section 1 (Parts 1–6) and Section 2 (Parts 1–9). | 
| [BSS_HW#1_810100205_Arman_Majidi.pdf](https://www.google.com/search?q=./BSS_HW%25231_810100205_Arman_Majidi.pdf) | Submitted Persian report containing analytical derivations, scatter/histogram/contour figures, and numerical results. | 
| [codes/section_1.m](https://www.google.com/search?q=./codes/section_1.m) | MATLAB script for Section 1 (Parts 1, 2, 4, and 5): noiseless source generation, scatter plot, extreme-vertex and `polyfit` mixing-matrix estimation, and 20-bin histograms of $x_1$ and $x_2$. | 
| [codes/section_1_noiseTest.m](https://www.google.com/search?q=./codes/section_1_noiseTest.m) | MATLAB script for Section 1 (Part 3): mixing-matrix estimation under $15\text{ dB}$ AWGN using both the extreme-vertex and `polyfit` boundary estimators. | 
| [codes/section_2.m](https://www.google.com/search?q=./codes/section_2.m) | MATLAB script for Section 2 (Parts 1, 2, 5, 6, 7, and 8): 3D mesh and 2D contour plots (linear and $\text{dB}$ scales), Steepest Descent, Newton's method, Alternating Minimization, and Gradient Projection on the unit circle. | 
| `README.md` | English problem statements, mathematical derivations, implementation details, results summary, and usage guide. | 

Keep all files in the `HW#1` folder with the `codes` subfolder preserved.

## 3. Assignment Problem Statement

### 3.1 General Instructions

* Exercises may be implemented in either **MATLAB** or **Python**.

* Submit both the written report and the source scripts (`.m` files or Python notebooks).

* The report must be cleanly typeset in Persian (handwritten mathematical derivations may be inserted as clean images if desired).

* When using MATLAB, write standard `.m` scripts rather than Live Scripts (`.mlx`).

### 3.2 Section 1: Geometric Blind Source Separation and Marginal Distributions of Uniform Mixtures

Generate $T = 1000$ samples of a random process $s_1$ uniformly distributed on $[-3, 3]$ and $T = 1000$ samples of an independent random process $s_2$ uniformly distributed on $[-2, 2]$. Combine the two sources via a linear instantaneous mixing model $X = AS$ using the mixing matrix

```
A = \begin{bmatrix} 0.6 & 0.8 \\ 0.8 & -0.6 \end{bmatrix}

```

to produce the observed mixtures $x_1$ and $x_2$.

1. **Part 1 (Observation Scatter Plot & Geometry):** Plot the scatter diagram of $x_2$ versus $x_1$. Explain how the geometric shape of this scatter plot relates to the mixing matrix $A$.

2. **Part 2 (Automated Mixing-Matrix Estimation):** Based on the geometric relationship from Part 1, propose a mathematical method to automatically estimate the mixing matrix $A$ from the observations, and implement your method in code.

3. **Part 3 (Robustness in the Presence of Noise):** Add low-power noise to the observation matrix $X$ and apply the estimation method proposed in Part 2. Does your method remain effective in the noisy scenario? If not, modify the method so that it works reliably under noise, report its performance, and propose a quantitative metric demonstrating that your method performs well in the presence of noise.

4. **Part 4 (Empirical Histogram & Analytical PDF of** $x_1$**, Noiseless):** Ignoring noise for Parts 4–6, plot the histogram of $x_1$ using $20$ bins. Using probability theory and statistical transformations, derive the exact analytical probability density function (PDF) of $x_1$ and verify that it matches the empirical histogram.

5. **Part 5 (Empirical Histogram & Analytical PDF of** $x_2$**, Noiseless):** Repeat Part 4 for the second observation $x_2$.

6. **Part 6 (Identifiability from Marginal Histograms Alone):** Suppose the sample size $T$ is sufficiently large that the empirical histograms in Parts 4 and 5 match the exact analytical marginal PDFs derived from probability theory, and suppose we also know the exact source distributions (both their uniformity and their support intervals $[-3, 3]$ and $[-2, 2]$). Under these assumptions, can the BSS problem be solved uniquely using only the marginal histograms of $x_1$ and $x_2$? Explain your reasoning.

### 3.3 Section 2: Multivariable Optimization and Constrained Minimization

Consider the two-variable quadratic objective function

```
f(x) = x_1^2 + x_2^2 - 4x_1 - 6x_2 + 13 + x_1 x_2,
\qquad
x = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \in \mathbb{R}^2.

```

1. **Part 1 (3D Surface Plot):** Using MATLAB's `mesh` command, plot $f(x_1, x_2)$ in 3D over the domain $-10 < x_1, x_2 < 10$. Plot the objective in decibels ($\text{dB}$) as well so that variations near the minimum are clearly visible.

2. **Part 2 (2D Contour Plot):** Using the `contour` command, plot the equipotential level curves of $f(x_1, x_2)$ in 2D and display the numerical level values directly on the contour lines (consult `help contour`).

3. **Part 3 (Gradient, Hessian, and Convexity Proof):** Compute the gradient vector $g(x) = \nabla f(x)$ and the Hessian matrix $H = \nabla^2 f(x)$. Using the Hessian matrix, prove that $f(x)$ is strictly convex ($x^T H x > 0$ for all nonzero vectors $x \in \mathbb{R}^2$) and therefore possesses a unique global minimum.

4. **Part 4 (Closed-Form Unconstrained Minimizer):** Find the unconstrained minimizer $x^*$ analytically by setting the gradient vector $g(x)$ to zero.

5. **Part 5 (Steepest Descent):** Starting from the initial point $x_1 = x_2 = 6$ ($x_0 = [6, 6]^T$), implement the **Steepest Descent** algorithm for step sizes $\mu = 0.01$ and $\mu = 0.1$. Does the algorithm converge in both cases, and which step size converges faster? Plot the convergence curves (objective value versus iteration number) for both step sizes on the same figure.

6. **Part 6 (Newton's Method):** Starting from $x_1 = x_2 = 6$, implement **Newton's method** to minimize $f(x)$. In how many iterations does the algorithm converge? Explain the mathematical reason for this behavior.

7. **Part 7 (Alternating Minimization):** Starting from $x_1 = x_2 = 6$, solve the unconstrained minimization problem using **Alternating Minimization** (coordinate descent) between $x_1$ and $x_2$. At each alternation, one variable is held fixed while optimizing over the other; because each univariate subproblem is a simple quadratic, use its closed-form minimizer rather than an inner iterative solver. Plot the optimization trajectory from the initial point to the optimum overlaid on the 2D contour plot in the $(x_1, x_2)$-plane.

8. **Part 8 (Projected Gradient Descent on the Unit Circle):** Impose the unit-norm equality constraint $x_1^2 + x_2^2 = 1$ (i.e., $\Vert{}x\Vert{}_2 = 1$). Starting from $x_1 = x_2 = 6$, minimize $f(x)$ subject to $\Vert{}x\Vert{}_2 = 1$ using the **Gradient Projection** approach combined with Steepest Descent at step size $\mu = 0.1$, and report the resulting optimal point.

9. **Part 9 (Analytical Solution via Lagrange Multipliers):** Solve the unit-circle constrained optimization problem from Part 8 analytically using the **method of Lagrange multipliers**, and verify that the analytical minimizer agrees with the numerical solution obtained in Part 8.

*(Note: As stated at the end of the assignment sheet, running any of the iterative algorithms above from different initial points converges to the same optimal solution.)*

## 4. Mathematical Formulation & Solutions

### 4.1 Section 1: Linear Mixing Geometry and Boundary-Slope Estimation

#### 4.1.1 Parallelogram Mapping of Independent Uniform Sources (Part 1)

Let $S = [s_1, s_2]^T$ denote the source vector with independent components $s_1 \sim \mathcal{U}(-3, 3)$ and $s_2 \sim \mathcal{U}(-2, 2)$. In the source plane $(s_1, s_2)$, the joint support of $S$ is the axis-aligned rectangle

```
\mathcal{R}_S = [-3, 3] \times [-2, 2],

```

whose four corners are $(\pm 3, \pm 2)$. Under the linear instantaneous mixing model

```
X = AS \implies
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
= \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix}
\begin{bmatrix} s_1 \\ s_2 \end{bmatrix}
= \begin{bmatrix} a_{11} \\ a_{21} \end{bmatrix} s_1
+ \begin{bmatrix} a_{12} \\ a_{22} \end{bmatrix} s_2
= \begin{bmatrix} a_{11} s_1 + a_{12} s_2 \\ a_{21} s_1 + a_{22} s_2 \end{bmatrix},

```

the rectangle $\mathcal{R}_S$ is mapped into a **parallelogram** $\mathcal{D}_X = A(\mathcal{R}_S)$ in the observation plane $(x_1, x_2)$. Specifically, for the given matrix

```
A = \begin{bmatrix} 0.6 & 0.8 \\ 0.8 & -0.6 \end{bmatrix},

```

note that $A$ is both **symmetric** ($A^T = A$) and **orthogonal** ($A^T A = I_2$, $\det(A) = -0.36 - 0.64 = -1$, and $A^{-1} = A$). Its columns $a_1 = [0.6, 0.8]^T$ and $a_2 = [0.8, -0.6]^T$ are orthonormal ($\Vert{}a_1\Vert{}_2 = \Vert{}a_2\Vert{}_2 = 1$, $a_1^T a_2 = 0$), so $\mathcal{D}_X$ is a rotated/reflected rectangle in $\mathbb{R}^2$:

* **Fixing** $s_1$ **and varying** $s_2 \in [-2, 2]$**:** Moves along the second mixing vector $a_2 = [a_{12}, a_{22}]^T$, producing two parallel boundary edges ($s_1 = \pm 3$) with slope

```
m^{(2)} = \frac{\Delta x_2}{\Delta x_1}\bigg|_{s_1 = \text{const}}
= \frac{a_{22}}{a_{12}} = \frac{-0.6}{0.8} = -\frac{3}{4} = -0.75.

```

* **Fixing** $s_2$ **and varying** $s_1 \in [-3, 3]$**:** Moves along the first mixing vector $a_1 = [a_{11}, a_{21}]^T$, producing two parallel boundary edges ($s_2 = \pm 2$) with slope

```
m^{(1)} = \frac{\Delta x_2}{\Delta x_1}\bigg|_{s_2 = \text{const}}
= \frac{a_{21}}{a_{11}} = \frac{0.8}{0.6} = \frac{4}{3} \approx 1.3333.

```

*(Note on page 2 of the Persian report: the bulleted text interchanges the labels* $a_{21}/a_{11}$ *and* $a_{22}/a_{12}$ *when referring to constant* $s_1$ *versus constant* $s_2$*, but the set of edge slopes* $\{a_{21}/a_{11}, a_{22}/a_{12}\}$ *and the subsequent algorithm are unaffected.)*

Evaluating $X = AS$ at the four corners $(s_1, s_2) \in \{(\pm 3, \pm 2)\}$ gives the exact vertices of the observation parallelogram $\mathcal{D}_X$:

```
\begin{aligned}
V_{\max x_1} &= A\begin{bmatrix} 3 \\ 2 \end{bmatrix} = \begin{bmatrix} 3.4 \\ 1.2 \end{bmatrix}, &
V_{\min x_1} &= A\begin{bmatrix} -3 \\ -2 \end{bmatrix} = \begin{bmatrix} -3.4 \\ -1.2 \end{bmatrix}, \\[4pt]
V_{\max x_2} &= A\begin{bmatrix} 3 \\ -2 \end{bmatrix} = \begin{bmatrix} 0.2 \\ 3.6 \end{bmatrix}, &
V_{\min x_2} &= A\begin{bmatrix} -3 \\ 2 \end{bmatrix} = \begin{bmatrix} -0.2 \\ -3.6 \end{bmatrix}.
\end{aligned}

```

#### 4.1.2 Automated Mixing-Matrix Estimation Algorithms (Parts 2 & 3)

Because each column of $A$ is determined up to scale and sign by the slope of its corresponding pair of parallel parallelogram edges, resolving the **scale ambiguity** requires a normalization convention. Assuming each spatial mixing vector has **unit Euclidean norm** ($\Vert{}a_k\Vert{}_2 = 1$) and positive first coordinate ($a_{1k} > 0$), a slope $m_k$ uniquely determines the unit vector

```
\hat{a}_k = \frac{1}{\sqrt{1 + m_k^2}} \begin{bmatrix} 1 \\ m_k \end{bmatrix}.

```

*(In `codes/section_1.m`, the unnormalized vectors* $[1, m_k]$ *are stored in the rows of `A_pred` and normalized row-wise; because the true* $A$ *is symmetric with* $a_{12} = a_{21} = 0.8 > 0$*, row-wise and column-wise normalization produce the exact same matrix up to transpose.)*

**Algorithm 1 — Direct Extreme-Vertex Slope Estimator (Part 2 Baseline):**

1. Find the indices of the four sample extrema $\min(x_1)$, $\max(x_1)$, $\min(x_2)$, and $\max(x_2)$ among the $T = 1000$ observations to obtain four empirical corner estimates:

```
\begin{aligned}
P_{\min x_1} &= \big(x_{1,\min},\ x_2(\text{ind}_{\min x_1})\big), &
P_{\max x_1} &= \big(x_{1,\max},\ x_2(\text{ind}_{\max x_1})\big), \\
P_{\min x_2} &= \big(x_1(\text{ind}_{\min x_2}),\ x_{2,\min}\big), &
P_{\max x_2} &= \big(x_1(\text{ind}_{\max x_2}),\ x_{2,\max}\big).
\end{aligned}

```

2. Compute the slopes of the two pairs of opposite edges and average each parallel pair:

```
\begin{aligned}
\hat{m}_1 &= \frac{1}{2}\left(
\frac{x_2(\text{ind}_{\max x_1}) - x_{2,\min}}{x_{1,\max} - x_1(\text{ind}_{\min x_2})}
+ \frac{x_{2,\max} - x_2(\text{ind}_{\min x_1})}{x_1(\text{ind}_{\max x_2}) - x_{1,\min}}
\right), \\[6pt]
\hat{m}_2 &= \frac{1}{2}\left(
\frac{x_2(\text{ind}_{\min x_1}) - x_{2,\min}}{x_{1,\min} - x_1(\text{ind}_{\min x_2})}
+ \frac{x_{2,\max} - x_2(\text{ind}_{\max x_1})}{x_1(\text{ind}_{\max x_2}) - x_{1,\max}}
\right).
\end{aligned}

```

3. Normalize $[1, \hat{m}_1]$ and $[1, \hat{m}_2]$ to unit Euclidean norm.

**Algorithm 2 — Refined Boundary-Regression Estimator via `linspace`, `unique`, and `polyfit` (Parts 2 & 3):**

Because Algorithm 1 relies solely on four extreme sample points, finite-sample variance near the corners—and especially unbounded outliers when Additive White Gaussian Noise (AWGN) is present—perturbs the vertex coordinates. Algorithm 2 (`slope_finder` in `section_1.m` and `section_1_noiseTest.m`) refines each edge slope by fitting a least-squares line to all boundary observations near the initial segment:

1. Obtain the four initial extreme points $P_{\min x_1}, P_{\max x_1}, P_{\min x_2}, P_{\max x_2}$ as in Step 1 of Algorithm 1.

2. Discretize each of the four candidate boundary segments connecting these extrema into `line_len` equally spaced query points using `linspace` (`line_len = 100000` in `section_1.m`; `line_len = 1000` in `section_1_noiseTest.m`).

3. For each query point on a segment, compute its Euclidean distance to all $T = 1000$ observation samples and find the closest observation index `ind_line`, retaining only matches whose distance is at most `dist_threshold` (`0.8` in `section_1.m`; `0.3` in `section_1_noiseTest.m`).

4. Remove duplicate sample indices via `unique(ind_line)` so that each near-boundary observation contributes once.

5. Perform first-order polynomial regression (`polyfit(x_1(ind_line), x_2(ind_line), 1)`) on the unique boundary points of each edge to extract the four refined slopes, average each pair of parallel slopes, and normalize to unit norm.

**Performance Metric under Noise (Part 3):**

In `section_1_noiseTest.m`, AWGN is added at $\mathrm{SNR} = 15\text{ dB}$ via `X = awgn(X, 15)`. Estimation quality can be quantified by the **entrywise absolute error** $\vert{}\hat{a}_{ij} - a_{ij}\vert{}$ (used in the report), the **relative Frobenius norm error**

```
E_{\mathrm{Frob}}(\hat{A}, A) = \frac{\|\hat{A} - A\|_F}{\|A\|_F},

```

or the **Amari performance index** (After optimal column permutation/sign matching). Under $15\text{ dB}$ AWGN, Algorithm 2 reduces the relative Frobenius error from $3.25\%$ (Algorithm 1) to $1.23\%$ (see Section 5.1).

#### 4.1.3 Analytical Marginal PDFs of $x_1$ and $x_2$ (Parts 4 & 5)

By statistical independence of $s_1 \sim \mathcal{U}(-3, 3)$ and $s_2 \sim \mathcal{U}(-2, 2)$, their joint PDF on $\mathcal{R}_S = [-3, 3] \times [-2, 2]$ is

```
f_{S_1, S_2}(s_1, s_2) = f_{S_1}(s_1) f_{S_2}(s_2)
= \frac{1}{3 - (-3)} \times \frac{1}{2 - (-2)}
= \frac{1}{6} \times \frac{1}{4} = \frac{1}{24}.

```

The Jacobian determinant of the linear transformation $X = AS$ is

```
J = \det(A) = \begin{vmatrix} 0.6 & 0.8 \\ 0.8 & -0.6 \end{vmatrix} = -0.36 - 0.64 = -1
\quad \Longrightarrow \quad |J| = 1.

```

Hence the joint PDF of $(X_1, X_2)$ is uniform over the parallelogram support $\mathcal{D}_X$:

```
f_{X_1, X_2}(x_1, x_2) = \frac{f_{S_1, S_2}(A^{-1}x)}{|J|} = \frac{1}{24}, \qquad (x_1, x_2) \in \mathcal{D}_X.

```

Since $A^{-1} = A$, the inverse relation $S = A^{-1}X$ is

```
\begin{cases}
s_1 = 0.6 x_1 + 0.8 x_2 \in [-3, 3], \\
s_2 = 0.8 x_1 - 0.6 x_2 \in [-2, 2].
\end{cases}

```

**Marginal PDF of** $X_1$ **(Part 4):**

Solving the support inequalities for $x_2$ as a function of $x_1$ gives

```
\begin{cases}
-3 \le 0.6 x_1 + 0.8 x_2 \le 3 \\
-2 \le 0.8 x_1 - 0.6 x_2 \le 2
\end{cases}
\Longleftrightarrow
\begin{cases}
-\dfrac{15}{4} - \dfrac{3}{4} x_1 \le x_2 \le \dfrac{15}{4} - \dfrac{3}{4} x_1 & (-3.75 - 0.75 x_1 \le x_2 \le 3.75 - 0.75 x_1), \\[6pt]
-\dfrac{10}{3} + \dfrac{4}{3} x_1 \le x_2 \le \dfrac{10}{3} + \dfrac{4}{3} x_1 & (-3.\overline{33} + 1.\overline{33} x_1 \le x_2 \le 3.\overline{33} + 1.\overline{33} x_1).
\end{cases}

```

The active upper and lower bounds switch at the $x_1$-coordinates of the four parallelogram vertices, namely $\{-3.4, -0.2, 0.2, 3.4\}$. Integrating $f_{X_1, X_2}(x_1, x_2) = \frac{1}{24}$ with respect to $x_2$ yields a symmetric **trapezoidal PDF**:

```
\boxed{
f_{X_1}(x_1) = \int_{-\infty}^{\infty} f_{X_1, X_2}(x_1, x_2)\,dx_2 =
\begin{cases}
\displaystyle \int_{-15/4 - 3x_1/4}^{10/3 + 4x_1/3} \frac{1}{24}\,dx_2 = \frac{1}{24}\left(\frac{85}{12} + \frac{25}{12}x_1\right) \approx \frac{7.08 + 2.08 x_1}{24}, & -3.4 \le x_1 < -0.2, \\[10pt]
\displaystyle \int_{-10/3 + 4x_1/3}^{10/3 + 4x_1/3} \frac{1}{24}\,dx_2 = \frac{1}{24}\left(\frac{20}{3}\right) = \frac{5}{18} \approx 0.2778, & -0.2 \le x_1 \le 0.2, \\[10pt]
\displaystyle \int_{-10/3 + 4x_1/3}^{15/4 - 3x_1/4} \frac{1}{24}\,dx_2 = \frac{1}{24}\left(\frac{85}{12} - \frac{25}{12}x_1\right) \approx \frac{7.08 - 2.08 x_1}{24}, & 0.2 < x_1 \le 3.4, \\[8pt]
0, & \text{otherwise.}
\end{cases}
}

```

*(In the Persian report,* $10/3$ *and* $4/3$ *are truncated to two decimal places as* $3.33$ *and* $1.33$*, giving plateau height* $6.66/24 = 0.2775$ *and graphical breakpoints* $\{-3.404, -0.202, 0.202, 3.404\}$*; page 7 of the report explicitly notes that the exact analytical breakpoints are* $\{-3.4, -0.2, 0.2, 3.4\}$*.)*

**Marginal PDF of** $X_2$ **(Part 5):**

Solving the support inequalities for $x_1$ as a function of $x_2$ gives

```
\begin{cases}
-5 - \dfrac{4}{3} x_2 \le x_1 \le 5 - \dfrac{4}{3} x_2 & (-5 - 1.\overline{33} x_2 \le x_1 \le 5 - 1.\overline{33} x_2), \\[6pt]
-\dfrac{5}{2} + \dfrac{3}{4} x_2 \le x_1 \le \dfrac{5}{2} + \dfrac{3}{4} x_2 & (-2.5 + 0.75 x_2 \le x_1 \le 2.5 + 0.75 x_2).
\end{cases}

```

The $x_2$-breakpoints are the $x_2$-coordinates of the four parallelogram vertices, $\{-3.6, -1.2, 1.2, 3.6\}$. Integrating $f_{X_1, X_2}(x_1, x_2) = \frac{1}{24}$ with respect to $x_1$ yields the symmetric **trapezoidal PDF**:

```
\boxed{
f_{X_2}(x_2) = \int_{-\infty}^{\infty} f_{X_1, X_2}(x_1, x_2)\,dx_1 =
\begin{cases}
\displaystyle \int_{-5 - 4x_2/3}^{2.5 + 0.75 x_2} \frac{1}{24}\,dx_1 = \frac{1}{24}\left(\frac{15}{2} + \frac{25}{12}x_2\right) \approx \frac{7.5 + 2.08 x_2}{24}, & -3.6 \le x_2 < -1.2, \\[10pt]
\displaystyle \int_{-2.5 + 0.75 x_2}^{2.5 + 0.75 x_2} \frac{1}{24}\,dx_1 = \frac{5}{24} \approx 0.2083, & -1.2 \le x_2 \le 1.2, \\[10pt]
\displaystyle \int_{-2.5 + 0.75 x_2}^{5 - 4x_2/3} \frac{1}{24}\,dx_1 = \frac{1}{24}\left(\frac{15}{2} - \frac{25}{12}x_2\right) \approx \frac{7.5 - 2.08 x_2}{24}, & 1.2 < x_2 \le 3.6, \\[8pt]
0, & \text{otherwise.}
\end{cases}
}

```

#### 4.1.4 Convolution Structure and Non-Uniqueness from Marginal Histograms Alone (Part 6)

For each channel $i \in \{1, 2\}$, define the scaled source variables $u_{i1} = a_{i1} s_1$ and $u_{i2} = a_{i2} s_2$. Because $s_1$ and $s_2$ are statistically independent, $x_i = u_{i1} + u_{i2}$ is the sum of two independent random variables with uniform densities $f_{U_{i1}}(u) = \frac{1}{6\vert{}a_{i1}\vert{}}\mathbf{1}_{[-3\vert{}a_{i1}\vert{}, 3\vert{}a_{i1}\vert{}]}(u)$ and $f_{U_{i2}}(u) = \frac{1}{4\vert{}a_{i2}\vert{}}\mathbf{1}_{[-2\vert{}a_{i2}\vert{}, 2\vert{}a_{i2}\vert{}]}(u)$. From the cumulative distribution function (CDF)

```
F_{X_i}(x_i) = \Pr\{a_{i1} s_1 + a_{i2} s_2 < x_i\}
= \int_{-\infty}^{\infty} f_{U_{i2}}(u_2) \left(\int_{-\infty}^{x_i - u_2} f_{U_{i1}}(u_1)\,du_1\right) du_2,

```

differentiating with respect to $x_i$ shows that the marginal PDF is the **convolution** of the two rectangular PDFs:

```
f_{X_i}(x_i) = \big(f_{U_{i1}} * f_{U_{i2}}\big)(x_i).

```

Convolving two centered rectangular pulses of half-widths $w_{i1} = 3\vert{}a_{i1}\vert{}$ and $w_{i2} = 2\vert{}a_{i2}\vert{}$ produces a symmetric trapezoid with positive breakpoints at

```
b_{i,\text{inner}} = \big|3|a_{i1}| - 2|a_{i2}|\big|,
\qquad
b_{i,\text{outer}} = 3|a_{i1}| + 2|a_{i2}|.

```

From the exact marginal histograms (and the known source supports $[-3, 3]$ and $[-2, 2]$), one can solve for the unordered pair of half-widths $\{w_{i1}, w_{i2}\} = \left\{\frac{b_{i,\text{outer}} + b_{i,\text{inner}}}{2}, \frac{b_{i,\text{outer}} - b_{i,\text{inner}}}{2}\right\}$ in each row $i$. Nevertheless, **the BSS problem cannot be uniquely solved from the marginal histograms alone**:

1. **Sign Ambiguity of Matrix Entries:** Because $s_1$ and $s_2$ are symmetrically distributed around zero, $f_{U_{ij}}$ depends only on $\vert{}a_{ij}\vert{}$. Flipping the sign of any individual entry $a_{ij}$ (for instance, replacing $a_{22} = -0.6$ with $+0.6$, which turns the nonsingular orthogonal matrix $A$ into a singular matrix of rank 1 if $a_{11}=a_{21}$ and $a_{12}=a_{22}$, or changing relative signs across rows) leaves the marginal PDFs $f_{X_1}(x_1)$ and $f_{X_2}(x_2)$ completely unchanged.

2. **Intra-Row Permutation Ambiguity:** Convolution is commutative ($f_{U_{i1}} * f_{U_{i2}} = f_{U_{i2}} * f_{U_{i1}}$). For a given row $i$, the trapezoid breakpoints determine the unordered set $\{3\vert{}a_{i1}\vert{}, 2\vert{}a_{i2}\vert{}\}$, but cannot distinguish whether the larger half-width belongs to $3\vert{}a_{i1}\vert{}$ or to $2\vert{}a_{i2}\vert{}$ without additional structural or joint-distribution constraints.

### 4.2 Section 2: Convex Quadratic Optimization and Unit-Circle Projection

#### 4.2.1 Gradient, Hessian, Strict Convexity, and Stationary Point (Parts 3 & 4)

Rewrite the quadratic objective in matrix-vector form:

```
f(x) = x_1^2 + x_2^2 + x_1 x_2 - 4x_1 - 6x_2 + 13
= \frac{1}{2} x^T H x + b^T x + 13,

```

where the gradient $g(x) = \nabla f(x)$ and Hessian $H = \nabla^2 f(x)$ are

```
\boxed{
g(x) = \nabla f(x) = \begin{bmatrix} \dfrac{\partial f}{\partial x_1} \\[8pt] \dfrac{\partial f}{\partial x_2} \end{bmatrix}
= \begin{bmatrix} 2x_1 + x_2 - 4 \\ x_1 + 2x_2 - 6 \end{bmatrix},
\qquad
H = \nabla^2 f(x) = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}.
}

```

**Strict Convexity Proof (Part 3):**

For any nonzero vector $x = [x_1, x_2]^T \in \mathbb{R}^2 \setminus \{0\}$, the quadratic form of the Hessian is

```
x^T H x = \begin{bmatrix} x_1 & x_2 \end{bmatrix} \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
= 2\left(x_1^2 + x_1 x_2 + x_2^2\right)
= 2\left[\left(x_1 + \frac{1}{2}x_2\right)^2 + \frac{3}{4}x_2^2\right] > 0.

```

Equivalently, viewing $x_1^2 + x_1 x_2 + x_2^2 = 0$ as a quadratic equation in $x_1$ for any $x_2 \neq 0$, its discriminant is $\Delta = x_2^2 - 4(1)(x_2^2) = -3x_2^2 < 0$, so it has no nonzero real root and remains strictly positive everywhere on $\mathbb{R}^2 \setminus \{0\}$ (the eigenvalues of $H$ are $\lambda_{\min} = 1 > 0$ and $\lambda_{\max} = 3 > 0$). Therefore, $H \succ 0$, proving that $f(x)$ is **strictly convex** and has a unique global minimizer.

**Unconstrained Global Minimizer (Part 4):**

Setting $\nabla f(x^*) = 0$ yields the $2 \times 2$ linear system

```
\begin{cases}
2x_1 + x_2 - 4 = 0, \\
x_1 + 2x_2 - 6 = 0
\end{cases}
\quad \Longrightarrow \quad
\boxed{
x^* = \begin{bmatrix} 2/3 \\ 8/3 \end{bmatrix} \approx \begin{bmatrix} 0.6667 \\ 2.6667 \end{bmatrix},
\qquad
f(x^*) = \frac{11}{3} \approx 3.6667.
}

```

#### 4.2.2 Steepest Descent, Newton's Method, and Alternating Minimization (Parts 5, 6, & 7)

All iterative methods in `codes/section_2.m` are initialized at $x_0 = [6, 6]^T$ (where $f(x_0) = 61$) and terminate when consecutive objective values satisfy

```
\big|f(x_{k+1}) - f(x_k)\big| < \varepsilon = 10^{-6}.

```

**Steepest Descent (Part 5):**

The constant-step Steepest Descent update is

```
x_{k+1} = x_k - \mu \nabla f(x_k) = (I - \mu H) x_k + \mu \begin{bmatrix} 4 \\ 6 \end{bmatrix}.

```

Because $H$ has eigenvalues $\lambda_1 = 1$ and $\lambda_2 = 3$, the error $e_k = x_k - x^*$ contracts at every step if and only if $\vert{}1 - \mu \lambda_i\vert{} < 1$ for both eigenvalues, i.e.,

```
0 < \mu < \frac{2}{\lambda_{\max}(H)} = \frac{2}{3}.

```

Both prescribed step sizes $\mu = 0.01$ and $\mu = 0.1$ satisfy this stability bound, so **both converge**:

* For $\mu = 0.1$, the slowest mode contracts by $1 - 0.1(1) = 0.9$ per iteration, satisfying $\vert{}f(x_{k+1}) - f(x_k)\vert{} < 10^{-6}$ in $59$ **iterations** with $x_{59} \approx [0.6687, 2.6647]^T$.

* For $\mu = 0.01$, the slowest mode contracts by $1 - 0.01(1) = 0.99$ per iteration, requiring $494$ **iterations** to reach $\vert{}f(x_{k+1}) - f(x_k)\vert{} < 10^{-6}$ with $x_{494} \approx [0.6736, 2.6597]^T$.

**Newton's Method (Part 6):**

The Newton update is

```
x_{k+1} = x_k - H^{-1} \nabla f(x_k).

```

Because $f(x)$ is a **quadratic polynomial** with constant nonsingular Hessian $H$, the gradient is affine: $\nabla f(x_k) = H(x_k - x^*)$. Substituting this identity into the Newton update gives

```
x_1 = x_0 - H^{-1} H(x_0 - x^*) = x^* = \begin{bmatrix} 2/3 \\ 8/3 \end{bmatrix}

```

in **exactly** $1$ **iteration** from any initial point $x_0 \in \mathbb{R}^2$.

**Alternating Minimization / Coordinate Descent (Part 7):**

Minimizing $f(x_1, x_2)$ with respect to one coordinate at a time while holding the other fixed gives two closed-form update rules:

```
\begin{aligned}
\frac{\partial f}{\partial x_1} = 2x_1 + x_2 - 4 = 0 &\quad \Longrightarrow \quad x_1^{\text{new}} = 2 - \frac{1}{2}x_2, \\[4pt]
\frac{\partial f}{\partial x_2} = x_1 + 2x_2 - 6 = 0 &\quad \Longrightarrow \quad x_2^{\text{new}} = 3 - \frac{1}{2}x_1.
\end{aligned}

```

In `codes/section_2.m`, each coordinate update counts as one alternation step: starting at $x_0 = [6, 6]^T$, odd steps update $x_1 \leftarrow 2 - x_2/2$ (horizontal movement on the contour plot) and even steps update $x_2 \leftarrow 3 - x_1/2$ (vertical movement on the contour plot). Each full cycle of two coordinate steps reduces the distance to $x^*$ by a factor of $1/4$, terminating after $14$ **coordinate updates** at $x \approx [0.6663, 2.6669]^T$.

#### 4.2.3 Unit-Circle Constrained Optimization: Gradient Projection vs. Lagrange Multipliers (Parts 8 & 9)

Consider the equality-constrained problem

```
\begin{aligned}
\min_{x \in \mathbb{R}^2} \quad & f(x) = x_1^2 + x_2^2 - 4x_1 - 6x_2 + 13 + x_1 x_2, \\
\text{subject to} \quad & g(x) = x_1^2 + x_2^2 - 1 = 0 \quad (\|x\|_2 = 1).
\end{aligned}

```

**Projected Gradient Descent (Part 8):**

At each iteration $k$ with $\mu = 0.1$, take a Steepest Descent step and project the resulting vector orthogonally onto the unit circle by normalizing its Euclidean norm:

```
\tilde{x}_{k+1} = x_k - \mu \nabla f(x_k),
\qquad
x_{k+1} = \frac{\tilde{x}_{k+1}}{\|\tilde{x}_{k+1}\|_2}.

```

In `codes/section_2.m`, this iteration converges under $\vert{}f(x_{k+1}) - f(x_k)\vert{} < 10^{-6}$ in $15$ **iterations** to

```
\boxed{
x_{\mathrm{GP}}^* \approx \begin{bmatrix} 0.4946 \\ 0.8691 \end{bmatrix},
\qquad
f(x_{\mathrm{GP}}^*) \approx 7.2375.
}

```

**Analytical Solution via Lagrange Multipliers (Part 9):**

By the method of Lagrange multipliers, any stationary point on the constraint curve $g(x) = x_1^2 + x_2^2 - 1 = 0$ satisfies $\nabla f(x) = \gamma \nabla g(x)$ for some multiplier $\gamma \in \mathbb{R}$:

```
\begin{cases}
2x_1 - 4 + x_2 = \gamma (2x_1), \\
2x_2 - 6 + x_1 = \gamma (2x_2), \\
x_1^2 + x_2^2 = 1.
\end{cases}

```

Assuming $x_1 \neq 0$ and $x_2 \neq 0$, divide the first equation by $2x_1$ and the second by $2x_2$:

```
\gamma = 1 - \frac{2}{x_1} + \frac{x_2}{2x_1} = 1 - \frac{3}{x_2} + \frac{x_1}{2x_2}
\quad \Longrightarrow \quad
\frac{2}{x_1} - \frac{x_2}{2x_1} = \frac{3}{x_2} - \frac{x_1}{2x_2}.

```

Multiplying through by $2x_1 x_2$ yields the hyperbola

```
x_1^2 - x_2^2 - 6x_1 + 4x_2 = 0.

```

Combining this relation with $x_1^2 + x_2^2 = 1$ (specifically, $x_2 = \frac{6x_1 + 1 - 2x_1^2}{4}$ substituted into $x_1^2 + x_2^2 = 1$, i.e., $4x_1^4 - 24x_1^3 + 48x_1^2 + 12x_1 - 15 = 0$) yields four roots:

1. **Constrained Global Minimizer:**

```
\boxed{
x_1^* \approx 0.494 \ (0.4942),\qquad x_2^* \approx 0.869 \ (0.8693) \quad \Longrightarrow \quad f(x^*) \approx 7.237,
}

```

which matches the Projected Gradient Descent solution $[0.4946, 0.8691]^T$ from Part 8. 2. **Constrained Global Maximizer:**

```
x_1 \approx -0.589 \ (-0.5898),\qquad x_2 \approx -0.807 \ (-0.8075) \quad \Longrightarrow \quad f(x) \approx 21.68.

```

3. **Complex Conjugate Roots (Infeasible in** $\mathbb{R}^2$**):**

```
x_1 \approx 3.047 \pm j\,1.893 \quad (\text{rejected}).

```

## 5. Results & Discussion

### 5.1 Section 1 Summary: Mixing-Matrix Estimation & Marginal Distributions

The true mixing matrix is $A = \begin{bmatrix} 0.6 & 0.8 \\ 0.8 & -0.6 \end{bmatrix}$ ($\Vert{}A\Vert{}_F = \sqrt{2} \approx 1.4142$). The table below summarizes the single-run mixing-matrix estimates reported in `BSS_HW#1_810100205_Arman_Majidi.pdf`:

| Scenario | Method | Estimated Matrix $\hat{A}$ | Max Entrywise Error $\max_{i,j}\vert{}\hat{a}_{ij}-a_{ij}\vert{}$ | Relative Frobenius Error $\Vert{}\hat{A}-A\Vert{}_F/\Vert{}A\Vert{}_F$ | 
| ----- | ----- | ----- | ----- | ----- | 
| Noiseless ($T=1000$) | Algorithm 1 (Extreme Vertices) | $\begin{bmatrix} 0.5986 & 0.8011 \\ 0.8154 & -0.5788 \end{bmatrix}$ | $0.0212$ | $1.87\%$ | 
| Noiseless ($T=1000$) | Algorithm 2 (`polyfit` Boundary Regression) | $\begin{bmatrix} 0.6004 & 0.7997 \\ 0.8086 & -0.5883 \end{bmatrix}$ | $0.0117$ | $1.03\%$ | 
| AWGN ($\mathrm{SNR}=15\text{ dB}$) | Algorithm 1 (Extreme Vertices) | $\begin{bmatrix} 0.5784 & 0.8158 \\ 0.7839 & -0.6209 \end{bmatrix}$ | $0.0216$ | $3.25\%$ | 
| AWGN ($\mathrm{SNR}=15\text{ dB}$) | Algorithm 2 (`polyfit` Boundary Regression) | $\begin{bmatrix} 0.5909 & 0.8068 \\ 0.8086 & -0.6120 \end{bmatrix}$ | $0.0120$ | $1.23\%$ | 

| Marginal Distribution | Support Interval | Plateau Interval | Plateau Density | Outer Slope Magnitude | 
| ----- | ----- | ----- | ----- | ----- | 
| $f_{X_1}(x_1)$ (Part 4) | $[-3.4,\ 3.4]$ | $[-0.2,\ 0.2]$ | $\dfrac{5}{18} \approx 0.2778$ ($0.2775$ in report) | $\dfrac{25}{288} \approx \dfrac{2.0833}{24}$ | 
| $f_{X_2}(x_2)$ (Part 5) | $[-3.6,\ 3.6]$ | $[-1.2,\ 1.2]$ | $\dfrac{5}{24} \approx 0.2083$ | $\dfrac{25}{288} \approx \dfrac{2.0833}{24}$ | 

### 5.2 Section 2 Summary: Optimization Methods Comparison

All runs start from $x_0 = [6, 6]^T$ with stopping tolerance $|f(x_{k+1}) - f(x_k)| < 10^{-6}$:

| Part | Method | Constraint | Step Size $\mu$ | Iterations | Final Point $x$ | Objective Value $f(x)$ | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| Part 4 | Closed-Form Stationary Point | Unconstrained | — | — | $[2/3,\ 8/3]^T \approx [0.6667,\ 2.6667]^T$ | $11/3 \approx 3.6667$ | 
| Part 5 | Steepest Descent | Unconstrained | $0.1$ | $59$ | $[0.6687,\ 2.6647]^T$ | $3.6667$ | 
| Part 5 | Steepest Descent | Unconstrained | $0.01$ | $494$ | $[0.6736,\ 2.6597]^T$ | $3.6667$ | 
| Part 6 | Newton's Method | Unconstrained | $1$ (exact) | $1$ | $[0.6667,\ 2.6667]^T$ | $3.6667$ | 
| Part 7 | Alternating Minimization | Unconstrained | Closed-form coordinate step | $14$ | $[0.6663,\ 2.6669]^T$ | $3.6667$ | 
| Part 8 | Gradient Projection | $\Vert{}x\Vert{}_2 = 1$ | $0.1$ | $15$ | $[0.4946,\ 0.8691]^T$ | $7.2375$ | 
| Part 9 | Lagrange Multipliers | $\Vert{}x\Vert{}_2 = 1$ | — | — | $[0.4942,\ 0.8693]^T$ ($[0.494,\ 0.869]^T$ in report) | $7.2375$ ($7.237$ in report) | 

### 5.3 Notes on the Submitted Report and Code

* **Permutation and sign ambiguity in Section 1 (Parts 2 & 3):** In general BSS, $X = AS$ has inherent column-permutation and scaling/sign ambiguities. In `codes/section_1.m`, because the first row of $A$ is strictly positive ($[0.6, 0.8]$) and the edges are ordered consistently by $(\min x_1, \max x_1, \min x_2, \max x_2)$, initializing the unnormalized basis vectors with $+1$ in the first component automatically matches the true sign and column order of $A$.

* **Decimal typo in Section 1 (Part 3 commentary):** On page 5 of the Persian report, the commentary states that each entry of $\hat{A}_{\text{alg2, noisy}}$ differs from the true value by about `0.1`; comparing $\hat{A}_{\text{alg2, noisy}} = \begin{bmatrix} 0.5909 & 0.8068 \\ 0.8086 & -0.6120 \end{bmatrix}$ with $A = \begin{bmatrix} 0.6 & 0.8 \\ 0.8 & -0.6 \end{bmatrix}$ shows that the actual entrywise error is $\approx 0.01$ (specifically between $0.0068$ and $0.0120$).

* **Typographical slips in Section 1 (Part 5 equations on page 9):** In the boxed inequality on page 9 of the Persian report, the lower bound of $x_1$ is written as $5 - 1.33 x_2 \le x_1 \le 5 - 1.33 x_2$ instead of $-5 - 1.33 x_2 \le x_1 \le 5 - 1.33 x_2$, and inside the piecewise integral of $f_{X_2}(x_2)$ the dummy variable is written as $x_1$ with $7.05 + 2.08 x_1$ in the third branch instead of $7.5 + 2.08 x_2$. The plotted trapezoidal density in Figure 1.9 and the exact breakpoints $\{-3.6, -1.2, 1.2, 3.6\}$ are unaffected.

* **Initial projection in `gradient_projection` (Section 2, Part 8):** In `codes/section_2.m` (lines 134–135), the first gradient step $\tilde{x}_1 = x_0 - \mu \nabla f(x_0)$ is computed from the unprojected initial vector $x_0 = [6, 6]^T$ before normalizing both $x_0$ and $\tilde{x}_1$ onto the unit circle; all subsequent iterations compute the gradient at the unit-norm iterate $x_k$. Because the constrained problem on the unit circle has a unique global minimizer, the sequence still converges rapidly to $[0.4946, 0.8691]^T$.

## 6. How to Use

### 6.1 Run the MATLAB Scripts

Open MATLAB, navigate to the `HW#1/codes` directory (or add `codes` to your MATLAB path), and run any of the three scripts:

```
% Section 1: Noiseless scatter plot, mixing-matrix estimation (Parts 1-2), and histograms (Parts 4-5)
section_1

% Section 1: Mixing-matrix estimation under 15 dB AWGN (Part 3)
section_1_noiseTest

% Section 2: 3D mesh, 2D contour, Steepest Descent, Newton, Alternating Minimization, and Gradient Projection
section_2

```

*(Note: `unifrnd` in `section_1.m` requires the Statistics and Machine Learning Toolbox, and `awgn` in `section_1_noiseTest.m` requires the Communications Toolbox.)*

### 6.2 Expected Console Output

Because `section_1.m` and `section_1_noiseTest.m` draw random uniform realizations without fixing `rng`, their printed $2 \times 2$ matrices vary slightly across runs around $A = \begin{bmatrix} 0.6 & 0.8 \\ 0.8 & -0.6 \end{bmatrix}$. Running `section_2.m` is completely deterministic and produces the following console output:

```
S.D: final value of x for mu = 0.10 is [0.668660, 2.664674] in iteration: 59
S.D: final value of x for mu = 0.01 is [0.673646, 2.659688] in iteration: 494

Newton: final value of x for is [0.666667, 2.666667] in iteration: 1

Alternation Minimization: final value of x for is [0.666260, 2.666870] in iteration: 14

G.P: final value of x for mu = 0.10 is [0.494565, 0.869141] in iteration: 15

```

## 7. References

1. Akhavan, S. *Blind Source Separation: Homework 1*. School of Electrical and Computer Engineering, University of Tehran, Spring 2023–2024 (`hw1.pdf`).

2. Majidi, A. *Blind Source Separation — Homework 1 Report* (`BSS_HW#1_810100205_Arman_Majidi.pdf`). University of Tehran, Spring 2023–2024.

3. Hyvärinen, A., Karhunen, J., & Oja, E. (2001). *Independent Component Analysis*. Wiley. Background on instantaneous linear mixing models and uniform source geometry.

4. Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press. Background on positive definite quadratic forms, Steepest Descent, Newton's method, and Lagrange duality.

5. [GitHub Docs: Writing mathematical expressions](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions).