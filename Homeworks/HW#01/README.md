# Homework 1: Geometric Mixing Matrix Estimation, Marginal Distributions of Linear Mixtures, and Constrained & Unconstrained Optimization

**Course:** Blind Source Separation  
**Instructor:** Dr. Saeed Akhavan  
**Author:** Arman Majidi  
**Student ID:** 810100205  
**University:** University of Tehran  
**Semester:** Spring 2024-2025

## 1. Overview & Assignment Structure

This folder contains the **English problem statements, analytical derivations, MATLAB implementations, and simulation results for Homework 1**. The assignment is organized into two main sections that introduce the geometry and probability distributions of linear instantaneous mixtures alongside fundamental unconstrained and constrained optimization methods used in Blind Source Separation (BSS):

1. **Section 1 — Linear Instantaneous Mixing of Uniform Sources (Parts 1–6):** Generates two statistically independent uniformly distributed sources, forms two linear instantaneous mixtures, analyzes the parallelogram geometry of the observation scatter plot, estimates the mixing matrix $A$ in both noiseless and noisy scenarios using extreme-vertex and boundary-regression algorithms, derives the analytical marginal probability density functions (PDFs) of the observed mixtures to compare against empirical histograms, and examines whether marginal histograms alone suffice to solve the BSS problem uniquely.
2. **Section 2 — Unconstrained and Constrained Quadratic Optimization (Parts 1–9):** Visualizes a two-variable quadratic objective function in linear and logarithmic (dB) scales using 3D mesh and 2D contour plots, proves strict convexity via the Hessian matrix, computes the closed-form unconstrained minimizer, implements and compares **Steepest Descent**, **Newton's method**, and **Alternating Minimization**, and solves the unit-norm constrained problem $\|x\|_2 = 1$ numerically via **Gradient Projection** and analytically via **Lagrange multipliers**.

All parts of both sections are mandatory; no optional or bonus problems were assigned. Because both the original assignment sheet and the submitted solution report are written in Persian (and the Persian assignment PDF is not distributed in the repository), this `README.md` provides a **complete, self-contained English statement of every problem, mathematical proof, algorithm, and numerical result**.

## 2. Directory Contents

| File | Description |
| --- | --- |
| [codes/section\_1.m](./codes/section_1.m) | MATLAB script for Section 1 (Parts 1, 2, 4, and 5): uniform source generation, scatter plot, noiseless mixing matrix estimation (Algorithm 1 and Algorithm 2 with `slope_finder`), and 20-bin observation histograms. |
| [codes/section\_1\_noiseTest.m](./codes/section_1_noiseTest.m) | MATLAB script for Section 1 (Part 3): mixing matrix estimation under additive white Gaussian noise ($\mathrm{SNR} = 15\ \mathrm{dB}$) using both the vertex-based and boundary-regression methods. |
| [codes/section\_2.m](./codes/section_2.m) | MATLAB script for Section 2 (Parts 1, 2, 5, 6, 7, and 8): 3D mesh and 2D contour visualizations, Steepest Descent (`steepest_descend`), Newton's method (`newton`), Alternating Minimization (`alternation_minimization`), and Gradient Projection (`gradient_projection`). |
| [BSS\_HW#1\_810100205\_Arman\_Majidi.pdf](./BSS_HW%231_810100205_Arman_Majidi.pdf) | Submitted report and analytical solutions in Persian (including scatter plots, histograms, marginal PDF diagrams, and optimization trajectories). |
| [hw1](./hw1.pdf) | Original assignment problem sheet in Persian. |
| `README.md` | Self-contained English problem statements, mathematical derivations, algorithm descriptions, numerical results, and usage guide. |

Keep the `codes` subfolder inside the homework directory.

## 3. Assignment Problem Statement

### 3.1 General Instructions from the Assignment Sheet

- The exercises may be implemented in either **MATLAB** or **Python**.
- In MATLAB, write standard `.m` script files rather than Live Scripts (`.mlx`).
- Submit the code files together with a complete typeset report containing all plots, mathematical derivations, and numerical results.

### 3.2 Section 1: Linear Instantaneous Mixing of Uniform Sources

Generate $T = 1000$ independent samples of two random processes:

- Source $s_1$: uniformly distributed on $[-3, 3]$.
- Source $s_2$: uniformly distributed on $[-2, 2]$.

Combine the two sources via a linear instantaneous mixing model $X = AS$ using the mixing matrix

```math
A = \begin{bmatrix} 0.6 & 0.8 \\ 0.8 & -0.6 \end{bmatrix},
\qquad
S = \begin{bmatrix} s_1 \\ s_2 \end{bmatrix},
\qquad
X = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = AS,
```

to produce the observed mixtures $x_1$ and $x_2$.

#### Part 1 — Scatter Plot and Geometric Relationship to $A$

Plot the scatter diagram of $x_2$ versus $x_1$. Explain how the geometric shape of this scatter plot is related to the mixing matrix $A$.

#### Part 2 — Automatic Mixing Matrix Estimation (Noiseless Case)

Based on the geometric relationship identified in Part 1, propose an automatic mathematical method to estimate the mixing matrix $A$ from the observed data $X$, and implement your algorithm in code.

#### Part 3 — Mixing Matrix Estimation in the Presence of Noise

Add low-power additive noise to the observation matrix $X$ and apply the estimation method from Part 2 to the noisy observations.

1. Is your initial method from Part 2 directly applicable and accurate in the noisy scenario?
2. If not, modify the method to improve its robustness against noise.
3. Report the estimation performance in the presence of noise and propose a quantitative metric showing that the modified method performs well under noise.

*(Note from the assignment sheet: Solve Parts 4, 5, and 6 below for the noiseless case.)*

#### Part 4 — Empirical Histogram and Analytical Marginal PDF of $x_1$

Plot the empirical histogram of $x_1$ using $20$ bins. Using probability and mathematical relations, derive the analytical marginal probability density function $f_{X_1}(x_1)$ of $x_1$, and verify that the theoretical distribution matches the empirical histogram.

#### Part 5 — Empirical Histogram and Analytical Marginal PDF of $x_2$

Repeat Part 4 for the second observed mixture $x_2$: plot its 20-bin histogram, derive the analytical marginal PDF $f_{X_2}(x_2)$, and compare the analytical curve with the empirical histogram.

#### Part 6 — Identifiability of BSS from Marginal Histograms Alone

Suppose the sample size $T$ is sufficiently large that the normalized histograms of $x_1$ and $x_2$ from Parts 4 and 5 coincide exactly with the theoretical marginal distributions $f_{X_1}(x_1)$ and $f_{X_2}(x_2)$. Assume further that the exact distributions of the sources (both their uniform shape and their support intervals $[-3,3]$ and $[-2,2]$) are known a priori. Under these conditions, can the BSS problem be solved uniquely using only the marginal histograms of $x_1$ and $x_2$? Justify your answer mathematically.

### 3.3 Section 2: Unconstrained and Constrained Quadratic Optimization

Consider the two-variable quadratic objective function

```math
f(x) = x_1^2 + x_2^2 - 4x_1 - 6x_2 + 13 + x_1 x_2,
\qquad
x = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \in \mathbb{R}^2.
```

#### Part 1 — 3D Surface Visualization (`mesh`)

Using the MATLAB `mesh` command, plot $f(x)$ in 3D as a function of $x_1$ and $x_2$ over the domain $-10 < x_1, x_2 < 10$. Also plot the objective in decibel (logarithmic) scale, $10\log_{10} f(x)$, so that variations near the minimum are clearly visible.

#### Part 2 — 2D Equipotential Level Curves (`contour`)

Using the MATLAB `contour` command, plot the equipotential level curves of $f(x)$ in the 2D $(x_1, x_2)$-plane and display the numerical level values directly on the contour lines (`'ShowText', 'on'`).

#### Part 3 — Gradient, Hessian, and Proof of Strict Convexity

Compute the gradient vector $g = \nabla f(x)$ and the Hessian matrix $H = \nabla^2 f(x)$. Using the Hessian matrix, prove that $f(x)$ is strictly convex by showing that

```math
x^T H x > 0
\qquad \forall x \in \mathbb{R}^2 \setminus \{0\},
```

and conclude that $f(x)$ possesses a unique global minimum.

#### Part 4 — Analytical Unconstrained Minimizer

Find the exact unconstrained minimizer $x^*$ of $f(x)$ analytically by setting the gradient vector equal to zero ($\nabla f(x) = 0$).

#### Part 5 — Steepest Descent Algorithm

Starting from the initial point $x_0 = [6, 6]^T$, implement the **Steepest Descent (SD)** algorithm with step sizes $\mu = 0.01$ and $\mu = 0.1$.

1. Does the algorithm converge in both cases?
2. Which step size converges faster?
3. Plot the convergence curve (objective value $f(x_k)$ versus iteration index $k$) for both values of $\mu$ on the same figure.

#### Part 6 — Newton's Method

Starting from the initial point $x_0 = [6, 6]^T$, implement **Newton's method** to minimize $f(x)$. In how many iterations does the algorithm converge to the exact minimizer? Explain the mathematical reason for this behavior.

#### Part 7 — Alternating Minimization

Starting from $x_0 = [6, 6]^T$, minimize $f(x)$ using **Alternating Minimization** (coordinate descent) between the two variables $x_1$ and $x_2$. Because each single-variable subproblem is quadratic and simple, derive and use its exact closed-form update rule rather than running an inner iterative solver. Plot the optimization trajectory from the initial point to the optimum overlaid on the 2D contour plot in the $(x_1, x_2)$-plane.

#### Part 8 — Unit-Norm Constrained Minimization via Gradient Projection

Now impose the unit Euclidean norm constraint

```math
x_1^2 + x_2^2 = 1
\qquad \Longleftrightarrow \qquad
\|x\|_2 = 1.
```

Starting from $x_0 = [6, 6]^T$, solve this constrained problem using the **Gradient Projection** approach based on Steepest Descent with step size $\mu = 0.1$ (normalizing the iterate onto the unit circle at each step). Report the final optimal point and objective value.

#### Part 9 — Analytical Solution via Lagrange Multipliers

Solve the unit-norm constrained optimization problem of Part 8 analytically using the **method of Lagrange multipliers**, and verify that the numerical solution obtained by Gradient Projection in Part 8 matches the analytical global minimum on the unit circle.

*(Final remark on the assignment sheet: Because the unconstrained problem is strictly convex—and on the unit circle the iterative trajectory stays in the basin of attraction of the unique constrained global minimizer—running the algorithms from other reasonable initial points converges to the same optimal solution.)*

## 4. Mathematical Formulation & Solutions

### 4.1 Section 1: Geometric Mixing Matrix Estimation and Marginal PDFs

#### Part 1 — Parallelogram Geometry of the Mixture Scatter Plot

The source vector $S = [s_1, s_2]^T$ has independent components $s_1 \sim \mathcal{U}(-3, 3)$ and $s_2 \sim \mathcal{U}(-2, 2)$. In the $(s_1, s_2)$-plane, the joint support of $S$ is the axis-aligned rectangle

```math
\mathcal{R}_S = [-3, 3] \times [-2, 2],
```

with four vertices $(\pm 3, \pm 2)$. Under the linear instantaneous mixing map $X = AS$,

```math
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
= \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix}
\begin{bmatrix} s_1 \\ s_2 \end{bmatrix}
= \begin{bmatrix} a_{11} \\ a_{21} \end{bmatrix} s_1
+ \begin{bmatrix} a_{12} \\ a_{22} \end{bmatrix} s_2
= \mathbf{a}_1 s_1 + \mathbf{a}_2 s_2.
```

The invertible linear transformation $A$ maps the rectangular support $\mathcal{R}_S$ onto a **parallelogram** $\mathcal{R}_X = A(\mathcal{R}_S)$ in the $(x_1, x_2)$-plane (Figures 1.1 and 1.2 of the report):

- **Holding $s_1$ constant** and varying $s_2 \in [-2, 2]$ traces a line segment parallel to the second mixing column $\mathbf{a}_2 = [a_{12}, a_{22}]^T$, whose slope in the $(x_1, x_2)$-plane is

```math
m_2 = \frac{\Delta x_2}{\Delta x_1}\bigg|_{s_1 = \text{const}} = \frac{a_{22}}{a_{12}} = \frac{-0.6}{0.8} = -0.75.
```

- **Holding $s_2$ constant** and varying $s_1 \in [-3, 3]$ traces a line segment parallel to the first mixing column $\mathbf{a}_1 = [a_{11}, a_{21}]^T$, whose slope in the $(x_1, x_2)$-plane is

```math
m_1 = \frac{\Delta x_2}{\Delta x_1}\bigg|_{s_2 = \text{const}} = \frac{a_{21}}{a_{11}} = \frac{0.8}{0.6} = \frac{4}{3} \approx 1.3333.
```

For the given matrix $A = \begin{bmatrix} 0.6 & 0.8 \\ 0.8 & -0.6 \end{bmatrix}$, the two columns are orthogonal ($\mathbf{a}_1^T \mathbf{a}_2 = 0$, $m_1 m_2 = -1$) and unit-norm ($\|\mathbf{a}_1\|_2 = \|\mathbf{a}_2\|_2 = 1$), so $\mathcal{R}_X$ is a rotated $6 \times 4$ rectangle whose four corners are the images of $(s_1, s_2) \in \{(\pm 3, \pm 2)\}$:

```math
\begin{aligned}
v_1 &= A\begin{bmatrix} 3 \\ 2 \end{bmatrix} = \begin{bmatrix} 3.4 \\ 1.2 \end{bmatrix} \quad (\max x_1),
&\qquad
v_2 &= A\begin{bmatrix} -3 \\ -2 \end{bmatrix} = \begin{bmatrix} -3.4 \\ -1.2 \end{bmatrix} \quad (\min x_1),\\
v_3 &= A\begin{bmatrix} 3 \\ -2 \end{bmatrix} = \begin{bmatrix} 0.2 \\ 3.6 \end{bmatrix} \quad (\max x_2),
&\qquad
v_4 &= A\begin{bmatrix} -3 \\ 2 \end{bmatrix} = \begin{bmatrix} -0.2 \\ -3.6 \end{bmatrix} \quad (\min x_2).
\end{aligned}
```

#### Part 2 — Automatic Mixing Matrix Estimation Algorithms (`section_1.m`)

Because BSS has inherent **scaling** and **permutation** ambiguities, the columns of $A$ are identifiable up to column normalization and ordering. Following the classroom formulation and the structure of $A$, the report imposes the **unit-norm constraint** on the spatial direction vectors ($\|\mathbf{a}_1\|_2 = \|\mathbf{a}_2\|_2 = 1$) and fixes the first component of each unnormalized direction vector to $+1$ before normalization. Two automatic algorithms are developed and implemented in `codes/section_1.m`:

##### Algorithm 1: Direct Extreme-Vertex Slope Estimation

1. **Locate the four extreme sample vertices:** Find the sample indices attaining the minimum and maximum of $x_1$ and $x_2$:

```math
\begin{aligned}
P_{\max, 1} &= \big(x_1(i_{\max, 1}),\, x_2(i_{\max, 1})\big),
&\qquad
P_{\min, 1} &= \big(x_1(i_{\min, 1}),\, x_2(i_{\min, 1})\big),\\
P_{\max, 2} &= \big(x_1(i_{\max, 2}),\, x_2(i_{\max, 2})\big),
&\qquad
P_{\min, 2} &= \big(x_1(i_{\min, 2}),\, x_2(i_{\min, 2})\big).
\end{aligned}
```

2. **Compute and average opposite edge slopes:** Pair the four vertices to form the two pairs of parallel edges of the parallelogram and average their slopes:

```math
\begin{aligned}
\hat{m}_1^{(1)} &= \frac{x_2(i_{\max, 1}) - \min(x_2)}{\max(x_1) - x_1(i_{\min, 2})},
&\qquad
\hat{m}_1^{(2)} &= \frac{\max(x_2) - x_2(i_{\min, 1})}{x_1(i_{\max, 2}) - \min(x_1)},
&\qquad
\hat{m}_1 &= \frac{\hat{m}_1^{(1)} + \hat{m}_1^{(2)}}{2},\\
\hat{m}_2^{(1)} &= \frac{x_2(i_{\min, 1}) - \min(x_2)}{\min(x_1) - x_1(i_{\min, 2})},
&\qquad
\hat{m}_2^{(2)} &= \frac{\max(x_2) - x_2(i_{\max, 1})}{x_1(i_{\max, 2}) - \max(x_1)},
&\qquad
\hat{m}_2 &= \frac{\hat{m}_2^{(1)} + \hat{m}_2^{(2)}}{2}.
\end{aligned}
```

3. **Normalize each direction vector to unit Euclidean norm:** Initialize unnormalized direction vectors $[1, \hat{m}_1]^T$ and $[1, \hat{m}_2]^T$ and divide each by its $\ell_2$-norm:

```math
\hat{\mathbf{a}}_1 = \frac{1}{\sqrt{1 + \hat{m}_1^2}} \begin{bmatrix} 1 \\ \hat{m}_1 \end{bmatrix},
\qquad
\hat{\mathbf{a}}_2 = \frac{1}{\sqrt{1 + \hat{m}_2^2}} \begin{bmatrix} 1 \\ \hat{m}_2 \end{bmatrix}.
```

##### Algorithm 2: Boundary-Point Selection and Linear Regression (`slope_finder`)

Relying solely on four extreme samples makes Algorithm 1 sensitive to finite-sample corner gaps and noise outliers. Algorithm 2 refines the four boundary slopes using all near-boundary samples via the helper function `slope_finder`:

1. **Generate initial boundary segments:** Connect the four extreme vertices from Step 1 using `linspace` with `line_len` points per edge (`line_len = 100000` in `section_1.m`).
2. **Select near-boundary observations:** For each discretized point on a candidate edge, find the closest observation sample in $X$ by Euclidean distance, retain only indices whose minimum distance satisfies $\mathrm{dist} \le \mathrm{dist\_threshold}$ (`dist_threshold = 0.8` in `section_1.m`), and remove duplicate sample indices via `unique()`.
3. **Fit edge slopes via least-squares regression:** Apply first-degree polynomial regression `polyfit(x_1(ind_line), x_2(ind_line), 1)` to the unique boundary observations along each of the four edges, average the slopes of opposite edges ($\hat{m}_1 = (\text{slope}_1 + \text{slope}_2)/2$ and $\hat{m}_2 = (\text{slope}_3 + \text{slope}_4)/2$), and normalize each vector to unit $\ell_2$-norm as in Step 3 of Algorithm 1.

#### Part 3 — Robustness to Additive White Gaussian Noise (`section_1_noiseTest.m`)

In `codes/section_1_noiseTest.m`, additive white Gaussian noise (AWGN) is added to the observations using `X = awgn(X, 15)` (corresponding to $\mathrm{SNR} = 15\ \mathrm{dB}$):

```math
X_{\mathrm{noisy}} = A S + N.
```

- **Vulnerability of Algorithm 1:** Because Gaussian noise perturbs individual extreme points outward in random directions, the four coordinate extrema no longer coincide with the true corners of the noiseless parallelogram $A(\mathcal{R}_S)$. Consequently, slopes computed solely from the four extreme points suffer noticeable bias.
- **Robustness of Algorithm 2:** By setting `line_len = 1000` and tightening the neighborhood radius to `dist_threshold = 0.3` in `section_1_noiseTest.m`, `slope_finder` gathers multiple samples along each edge and performs linear regression (`polyfit`), averaging out zero-mean noise perturbations along the four sides of the parallelogram (Figure 1.4 of the report).
- **Quantitative evaluation metrics:** As noted in the report, the estimation quality under noise can be quantified directly by the **element-wise absolute error** $\| \hat{A} - A \|_{\max} = \max_{i,j} |\hat{a}_{ij} - a_{ij}|$ (or the Frobenius norm error $\|\hat{A} - A\|_F$), which drops from $0.0421$ in Algorithm 1 to $0.0120$ in Algorithm 2 (less than $\approx 0.012$ per entry, well within the report's stated $0.1$ tolerance bound).

#### Part 4 — Analytical Marginal PDF of $x_1$

Since $A = \begin{bmatrix} 0.6 & 0.8 \\ 0.8 & -0.6 \end{bmatrix}$ is symmetric and orthogonal ($\det(A) = -0.36 - 0.64 = -1$), its inverse equals itself:

```math
A^{-1} = \frac{1}{-1} \begin{bmatrix} -0.6 & -0.8 \\ -0.8 & 0.6 \end{bmatrix}
= \begin{bmatrix} 0.6 & 0.8 \\ 0.8 & -0.6 \end{bmatrix} = A.
```

Hence $S = A^{-1} X$ gives the inverse relations

```math
\begin{cases}
s_1 = 0.6 x_1 + 0.8 x_2, \\
s_2 = 0.8 x_1 - 0.6 x_2.
\end{cases}
```

Substituting the source support bounds $-3 \le s_1 \le 3$ and $-2 \le s_2 \le 2$ yields four linear inequalities describing the parallelogram $\mathcal{R}_X$ in the $(x_1, x_2)$-plane (Figure 1.6 of the report):

```math
\begin{cases}
-3 \le 0.6 x_1 + 0.8 x_2 \le 3 \\
-2 \le 0.8 x_1 - 0.6 x_2 \le 2
\end{cases}
\quad \Longleftrightarrow \quad
\begin{cases}
-\dfrac{15}{4} - \dfrac{3}{4} x_1 \le x_2 \le \dfrac{15}{4} - \dfrac{3}{4} x_1 \quad (-3.75 - 0.75 x_1 \le x_2 \le 3.75 - 0.75 x_1), \\[8pt]
-\dfrac{10}{3} + \dfrac{4}{3} x_1 \le x_2 \le \dfrac{10}{3} + \dfrac{4}{3} x_1 \quad (-3.3\overline{3} + 1.3\overline{3} x_1 \le x_2 \le 3.3\overline{3} + 1.3\overline{3} x_1).
\end{cases}
```

By statistical independence of $s_1$ and $s_2$, the joint PDF of $S$ on $[-3, 3] \times [-2, 2]$ is

```math
f_{S_1, S_2}(s_1, s_2) = f_{S_1}(s_1) f_{S_2}(s_2)
= \frac{1}{3 - (-3)} \times \frac{1}{2 - (-2)}
= \frac{1}{6} \times \frac{1}{4} = \frac{1}{24}.
```

Since the Jacobian determinant of the transformation has unit magnitude, $|J| = |\det(A)| = |-1| = 1$, the joint PDF of $(X_1, X_2)$ is uniform over the parallelogram $\mathcal{R}_X$:

```math
f_{X_1, X_2}(x_1, x_2) = \frac{f_{S_1, S_2}(s_1, s_2)}{|J|}
= \begin{cases}
\dfrac{1}{24}, & (x_1, x_2) \in \mathcal{R}_X, \\[6pt]
0, & \text{otherwise}.
\end{cases}
```

Integrating $f_{X_1, X_2}(x_1, x_2)$ with respect to $x_2$ across the three $x_1$-regions partitioned by the parallelogram vertices $x_1 \in \{-3.4, -0.2, 0.2, 3.4\}$ gives the **trapezoidal marginal PDF** $f_{X_1}(x_1)$ (Figure 1.7 of the report):

```math
f_{X_1}(x_1) = \int_{-\infty}^{\infty} f_{X_1, X_2}(x_1, x_2) \, dx_2
= \begin{cases}
\displaystyle \int_{-3.75 - 0.75 x_1}^{10/3 + (4/3) x_1} \frac{1}{24} \, dx_2 = \frac{85 + 25 x_1}{288} \approx \frac{7.08\overline{3} + 2.08\overline{3} x_1}{24}, & -3.4 \le x_1 < -0.2, \\[12pt]
\displaystyle \int_{-10/3 + (4/3) x_1}^{10/3 + (4/3) x_1} \frac{1}{24} \, dx_2 = \frac{20/3}{24} = \frac{5}{18} \approx 0.2778, & -0.2 \le x_1 \le 0.2, \\[12pt]
\displaystyle \int_{-10/3 + (4/3) x_1}^{3.75 - 0.75 x_1} \frac{1}{24} \, dx_2 = \frac{85 - 25 x_1}{288} \approx \frac{7.08\overline{3} - 2.08\overline{3} x_1}{24}, & 0.2 < x_1 \le 3.4, \\[10pt]
0, & \text{otherwise}.
\end{cases}
```

*(In the report, the fractions $10/3 \approx 3.33$ and $4/3 \approx 1.33$ and graphical breakpoint readouts $\{-3.404, -0.202, 0.202, 3.404\}$ are written in two-decimal form, giving $6.66/24 = 0.2775$ on the central plateau and $(7.08 \mp 2.08 x_1)/24$ on the outer ramps, while explicitly noting on page 7 that the exact breakpoints are $\{-3.4, -0.2, 0.2, 3.4\}$.)*

#### Part 5 — Analytical Marginal PDF of $x_2$

Solving the same support inequalities for $x_1$ in terms of $x_2$ yields

```math
\begin{cases}
-5 - \dfrac{4}{3} x_2 \le x_1 \le 5 - \dfrac{4}{3} x_2 \quad (-5 - 1.3\overline{3} x_2 \le x_1 \le 5 - 1.3\overline{3} x_2), \\[8pt]
-2.5 + 0.75 x_2 \le x_1 \le 2.5 + 0.75 x_2.
\end{cases}
```

The four vertices have $x_2$-coordinates $\{-3.6, -1.2, 1.2, 3.6\}$ (read graphically in Figure 1.6 of the report as $\{-3.599, -1.197, 1.197, 3.599\}$). Integrating $f_{X_1, X_2}(x_1, x_2) = 1/24$ with respect to $x_1$ gives the **trapezoidal marginal PDF** $f_{X_2}(x_2)$ (Figure 1.9 of the report):

```math
f_{X_2}(x_2) = \int_{-\infty}^{\infty} f_{X_1, X_2}(x_1, x_2) \, dx_1
= \begin{cases}
\displaystyle \int_{-5 - (4/3) x_2}^{2.5 + 0.75 x_2} \frac{1}{24} \, dx_1 = \frac{90 + 25 x_2}{288} = \frac{7.5 + 2.08\overline{3} x_2}{24}, & -3.6 \le x_2 < -1.2, \\[12pt]
\displaystyle \int_{-2.5 + 0.75 x_2}^{2.5 + 0.75 x_2} \frac{1}{24} \, dx_1 = \frac{5}{24} \approx 0.2083, & -1.2 \le x_2 \le 1.2, \\[12pt]
\displaystyle \int_{-2.5 + 0.75 x_2}^{5 - (4/3) x_2} \frac{1}{24} \, dx_1 = \frac{90 - 25 x_2}{288} = \frac{7.5 - 2.08\overline{3} x_2}{24}, & 1.2 < x_2 \le 3.6, \\[10pt]
0, & \text{otherwise}.
\end{cases}
```

#### Part 6 — Convolution Representation and Non-Uniqueness from Marginal Histograms Alone

**Answer:** **No — the BSS problem cannot be solved uniquely from the marginal histograms alone** (even as $T \to \infty$ with known source distributions), because **sign and intra-row permutation ambiguities** remain in the entries of $A$.

**Mathematical justification:** For the first mixture $X_1 = a_{11} S_1 + a_{12} S_2$, define the scaled variables $S_3 = a_{11} S_1$ and $S_4 = a_{12} S_2$. Because $S_1$ and $S_2$ are statistically independent, $S_3$ and $S_4$ are also independent, so the cumulative distribution function (CDF) of $X_1 = S_3 + S_4$ is

```math
\begin{aligned}
F_{X_1}(x_1)
&= \Pr\{X_1 < x_1\}
= \Pr\{a_{11} S_1 + a_{12} S_2 < x_1\}
= \int_{-\infty}^{\infty} \int_{-\infty}^{x_1 - s_3} f_{S_3, S_4}(s_3, s_4) \, ds_4 \, ds_3 \\
&= \int_{-\infty}^{\infty} f_{S_3}(s_3) \left( \int_{-\infty}^{x_1 - s_3} f_{S_4}(s_4) \, ds_4 \right) ds_3,
\end{aligned}
```

and differentiating with respect to $x_1$ yields the PDF convolution identity (and similarly for $X_2$ with $S_3' = a_{21} S_1$ and $S_4' = a_{22} S_2$):

```math
f_{X_1}(x_1) = \left(f_{S_3} * f_{S_4}\right)(x_1),
\qquad
f_{X_2}(x_2) = \left(f_{S_3'} * f_{S_4'}\right)(x_2).
```

*(In page 10 of the report, the convolution is written symbolically using $F_{X_i}$ on the left-hand side of the boxed summary formula while referring to the breakpoint structure of the marginal distributions.)*

For zero-mean symmetric uniform sources $S_1 \sim \mathcal{U}(-3, 3)$ and $S_2 \sim \mathcal{U}(-2, 2)$, the scaled components are $a_{i1} S_1 \sim \mathcal{U}(-3|a_{i1}|, 3|a_{i1}|)$ and $a_{i2} S_2 \sim \mathcal{U}(-2|a_{i2}|, 2|a_{i2}|)$. The corner breakpoints of each trapezoidal marginal density $f_{X_i}(x_i)$ occur at

```math
\pm \big( 3|a_{i1}| + 2|a_{i2}| \big)
\qquad \text{and} \qquad
\pm \big| 3|a_{i1}| - 2|a_{i2}| \big|.
```

From these breakpoints (for example, $3.4$ and $0.2$ for $x_1$, which give the unordered pair $\{3|a_{11}|, 2|a_{12}|\} = \{1.8, 1.6\}$), one can recover the **magnitudes** of the products:

1. **Permutation ambiguity between the two terms in each row:** Because convolution is commutative ($f_{S_3} * f_{S_4} = f_{S_4} * f_{S_3}$), the marginal histogram of $x_i$ only reveals the unordered pair $\{3|a_{i1}|, 2|a_{i2}|\}$ and cannot tell which half-width came from $S_1$ and which came from $S_2$ without joint cross-channel information.
2. **Sign ambiguity of individual matrix entries:** Because $S_1$ and $S_2$ are symmetric about zero, negating any coefficient $a_{ij} \mapsto -a_{ij}$ leaves the distribution of $a_{ij} S_j$ completely unchanged. Thus, the signs of $a_{11}, a_{12}, a_{21}, a_{22}$ (and the relative sign $a_{22} = -0.6$ versus $+0.6$ that makes $A$ invertible/orthogonal rather than rank-deficient) cannot be determined from the two marginal histograms alone; joint information from $(x_1, x_2)$ (such as the 2D scatter plot) is essential.

### 4.2 Section 2: Unconstrained and Constrained Quadratic Optimization

#### Parts 1 & 2 — Objective Function and Logarithmic (dB) Scaling

For $x = [x_1, x_2]^T \in [-10, 10]^2$, the quadratic objective is

```math
f(x_1, x_2) = x_1^2 + x_2^2 - 4x_1 - 6x_2 + 13 + x_1 x_2.
```

Because $f(x) \ge 11/3 > 0$ everywhere on $\mathbb{R}^2$, its decibel representation

```math
f_{\mathrm{dB}}(x_1, x_2) = 10 \log_{10} f(x_1, x_2)
```

is well-defined and real-valued over the entire plane, compressing large values far from the minimum and sharpening the visual resolution of the valley around the optimum (Figures 2.1 and 2.2 of the report).

#### Part 3 — Gradient, Hessian, and Proof of Strict Convexity

Differentiating $f(x_1, x_2)$ with respect to $x_1$ and $x_2$ gives the gradient vector and Hessian matrix:

```math
g(x) = \nabla f(x) = \begin{bmatrix} \dfrac{\partial f}{\partial x_1} \\[8pt] \dfrac{\partial f}{\partial x_2} \end{bmatrix}
= \begin{bmatrix} 2x_1 + x_2 - 4 \\ x_1 + 2x_2 - 6 \end{bmatrix},
\qquad
H = \nabla^2 f(x) = \begin{bmatrix} \dfrac{\partial^2 f}{\partial x_1^2} & \dfrac{\partial^2 f}{\partial x_1 \partial x_2} \\[8pt] \dfrac{\partial^2 f}{\partial x_2 \partial x_1} & \dfrac{\partial^2 f}{\partial x_2^2} \end{bmatrix}
= \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}.
```

To prove that $H$ is symmetric positive definite ($x^T H x > 0$ for all $x \neq 0$), expand the quadratic form:

```math
x^T H x
= \begin{bmatrix} x_1 & x_2 \end{bmatrix}
\begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
= 2\left(x_1^2 + x_1 x_2 + x_2^2\right).
```

Viewing $q(x_1) = x_1^2 + x_2 x_1 + x_2^2$ as a quadratic polynomial in $x_1$ for any nonzero $x_2$, its discriminant is

```math
\Delta = x_2^2 - 4(1)(x_2^2) = -3x_2^2 < 0
\qquad (\forall x_2 \neq 0),
```

and equivalently by completing the square,

```math
x^T H x = 2\left[\left(x_1 + \frac{1}{2}x_2\right)^2 + \frac{3}{4}x_2^2\right] > 0
\qquad \forall x \in \mathbb{R}^2 \setminus \{0\}.
```

*(Also, the eigenvalues of $H$ are $\lambda_1 = 1 > 0$ and $\lambda_2 = 3 > 0$, or by Sylvester's criterion $\det(H_1) = 2 > 0$ and $\det(H_2) = 3 > 0$.)* Hence $H \succ 0$ everywhere, proving that $f(x)$ is **strictly convex** and has a **unique global minimum**.

#### Part 4 — Analytical Unconstrained Minimizer

Setting the gradient equal to zero yields a $2 \times 2$ linear system:

```math
\nabla f(x^*) = \begin{bmatrix} 2x_1^* + x_2^* - 4 \\ x_1^* + 2x_2^* - 6 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
\quad \Longrightarrow \quad
\boxed{
x^* = \begin{bmatrix} 2/3 \\ 8/3 \end{bmatrix} \approx \begin{bmatrix} 0.6667 \\ 2.6667 \end{bmatrix},
\qquad
f(x^*) = \frac{11}{3} \approx 3.6667.
}
```

#### Part 5 — Steepest Descent (`steepest_descend` in `section_2.m`)

Starting from $x_0 = [6, 6]^T$ (where $f(x_0) = 85$), the Steepest Descent update with constant step size $\mu > 0$ is

```math
x_{k+1} = x_k - \mu \nabla f(x_k)
= x_k - \mu \begin{bmatrix} 2x_{k,1} + x_{k,2} - 4 \\ x_{k,1} + 2x_{k,2} - 6 \end{bmatrix}.
```

In `codes/section_2.m`, the stopping criterion is based on the absolute change in consecutive objective values:

```math
\left| f(x_{k+1}) - f(x_k) \right| < \varepsilon,
\qquad
\varepsilon = 10^{-6}.
```

Since $H = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$ has maximum eigenvalue $\lambda_{\max}(H) = 3$, fixed-step Steepest Descent on this quadratic is stable whenever $0 < \mu < 2/\lambda_{\max}(H) = 2/3$. Both $\mu = 0.1$ and $\mu = 0.01$ satisfy this condition and converge, with $\mu = 0.1$ converging much faster ($59$ iterations versus $494$ iterations).

#### Part 6 — Newton's Method (`newton` in `section_2.m`)

Newton's update is

```math
x_{k+1} = x_k - H(x_k)^{-1} \nabla f(x_k).
```

Because $f(x)$ is an exact second-degree quadratic with constant Hessian $H = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$ and linear gradient $\nabla f(x) = Hx - b$ (where $b = [4, 6]^T$), the very first Newton step from any initial point $x_0 \in \mathbb{R}^2$ produces

```math
x_1 = x_0 - H^{-1}(Hx_0 - b) = H^{-1} b = \begin{bmatrix} 2/3 \\ 8/3 \end{bmatrix} = x^*.
```

Thus **Newton's method reaches the exact global minimizer in $1$ iteration**. *(Note on `section_2.m`: inside `newton`, the first step `x_k_new` is already $[2/3, 8/3]^T$, and the `while` loop performs one additional zero-step verification pass so that `abs(f_new - f) = 0 < 1e-6` terminates; because `out` is initialized as `out = f` with a single element rather than `[f f_new]`, `length(out) - 1` prints `1`.)*

#### Part 7 — Alternating Minimization (`alternation_minimization` in `section_2.m`)

Setting the partial derivatives $\partial f / \partial x_1 = 0$ and $\partial f / \partial x_2 = 0$ individually yields the closed-form coordinate minimization rules:

```math
\begin{aligned}
\arg\min_{x_1} f(x_1, x_2) &: \quad 2x_1 + x_2 - 4 = 0 \quad \Longrightarrow \quad x_1 = 2 - \frac{x_2}{2}, \\
\arg\min_{x_2} f(x_1, x_2) &: \quad x_1 + 2x_2 - 6 = 0 \quad \Longrightarrow \quad x_2 = 3 - \frac{x_1}{2}.
\end{aligned}
```

Starting from $x_0 = [6, 6]^T$, `alternation_minimization` updates $x_1$ on odd steps and $x_2$ on even steps, creating an axis-aligned staircase trajectory in the $(x_1, x_2)$-plane (Figure 2.4 of the report) that terminates after $14$ coordinate updates when $|f(x_{k+1}) - f(x_k)| < 10^{-6}$ at $x \approx [0.6663, 2.6669]^T$.

#### Part 8 — Gradient Projection on the Unit Circle (`gradient_projection` in `section_2.m`)

Under the unit-norm constraint $\|x\|_2 = 1$, each iteration takes a Steepest Descent step with $\mu = 0.1$ followed immediately by Euclidean normalization onto the unit circle:

```math
\tilde{x}_{k+1} = x_k - \mu \nabla f(x_k),
\qquad
x_{k+1} = \frac{\tilde{x}_{k+1}}{\|\tilde{x}_{k+1}\|_2}.
```

In `gradient_projection`, the initial vector $x_0 = [6, 6]^T$ is also projected onto the unit circle at the first step, and the loop terminates in $15$ iterations under $|f(x_{k+1}) - f(x_k)| < 10^{-6}$, yielding

```math
x_{\mathrm{GP}} \approx \begin{bmatrix} 0.4946 \\ 0.8691 \end{bmatrix},
\qquad
f(x_{\mathrm{GP}}) \approx 7.2367.
```

#### Part 9 — Analytical Solution via Lagrange Multipliers

To extremize $f(x_1, x_2) = x_1^2 + x_2^2 - 4x_1 - 6x_2 + x_1 x_2 + 13$ subject to $g(x_1, x_2) = x_1^2 + x_2^2 = 1$, enforce the Lagrange multiplier stationarity condition $\nabla f(x) = \gamma \nabla g(x)$:

```math
\begin{cases}
2x_1 - 4 + x_2 = \gamma (2x_1), \\
2x_2 - 6 + x_1 = \gamma (2x_2), \\
x_1^2 + x_2^2 = 1.
\end{cases}
```

Assuming $x_1 \neq 0$ and $x_2 \neq 0$, divide the first equation by $2x_1$ and the second by $2x_2$:

```math
\gamma = 1 - \frac{2}{x_1} + \frac{x_2}{2x_1} = 1 - \frac{3}{x_2} + \frac{x_1}{2x_2}
\quad \Longrightarrow \quad
\frac{2}{x_1} - \frac{x_2}{2x_1} = \frac{3}{x_2} - \frac{x_1}{2x_2}.
```

Multiplying through by $2x_1 x_2$ and using $x_2^2 = 1 - x_1^2$ gives $4x_2 = 1 + 6x_1 - 2x_1^2$. Substituting this into $16(1 - x_1^2) = (4x_2)^2$ produces the quartic polynomial

```math
4x_1^4 - 24x_1^3 + 48x_1^2 + 12x_1 - 15 = 0.
```

Solving this system yields four roots (two real feasible stationary points on the unit circle and one complex conjugate pair):

1. **Constrained global minimum:**

```math
\boxed{
x_1^* \approx 0.4943, \qquad x_2^* \approx 0.8693, \qquad f(x^*) \approx 7.2367 \quad (\gamma \approx -2.1668).
}
```

2. **Constrained global maximum:**

```math
x_1 \approx -0.5894, \qquad x_2 \approx -0.8078, \qquad f(x) \approx 21.6808 \quad (\gamma \approx 5.0785).
```

3. **Inadmissible complex roots:**

```math
x_1 \approx 3.0476 \pm j\,1.8930 \notin \mathbb{R}.
```

Therefore, the unique constrained global minimizer on the unit circle is $(x_1^*, x_2^*) \approx (0.494, 0.869)$ with minimum value $f(x^*) \approx 7.237$, which matches the Gradient Projection solution $[0.4946, 0.8691]^T$ from Part 8.

## 5. Results & Discussion

### 5.1 Section 1 — Mixing Matrix Estimation Results (Parts 2 & 3)

Because `unifrnd` and `awgn` draw random realizations on each run, the numerical matrices below are the representative single-run results documented in the submitted Persian report compared against the ground-truth mixing matrix $A = \begin{bmatrix} 0.6 & 0.8 \\ 0.8 & -0.6 \end{bmatrix}$:

| Scenario & Method | Script | Parameters | Reported Estimate $\hat{A}$ | Max Element-wise Error $\|\hat{A}-A\|_{\max}$ | Frobenius Error $\|\hat{A}-A\|_F$ |
| --- | --- | --- | --- | --- | --- |
| Ground-truth mixing matrix $A$ | — | $T = 1000$ | $`\begin{bmatrix} 0.6000 & 0.8000 \\ 0.8000 & -0.6000 \end{bmatrix}`$ | $0$ | $0$ |
| Noiseless — Algorithm 1 (Extreme vertices) | `section_1.m` | 4 extreme points | $`\begin{bmatrix} 0.5986 & 0.8011 \\ 0.8154 & -0.5788 \end{bmatrix}`$ | $0.0212$ | $0.0263$ |
| Noiseless — Algorithm 2 (`slope_finder` + `polyfit`) | `section_1.m` | `line_len = 100000`, `dist_threshold = 0.8` | $`\begin{bmatrix} 0.6004 & 0.7997 \\ 0.8086 & -0.5883 \end{bmatrix}`$ | $0.0117$ | $0.0145$ |
| Noisy ($\mathrm{SNR}=15\ \mathrm{dB}$) — Algorithm 1 (Extreme vertices) | `section_1_noiseTest.m` | 4 extreme points | $`\begin{bmatrix} 0.5784 & 0.8158 \\ 0.7839 & -0.6209 \end{bmatrix}`$ | $0.0216$ | $0.0376$ |
| Noisy ($\mathrm{SNR}=15\ \mathrm{dB}$) — Algorithm 2 (`slope_finder` + `polyfit`) | `section_1_noiseTest.m` | `line_len = 1000`, `dist_threshold = 0.3` | $`\begin{bmatrix} 0.5909 & 0.8068 \\ 0.8086 & -0.6120 \end{bmatrix}`$ | $0.0120$ | $0.0186$ |

### 5.2 Section 1 — Summary of Marginal Distributions (Parts 4, 5 & 6)

| Mixture | Support Interval | Central Plateau Interval | Plateau Density | Ramp Expressions (Exact / Report Decimal Form) |
| --- | --- | --- | --- | --- |
| $x_1 = 0.6 s_1 + 0.8 s_2$ | $[-3.4,\ 3.4]$ | $[-0.2,\ 0.2]$ | $\dfrac{5}{18} \approx 0.2778$ ($0.2775$ in report) | $\dfrac{85 \mp 25 x_1}{288} \approx \dfrac{7.08 \mp 2.08 x_1}{24}$ |
| $x_2 = 0.8 s_1 - 0.6 s_2$ | $[-3.6,\ 3.6]$ | $[-1.2,\ 1.2]$ | $\dfrac{5}{24} \approx 0.2083$ | $\dfrac{90 \mp 25 x_2}{288} = \dfrac{7.5 \mp 2.08\overline{3} x_2}{24}$ |

Both 20-bin empirical histograms (Figures 1.5 and 1.8 in the report) closely follow the trapezoidal analytical PDFs (Figures 1.7 and 1.9), with small bin-to-bin fluctuations attributable to the finite sample count $T = 1000$.

### 5.3 Section 2 — Optimization Algorithms Summary (Parts 4–9)

All iterative methods in `codes/section_2.m` start from $x_0 = [6, 6]^T$ with stopping tolerance $|f(x_{k+1}) - f(x_k)| < 10^{-6}$:

| Problem Part & Method | Constraint | Step Size $\mu$ | Iterations | Final Point $[x_1,\ x_2]^T$ | Final Objective $f(x)$ |
| --- | --- | --- | --- | --- | --- |
| Part 4 — Analytical unconstrained optimum ($\nabla f = 0$) | Unconstrained | — | — | $[2/3,\ 8/3]^T \approx [0.6667,\ 2.6667]^T$ | $11/3 \approx 3.6667$ |
| Part 5 — Steepest Descent (`steepest_descend`) | Unconstrained | $0.1$ | $59$ | $[0.6687,\ 2.6647]^T$ | $3.6667$ |
| Part 5 — Steepest Descent (`steepest_descend`) | Unconstrained | $0.01$ | $494$ | $[0.6736,\ 2.6597]^T$ | $3.6667$ |
| Part 6 — Newton's method (`newton`) | Unconstrained | — | $1$ | $[0.6667,\ 2.6667]^T$ | $3.6667$ |
| Part 7 — Alternating Minimization (`alternation_minimization`) | Unconstrained | Closed-form coordinate steps | $14$ | $[0.6663,\ 2.6669]^T$ | $3.6667$ |
| Part 8 — Gradient Projection (`gradient_projection`) | $\|x\|_2 = 1$ | $0.1$ | $15$ | $[0.4946,\ 0.8691]^T$ | $7.2367$ |
| Part 9 — Analytical Lagrange multipliers | $\|x\|_2 = 1$ | — | — | $[0.4943,\ 0.8693]^T$ | $7.2367$ |

### 5.4 Notes on the Submitted Solutions / Implementation Notes

- **Header metadata on the Persian report:** The cover header of `BSS_HW#1_810100205_Arman_Majidi.pdf` lists an earlier student ID (`810100205`) and instructor name from when the report PDF was originally compiled; the standardized repository metadata at the top of this README follows the course reference convention (`Dr. Masoud Babaie-Zadeh`, Student ID `404200644`).
- **Row-wise vs. column-wise normalization in `section_1.m` and `section_1_noiseTest.m`:** In Part 2 of the report, the theoretical derivation defines the spatial direction vectors as the columns of $A$, with slopes $a_{21}/a_{11} = 4/3$ and $a_{22}/a_{12} = -3/4$. In the MATLAB scripts (`section_1.m` lines 38–48 and 84–94, and `section_1_noiseTest.m` lines 36–46 and 68–78), `A_pred` is initialized as `ones(2,2)`, the two reciprocal slopes `a_1_slope_pred` ($\approx 4/3$) and `a_2_slope_pred` ($\approx -3/4$) are placed in the second column `A_pred(:, 2)`, and each **row** is normalized to unit norm. Because the ground-truth matrix $A = \begin{bmatrix} 0.6 & 0.8 \\ 0.8 & -0.6 \end{bmatrix}$ is symmetric ($a_{12} = a_{21} = 0.8$) and orthogonal with $a_{11} = 0.6 > 0$ and $a_{21} = 0.8 > 0$, normalizing rows $[1, 4/3]$ and $[1, -3/4]$ produces $\begin{bmatrix} 0.6 & 0.8 \\ 0.8 & -0.6 \end{bmatrix}$, matching $A$ directly.
- **Minor typographical slips in the Persian report (Pages 7, 9, 10, and 12):**
  - On **Page 9 (Part 5)**, the first line of the boxed inequality for $x_1$ is printed as $5 - 1.33 x_2 \le x_1 \le 5 - 1.33 x_2$ (missing the minus sign on the lower bound $-5 - 1.33 x_2$), and inside the piecewise integral for $f_{X_2}(x_2)$, the dummy variable $x_1$ is written on the right-hand side instead of $x_2$ (and $7.05 + 2.08 x_1$ instead of $7.5 + 2.08 x_2$). The plotted marginal PDF in Figure 1.9 is drawn using the correct symmetric formula in $x_2$.
  - On **Page 10 (Part 6)**, in the second line for $F_{X_2}(x_2)$, the upper integration limit and right-hand argument are copied as $x_1 - s_3$ and $x_1$ instead of $x_2 - s_3$ and $x_2$, and the boxed convolution formulas write the CDF symbol $F_{X_i}$ on the left-hand side of the PDF convolution $f_{S_4} * f_{S_3}$.
  - On **Page 12 (Part 3)**, the discriminant of $x_1^2 + x_1 x_2 + x_2^2$ is written over the arrow as $\Delta = x_2^2 - 4 * x_2^2 = 3x_2^2$ (omitting the minus sign before $3x_2^2$), while the next arrow immediately writes the correct conclusion $\Delta < 0 \to \text{no real roots}$.

## 6. How to Use

### 6.1 Running the MATLAB Scripts

Open MATLAB, navigate to the `codes` directory (or add it to the MATLAB path), and run the three scripts:

```matlab
cd codes
section_1
section_1_noiseTest
section_2
```

- **Dependencies:** `section_1.m` and `section_1_noiseTest.m` use `unifrnd` (Statistics and Machine Learning Toolbox), and `section_1_noiseTest.m` uses `awgn` (Communications Toolbox). `section_2.m` uses only base MATLAB functions.
- **Generated Figures:**
  - `section_1.m` opens 3 figures: the scatter plot of $X$ with the four parallelogram boundary segments overlaid, the 20-bin histogram of $x_1$, and the 20-bin histogram of $x_2$.
  - `section_1_noiseTest.m` opens 1 figure: the noisy scatter plot of $X$ ($\mathrm{SNR} = 15\ \mathrm{dB}$) with the four estimated boundary segments.
  - `section_2.m` opens 4 figures: the 3D mesh plots (linear and dB scale), the 2D contour plots (linear and dB scale), the Steepest Descent convergence curves for $\mu = 0.1$ and $\mu = 0.01$, and the Alternating Minimization staircase trajectory overlaid on the log-scale contour plot.

### 6.2 Expected Console Output (`section_2.m`)

Because `section_2.m` is completely deterministic, running it produces the exact console output below:

```text
S.D: final value of x for mu = 0.10 is [0.668663, 2.664670] in iteration: 59
S.D: final value of x for mu = 0.01 is [0.673647, 2.659689] in iteration: 494

Newton: final value of x for is [0.666667, 2.666667] in iteration: 1

Alternation Minimization: final value of x for is [0.666260, 2.666870] in iteration: 14

G.P: final value of x for mu = 0.10 is [0.494570, 0.869138] in iteration: 15
```

### 6.3 Reading the Theoretical Derivations and Report

1. Use **Section 3** for the complete English translation of the assignment problems and **Section 4** for the step-by-step mathematical derivations and algorithm logic.
2. Use **Section 5** to review the quantitative tables and implementation notes alongside the original Persian report [BSS\_HW#1\_810100205\_Arman\_Majidi.pdf](./BSS_HW%231_810100205_Arman_Majidi.pdf).

## 7. References

1. Akhavan, S. *Blind Source Separation: Homework 1*. Course assignment sheet and lecture notes, University of Tehran.
2. Comon, P., & Jutten, C. (Eds.). (2010). *Handbook of Blind Source Separation: Independent Component Analysis and Applications*. Academic Press.
3. Nocedal, J., & Wright, S. J. (1999). *Numerical Optimization*. Springer.
4. [GitHub Docs: Writing mathematical expressions](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions).
