# Homework 10: Independent Component Analysis (ICA) via Deflationary Kurtosis & Outlier-Robust Negentropy Maximization Using Gradient Projection and FastICA Fixed-Point Algorithms

**Course:** Blind Source Separation  
**Instructor:** Dr. Saeed Akhavan  
**Author:** Arman Majidi  
**Student ID:** 810100205  
**University:** University of Tehran  
**Semester:** Spring 2024-2025

## 1. Overview & Assignment Structure

This folder contains the **English problem statements, analytical derivations, MATLAB implementations, and simulation results for Homework 10**. The assignment investigates **Independent Component Analysis (ICA)** for noisy linear instantaneous mixtures of three statistically independent sources by combining spatial whitening with **deflationary non-Gaussianity maximization** across two contrast criteria and two numerical optimization schemes:

1. **Preliminary Signal Modeling & Visualization (Part 0):** Loads the $3 \times 3$ mixing matrix $A$, the $3 \times 1001$ source matrix $S$, and the $3 \times 1001$ additive noise matrix $\mathrm{Noise}$ from `hw10.mat`, synthesizes both the noiseless observations $X_{\mathrm{noNoise}} = A S$ and the noisy observations $X = A S + \mathrm{Noise}$, and visualizes the three source waveforms alongside the noiseless and noisy sensor mixtures.
2. **Section 1 — Deflationary Kurtosis Maximization via Gradient Projection (`Section1.m`):** Applies PCA-based spatial whitening $Z = W X$ and extracts the rows of the orthonormal separating matrix $B \in \mathbb{R}^{3 \times 3}$ sequentially (deflation) by maximizing the absolute kurtosis score $\left|\mathrm{kurt}(y)\right|$ using the **Gradient Projection (GP)** algorithm with Gram-Schmidt orthogonalization and unit-norm projection. Evaluates the global transfer matrix $P = B W A$, resolves permutation, sign, and scaling ambiguities via cross-correlation matching and energy normalization, computes the relative squared Frobenius source reconstruction error $E = \|\hat{S} - S\|_F^2 / \|S\|_F^2$, and examines the score-function convergence trajectories over $5\,000$ and $50\,000$ iterations.
3. **Section 2 — Deflationary Kurtosis Maximization via Fixed-Point Iteration / Kurtosis FastICA (`Section2.m`):** Replaces gradient ascent with the **fixed-point iteration** scheme for kurtosis maximization (kurtosis-based **FastICA**), evaluating the global transfer matrix $P = B W A$, aligned source reconstructions $\hat{S}$, relative error $E$, and score-function convergence curves.
4. **Section 3 — Outlier-Robust Negentropy Maximization via Gradient Projection (`Section3.m`):** Replaces the fourth-order kurtosis contrast (which is sensitive to outliers) with an outlier-robust non-quadratic negentropy contrast function $G(y) = -\exp(-y^2 / 2)$ optimized sequentially via **Gradient Projection (GP)**, reporting the transfer matrix $P = B W A$, source waveforms, relative error $E$, and convergence curves.
5. **Section 4 — Outlier-Robust Negentropy Maximization via Modified Fixed-Point Iteration / Final FastICA (`Section4.m`):** Implements the **modified fixed-point (final FastICA)** update rule $b \leftarrow \mathbb{E}\{Z g(y)\} - \mathbb{E}\{g'(y)\} b$ using the Gaussian contrast $G(y) = -\exp(-y^2 / 2)$ ($g(y) = y \exp(-y^2 / 2)$ and $g'(y) = (1 - y^2)\exp(-y^2 / 2)$), reporting all separation metrics and convergence curves.
6. **Section 5 — Controlled Empirical Comparison of the Four ICA Algorithms:** Benchmarks all four methods from an identical initial matrix `B_init` stored in `B_init.mat` at both $\mathrm{iterNum} = 5\,000$ and $\mathrm{iterNum} = 50\,000$, comparing wall-clock execution time (`tic`/`toc`), iterations required for row convergence, and final source separation error $E$.

All five sections of the assignment are mandatory; no optional or bonus problems were assigned. Because both the original assignment sheet and the submitted solution report are written in Persian, this `README.md` provides a **complete, self-contained English reference for every problem statement, mathematical derivation, algorithm, and numerical result**.

## 2. Directory Contents

| File | Description |
| --- | --- |
| [codes/Section1.m](./codes/Section1.m) | MATLAB script for Part 0 and Section 1 (Subparts 1-1, 1-2, and 1-3): waveform plots of $S$, $X_{\mathrm{noNoise}}$, and $X$, spatial whitening $Z = W X$, deflationary kurtosis maximization via Gradient Projection (GP), permutation/scale alignment, relative error $E$, and convergence plots. |
| [codes/Section2.m](./codes/Section2.m) | MATLAB script for Section 2: deflationary kurtosis maximization via fixed-point iteration (kurtosis-based FastICA), evaluation of $P = B W A$ and relative error $E$, source overlay plots, and convergence curves. |
| [codes/Section3.m](./codes/Section3.m) | MATLAB script for Section 3: outlier-robust negentropy maximization with contrast $G(y) = -\exp(-y^2 / 2)$ via Gradient Projection (GP), evaluation of $P = B W A$ and relative error $E$, source overlay plots, and convergence curves. |
| [codes/Section4.m](./codes/Section4.m) | MATLAB script for Section 4: outlier-robust negentropy maximization with contrast $G(y) = -\exp(-y^2 / 2)$ via modified fixed-point iteration (final FastICA), evaluation of $P = B W A$ and relative error $E$, source overlay plots, and convergence curves. |
| [codes/hw10.mat](./codes/hw10.mat) | Primary dataset containing the $3 \times 3$ mixing matrix `A`, the $3 \times 1001$ ground-truth source matrix `S`, and the $3 \times 1001$ additive noise matrix `Noise`. |
| [codes/B\_init.mat](./codes/B_init.mat) | Saved $3 \times 3$ initialization matrix `B_init` used in Section 5 to benchmark all four ICA algorithms from an identical starting point. |
| [BSS\_HW#10\_810100205\_Arman\_Majidi.pdf](./BSS_HW%2310_810100205_Arman_Majidi.pdf) | Submitted report and analytical solutions in Persian (including BSS block diagrams, mathematical update rules, source and observation plots, permutation matrices, convergence plots at $5\,000$ and $50\,000$ iterations, and runtime/error comparisons). |
| [hw10.pdf](./hw10.pdf) | Original assignment problem sheet in Persian. |
| `README.md` | Self-contained English problem statements, mathematical derivations, algorithm descriptions, numerical results, and usage guide. |

Keep the `codes` subfolder inside the homework directory.

## 3. Assignment Problem Statement

### 3.1 General Problem Setup and Preliminary Visualization (Part 0)

In this assignment, we solve the **Blind Source Separation (BSS)** problem under the assumption that the underlying source signals are mutually **statistically independent** (Independent Component Analysis — ICA).

The mixing matrix $A \in \mathbb{R}^{3 \times 3}$, the source matrix $S \in \mathbb{R}^{3 \times 1001}$, and the additive noise matrix $\mathrm{Noise} \in \mathbb{R}^{3 \times 1001}$ are provided in the file `hw10.mat`. Specifically, the mixing matrix in `hw10.mat` has unit-norm columns given by

```math
A = \frac{1}{\sqrt{14}} \begin{bmatrix}
1 & 1 & 1 \\
2 & -2 & -3 \\
3 & 3 & -2
\end{bmatrix}
\approx \begin{bmatrix}
0.2673 & 0.2673 & 0.2673 \\
0.5345 & -0.5345 & -0.8018 \\
0.8018 & 0.8018 & -0.5345
\end{bmatrix},
```

and the three rows of $S$ are zero-mean unit-energy signals ($\|s_i\|_2 = 1$ for $i = 1, 2, 3$, so $\|S\|_F^2 = 3$): a smooth periodic wave $s_1(t)$, a triangular wave $s_2(t)$, and a sparse spiky pulse train $s_3(t)$.

First, obtain the noisy observation matrix $X$ using the linear instantaneous mixing model:

```math
X = A S + \mathrm{Noise}.
```

Plot the original sources $S$, the noiseless observations $X_{\mathrm{noNoise}} = A S$, and the noisy observations $X = A S + \mathrm{Noise}$ to inspect their time-domain waveforms.

Next, approach the problem from a blind source separation perspective: assume that only the noisy observation matrix $X$ is known along with the number of sources ($M = 3$). Our strategy is to multiply the spatially whitened observation matrix $Z = W X$ by an orthonormal separating matrix $B \in \mathbb{R}^{3 \times 3}$ ($Y = B Z$) so that the extracted output signals $y_m = b_m^T Z$ ($m = 1, 2, 3$) are as **non-Gaussian** as possible.

### 3.2 Section 1: Deflationary Kurtosis Maximization via Gradient Projection (GP)

Implement the first ICA method presented in class based on maximizing the output **kurtosis** ($\left|\mathrm{kurt}(y)\right|$). Specifically, using a **deflation** approach, extract the rows of the separating matrix $B$ one by one and perform the associated optimizations using the **Gradient Projection (GP)** algorithm. Note that the overall separating matrix mapping $X$ to the separated outputs is the product of the final orthonormal matrix $B$ obtained from the algorithm and the spatial whitening matrix $W$ ($B_{\mathrm{total}} = B W$).

#### Subpart 1-1 — Global Transfer / Permutation Matrix Verification

Multiply the final overall separating matrix ($B W$) obtained from the algorithm by the true mixing matrix $A$, and report the resulting $3 \times 3$ product matrix:

```math
P = B W A.
```

Ideally, this product matrix should be close to a generalized **permutation matrix**, meaning that each row and each column contains only one dominant non-zero entry.

#### Subpart 1-2 — Ambiguity Removal, Waveform Comparison, and Reconstruction Error

Resolve both the **permutation ambiguity** and the **scaling (and sign) ambiguity** of the estimated sources—specifically, normalize the energy of each estimated source so that it equals the energy of the corresponding true source. Plot the aligned estimated sources $\hat{S}$ overlaid on the true sources $S$, and compute and report the relative squared Frobenius error:

```math
E = \frac{\|\hat{S} - S\|_F^2}{\|S\|_F^2}.
```

#### Subpart 1-3 — Convergence Curve Visualization

Plot the convergence curve (the objective / score function value versus the iteration index) for each extracted row of $B$.

### 3.3 Section 2: Deflationary Kurtosis Maximization via Fixed-Point Iteration (Kurtosis FastICA)

Implement the second method presented in class, which performs kurtosis maximization using a **fixed-point** iteration approach (kurtosis-based **FastICA**). Report all results (the global transfer matrix $P = B W A$, the overlaid source waveforms, the relative error $E$, and the convergence curves) in the exact same format as Section 1.

### 3.4 Section 3: Outlier-Robust Negentropy Maximization via Gradient Projection (GP)

Implement the third method presented in class, which is insensitive (robust) to outlier data points, using the non-quadratic contrast function

```math
G(y) = -\exp\!\left(-\frac{y^2}{2}\right)
```

and the **Gradient Projection (GP)** optimization algorithm. Report all results in the exact same format as Section 1.

### 3.5 Section 4: Outlier-Robust Negentropy Maximization via Modified Fixed-Point Iteration (Final FastICA)

Implement the fourth method presented in class *(labeled as item 4 on the assignment sheet with the text "روش سوم..." carried over from item 3)*, which is robust to outliers with

```math
G(y) = -\exp\!\left(-\frac{y^2}{2}\right)
```

and uses the **"modified fixed-point" (final FastICA)** optimization algorithm. Report all results in the exact same format as Section 1.

### 3.6 Section 5: Empirical Comparison of the Four ICA Algorithms

Empirically compare the four methods implemented in Sections 1 through 4 and state which algorithm converges fastest and which achieves the best source separation quality (in terms of the error parameter $E$).

## 4. Mathematical Formulation & Solutions

### 4.1 Two-Stage ICA Architecture: Spatial Whitening and Orthonormal Deflation

Across all four sections (Figures 1.4, 2.1, 3.1, and 4.1 of the report), the noisy instantaneous BSS system is modeled as the cascade

```math
S \longrightarrow \boxed{A} \longrightarrow X \longrightarrow \boxed{W} \longrightarrow Z \longrightarrow \boxed{B} \longrightarrow Y,
```

where $X = A S + \mathrm{Noise} \in \mathbb{R}^{M \times T}$ ($M = 3$, $T = 1001$; overall input $\mathrm{SNR} \approx 22.11\ \mathrm{dB}$).

#### Stage 1 — Spatial Whitening via PCA (`eig`)

In `Section1.m` through `Section4.m`, the sample spatial correlation matrix $R_X \in \mathbb{R}^{3 \times 3}$ is computed and decomposed into its sorted eigenvalue/eigenvector matrices $(\lambda_1 \ge \lambda_2 \ge \lambda_3 > 0)$:

```math
R_X = X X^T = U D U^T,
\qquad
D = \begin{bmatrix}
1.8552 & 0 & 0 \\
0 & 1.2133 & 0 \\
0 & 0 & 0.0514
\end{bmatrix}.
```

The $3 \times 3$ spatial whitening matrix $W$ and whitened data matrix $Z \in \mathbb{R}^{3 \times T}$ are then formed as

```math
W = D^{-1/2} U^T
= \begin{bmatrix}
-0.1476 & 0.6974 & 0.1757 \\
-0.3460 & 0.1355 & -0.8284 \\
3.9787 & 1.2111 & -1.4637
\end{bmatrix},
\qquad
Z = W X,
```

which satisfies $Z Z^T = D^{-1/2} U^T (X X^T) U D^{-1/2} = I_3$.

#### Stage 2 — Sequential (Deflationary) Extraction of Orthonormal Rows of $B$

Given spatially whitened data $Z$, the remaining separating transformation $B = \begin{bmatrix} b_1 & b_2 & b_3 \end{bmatrix}^T \in \mathbb{R}^{3 \times 3}$ is restricted to be **orthonormal** ($B B^T = I_3$). Using a **deflation** strategy, the rows $b_i^T$ ($i = 1, 2, \dots, M$) are estimated one by one by maximizing a single-unit non-Gaussianity objective $H(y_i)$ where $y_i = b_i^T Z$:

```math
f(B) = \sum_{m=1}^{M} H(y_m),
\qquad y_m = b_m^T Z.
```

To prevent the $i$-th vector $b_i$ from converging to any of the previously extracted directions $\{b_1, \dots, b_{i-1}\}$ and to maintain unit norm $\|b_i\|_2 = 1$, every iteration projects $b_i$ onto the orthogonal complement of $\mathrm{span}\{b_1, \dots, b_{i-1}\}$ via Gram-Schmidt projection followed by Euclidean normalization:

```math
\begin{aligned}
b_i &\leftarrow \left(I_M - \sum_{j=1}^{i-1} b_j b_j^T\right) b_i
= \left(I_M - B_{1:i-1,:}^T B_{1:i-1,:}\right) b_i, \\
b_i &\leftarrow \frac{b_i}{\|b_i\|_2},
\end{aligned}
```

implemented in MATLAB across all four scripts as:

```matlab
b = (eye(size(B)) - B(1:row-1, :).' * B(1:row-1, :)) * b;
b = b ./ norm(b, "fro");
```

**Why the 3rd Row ($b_3$) Converges in a Single Iteration:** In a $3$-source system ($M = 3$), once the first two mutually orthogonal unit vectors $b_1, b_2 \in \mathbb{R}^3$ have been determined, their orthogonal complement $\mathrm{span}\{b_1, b_2\}^\perp$ in $\mathbb{R}^3$ has dimension $3 - 2 = 1$. Consequently, on the very first iteration ($k = 1$) for $i = 3$, the projection operator $I_3 - b_1 b_1^T - b_2 b_2^T$ maps any non-orthogonal initial vector $b_3^{(0)}$ directly onto the unique line orthogonal to $b_1$ and $b_2$, and normalization fixes $b_3 = \pm (b_1 \times b_2)$. Thus $b_3$ converges in $1$ iteration in every deflationary method, and its score function remains completely flat thereafter (as noted on Pages 6, 14, and 17 of the report).

#### Stage 3 — Post-Processing: Resolving Permutation, Sign, and Scaling Ambiguities

Because ICA can only recover independent components up to row permutation, sign flips, and amplitude scaling, all four scripts align the estimated source matrix $\hat{S}$ with the ground-truth source matrix $S$ before computing $E$ (`Section1.m` lines 86–97, `Section2.m` and `Section3.m` lines 42–53, `Section4.m` lines 41–52):

1. **Compute raw source estimates and Frobenius-normalize:**

```matlab
S_hat = B * X;
S_hat = S_hat ./ norm(S_hat, "fro");
```

2. **Greedy cross-correlation matching and sign/energy alignment:** For each extracted row $\hat{s}_i^{\mathrm{raw}}$ ($i = 1, 2, 3$), compute its inner-product correlation vector $\mathbf{c}_i = \hat{s}_i^{\mathrm{raw}} S_{\mathrm{temp}}^T$ with the remaining unmatched rows of $S$, find the index $\mathrm{ind} = \arg\max_j |c_{i,j}|$, zero out $S_{\mathrm{temp}}(\mathrm{ind}, :)$ so each ground-truth source is matched exactly once, multiply by $\mathrm{sign}(c_{i, \mathrm{ind}})$ to resolve the sign ambiguity, and rescale the row norm to match the true source energy:

```math
\hat{S}(\mathrm{ind}, :) = \mathrm{sign}\!\left(\hat{s}_i^{\mathrm{raw}} s_{\mathrm{ind}}^T\right) \frac{\|s_i\|_2}{\|\hat{s}_i^{\mathrm{raw}}\|_2} \, \hat{s}_i^{\mathrm{raw}}.
```

*(Because all three ground-truth rows in `hw10.mat` have identical unit energy $\|s_1\|_2 = \|s_2\|_2 = \|s_3\|_2 = 1$, scaling by $\|s_i\|_2$ and $\|s_{\mathrm{ind}}\|_2$ are numerically identical.)*

### 4.2 Section 1: Deflationary Kurtosis Maximization via Gradient Projection (`Section1.m`)

For a zero-mean scalar random variable $y = b^T Z$, the fourth-order **kurtosis** (fourth cumulant) is defined as

```math
\mathrm{kurt}(y) = \mathbb{E}\!\left\{y^4\right\} - 3\left(\mathbb{E}\!\left\{y^2\right\}\right)^2,
```

which equals zero for any Gaussian random variable $y \sim \mathcal{N}(0, \sigma^2)$. To extract both sub-Gaussian ($\mathrm{kurt}(y) < 0$) and super-Gaussian ($\mathrm{kurt}(y) > 0$) sources, we maximize the absolute kurtosis score function:

```math
H(y) = \left|\mathrm{kurt}(y)\right| = \left|\mathrm{kurt}\!\left(b^T Z\right)\right|.
```

Differentiating $H(b^T Z) = |\mathrm{kurt}(b^T Z)|$ with respect to $b$ under the unit-variance constraint $\mathbb{E}\{y^2\} = b^T \mathbb{E}\{Z Z^T\} b = 1$ yields the gradient direction (up to a positive constant factor $4$ absorbed into the step size $\mu$):

```math
\frac{\partial f}{\partial b} = \mathrm{sign}\!\left(\mathrm{kurt}\!\left(b^T Z\right)\right) \left[ \mathbb{E}\!\left\{Z \left(b^T Z\right)^3\right\} - 3 b \right].
```

In `codes/Section1.m` (lines 73–85), the **Gradient Projection (GP)** update rule with step size $\mu = 4$ and $\mathrm{iterNum} = 50\,000$ is:

```math
\begin{aligned}
y^{(k)} &= \left(b_i^{(k)}\right)^T Z, \\
\mathrm{kurt}\!\left(y^{(k)}\right) &= \frac{1}{T} \sum_{t=1}^{T} \left(y^{(k)}(t)\right)^4 - 3 \left(\frac{1}{T} \sum_{t=1}^{T} \left(y^{(k)}(t)\right)^2\right)^2, \\
\tilde{b}_i^{(k+1)} &= b_i^{(k)} + \mu \, \mathrm{sign}\!\left(\mathrm{kurt}\!\left(y^{(k)}\right)\right) \left( Z \left(\left(y^{(k)}\right)^3\right)^T - 3 b_i^{(k)} \right), \\
\hat{b}_i^{(k+1)} &= \left(I_3 - \sum_{j=1}^{i-1} b_j b_j^T\right) \tilde{b}_i^{(k+1)}, \\
b_i^{(k+1)} &= \frac{\hat{b}_i^{(k+1)}}{\left\|\hat{b}_i^{(k+1)}\right\|_2}.
\end{aligned}
```

### 4.3 Section 2: Deflationary Kurtosis Maximization via Fixed-Point Iteration (`Section2.m`)

Setting the constrained stationary condition of the kurtosis objective to zero ($\frac{\partial |\mathrm{kurt}(b^T Z)|}{\partial b} \propto \mathbb{E}\{Z(b^T Z)^3\} - 3b = \lambda b$) motivates the **fixed-point iteration** (kurtosis-based **FastICA**), which replaces the step-size-dependent gradient ascent update with the direct fixed-point map:

```math
\begin{aligned}
\tilde{b}_i^{(k+1)} &= \mathbb{E}\!\left\{Z \left(\left(b_i^{(k)}\right)^T Z\right)^3\right\} - 3 b_i^{(k)}, \\
\hat{b}_i^{(k+1)} &= \left(I_3 - \sum_{j=1}^{i-1} b_j b_j^T\right) \tilde{b}_i^{(k+1)}, \\
b_i^{(k+1)} &= \frac{\hat{b}_i^{(k+1)}}{\left\|\hat{b}_i^{(k+1)}\right\|_2}.
\end{aligned}
```

In `codes/Section2.m` (lines 25–41), the stopping condition is written as `while (1 - norm(b_new * b.') > thr) || (iter < iterNum)` with `thr = 1e-15` and `iterNum = 5e4` (so the loop always runs for `iterNum` iterations), and line 33 evaluates `b = Z * (y.^3).' - 3*b` followed by Gram-Schmidt projection and unit-norm normalization.

### 4.4 Section 3: Outlier-Robust Negentropy Maximization via Gradient Projection (`Section3.m`)

Because fourth-order sample kurtosis $\mathbb{E}\{y^4\} - 3(\mathbb{E}\{y^2\})^2$ raises sample values to the fourth power, it is highly sensitive to impulsive noise and outliers. Hyvärinen's **negentropy approximation** replaces kurtosis with expectations of smooth, bounded-growth non-quadratic contrast functions $G(y)$:

```math
J_G(b) \propto \Big[ \mathbb{E}\!\left\{G\!\left(b^T Z\right)\right\} - \mathbb{E}\!\left\{G(v)\right\} \Big]^2,
\qquad v \sim \mathcal{N}(0, 1).
```

Using the Gaussian-type outlier-robust contrast function specified on the assignment sheet,

```math
G(y) = -\exp\!\left(-\frac{y^2}{2}\right),
\qquad
g(y) = G'(y) = y \exp\!\left(-\frac{y^2}{2}\right),
\qquad
g'(y) = G''(y) = \left(1 - y^2\right) \exp\!\left(-\frac{y^2}{2}\right),
```

the derivative of $f(b) = [f_1(b)]^2$ with respect to $b$ is proportional to

```math
\frac{\partial f}{\partial b} \propto f_1(b) \, \mathbb{E}\!\left\{Z g\!\left(b^T Z\right)\right\}.
```

In `codes/Section3.m` (lines 26–41), a reference Gaussian vector `v = randn(1, T)` is drawn for each row, the scalar score is tracked as `f = mean(-exp(-y.^2/2) - mean(exp(-v.^2/2)))^2`, the direction vector is formed as `df = sqrt(f) * (Z * g.')` where `g = y .* exp(-y.^2/2)`, and the **Gradient Projection** step is executed with $\mu = 4$ followed by Gram-Schmidt orthogonalization and unit-norm projection:

```math
\begin{aligned}
\tilde{b}_i^{(k+1)} &= b_i^{(k)} - \mu \sqrt{f\!\left(b_i^{(k)}\right)} \, Z \, g\!\left(\left(b_i^{(k)}\right)^T Z\right)^T, \\
\hat{b}_i^{(k+1)} &= \left(I_3 - \sum_{j=1}^{i-1} b_j b_j^T\right) \tilde{b}_i^{(k+1)}, \\
b_i^{(k+1)} &= \frac{\hat{b}_i^{(k+1)}}{\left\|\hat{b}_i^{(k+1)}\right\|_2}.
\end{aligned}
```

### 4.5 Section 4: Outlier-Robust Negentropy Maximization via Modified Fixed-Point Iteration (`Section4.m`)

Applying Newton's method to the Kuhn-Tucker stationary condition $\mathbb{E}\{Z g(b^T Z)\} - \beta b = 0$ on the unit sphere and approximating the Jacobian matrix $\mathbb{E}\{Z Z^T g'(b^T Z)\} \approx \mathbb{E}\{Z Z^T\}\mathbb{E}\{g'(b^T Z)\} = \mathbb{E}\{g'(b^T Z)\} I$ yields Hyvärinen's **modified fixed-point algorithm (final FastICA)**:

```math
\begin{aligned}
\tilde{b}_i^{(k+1)} &= \mathbb{E}\!\left\{Z g\!\left(\left(b_i^{(k)}\right)^T Z\right)\right\} - \mathbb{E}\!\left\{g'\!\left(\left(b_i^{(k)}\right)^T Z\right)\right\} b_i^{(k)}, \\
\hat{b}_i^{(k+1)} &= \left(I_3 - \sum_{j=1}^{i-1} b_j b_j^T\right) \tilde{b}_i^{(k+1)}, \\
b_i^{(k+1)} &= \frac{\hat{b}_i^{(k+1)}}{\left\|\hat{b}_i^{(k+1)}\right\|_2},
\end{aligned}
```

where for $G(y) = -\exp(-y^2 / 2)$:

```math
g(y) = y \exp\!\left(-\frac{y^2}{2}\right),
\qquad
g'(y) = \left(1 - y^2\right) \exp\!\left(-\frac{y^2}{2}\right).
```

In `codes/Section4.m` (lines 31–35), this update is implemented at each iteration as:

```matlab
g = y .* exp(-y.^2/2);
dg = (1 - y.^2) .* exp(-y.^2/2);
b = (Z * g.') - mean(dg .* b, 2);
b = (eye(size(B)) - B(1:row-1, :).' * B(1:row-1, :)) * b;
b = b ./ norm(b, "fro");
```

Because `mean(dg .* b, 2)` subtracts the first-order linear component $\mathbb{E}\{g'(y)\} b$ from `Z * g.'`, the remaining update isolates the higher-order non-Gaussian component directly without requiring a step size $\mu$, yielding very fast cubic/quadratic fixed-point convergence.

## 5. Results & Discussion

### 5.1 Summary of Single-Run Results Reported in Sections 1–4 of the Persian Report

In Sections 1 through 4 of the submitted report, each script (`Section1.m` to `Section4.m`) was executed with random initialization `B = randn(size(A))`. The resulting global transfer matrices `permutation_matrix = B * W * A` and relative reconstruction errors $E = \|\hat{S} - S\|_F^2 / \|S\|_F^2$ documented on Pages 3–4, 8, 12, and 15–16 of the report are summarized below:

| Section & Algorithm | Contrast Function | Optimization Method | Reported Global Matrix $P = B W A$ | Highlighted Off-Permutation Entries | Reported Error $E$ |
| --- | --- | --- | --- | --- | --- |
| **Section 1** (`Section1.m`) | Kurtosis $\left\|\mathrm{kurt}(y)\right\|$ | Gradient Projection ($\mu = 4$) | $`\begin{bmatrix} -0.9897 & -0.0699 & 0.0751 \\ -0.0421 & 1.0721 & -0.9009 \\ 0.0179 & 0.4248 & 0.6969 \end{bmatrix}`$ | $P_{2,3} = -0.9009$ (and $P_{3,2} = 0.4248$) | $0.21409$ |
| **Section 2** (`Section2.m`) | Kurtosis $\left\|\mathrm{kurt}(y)\right\|$ | Fixed-Point (Kurtosis FastICA) | $`\begin{bmatrix} -0.1437 & -1.1435 & 0.6589 \\ 0.6250 & -0.0132 & 0.7620 \\ -0.7552 & 0.1647 & 0.5369 \end{bmatrix}`$ | $P_{1,3} = 0.6589,\ P_{2,1} = 0.6250,\ P_{3,3} = 0.5369$ | $0.30682$ |
| **Section 3** (`Section3.m`) | Negentropy $G(y) = -\exp(-y^2/2)$ | Gradient Projection ($\mu = 4$) | $`\begin{bmatrix} -0.0434 & 1.0687 & -0.9062 \\ -0.9873 & -0.1091 & 0.0149 \\ -0.0700 & 0.4252 & 0.6939 \end{bmatrix}`$ | $P_{1,3} = -0.9062,\ P_{3,2} = 0.4252$ | $0.20242$ |
| **Section 4** (`Section4.m`) | Negentropy $G(y) = -\exp(-y^2/2)$ | Modified Fixed-Point (Final FastICA) | $`\begin{bmatrix} 0.9897 & 0.0699 & -0.0751 \\ -0.0421 & 1.0721 & -0.9009 \\ 0.0179 & 0.4248 & 0.6970 \end{bmatrix}`$ | $P_{2,3} = -0.9009,\ P_{3,2} = 0.4248$ | $0.20242$ |

As shown in Figures 1.5, 2.2, 3.2, and 4.2 of the report, in Sections 1, 3, and 4 the three separated and energy-normalized waveforms $\hat{s}_1(t)$ (smooth periodic wave), $\hat{s}_2(t)$ (triangular wave), and $\hat{s}_3(t)$ (sparse pulse train) closely track the ground-truth sources $s_1(t), s_2(t), s_3(t)$ despite the additive observation noise, achieving a consistent minimum error floor of $E = 0.20242$.

### 5.2 Section 5 — Controlled Benchmark Using `B_init.mat` ($\mathrm{iterNum} = 5\,000$ vs. $50\,000$)

To perform a fair, controlled comparison across all four algorithms in Section 5 (Pages 18–20 of the report), the random initialization `B = randn(size(A))` was replaced with the fixed initial matrix `B_init` loaded from `codes/B_init.mat`:

```math
B_{\mathrm{init}} = \begin{bmatrix}
0.3183 & 0.5065 & 0.0190 \\
0.1069 & 0.0269 & 0.6555 \\
0.1971 & 0.7459 & 0.2330
\end{bmatrix},
```

and the execution time of each script was measured via `tic` and `toc` at both $\mathrm{iterNum} = 5\,000$ and $\mathrm{iterNum} = 50\,000$:

| Algorithm | $\mathrm{iterNum} = 5\,000$ Runtime (s) | $\mathrm{iterNum} = 5\,000$ Error ($E$) | $\mathrm{iterNum} = 5\,000$ Max Row Convergence Iterations | $\mathrm{iterNum} = 50\,000$ Runtime (s) | $\mathrm{iterNum} = 50\,000$ Error ($E$) | $\mathrm{iterNum} = 50\,000$ Max Row Convergence Iterations |
| --- | --- | --- | --- | --- | --- | --- |
| **Section 1:** Kurtosis + Gradient Projection (`kurt(deflation)`) | $1.364805\ \mathrm{s}$ | $0.24635$ | $\approx 5\,000$ iterations | $13.699159\ \mathrm{s}$ | $0.20243$ | $\approx 5\,000$ iterations |
| **Section 2:** Kurtosis + Fixed-Point (`kurt(fixed-point)`) | $0.824397\ \mathrm{s}$ | $0.49648$ | Very few iterations | $8.677623\ \mathrm{s}$ | $0.49648$ | Very few iterations |
| **Section 3:** Negentropy $G(y)$ + Gradient Projection (`fastICA(G.P)`) | $0.700543\ \mathrm{s}$ | $0.22055$ | $\approx 3\,000$ iterations | $6.883684\ \mathrm{s}$ | $0.20242$ | $\approx 2\,000$ iterations |
| **Section 4:** Negentropy $G(y)$ + Modified Fixed-Point (`fastICA(fixed-point)`) | $1.255933\ \mathrm{s}$ | $0.20242$ | Very few iterations | $12.379115\ \mathrm{s}$ | $0.20242$ | Very few iterations |

### 5.3 Overall Comparative Conclusions (Section 5)

1. **Convergence Speed (Iterations to Converge vs. Per-Iteration Cost):**
   - Although the modified fixed-point FastICA algorithm (Section 4) has a slightly higher per-iteration runtime than Section 2 and Section 3 when forced to run for a fixed number of iterations ($1.256\ \mathrm{s}$ vs. $0.701\ \mathrm{s}$ at $5\,000$ iterations because it evaluates both $g(y)$ and $g'(y)$ at every step), it reaches its stationary optimum in **very few iterations** (an almost vertical initial jump on the convergence plot), already achieving the optimal error $E = 0.20242$ well before $5\,000$ iterations.
   - By contrast, the two Gradient Projection methods (Sections 1 and 3) require thousands of iterations ($\approx 2\,000\text{–}3\,000$ iterations for Section 3 and $\approx 5\,000$ iterations for Section 1) to fully converge from `B_init`, which is why their error $E$ improves from $0.24635 \to 0.20243$ (Section 1) and $0.22055 \to 0.20242$ (Section 3) when increasing $\mathrm{iterNum}$ from $5\,000$ to $50\,000$.
   - Ranking the four algorithms from **fastest to slowest convergence** (fewest iterations required to reach steady state, Page 20 of the report):

```math
\text{Convergence Time (Iterations):} \quad
\mathrm{FastICA}_{\text{fixed-point}}\text{ (Sec. 4)} \approx \mathrm{Kurt}_{\text{fixed-point}}\text{ (Sec. 2)} < \mathrm{Negentropy}_{\text{GP}}\text{ (Sec. 3)} < \mathrm{Kurt}_{\text{GP}}\text{ (Sec. 1)}.
```

2. **Source Separation Quality (Reconstruction Error $E$):**
   - At convergence ($\mathrm{iterNum} = 50\,000$), **Section 4** (Modified Fixed-Point FastICA) and **Section 3** (Outlier-Robust GP) achieve the lowest reconstruction error ($E = 0.20242$), closely followed by **Section 1** (Kurtosis GP, $E = 0.20243$), whereas **Section 2** (`Section2.m`, $E = 0.49648$ with `B_init` and $E = 0.30682$ in the random-start run) performs substantially worse.
   - Ranking the four algorithms from **lowest (best) to highest (worst) error $E$**:

```math
\text{Error }(E): \quad
\mathrm{FastICA}_{\text{fixed-point}}\text{ (Sec. 4)} \approx \mathrm{Negentropy}_{\text{GP}}\text{ (Sec. 3)} \le \mathrm{Kurt}_{\text{GP}}\text{ (Sec. 1)} < \mathrm{Kurt}_{\text{fixed-point}}\text{ (Sec. 2)}.
```

### 5.4 Notes on the Submitted Solutions / Implementation Notes

- **Unnormalized Spatial Correlation Matrix $R_X = X X^T$ and Score Function Scaling:** In all four scripts (`Section1.m` line 60, `Section2.m`–`Section4.m` line 13), the sample spatial correlation matrix is computed as `R_x = X * X.'` without dividing by $T = 1001$. Consequently, the whitened signals $Z = W X$ satisfy $Z Z^T = I_3$ (unit energy per row) rather than $\frac{1}{T} Z Z^T = I_3$ (unit variance per row). For any unit-norm vector $b$ ($\|b\|_2 = 1$), the output $y = b^T Z$ has total energy $\|y\|_2^2 = 1$ and sample mean-square $\frac{1}{T} \sum_{t=1}^T y(t)^2 = \frac{1}{1001} \approx 9.99 \times 10^{-4}$. This scaling accounts for two observations in the report:
  1. In **Sections 1 and 2** (Figures 1.6, 1.7, 2.3, and 2.4), the sample kurtosis $|\mathrm{kurt}(y)| = \left|\frac{1}{T}\sum y^4 - 3\left(\frac{1}{T}\sum y^2\right)^2\right|$ scales as $T^{-2} \approx 10^{-6}$, taking numerical values in the range $[10^{-7},\ 1.5 \times 10^{-6}]$.
  2. In **Sections 3 and 4** (Figures 3.3, 3.4, 4.3, and 4.4), because $y(t) \sim \mathcal{O}(T^{-1/2})$ is small, $\exp(-y(t)^2 / 2) \approx 1 - \frac{1}{2} y(t)^2 + \frac{1}{8} y(t)^4$. In `Section3.m` (line 31) and `Section4.m` (line 30), the score function is computed as `f = mean(-exp(-y.^2/2) - mean(exp(-v.^2/2)))^2` (subtracting $+\mathbb{E}\{\exp(-v^2/2)\} \approx 1/\sqrt{2} \approx 0.7071$ instead of $\mathbb{E}\{G(v)\} = -\mathbb{E}\{\exp(-v^2/2)\}$), so $f \approx (-1 - 0.7071)^2 \approx 2.914$, and maximizing $|\mathrm{kurt}(y)|$ (increasing $\sum y^4$) slightly decreases $f(b)$ in the $7$-th decimal place (from $\approx 2.9578868$ down to $\approx 2.9578867$), which is why the convergence curves for $b_1$ and $b_2$ in Figures 3.3, 3.4, 4.3, and 4.4 decrease monotonically before leveling off.
- **Why `Section2.m` Stops Updating After 2 Iterations (`b = b_new` Inside the Loop):** In `codes/Section2.m` (lines 26–39), `b_new` is initialized before the `while` loop as `b_new = B(row, :).'`, and line 31 inside the `while` loop executes `b = b_new;` without ever updating `b_new = b` at the end of the iteration. As a result, `b` is reset to the initial random vector `b_new` at the start of every iteration, so `Section2.m` performs only a single one-step update from the random initial vector (producing the immediate flat line in Figures 2.3 and 2.4 and the higher reconstruction error $E = 0.30682$ / $E = 0.49648$). Furthermore, the `while` condition `(1 - norm(b_new * b.') > thr) || (iter < iterNum)` uses logical OR (`||`) instead of logical AND (`&&`) and the outer product `b_new * b.'` instead of the inner product `abs(b_new.' * b)`, ensuring that the loop always executes all `iterNum = 50000` iterations.
- **Direct Application of `B` to Unwhitened Observations (`S_hat = B * X`) and Off-Diagonal Entries in `B * W * A`:** In all four scripts (`Section1.m` line 86, `Section2.m` and `Section3.m` line 42, `Section4.m` line 41), the estimated sources are computed as `S_hat = B * X` rather than `S_hat = B * Z` (`B * W * X`), whereas the permutation matrix is computed as `permutation_matrix = B * W * A`. In addition, because the sample expectations `Z * (y.^3).'` (in `Section1.m` line 78) and `Z * g.'` (in `Section3.m` line 33 and `Section4.m` line 33) are unnormalized sums over $t = 1, \dots, T$ rather than sample means $\frac{1}{T} Z (\cdot)^T$, the stationary fixed points of Sections 1, 3, and 4 converge to the same orthonormal matrix $B$ (up to row ordering and sign, yielding $E = 0.20242$ and the highlighted off-diagonal entry $P_{2,3} \approx -0.9009$ in `B * W * A`).
- **Minor Typographical Slips in the Persian Report (Pages 3, 7, 11, 15, and 20):**
  - On **Pages 3, 7, 11, and 15**, the Gram-Schmidt orthogonal projection formula inside the large brace is written in shorthand as $b_i = \left(I - \begin{bmatrix} b_1^T & b_2^T & \cdots & b_{i-1}^T \end{bmatrix}^T\right)$, omitting the trailing factor $\begin{bmatrix} b_1 & b_2 & \cdots & b_{i-1} \end{bmatrix} b_i$; all four MATLAB scripts implement the exact projection `(eye(size(B)) - B(1:row-1, :).' * B(1:row-1, :)) * b`.
  - On **Pages 3 and 7**, the kurtosis definition is printed as $\mathrm{kurt}(y) = E\{y^4\} - 3(E\{y^2\})$ (omitting the square exponent on $(E\{y^2\})^2$), whereas `Section1.m` (line 77) and `Section2.m` (line 32) compute `mean(y.^4, 2) - 3 * mean(y.^2, 2)^2`.
  - On **Pages 11 and 15**, the introductory sentence and Figure 3.1 / 4.1 captions retain the word `kurt` from Sections 1–2 ("روش بیشینه سازی kurt با رویکرد GP / fixed-point (اصلاح شده)"), whereas the displayed equations and code (`Section3.m` and `Section4.m`) use the negentropy contrast $G(y) = -\exp(-y^2 / 2)$.
  - On **Page 20**, the final inequality for `Error (E)` writes `fastICA(G.P)` twice (`fastICA(fixed-point) ≈ fastICA(G.P) ≤ kurt(deflation) < fastICA(G.P)`); as stated in the paragraph on Page 19, the rightmost (worst-error) term is `kurt(fixed-point)` (the Section 2 algorithm, with $E = 0.49648$).

## 6. How to Use

### 6.1 Running the MATLAB Scripts

Open MATLAB, navigate to the `codes` directory, and run any of the four scripts:

```matlab
cd codes
Section1
Section2
Section3
Section4
```

- **Dependencies:** All four scripts use only base MATLAB functions (`eig`, `sort`, `vecnorm`, `norm`, `mean`, `randn`, `plot`, `subplot`). No external toolboxes are required.
- **Reproducing the Section 5 Benchmark with `B_init.mat`:** By default, each script initializes `B = randn(size(A))` with `iterNum = 5e4` ($50\,000$ iterations). To reproduce the exact deterministic benchmark reported in Section 5 (Pages 18–19 of the report), replace `B = randn(size(A));` in each script with:

```matlab
load B_init.mat
B = B_init;
```

- **Generated Figures:**
  - `Section1.m` opens **5 figures**:
    1. Three-panel subplot of the original sources $s_1(t), s_2(t), s_3(t)$ (Figure 1.1 in the report).
    2. Three-panel subplot of the noiseless mixtures $X_{\mathrm{noNoise}}(t) = A S(t)$ (Figure 1.2 in the report).
    3. Three-panel subplot of the noisy mixtures $X(t) = A S(t) + \mathrm{Noise}(t)$ (Figure 1.3 in the report).
    4. Three-panel overlay plot of the estimated sources $\hat{s}_i(t)$ (magenta) and true sources $s_i(t)$ (black) (Figure 1.5 in the report).
    5. Three-panel convergence plot of the score function $|\mathrm{kurt}(y)|$ for $b_1, b_2, b_3$ across all iterations (Figures 1.6 and 1.7 in the report).
  - `Section2.m`, `Section3.m`, and `Section4.m` each open **2 figures**:
    1. Three-panel overlay plot of $\hat{s}_i(t)$ versus $s_i(t)$ (Figures 2.2, 3.2, and 4.2 in the report).
    2. Three-panel convergence plot of the score function for $b_1, b_2, b_3$ across all iterations (Figures 2.3–2.4, 3.3–3.4, and 4.3–4.4 in the report).

### 6.2 Expected Console Output

Because `B = randn(size(A))` draws a fresh random initial matrix on each standalone run (and `Section3.m` / `Section4.m` draw `v = randn(1, T)` for the reference Gaussian shift in the score plot), the printed error `E` in standalone runs matches the report values up to the random initialization seed. For the representative runs documented in Sections 1–4 of the report, the console outputs are:

```text
% Section1.m (Report Section 1 run)
Part 2:
Error (E) = 0.21409

% Section2.m (Report Section 2 run)
Part 2:
Error (E) = 0.30682

% Section3.m (Report Section 3 run)
Part 2:
Error (E) = 0.20242

% Section4.m (Report Section 4 run)
Part 2:
Error (E) = 0.20242
```

When initialized deterministically with `B = B_init` from `B_init.mat` at `iterNum = 5e4` (Report Section 5), the four scripts print:

```text
% Section1.m with B_init (iterNum = 50000)
Part 2:
Error (E) = 0.20243

% Section2.m with B_init (iterNum = 50000)
Part 2:
Error (E) = 0.49648

% Section3.m with B_init (iterNum = 50000)
Part 2:
Error (E) = 0.20242

% Section4.m with B_init (iterNum = 50000)
Part 2:
Error (E) = 0.20242
```

### 6.3 Reading the Theoretical Derivations and Report

1. Use **Section 3** for the complete English translation of the assignment problems and **Section 4** for the step-by-step mathematical derivations of spatial whitening, deflationary Gram-Schmidt projection, kurtosis maximization, and outlier-robust FastICA.
2. Use **Section 5** to review the permutation matrices, error benchmarks, runtime comparisons, and implementation notes alongside the original Persian report [BSS\_HW#10\_810100205\_Arman\_Majidi.pdf](./BSS_HW%2310_810100205_Arman_Majidi.pdf) and assignment sheet [hw10.pdf](./hw10.pdf).

## 7. References

1. Akhavan, S. *Blind Source Separation: Homework 10*. Course assignment sheet and lecture notes, University of Tehran.
2. Hyvärinen, A., & Oja, E. (1997). A fast fixed-point algorithm for independent component analysis. *Neural Computation*, 9(7), 1483–1492.
3. Hyvärinen, A. (1999). Fast and robust fixed-point algorithms for independent component analysis. *IEEE Transactions on Neural Networks*, 10(3), 626–634.
4. Hyvärinen, A., Karhunen, J., & Oja, E. (2001). *Independent Component Analysis*. John Wiley & Sons.
5. Comon, P., & Jutten, C. (Eds.). (2010). *Handbook of Blind Source Separation: Independent Component Analysis and Applications*. Academic Press.
