# Homework 10: Independent Component Analysis (ICA) via Deflationary Kurtosis & Negentropy Maximization Using Gradient Projection and Fixed-Point FastICA

**Course:** Blind Source Separation

**Instructor:** Dr. Saeed Akhavan

**Author:** Arman Majidi

**Student ID:** 810100205

**University:** University of Tehran

**Semester:** Spring 2024-2025

## 1. Overview & Assignment Structure

This folder contains the **English problem statements, analytical derivations, MATLAB implementations, and simulation results for Homework 10**. The assignment is organized into five main sections that investigate **Independent Component Analysis (ICA)** for noisy linear instantaneous mixtures by maximizing output non-Gaussianity (via fourth-order **kurtosis** and outlier-robust Gaussian **negentropy** approximations) using **deflationary** row-by-row extraction with both **Gradient Projection (GP)** and **fixed-point (FastICA)** optimization schemes:

1. **Section 1 — Deflationary Kurtosis Maximization via Gradient Projection (Parts 0–3):** Generates the noiseless and noisy 3-channel observation mixtures $X = A S + \mathrm{Noise}$ from `hw10.mat`, performs spatial whitening $Z = W X$ via eigenvalue decomposition of the sample correlation matrix $R_X$, extracts the rows of the orthonormal separating matrix $B$ sequentially (deflation) by maximizing $\vert{}\mathrm{kurt}(y)\vert{}$ via Gradient Projection with Gram-Schmidt orthogonalization, evaluates the overall mixing-unmixing product $B W A$, resolves permutation, sign, and energy scaling ambiguities to reconstruct $\hat{S}$, computes the relative squared Frobenius error $E = \Vert{}\hat{S} - S\Vert{}_F^2 / \Vert{}S\Vert{}_F^2$, and analyzes the objective function convergence trajectories over $5\,000$ and $50\,000$ iterations.

2. **Section 2 — Deflationary Kurtosis Maximization via Fixed-Point Iteration / Kurtosis FastICA (Parts 1–3):** Replaces the gradient ascent step with the cubic fixed-point FastICA update rule for kurtosis maximization under deflationary orthogonalization, and evaluates the resulting permutation product $B W A$, waveform reconstruction error $E$, and convergence behavior.

3. **Section 3 — Outlier-Robust Negentropy Maximization with** $G(y) = -\exp(-y^2 / 2)$ **via Gradient Projection (Parts 1–3):** Replaces the fourth-order kurtosis contrast (which is sensitive to outliers) with a bounded non-quadratic negentropy contrast function $G(y) = -\exp(-y^2 / 2)$, optimizes the rows of $B$ sequentially via Gradient Projection, and reports the resulting permutation matrix, source reconstructions, relative error $E$, and convergence plots.

4. **Section 4 — Outlier-Robust Negentropy Maximization with** $G(y) = -\exp(-y^2 / 2)$ **via Modified Fixed-Point FastICA (Parts 1–3):** Implements the full Newton-based modified fixed-point FastICA update rule using $g(y) = y\exp(-y^2 / 2)$ and $g'(y) = (1 - y^2)\exp(-y^2 / 2)$ with deflationary orthogonalization, achieving fast convergence and accurate source separation.

5. **Section 5 — Empirical Comparison of Convergence Speed and Separation Quality:** Benchmarks all four ICA algorithms from an identical initial matrix `B_init.mat` at both $5\,000$ and $50\,000$ iterations, comparing execution runtime (`tic`–`toc`), iterations required for row convergence, and separation error $E$.

All five sections of the assignment are mandatory; no optional or bonus problems were assigned. Because both the original assignment sheet and the submitted solution report are written in Persian, this `README.md` provides a **complete, self-contained English reference for every problem statement, mathematical derivation, algorithm, and numerical result**.

## 2. Directory Contents

| File | Description | 
| ----- | ----- | 
| [codes/Section1.m](https://www.google.com/search?q=./codes/Section1.m) | MATLAB script for Section 1 (Parts 0–3): time-domain visualization of $S$, $X_{\mathrm{noNoise}}$, and noisy $X$; spatial whitening $Z = W X$; deflationary kurtosis maximization via Gradient Projection ($\mu = 4$); permutation/sign/energy alignment; error $E$ calculation; and convergence plots. | 
| [codes/Section2.m](https://www.google.com/search?q=./codes/Section2.m) | MATLAB script for Section 2 (Parts 0–3): spatial whitening and deflationary kurtosis maximization via the cubic fixed-point (FastICA) update rule, source alignment, error $E$ evaluation, and convergence plots. | 
| [codes/Section3.m](https://www.google.com/search?q=./codes/Section3.m) | MATLAB script for Section 3 (Parts 0–3): spatial whitening and deflationary negentropy maximization with $G(y) = -\exp(-y^2 / 2)$ via Gradient Projection ($\mu = 4$), source alignment, error $E$ evaluation, and convergence plots. | 
| [codes/Section4.m](https://www.google.com/search?q=./codes/Section4.m) | MATLAB script for Section 4 (Parts 0–3): spatial whitening and deflationary negentropy maximization with $G(y) = -\exp(-y^2 / 2)$ via the modified fixed-point (final FastICA) algorithm, source alignment, error $E$ evaluation, and convergence plots. | 
| [codes/hw10.mat](https://www.google.com/search?q=./codes/hw10.mat) | Dataset containing the $3 \times 3$ mixing matrix `A`, the $3 \times 1001$ ground-truth source matrix `S`, and the $3 \times 1001$ additive sensor noise matrix `Noise`. | 
| [codes/B_init.mat](https://www.google.com/search?q=./codes/B_init.mat) | Saved $3 \times 3$ initial matrix `B_init` used in Section 5 of the report to benchmark all four algorithms from identical initial conditions. | 
| [BSS_HW#10_810100205_Arman_Majidi.pdf](https://www.google.com/search?q=./BSS_HW%252310_810100205_Arman_Majidi.pdf) | Submitted report and analytical solutions in Persian (including source and mixture waveforms, BSS block diagrams, permutation matrices, reconstructed waveforms, convergence plots at $5\,000$ and $50\,000$ iterations, and comparative benchmarks). | 
| [hw10.pdf](https://www.google.com/search?q=./hw10.pdf) | Original assignment problem sheet in Persian. | 
| `README.md` | Self-contained English problem statements, mathematical derivations, algorithm descriptions, numerical results, and usage guide. | 

Keep the `codes` subfolder inside the homework directory.

## 3. Assignment Problem Statement

### 3.1 General Problem Setup and Preliminary Visualization (Part 0)

In this assignment, we solve the Blind Source Separation (BSS) problem under the assumption that the underlying sources are statistically independent (**Independent Component Analysis — ICA**).

The mixing matrix $A \in \mathbb{R}^{3 \times 3}$, the source matrix $S \in \mathbb{R}^{3 \times T}$ ($T = 1001$), and the additive noise matrix $\mathrm{Noise} \in \mathbb{R}^{3 \times T}$ are provided in the file `hw10.mat`:

1. First, compute the noisy observation matrix via the linear instantaneous mixing model

```
X = A S + \mathrm{Noise},

```

along with the noiseless mixtures $X_{\mathrm{noNoise}} = A S$. 2. Plot the original sources $S$, the noiseless observations $X_{\mathrm{noNoise}}$, and the noisy observations $X$ to inspect their time-domain waveforms. 3. Next, view the problem strictly from a Blind Source Separation perspective: assume that only the noisy observation matrix $X$ is available and that the number of sources ($M = 3$) is known. Our strategy is to apply a spatial whitening transformation $W$ to obtain whitened observations $Z = W X$, and then multiply $Z$ by an orthonormal separating matrix $B$ so that the resulting outputs $Y = B Z$ are **as non-Gaussian as possible**.

### 3.2 Section 1: Deflationary Kurtosis Maximization via Gradient Projection (GP)

Implement the first ICA method presented in class, which is based on maximizing the output **kurtosis**. Specifically, extract the rows of the separating matrix $B$ one by one using a **deflation** approach, and perform the associated constrained optimizations using the **Gradient Projection (GP)** algorithm. Note that the final overall separating matrix applied to $X$ is the product of the final orthonormal matrix $B$ obtained from the algorithm and the spatial whitening matrix $W$ ($B_{\mathrm{total}} = B W$).

#### Part 1-1 — Generalized Permutation Matrix Verification

Multiply the final overall separating matrix ($B W$) obtained by the algorithm by the true mixing matrix $A$, and report the resulting product matrix:

```
P = B W A.

```

The resulting matrix should be close to a generalized **permutation matrix**, meaning that each row and each column should contain only one dominant non-zero entry.

#### Part 1-2 — Ambiguity Removal, Source Reconstruction Plots, and Relative Error $E$

Resolve both the **permutation ambiguity** and the **scaling (and sign) ambiguity** of the estimated sources—specifically, set the energy ($\ell_2$-norm) of each estimated source equal to the energy of the corresponding true source. Plot the aligned estimated sources $\hat{S}$ overlaid on the true sources $S$, and compute and report the relative squared Frobenius reconstruction error:

```
E = \frac{\left\|\hat{S} - S\right\|_F^2}{\left\|S\right\|_F^2}.

```

#### Part 1-3 — Objective Function Convergence Curves

Plot the convergence curve (the objective / score function value versus the iteration index) for each extracted row of $B$.

### 3.3 Section 2: Deflationary Kurtosis Maximization via Fixed-Point Iteration (Kurtosis FastICA)

Implement the second ICA method presented in class, which maximizes output **kurtosis** using a **fixed-point** iteration approach (**FastICA**). Report all results analogous to Section 1 (Parts 1-1, 1-2, and 1-3).

### 3.4 Section 3: Outlier-Robust Negentropy Maximization via Gradient Projection (GP)

Implement the third ICA method presented in class, which is robust against outliers, using the non-quadratic contrast function

```
G(y) = -\exp\!\left(-\frac{y^2}{2}\right),

```

and the **Gradient Projection (GP)** optimization algorithm under deflation. Report all results analogous to Section 1 (Parts 1-1, 1-2, and 1-3).

### 3.5 Section 4: Outlier-Robust Negentropy Maximization via Modified Fixed-Point Iteration (Final FastICA)

Implement the outlier-robust ICA method presented in class with

```
G(y) = -\exp\!\left(-\frac{y^2}{2}\right),

```

using the **modified fixed-point** optimization algorithm (the final Newton-based **FastICA** algorithm) under deflation. Report all results analogous to Section 1 (Parts 1-1, 1-2, and 1-3).

### 3.6 Section 5: Empirical Comparison of Convergence Speed and Separation Quality

Compare the four methods above empirically and state which of the four algorithms converges fastest and which achieves the best source separation quality (the lowest error parameter $E$).

## 4. Mathematical Formulation & Solutions

### 4.1 Preprocessing, Dataset Structure, and Spatial Whitening (Part 0)

In `codes/hw10.mat`, the $M = 3$ sources over $T = 1001$ samples consist of three distinct non-Gaussian waveforms (Figure 1.1 on Page 1 of the report):

* $s_1(t)$ (`S(1, :)`): a smooth low-frequency sinusoidal waveform (sub-Gaussian, negative kurtosis) with amplitude $\approx 0.045$.

* $s_2(t)$ (`S(2, :)`): a periodic symmetric triangular waveform (sub-Gaussian, negative kurtosis) with amplitude $\approx 0.05$.

* $s_3(t)$ (`S(3, :)`): a sparse, sharp positive pulse train (super-Gaussian, positive kurtosis) taking values in $[0, 0.1]$.

The $3 \times 3$ mixing matrix $A$ in `hw10.mat` has unit-norm columns ($\Vert{}a_j\Vert{}_2 = 1$):

```
A = \frac{1}{\sqrt{14}} \begin{bmatrix}
1 & 1 & 1 \\
2 & -2 & -3 \\
3 & 3 & -2
\end{bmatrix}
\approx \begin{bmatrix}
0.2673 & 0.2673 & 0.2673 \\
0.5345 & -0.5345 & -0.8018 \\
0.8018 & 0.8018 & -0.5345
\end{bmatrix}.

```

The noisy observation matrix $X \in \mathbb{R}^{3 \times 1001}$ is formed as $X = A S + \mathrm{Noise}$ (Figures 1.2 and 1.3 on Page 2 of the report).

To reduce the ICA estimation problem to finding an **orthogonal** separating matrix $B \in \mathbb{R}^{3 \times 3}$ ($B B^T = I_3$), all four scripts (`Section1.m`–`Section4.m`) first whiten the observations $X$ using Principal Component Analysis (PCA). Computing the eigen-decomposition of the sample correlation matrix $R_X = X X^T = U D U^T$ with eigenvalues ordered $\lambda_1 \ge \lambda_2 \ge \lambda_3 > 0$, the spatial whitening matrix $W \in \mathbb{R}^{3 \times 3}$ and whitened signal matrix $Z \in \mathbb{R}^{3 \times T}$ are given by

```
W = D^{-1/2} U^T,
\qquad
Z = W X \in \mathbb{R}^{3 \times T},
\qquad
Z Z^T = D^{-1/2} U^T \left(X X^T\right) U D^{-1/2} = I_3.

```

As illustrated in the two-stage BSS block diagrams in the report (Figures 1.4, 2.1, 3.1, and 4.1), the complete linear separation cascade is

```
S \xrightarrow{\quad A \quad} X \xrightarrow{\quad W \quad} Z \xrightarrow{\quad B \quad} Y = B Z = B W X.

```

In the noiseless limit where the sources are uncorrelated with unit variance, $W A$ is an orthogonal matrix, so choosing $B = (W A)^{-1} = (W A)^T$ yields $B B^T = I_3$ and $Y = B W A S = P D_s S$, where $P$ is a permutation matrix and $D_s$ is a diagonal sign/scaling matrix.

### 4.2 Deflationary Orthogonalization and Ambiguity Resolution

#### Row-by-Row Deflation via Gram-Schmidt Projection

Rather than optimizing all $M = 3$ rows of $B = [b_1, b_2, b_3]^T$ simultaneously under symmetric orthogonality constraints, the **deflationary** approach extracts the unit-norm row vectors $b_i^T \in \mathbb{R}^{1 \times 3}$ sequentially for $i = 1, 2, \dots, M$.

Suppose the first $i - 1$ orthonormal vectors $b_1, \dots, b_{i-1} \in \mathbb{R}^{3 \times 1}$ have already been estimated and stored in the first $i - 1$ rows of $B$, denoted $B_{1:i-1} = [b_1, \dots, b_{i-1}]^T \in \mathbb{R}^{(i-1) \times 3}$. To ensure that the $i$-th candidate vector $b_i$ extracts a **new** independent component orthogonal to the subspace spanned by $\{b_1, \dots, b_{i-1}\}$, each iteration projects $b_i$ onto the orthogonal complement of $\mathrm{span}\{b_1, \dots, b_{i-1}\}$ and normalizes it to unit Euclidean norm:

```
\begin{aligned}
b_i &\leftarrow \left(I_M - \sum_{j=1}^{i-1} b_j b_j^T\right) b_i = \left(I_M - B_{1:i-1}^T B_{1:i-1}\right) b_i, \\
b_i &\leftarrow \frac{b_i}{\left\|b_i\right\|_2}.
\end{aligned}

```

In MATLAB (`Section1.m` lines 80–81, `Section2.m` lines 34–35, `Section3.m` lines 35–36, `Section4.m` lines 34–35), this is implemented cleanly for all $i = 1, 2, 3$ as:

```
b = (eye(size(B)) - B(1:row-1, :).' * B(1:row-1, :)) * b;
b = b ./ norm(b, "fro");

```

*(Note: For* $i = 1$*, `B(1:0, :)` is an empty* $0 \times 3$ *matrix whose product `B(1:0, :).' * B(1:0, :)` is the* $3 \times 3$ *zero matrix, so no separate `if` branch is needed. For the final row* $i = M = 3$ *in* $\mathbb{R}^3$*, the orthogonal complement of* $\mathrm{span}\{b_1, b_2\}$ *is one-dimensional, so* $b_3 = \pm (b_1 \times b_2)$ *is uniquely determined up to sign on the very first projection step.)*

#### Post-Separation Permutation, Sign, and Energy Alignment

Because ICA has inherent **permutation**, **sign**, and **scaling** ambiguities, computing the relative reconstruction error $E = \Vert{}\hat{S} - S\Vert{}_F^2 / \Vert{}S\Vert{}_F^2$ requires matching each estimated row of $\hat{S}$ to its corresponding ground-truth source in $S$:

1. For each extracted source row $\hat{s}_{\mathrm{raw}, i}$ ($i = 1, \dots, M$), compute its inner-product correlation vector with the remaining unmatched rows of $S$:

```
c_j = \hat{s}_{\mathrm{raw}, i} s_j^T,
\qquad
\mathrm{ind} = \arg\max_{j} \left|c_j\right|.

```

2. Zero out the matched row $s_{\mathrm{ind}}$ in a temporary copy `S_temp` so it cannot be matched twice, align the polarity using $\mathrm{sign}(c_{\mathrm{ind}})$, and rescale the estimated signal energy to match the ground-truth source norm:

```
\hat{s}_{\mathrm{ind}} \leftarrow \mathrm{sign}(c_{\mathrm{ind}}) \, \frac{\left\|s_{\mathrm{ind}}\right\|_2}{\left\|\hat{s}_{\mathrm{raw}, i}\right\|_2} \, \hat{s}_{\mathrm{raw}, i}.

```

### 4.3 Section 1: Deflationary Kurtosis Maximization via Gradient Projection (`Section1.m`)

For a zero-mean scalar random variable $y = b^T Z$, the fourth-order cumulant (**kurtosis**) is defined as

```
\mathrm{kurt}(y) = \mathbb{E}\!\left\{y^4\right\} - 3\left(\mathbb{E}\!\left\{y^2\right\}\right)^2.

```

If $y \sim \mathcal{N}(0, \sigma^2)$ is Gaussian, $\mathbb{E}\{y^4\} = 3\sigma^4 = 3(\mathbb{E}\{y^2\})^2$, so $\mathrm{kurt}(y) = 0$. By the Central Limit Theorem, a linear combination of independent sources is more Gaussian than any individual source; therefore, maximizing the absolute kurtosis

```
H(y) = \left|\mathrm{kurt}(y)\right| = \left|\mathrm{kurt}\!\left(b^T Z\right)\right|

```

over the unit sphere $\Vert{}b\Vert{}_2 = 1$ (where $\mathbb{E}\{y^2\} = b^T \mathbb{E}\{Z Z^T\} b = \Vert{}b\Vert{}_2^2 = 1$ under unit-variance whitening) isolates a single independent source.

Differentiating $f(b) = \vert{}\mathrm{kurt}(b^T Z)\vert{}$ with respect to $b$ subject to unit variance yields the gradient direction (Page 3 of the report):

```
\frac{\partial f}{\partial b} = \mathrm{sign}\!\left(\mathrm{kurt}\!\left(b^T Z\right)\right) \left[ \mathbb{E}\!\left\{Z \left(b^T Z\right)^3\right\} - 3b \right]
\quad \text{(up to a constant factor of } 4\text{)}.

```

In `codes/Section1.m` (lines 73–85), each row $b_i$ ($i = 1, 2, 3$) is updated over $\mathrm{iterNum}$ iterations via **Gradient Projection** with step size $\mu = 4$:

```
\begin{aligned}
y &= b_i^{(k)T} Z, \\
\mathrm{kurt}(y) &= \frac{1}{T} \sum_{t=1}^T y(t)^4 - 3 \left(\frac{1}{T} \sum_{t=1}^T y(t)^2\right)^2, \\
\tilde{b}_i^{(k+1)} &= b_i^{(k)} + \mu \, \mathrm{sign}\!\left(\mathrm{kurt}(y)\right) \left( Z \left(y^3\right)^T - 3 b_i^{(k)} \right), \\
\bar{b}_i^{(k+1)} &= \left(I_3 - B_{1:i-1}^T B_{1:i-1}\right) \tilde{b}_i^{(k+1)}, \\
b_i^{(k+1)} &= \frac{\bar{b}_i^{(k+1)}}{\left\|\bar{b}_i^{(k+1)}\right\|_2}.
\end{aligned}

```

### 4.4 Section 2: Deflationary Kurtosis Maximization via Fixed-Point FastICA (`Section2.m`)

At a stationary point of $\vert{}\mathrm{kurt}(b^T Z)\vert{}$ on the unit sphere $\Vert{}b\Vert{}_2 = 1$, Lagrange's theorem requires that the gradient $\frac{\partial f}{\partial b}$ be collinear with $b$:

```
\mathbb{E}\!\left\{Z \left(b^T Z\right)^3\right\} - 3b = \lambda b.

```

Solving this fixed-point relation directly (or equivalently applying Newton's method to the Lagrangian with $\mathbb{E}\{Z Z^T\} = I$) yields the **kurtosis-based FastICA** fixed-point update rule (Page 7 of the report):

```
\begin{aligned}
\tilde{b}_i^{(k+1)} &\leftarrow \mathbb{E}\!\left\{Z \left(b_i^{(k)T} Z\right)^3\right\} - 3 b_i^{(k)}, \\
\bar{b}_i^{(k+1)} &\leftarrow \left(I_3 - B_{1:i-1}^T B_{1:i-1}\right) \tilde{b}_i^{(k+1)}, \\
b_i^{(k+1)} &\leftarrow \frac{\bar{b}_i^{(k+1)}}{\left\|\bar{b}_i^{(k+1)}\right\|_2}.
\end{aligned}

```

### 4.5 Section 3: Outlier-Robust Negentropy Maximization via Gradient Projection (`Section3.m`)

Because fourth-order kurtosis $\mathbb{E}\{y^4\} - 3(\mathbb{E}\{y^2\})^2$ raises sample values to the fourth power, it is highly sensitive to erratic noise spikes and outliers. To achieve statistical robustness against outliers, Hyvärinen's general contrast framework approximates **negentropy** $J(y) = H(y_{\mathrm{gauss}}) - H(y)$ using a smooth, bounded non-quadratic function $G(y)$:

```
J(y) \propto f(b) = \left[ \mathbb{E}\!\left\{G(y)\right\} - \mathbb{E}\!\left\{G(v)\right\} \right]^2,
\qquad
y = b^T Z,

```

where $v \sim \mathcal{N}(0, \sigma_y^2)$ is a zero-mean Gaussian random variable with the same variance as $y$ ($\sigma_y^2 = 1$ for standard normal $v \sim \mathcal{N}(0, 1)$).

Using the Gaussian contrast function assigned in Problem 3:

```
G(y) = -\exp\!\left(-\frac{y^2}{2}\right),
\qquad
g(y) = G'(y) = y \exp\!\left(-\frac{y^2}{2}\right),
\qquad
g'(y) = G''(y) = \left(1 - y^2\right)\exp\!\left(-\frac{y^2}{2}\right),

```

and defining $f_1(b) = \mathbb{E}\{G(b^T Z)\} - \mathbb{E}\{G(v)\}$ so that $f(b) = \left[f_1(b)\right]^2$, the chain rule gives the gradient with respect to $b$ (Page 11 of the report):

```
\frac{\partial f}{\partial b} = 2 f_1(b) \frac{\partial f_1}{\partial b} \propto f_1(b) \, \mathbb{E}\!\left\{Z g\!\left(b^T Z\right)\right\},
\qquad
g(y) = y \exp\!\left(-\frac{y^2}{2}\right).

```

In `codes/Section3.m` (lines 26–40), each row $b_i$ is updated via **Gradient Projection** with step size $\mu = 4$:

```
\begin{aligned}
\tilde{b}_i^{(k+1)} &\leftarrow b_i^{(k)} - \mu \sqrt{f\big(b_i^{(k)}\big)} \, \left(Z g(y)^T\right), \\
\bar{b}_i^{(k+1)} &\leftarrow \left(I_3 - B_{1:i-1}^T B_{1:i-1}\right) \tilde{b}_i^{(k+1)}, \\
b_i^{(k+1)} &\leftarrow \frac{\bar{b}_i^{(k+1)}}{\left\|\bar{b}_i^{(k+1)}\right\|_2}.
\end{aligned}

```

### 4.6 Section 4: Outlier-Robust Negentropy Maximization via Modified Fixed-Point FastICA (`Section4.m`)

To accelerate convergence of the outlier-robust negentropy contrast $f(b) \propto [\mathbb{E}\{G(b^T Z)\} - \mathbb{E}\{G(v)\}]^2$ without having to tune a learning rate $\mu$, we apply **Newton's method** to the Kuhn-Tucker stationarity condition

```
F(b) = \mathbb{E}\!\left\{Z g\!\left(b^T Z\right)\right\} - \beta b = 0
\qquad \text{subject to} \qquad
\|b\|_2 = 1.

```

Because the data $Z$ are spatially whitened ($\mathbb{E}\{Z Z^T\} = I$), the Jacobian matrix of $F(b)$ is well approximated by a scalar multiple of the identity matrix:

```
\frac{\partial F}{\partial b} = \mathbb{E}\!\left\{Z Z^T g'\!\left(b^T Z\right)\right\} - \beta I \approx \left(\mathbb{E}\!\left\{g'\!\left(b^T Z\right)\right\} - \beta\right) I.

```

Taking a Newton step $b \leftarrow b - \left(\frac{\partial F}{\partial b}\right)^{-1} F(b)$ and multiplying through by the scalar $\beta - \mathbb{E}\{g'(b^T Z)\}$ yields the **modified fixed-point (final FastICA)** update rule (Page 15 of the report):

```
\begin{aligned}
\tilde{b}_i^{(k+1)} &\leftarrow \mathbb{E}\!\left\{Z g(y)\right\} - \mathbb{E}\!\left\{g'(y)\right\} b_i^{(k)}, \\
\bar{b}_i^{(k+1)} &\leftarrow \left(I_3 - B_{1:i-1}^T B_{1:i-1}\right) \tilde{b}_i^{(k+1)}, \\
b_i^{(k+1)} &\leftarrow \frac{\bar{b}_i^{(k+1)}}{\left\|\bar{b}_i^{(k+1)}\right\|_2},
\end{aligned}

```

where for $G(y) = -\exp(-y^2 / 2)$:

```
g(y) = y \exp\!\left(-\frac{y^2}{2}\right),
\qquad
g'(y) = \left(1 - y^2\right) \exp\!\left(-\frac{y^2}{2}\right).

```

In `codes/Section4.m` (lines 31–35), this update is implemented at each iteration as:

```
g = y .* exp(-y.^2/2);
dg = (1 - y.^2) .* exp(-y.^2/2);
b = (Z * g.') - mean(dg .* b, 2);
b = (eye(size(B)) - B(1:row-1, :).' * B(1:row-1, :)) * b;
b = b ./ norm(b, "fro");

```

## 5. Results & Discussion

### 5.1 Summary of Single-Run Permutation Matrices and Reconstruction Errors (Sections 1–4)

In Sections 1–4 of the Persian report, each script (`Section1.m`–`Section4.m`) was executed with a random unit-row initialization `B = randn(size(A)); B = B ./ vecnorm(B, 2, 2);`. The table below summarizes the contrast functions, optimization rules, resulting overall system matrices $\mathrm{permutation\_matrix} = B W A$, and single-run relative reconstruction errors $E = \Vert{}\hat{S} - S\Vert{}_F^2 / \Vert{}S\Vert{}_F^2$ documented on Pages 3–4, 7–8, 11–12, and 15–16 of the report:

| Section & Algorithm | Script | Contrast / Score Function | Vector Update Rule (Before Deflation & Normalization) | Reported Product Matrix $\mathrm{permutation\_matrix} = B W A$ | Single-Run Error $E$ | 
| ----- | ----- | ----- | ----- | ----- | ----- | 
| **Section 1:** Kurtosis + Gradient Projection (GP) | `Section1.m` | $\left\Vert{}\mathrm{kurt}(y)\right\Vert{}$ | $b \leftarrow b + \mu\,\mathrm{sign}(\mathrm{kurt}(y))\left(\mathbb{E}\{Z y^3\} - 3b\right)$ ($\mu = 4$) | $\begin{bmatrix} -0.9897 & -0.0699 & 0.0751 \\ -0.0421 & 1.0721 & -0.9009 \\ 0.0179 & 0.4248 & 0.6969 \end{bmatrix}$ | $0.21409$ | 
| **Section 2:** Kurtosis + Fixed-Point (FastICA) | `Section2.m` | $\left\Vert{}\mathrm{kurt}(y)\right\Vert{}$ | $b \leftarrow \mathbb{E}\{Z y^3\} - 3b$ | $\begin{bmatrix} -0.1437 & -1.1435 & 0.6589 \\ 0.6250 & -0.0132 & 0.7620 \\ -0.7552 & 0.1647 & 0.5369 \end{bmatrix}$ | $0.30682$ | 
| **Section 3:** Negentropy $G(y) = -\exp(-y^2/2)$ + GP | `Section3.m` | $\left[\mathbb{E}\{G(y)\} - \mathbb{E}\{G(v)\}\right]^2$ | $b \leftarrow b - \mu \sqrt{f(b)}\,\mathbb{E}\{Z g(y)\}$ ($\mu = 4$) | $\begin{bmatrix} -0.0434 & 1.0687 & -0.9062 \\ -0.9873 & -0.1091 & 0.0149 \\ -0.0700 & 0.4252 & 0.6939 \end{bmatrix}$ | $0.20242$ | 
| **Section 4:** Negentropy $G(y) = -\exp(-y^2/2)$ + Modified Fixed-Point (Final FastICA) | `Section4.m` | $\left[\mathbb{E}\{G(y)\} - \mathbb{E}\{G(v)\}\right]^2$ | $b \leftarrow \mathbb{E}\{Z g(y)\} - \mathbb{E}\{g'(y)\}b$ | $\begin{bmatrix} 0.9897 & 0.0699 & -0.0751 \\ -0.0421 & 1.0721 & -0.9009 \\ 0.0179 & 0.4248 & 0.6970 \end{bmatrix}$ | $0.20242$ | 

### 5.2 Analysis of the Permutation Matrices and Reconstructed Waveforms

* **Dominant vs. Cross-Talk Entries in** $B W A$**:** In Sections 1, 3, and 4, the row extracting the sinusoidal source $s_1(t)$ achieves near-ideal isolation ($[\mp 0.9897, \mp 0.0699, \pm 0.0751]$ in Sections 1 and 4, and $[-0.9873, -0.1091, 0.0149]$ in Section 3), whereas the remaining two rows exhibit cross-talk entries (highlighted with blue boxes in the report: $-0.9009$ and $0.4248$ in Sections 1 and 4; $-0.9062$ and $0.4252$ in Section 3) that prevent $B W A$ from being a purely diagonal/permutation matrix.

* **Visual Waveform Comparison (Figures 1.5, 2.2, 3.2, and 4.2):** Across all four methods, the reconstructed signals $\hat{s}_1(t)$ (sinusoid), $\hat{s}_2(t)$ (triangular wave), and $\hat{s}_3(t)$ (sparse positive pulses) closely track the phase and shape of the ground-truth waveforms $s_1(t), s_2(t), s_3(t)$, with visible additive noise ripples and slight cross-talk artifacts inherited from the noisy mixtures $X$.

### 5.3 Convergence Behavior at $5\,000$ and $50\,000$ Iterations

In Part 3 of each section, the objective function trajectories for $b_1$, $b_2$, and $b_3$ are plotted over both $\mathrm{iterNum} = 5\,000$ (Figures 1.6, 2.3, 3.3, 4.3) and $\mathrm{iterNum} = 50\,000$ (Figures 1.7, 2.4, 3.4, 4.4):

1. **One-Iteration Convergence of the Third Row (**$b_3$**):** Across all four algorithms, the convergence curve of $b_3$ (bottom subplot of Figures 1.6–4.4) jumps to its final constant plateau in **a single iteration**. As explained on Pages 6, 14, and 17 of the report, because $B \in \mathbb{R}^{3 \times 3}$ is restricted to be orthonormal and $b_1, b_2 \in \mathbb{R}^3$ have already been fixed, the orthogonal complement $\mathrm{span}\{b_1, b_2\}^\perp$ in $\mathbb{R}^3$ has dimension $3 - 2 = 1$. Thus, the Gram-Schmidt projection $b_3 \leftarrow (I_3 - b_1 b_1^T - b_2 b_2^T) b_3$ followed by unit normalization uniquely locks $b_3$ onto $\pm (b_1 \times b_2)$ on the very first step.

2. **Gradient Projection vs. Fixed-Point Convergence Trajectories:**

   * In **Section 1** (Kurtosis GP, Figures 1.6 and 1.7), $b_1$ and $b_2$ require roughly $3\,000\text{–}5\,000$ iterations to reach their steady-state kurtosis plateaus ($\approx 1.4 \times 10^{-6}$ and $8.5 \times 10^{-7}$). Notably, in Figure 1.6 ($b_1$), the score $\vert{}\mathrm{kurt}(y_1)\vert{}$ initially dips to zero near iteration $450$ as $y_1$ passes from negative to positive kurtosis before climbing to its super-Gaussian maximum.

   * In **Section 3** (Negentropy GP, Figures 3.3 and 3.4), $b_1$ and $b_2$ converge smoothly within approximately $2\,000\text{–}3\,000$ iterations.

   * In **Sections 2 and 4** (Fixed-Point methods, Figures 2.3, 2.4, 4.3, and 4.4), the score functions for $b_1$, $b_2$, and $b_3$ reach their steady-state plateaus in only a few iterations (an almost vertical step at the start of the plot).

### 5.4 Section 5 — Controlled Benchmark Using `B_init.mat` ($5\,000$ vs. $50\,000$ Iterations)

To compare the four algorithms on an equal footing without random initialization variance, all four scripts were initialized with the same saved matrix `B_init` from `codes/B_init.mat` and timed using `tic`–`toc` (Pages 18–19 of the report):

| Algorithm | Iteration Count `iterNum` | Elapsed Time $[s]$ | Relative Error $E$ | Max Iterations to Converge a Single Row | 
| ----- | ----- | ----- | ----- | ----- | 
| **Section 1:** Kurtosis + Gradient Projection (`kurt(deflation)`) | $5\,000$ | $1.364805$ | $0.24635$ | $\approx 5\,000$ iterations | 
| **Section 2:** Kurtosis + Fixed-Point (`kurt(fixed-point)`) | $5\,000$ | $0.824397$ | $0.49648$ | Very few (1 step) | 
| **Section 3:** Negentropy + Gradient Projection (`fastICA(G.P)`) | $5\,000$ | $0.700543$ | $0.22055$ | $\approx 3\,000$ iterations | 
| **Section 4:** Negentropy + Modified Fixed-Point (`fastICA(fixed-point)`) | $5\,000$ | $1.255933$ | $0.20242$ | Very few | 
| **Section 1:** Kurtosis + Gradient Projection (`kurt(deflation)`) | $50\,000$ | $13.699159$ | $0.20243$ | $\approx 5\,000$ iterations | 
| **Section 2:** Kurtosis + Fixed-Point (`kurt(fixed-point)`) | $50\,000$ | $8.677623$ | $0.49648$ | Very few (1 step) | 
| **Section 3:** Negentropy + Gradient Projection (`fastICA(G.P)`) | $50\,000$ | $6.883684$ | $0.20242$ | $\approx 2\,000$ iterations | 
| **Section 4:** Negentropy + Modified Fixed-Point (`fastICA(fixed-point)`) | $50\,000$ | $12.379115$ | $0.20242$ | Very few | 

### 5.5 Overall Comparative Ranking (Section 5 Conclusion)

Based on the empirical benchmarks in Section 5 (Pages 19–20 of the report):

1. **Convergence Speed (Iterations to Converge):** Although the modified fixed-point FastICA algorithm (`Section4.m`) has a slightly higher per-iteration computational cost (evaluating both $g(y)$ and $g'(y)$ at each step, taking $1.256\ \mathrm{s}$ for $5\,000$ fixed iterations), its **number of iterations to converge** is drastically smaller than the Gradient Projection methods—converging in just a handful of iterations instead of $2\,000\text{–}5\,000$ iterations:

```
\text{Convergence Time (Iterations):} \quad
\mathrm{fastICA}(\text{fixed-point}) \approx \mathrm{kurt}(\text{fixed-point}) < \mathrm{fastICA}(\text{GP}) < \mathrm{kurt}(\text{deflation GP}).

```

2. **Separation Quality (Reconstruction Error** $E$**):** At $5\,000$ iterations, the modified fixed-point FastICA (`Section4.m`) already achieves the best separation error ($E = 0.20242$), whereas Negentropy GP (`Section3.m`, $E = 0.22055$) and Kurtosis GP (`Section1.m`, $E = 0.24635$) require $50\,000$ iterations to reach the same asymptotic error floor ($E = 0.20242\text{–}0.20243$). The kurtosis fixed-point script (`Section2.m`) exhibits the largest error ($E = 0.49648$):

```
\text{Separation Error } (E): \quad
\mathrm{fastICA}(\text{fixed-point}) \approx \mathrm{fastICA}(\text{GP}) \le \mathrm{kurt}(\text{deflation GP}) < \mathrm{kurt}(\text{fixed-point}).

```

### 5.6 Notes on the Submitted Solutions / Implementation Notes

* **Unmixing Applied to `X` Rather Than `Z` in `S_hat = B * X` (`Section1.m`–`Section4.m`):** In all four scripts (`Section1.m` line 86, `Section2.m` line 42, `Section3.m` line 42, and `Section4.m` line 41), the estimated sources are computed as `S_hat = B * X;` rather than `S_hat = B * Z;` (or `S_hat = B * W * X;`), even though the subsequent line (`permutation_matrix = B * W * A;`) properly includes the whitening matrix $W$ in the total separating transformation $B_{\mathrm{total}} = B W$. Because $B$ is an orthogonal matrix designed to separate the **whitened** mixtures $Z = W X$, multiplying $B$ directly by the unwhitened observations $X$ leaves the spatial covariance distortion $W^{-1}$ uncompensated in `S_hat`, which accounts for the residual error floor $E \approx 0.20242$ across Sections 1, 3, and 4.

* **Unnormalized Correlation Matrix** $R_X = X X^T$ **and Mean Centering:** In all four scripts, the sample correlation matrix is computed as `R_x = X * X.'` without subtracting the sample mean of $X$ (note that source $s_3(t)$ is a strictly non-negative pulse train with a positive DC mean) and without dividing by $T = 1001$. Consequently, $Z Z^T = I_3$ (rather than $\frac{1}{T} Z Z^T = I_3$), so each row of $Z$ has unit Euclidean norm ($\Vert{}z_i\Vert{}_2 = 1$) and sample variance $\frac{1}{T} \sum_{t=1}^T z_i(t)^2 = \frac{1}{1001} \approx 10^{-3}$. Because `kurt = mean(y.^4, 2) - 3 * mean(y.^2, 2)^2` divides by $T^2$, the plotted kurtosis score function values in Sections 1 and 2 have magnitude $\sim 10^{-7}\text{–}10^{-6}$.

* **Single-Step Reset in `Section2.m` (Lines 30–36):** In `codes/Section2.m`, the inner `while` loop is written as:

  ```
  b_new = B(row, :).';
  b = inf * ones(M,1);
  y = b_new.' * Z;
  iter = 0;
  while (1 - norm(b_new * b.') > thr) || (iter < iterNum)
      b = b_new;
      kurt = mean(y.^4, 2) - 3 * mean(y.^2, 2)^2;
      b = Z * (y.^3).' - 3*b;
      b = (eye(size(B)) - B(1:row-1, :).' * B(1:row-1, :)) * b;
      b = b ./ norm(b,"fro");
      y = b.' * Z;
      iter = iter + 1;
      score_func(row, iter) = abs(kurt);
  end
  
  ```

  Because `b_new` is never updated to the new iterate `b` at the end of the loop body, `b = b_new;` at line 31 resets `b` back to the initial random vector `B(row, :).'` on every iteration. Consequently, `Section2.m` repeatedly computes the exact same **single** fixed-point step from the initial vector `B(row, :).'` for all $50\,000$ iterations, which explains why its score function jumps at iteration $2$ and stays completely constant thereafter (Figures 2.3 and 2.4) and why its separation error ($E = 0.30682$ in Section 2 and $E = 0.49648$ in Section 5) is higher than the other three methods.

* **Sign of the Reference Term** $\mathbb{E}\{G(v)\}$ **and Gradient Update in `Section3.m` and `Section4.m`:** In `Section3.m` (line 31) and `Section4.m` (line 30), the score function is written as `f = mean(-exp(-y.^2/2) - mean(exp(-v.^2/2)))^2;`, which uses a minus sign before `mean(exp(-v.^2/2))` instead of `+` (since $G(v) = -\exp(-v^2/2)$, $-G(v) = +\exp(-v^2/2)$). Because $v$ is fixed outside the loop, $\mathbb{E}\{\exp(-v^2/2)\} \approx 1/\sqrt{2} \approx 0.707$ acts as a positive constant offset inside the square, shifting the plotted objective value to $\approx (1 + 0.707)^2 \approx 2.91\text{–}2.96$. In `Section3.m` (lines 33–34), subtracting `mu * sqrt(f) * (Z * g.')` decreases this shifted quantity `f` slightly while driving $b$ toward the non-Gaussian stationary point, which is why the convergence curves in Figures 3.3 and 3.4 are decreasing.

* **Energy Rescaling Index in the Ambiguity Removal Loop (`Section1.m` line 95, `Section2.m` line 51, `Section3.m` line 51, `Section4.m` line 50):** In the post-processing loop that resolves permutation and scaling ambiguities, the energy rescaling line is written as `S_hat(ind, :) = S_hat(ind, :) * norm(S(i, :), "fro") ./ norm(S_hat(ind, :), "fro");`, which multiplies by the Frobenius norm of row `i` (`norm(S(i, :), "fro")`) rather than the matched source row `ind` (`norm(S(ind, :), "fro")`). Whenever row `i` of $B$ matches source `ind = i` (or because the three sources in `hw10.mat` have very similar $\ell_2$-norms: $\|s_1\|_2 = 1.0000$, $\|s_2\|_2 = 0.9133$, $\|s_3\|_2 = 1.0156$), the visual overlay and error $E$ are only minimally affected.

* **Minor Typographical Slips in the Persian Report (Pages 3, 7, 11, 15, and 20):**

  * On **Pages 3 and 7**, the kurtosis definition is written as $\mathrm{kurt}(y) = E\{y^4\} - 3(E\{y^2\})$ (omitting the exponent $2$ on $(E\{y^2\})^2$), whereas `Section1.m` (line 77) and `Section2.m` (line 32) compute the exact squared term `3 * mean(y.^2, 2)^2`.

  * On **Pages 3, 7, 11, and 15**, the Gram-Schmidt deflation step inside the large curly brace is written in shorthand notation as $b_i = \left(I - \begin{bmatrix} b_1^T \\ b_2^T \\ \vdots \\ b_{i-1}^T \end{bmatrix}\right)$, omitting the right-hand factor $B_{1:i-1} b_i$, whereas the MATLAB scripts implement the full projection `(eye(size(B)) - B(1:row-1, :).' * B(1:row-1, :)) * b`.

  * On **Page 11 (Section 3)** and **Page 15 (Section 4)**, the introductory sentence under "Part 1" is copied from Section 1/2 and says `روش بیشینه سازی kurt` ("kurtosis maximization method") instead of negentropy maximization with $G(y) = -\exp(-y^2 / 2)$, and the caption of Figure 4.1 on Page 15 says `kurt deflation (اصلاح شده)` instead of modified fixed-point FastICA.

  * On **Page 20 (Section 5 summary)**, the final inequality for `Error (E)` is printed as `fastICA(fixed-point) ≈ fastICA(G.P) ≤ kurt(deflation) < fastICA(G.P)`, where the last term on the right was accidentally repeated as `fastICA(G.P)` instead of `kurt(fixed-point)` (as stated in the narrative at the bottom of Page 19, where the algorithm of Section 2, `kurt(fixed-point)`, has the largest error $E = 0.49648$).

## 6. How to Use

### 6.1 Running the MATLAB Scripts

Open MATLAB, navigate to the `codes` directory (or add it to the MATLAB path), and run the four scripts:

```
cd codes
Section1
Section2
Section3
Section4

```

* **Dependencies:** All four scripts (`Section1.m`–`Section4.m`) use only base MATLAB functions (`eig`, `sort`, `randn`, `vecnorm`, `norm`, `mean`, `sign`, `exp`, `sqrt`, `plot`, `subplot`) and require no additional toolboxes.

* **Iteration Count Configuration:** By default, the scripts inside `codes/` are saved with `iterNum = 5e4` ($50\,000$ iterations, corresponding to Figures 1.7, 2.4, 3.4, and 4.4 of the report). To reproduce the $5\,000$-iteration plots (Figures 1.6, 2.3, 3.3, and 4.3), change `iterNum = 5e4;` to `iterNum = 5e3;`.

* **Reproducing the Section 5 Benchmark (`B_init.mat`):** To reproduce the controlled benchmark in Section 5 of the report, replace the random initialization lines `B = randn(size(A)); B = B ./ vecnorm(B, 2, 2);` in each script with:

  ```
  load B_init.mat
  B = B_init;
  
  ```

* **Generated Figures:**

  * `Section1.m` opens **5 figures**:

    1. Three-panel subplot of the ground-truth sources $s_1(t), s_2(t), s_3(t)$ (Figure 1.1).

    2. Three-panel subplot of the noiseless observations $X_{\mathrm{noNoise}}(t) = A S(t)$ (Figure 1.2).

    3. Three-panel subplot of the noisy observations $X(t) = A S(t) + \mathrm{Noise}(t)$ (Figure 1.3).

    4. Three-panel overlay comparison of the estimated sources $\hat{s}_i(t)$ (magenta) and true sources $s_i(t)$ (black) (Figure 1.5).

    5. Three-panel convergence plot of the score function $|\mathrm{kurt}(y_i)|$ versus iteration index for $b_1, b_2, b_3$ (Figure 1.6 / 1.7).

  * `Section2.m`, `Section3.m`, and `Section4.m` each open **2 figures**:

    1. Three-panel overlay comparison of the estimated sources $\hat{s}_i(t)$ (magenta) and true sources $s_i(t)$ (black) (Figures 2.2, 3.2, and 4.2).

    2. Three-panel convergence plot of the score function versus iteration index for $b_1, b_2, b_3$ (Figures 2.3/2.4, 3.3/3.4, and 4.3/4.4).

### 6.2 Expected Console Output (`Section1.m`–`Section4.m`)

Because `B` is initialized randomly via `randn(size(A))` in the standalone scripts (and `v = randn(1, T)` is drawn randomly in `Section3.m` and `Section4.m`), the printed error `E` varies slightly across runs if `B_init.mat` is not loaded. Representative console outputs matching the single-run results in Sections 1–4 of the Persian report are:

```
% Section1.m
Part 2:
Error (E) = 0.21409

% Section2.m
Part 2:
Error (E) = 0.30682

% Section3.m
Part 2:
Error (E) = 0.20242

% Section4.m
Part 2:
Error (E) = 0.20242

```

When initialized deterministically with `load B_init.mat; B = B_init;` at `iterNum = 50000` (Section 5 of the report), the console outputs are:

```
% Section1.m (with B_init, iterNum = 50000)
Part 2:
Error (E) = 0.20243

% Section2.m (with B_init, iterNum = 50000)
Part 2:
Error (E) = 0.49648

% Section3.m (with B_init, iterNum = 50000)
Part 2:
Error (E) = 0.20242

% Section4.m (with B_init, iterNum = 50000)
Part 2:
Error (E) = 0.20242

```

### 6.3 Reading the Theoretical Derivations and Report

1. Use **Section 3** for the complete English translation of the assignment problems and **Section 4** for the step-by-step mathematical derivations of spatial whitening, deflationary Gram-Schmidt projection, kurtosis-based GP/FastICA, and negentropy-based GP/FastICA.

2. Use **Section 5** to review the permutation matrices, error benchmarks, convergence analyses, and implementation notes alongside the original Persian report [BSS_HW#10_810100205_Arman_Majidi.pdf](https://www.google.com/search?q=./BSS_HW%252310_810100205_Arman_Majidi.pdf) and assignment sheet [hw10.pdf](https://www.google.com/search?q=./hw10.pdf).

## 7. References

1. Akhavan, S. *Blind Source Separation: Homework 10*. Course assignment sheet and lecture notes, University of Tehran.

2. Hyvärinen, A., & Oja, E. (1997). A fast fixed-point algorithm for independent component analysis. *Neural Computation*, 9(7), 1483–1492.

3. Hyvärinen, A. (1999). Fast and robust fixed-point algorithms for independent component analysis. *IEEE Transactions on Neural Networks*, 10(3), 626–634.

4. Hyvärinen, A., Karhunen, J., & Oja, E. (2001). *Independent Component Analysis*. John Wiley & Sons.

5. Comon, P., & Jutten, C. (Eds.). (2010). *Handbook of Blind Source Separation: Independent Component Analysis and Applications*. Academic Press.