# Homework 9: Independent Component Analysis (ICA) via Mutual Information Minimization — Deflationary Orthonormal Separation after Spatial Whitening and Equivariant Relative-Gradient Separation

**Course:** Blind Source Separation  
**Instructor:** Dr. Saeed Akhavan  
**Author:** Arman Majidi  
**Student ID:** 810100205  
**University:** University of Tehran  
**Semester:** Spring 2024-2025

## 1. Overview & Assignment Structure

This folder contains the **English problem statements, analytical derivations, MATLAB implementations, and simulation results for Homework 9**. The assignment is organized into three main sections that solve the noisy instantaneous Blind Source Separation (BSS) problem under the statistical independence assumption by minimizing the Kullback-Leibler Divergence (mutual information) between the separated outputs:

1. **Section 1 — Deflationary Orthonormal ICA via Kullback-Leibler Divergence Minimization after Spatial Whitening (Parts 0–3):** Generates the noiseless and noisy $`3 \times 1001`$ mixtures from `hw9.mat`, applies eigenvalue-based spatial whitening $`Z = D^{-1/2} U^T X`$ so that the separating matrix $`B`$ in the whitened domain is orthonormal, estimates each marginal score function $`\psi_{Y_m}(y_m)`$ via 5th-order polynomial Mean Square Error (MSE) regression, extracts the rows of $`B`$ sequentially using steepest descent and Gram-Schmidt orthogonal projection (deflation), evaluates the composite system matrix $`P = B (D^{-1/2} U^T) A`$, resolves permutation and scaling ambiguities to compute the relative reconstruction error $`E`$, and analyzes the convergence trajectory over $`500`$ and $`5000`$ iterations.
2. **Section 2 — Equivariant (Natural / Relative Gradient) ICA without Pre-Whitening (Parts 0–3):** Implements the third classroom ICA method—whose asymptotic convergence behavior is independent of the conditioning of the mixing matrix $`A`$ (the equivariant property)—directly on the noisy observations $`X`$ using polynomial MSE score function estimation and the matrix-multiplied relative gradient $`\frac{\partial f}{\partial B} = D B`$, reports the resulting product matrix $`P = B A`$ and relative reconstruction error $`E`$, and examines the convergence behavior over $`500`$ and $`5000`$ iterations.
3. **Section 3 — Comparative and Empirical Evaluation of the Three KLD-Based ICA Algorithms (Homework 8 & Homework 9):** Compares standard Gradient-Projection $`D_{\mathrm{KL}}`$ minimization (from Homework 8), Deflationary Orthonormal ICA (Section 1), and Equivariant ICA (Section 2) in terms of separation quality ($`E`$), runtime (`tic`/`toc`), convergence speed, and computational structure.

All parts of all three sections are mandatory; no optional or bonus problems were assigned. Because both the original assignment sheet and the submitted solution report are written in Persian, this `README.md` provides a **complete, self-contained English reference for every problem statement, mathematical derivation, algorithm, and numerical result**.

## 2. Directory Contents

| File | Description |
| --- | --- |
| [codes/Section1.m](./codes/Section1.m) | MATLAB script for Section 1 (Parts 0–3): noisy/noiseless signal plotting, PCA spatial whitening ($`Z = D^{-1/2} U^T X`$), 5th-order polynomial MSE score estimation, deflationary row-by-row ICA with Gram-Schmidt orthogonalization, permutation/scale alignment, error $`E`$ calculation, and convergence plot. |
| [codes/Section2.m](./codes/Section2.m) | MATLAB script for Section 2 (Parts 0–3): equivariant ICA on unwhitened noisy observations $`X`$ using relative gradient updates $`\frac{\partial f}{\partial B} = D B`$, 5th-order polynomial MSE score estimation, permutation/scale alignment, error $`E`$ calculation, and convergence plot. |
| [codes/hw9.mat](./codes/hw9.mat) | Dataset containing the $`3 \times 3`$ mixing matrix `A`, the $`3 \times 1001`$ ground-truth source matrix `S`, and the $`3 \times 1001`$ additive Gaussian noise matrix `Noise`. |
| [BSS\_HW#9\_810100205\_Arman\_Majidi.pdf](./BSS_HW%239_810100205_Arman_Majidi.pdf) | Submitted report and analytical solutions in Persian (including source/observation waveforms, BSS block diagrams, estimated permutation matrices, overlaid reconstructed sources, $`500`$ and $`5000`$-iteration convergence curves, and comparative analysis). |
| [hw9.pdf](./hw9.pdf) | Original assignment problem sheet in Persian. |
| `README.md` | Self-contained English problem statements, mathematical derivations, algorithm descriptions, numerical results, and usage guide. |

Keep the `codes` subfolder inside the homework directory.

## 3. Assignment Problem Statement

### 3.1 General Problem Setup and Preprocessing (Part 0)

In this assignment, we solve the Blind Source Separation (BSS) problem under the assumption of mutually statistically independent sources. The mixing matrix $`A`$, the source matrix $`S`$, and the additive noise matrix $`\mathrm{Noise}`$ are provided in the file `hw9.mat`:

- **Mixing Matrix (`A`):** A $`3 \times 3`$ full-rank matrix with unit-norm columns ($`\det(A) \approx 0.3818`$, $`\mathrm{cond}(A) \approx 5.0327`$):

```math
A = \frac{1}{\sqrt{14}} \begin{bmatrix}
1 & 1 & 1 \\
2 & -2 & -3 \\
3 & 3 & -2
\end{bmatrix}
\approx \begin{bmatrix}
0.267261 & 0.267261 & 0.267261 \\
0.534522 & -0.534522 & -0.801784 \\
0.801784 & 0.801784 & -0.534522
\end{bmatrix}.
```

- **Source Matrix (`S`):** A $`3 \times 1001`$ matrix ($`M = 3`$ sources, $`T = 1001`$ time samples) containing three zero-mean, unit-energy deterministic/periodic waveforms ($`\|s_m\|_2 = 1`$ for $`m = 1, 2, 3`$, so $`\|S\|_F^2 = 3`$): a sinusoidal wave $`s_1(t)`$, a triangular wave $`s_2(t)`$, and a spiky pulse train $`s_3(t)`$.
- **Additive Noise Matrix (`Noise`):** A $`3 \times 1001`$ zero-mean Gaussian noise matrix with $`\|\mathrm{Noise}\|_F^2 \approx 0.01914`$, corresponding to an input signal-to-noise ratio of $`\mathrm{SNR} = 10\log_{10}\!\left(\|A S\|_F^2 / \|\mathrm{Noise}\|_F^2\right) \approx 22.11\ \mathrm{dB}`$.

First, construct the noisy observation matrix $`X`$ via

```math
X = A S + \mathrm{Noise},
```

and plot the original sources $`S`$, the noiseless observations $`X_{\mathrm{noNoise}} = A S`$, and the noisy observations $`X`$ to inspect their time-domain waveforms. Next, approach the problem from a blind source separation perspective: assuming only the observed matrix $`X`$ and the number of sources $`M = 3`$ are known, our strategy is to find a separating matrix $`B`$ such that the outputs $`Y = B X`$ (or $`Y = B Z`$ after spatial whitening) are as statistically independent as possible.

### 3.2 Section 1: Deflationary Orthonormal ICA via KLD Minimization after Spatial Whitening

Implement the **second method presented in class** based on minimizing the Kullback-Leibler Divergence ($`D_{\mathrm{KL}}`$) **after spatially whitening the data (the deflation mode)**. Note that the overall separating matrix mapping $`X`$ to the separated outputs is the product of the final orthonormal matrix $`B`$ obtained from the algorithm and the spatial whitening matrix $`W = D^{-1/2} U^T`$.

#### Part 1.1 — Composite Separation-Mixing Product (Permutation Matrix Check)

Multiply the final overall separating matrix obtained by your algorithm by the true mixing matrix $`A`$ and report the resulting $`3 \times 3`$ matrix. Ideally, this product should be close to a generalized **permutation matrix** (having only one dominant non-zero entry in each row and each column).

#### Part 1.2 — Ambiguity Removal, Source Reconstruction Plots, and Relative Error

Resolve both the **permutation ambiguity** (channel ordering and sign) and the **scaling ambiguity** of the estimated sources so that the energy ($`\ell_2`$-norm) of each estimated source equals the energy of the corresponding true source. Plot the aligned estimated sources $`\hat{S}`$ overlaid on the true sources $`S`$, and report the normalized squared Frobenius reconstruction error:

```math
E = \frac{\left\|\hat{S} - S\right\|_F^2}{\left\|S\right\|_F^2}.
```

#### Part 1.3 — Convergence Curve

Plot the convergence curve of the algorithm (the objective / gradient norm metric versus iteration index).

### 3.3 Section 2: Equivariant ICA via KLD Minimization

Implement the **third method presented in class**, whose convergence behavior does not depend on the mixing/separating matrix conditioning (**the equivariant method**). Report all results following the exact same structure as Section 1 (Parts 2.1, 2.2, and 2.3: product matrix $`B A`$, overlaid source waveforms, relative error $`E`$, and convergence plots).

### 3.4 Section 3: Empirical and Comparative Evaluation of Three ICA Methods (HW8 & HW9)

Provide an empirical and comparative analysis of the **three KLD-based ICA methods** implemented across Homework 8 and Homework 9 (the two methods above plus the direct $`D_{\mathrm{KL}}`$ minimization method implemented in Homework 8). State which method converged fastest and which method achieved the best separation quality (lowest error parameter $`E`$).

## 4. Mathematical Formulation & Solutions

### 4.1 Section 1: Deflationary Orthonormal ICA after Spatial Whitening (`Section1.m`)

#### Step 1 — Spatial Whitening of the Observations

Given the noisy observation matrix $`X \in \mathbb{R}^{M \times T}`$ ($`M = 3`$, $`T = 1001`$), `Section1.m` (lines 51–56) forms the unnormalized sample correlation matrix $`R_X = X X^T \in \mathbb{R}^{3 \times 3}`$, computes its eigenvalue decomposition $`R_X = U D U^T`$ with eigenvalues sorted in descending order ($`\lambda_1 = 1.8552`$, $`\lambda_2 = 1.2133`$, $`\lambda_3 = 0.0514`$), and applies the spatial whitening transformation $`W = D^{-1/2} U^T`$:

```math
Z = W X = D^{-1/2} U^T X \in \mathbb{R}^{3 \times 1001},
\qquad
R_Z = Z Z^T = D^{-1/2} U^T \left(X X^T\right) U D^{-1/2} = I_3.
```

As illustrated in the two-stage BSS block diagram (Figure 1.4 on Page 3 of the report):

```math
S \xrightarrow{\quad A \quad} X \xrightarrow{\quad W = D^{-1/2} U^T \quad} Z \xrightarrow{\quad B \quad} Y = B Z = B W X.
```

#### Step 2 — Simplification of the KLD Objective under Orthonormality

To make the separated components $`Y = [y_1, \dots, y_M]^T = B Z`$ mutually independent, we minimize the Kullback-Leibler Divergence between the joint PDF $`p_Y(y)`$ and the product of marginal PDFs $`\prod_{m=1}^M p_{Y_m}(y_m)`$:

```math
\begin{aligned}
D_{\mathrm{KL}}\!\left(p_Y(y) \,\Big\|\, \prod_{m=1}^M p_{Y_m}(y_m)\right)
&= -H(Y) + \sum_{m=1}^M H(y_m) \\
&= -H(Z) - \log|\det(B)| + \sum_{m=1}^M H(y_m).
\end{aligned}
```

Because $`Z`$ is spatially white ($`Z Z^T = I_M`$) and we constrain the whitened outputs $`Y = B Z`$ to remain uncorrelated with unit norm ($`Y Y^T = B Z Z^T B^T = B B^T = I_M`$), the separating matrix $`B`$ in the whitened space must be **orthogonal** ($`B B^T = I_M`$, $`|\det(B)| = 1`$). Consequently, the term $`-\log|\det(B)| = 0`$ and its matrix-inverse gradient term $`-B^{-T}`$ vanish completely, reducing the objective function (Page 3 of the report) to the sum of marginal differential entropies:

```math
f(B) = \sum_{m=1}^M H(y_m),
\qquad
\text{subject to} \quad B B^T = I_M.
```

#### Step 3 — Polynomial MSE Score Function Estimation

Differentiating the marginal differential entropy $`H(y_m) = -\mathbb{E}\{\log p_{Y_m}(y_m)\}`$ with respect to the $`m`$-th row $`b_m^T`$ of $`B`$ (where $`y_m = b_m^T z`$) requires the **marginal score function**:

```math
\psi_Y(y) = -\frac{p_Y'(y)}{p_Y(y)} = -\frac{d}{dy} \log p_Y(y).
```

Following the classroom parametric MSE score estimation approach (Page 3 of the report), $`\psi_Y(y)`$ is approximated as a linear combination of 5th-order polynomial basis functions:

```math
\hat{\psi}_Y(y) = \theta^T K(y),
\qquad
K(y) = \begin{bmatrix} 1 \\ y \\ y^2 \\ y^3 \\ y^4 \\ y^5 \end{bmatrix} \in \mathbb{R}^{6 \times 1},
\qquad
K'(y) = \frac{dK(y)}{dy} = \begin{bmatrix} 0 \\ 1 \\ 2y \\ 3y^2 \\ 4y^3 \\ 5y^4 \end{bmatrix} \in \mathbb{R}^{6 \times 1}.
```

Minimizing the Mean Square Error $`\mathbb{E}\!\left\{\left(\psi_Y(y) - \theta^T K(y)\right)^2\right\}`$ and using integration by parts ($`\mathbb{E}\{\psi_Y(y) K(y)\} = \mathbb{E}\{K'(y)\}`$ under vanishing boundary tails) yields the quadratic cost in $`\theta`$:

```math
\begin{aligned}
g(\theta) &= \mathbb{E}\!\left\{\theta^T K(y) K(y)^T \theta\right\} - 2\,\mathbb{E}\!\left\{\theta^T K'(y)\right\}, \\
\frac{\partial g}{\partial \theta} &= 2\,\mathbb{E}\!\left\{K(y) K(y)^T\right\} \theta - 2\,\mathbb{E}\!\left\{K'(y)\right\} = 0.
\end{aligned}
```

Solving for the optimal parameter vector $`\hat{\theta}`$ gives the closed-form estimator implemented in `Section1.m` (lines 65–66):

```math
\hat{\theta} = \left(\mathbb{E}\!\left\{K(y) K(y)^T\right\}\right)^{-1} \mathbb{E}\!\left\{K'(y)\right\}
\approx \left(\frac{1}{T} K_y K_y^T\right)^{-1} \left(\frac{1}{T} \sum_{t=1}^T K'(y(t))\right),
\qquad
\hat{\psi}_Y(y) = \hat{\theta}^T K(y).
```

#### Step 4 — Deflationary Row-by-Row Update and Gram-Schmidt Orthogonal Projection

For the $`m`$-th row $`b_m^T`$ of $`B`$ (represented as a column vector $`b_m = B(m, :)^T \in \mathbb{R}^{M \times 1}`$), the gradient of $`H(y_m)`$ with respect to $`b_m^T`$ is

```math
\frac{\partial f}{\partial b_m^T} = \mathbb{E}\!\left\{\psi_{Y_m}(y_m)\, z^T\right\} \approx \frac{1}{T} \hat{\psi}_{Y_m}(y_m) Z^T \in \mathbb{R}^{1 \times M}.
```

In the **deflationary scheme** (`Section1.m` lines 60–73), the rows $`m = 1, 2, \dots, M`$ are optimized sequentially over `iterNum = 500` iterations with step size $`\mu = 0.2`$. At each iteration for row $`m`$:

1. **Gradient descent step:**

```math
b_m \leftarrow b_m - \mu \left(\frac{\partial f}{\partial b_m^T}\right)^T.
```

2. **Gram-Schmidt orthogonal projection onto the subspace orthogonal to previously found rows $`b_1, \dots, b_{m-1}`$:** Let $`B_{1:m-1} = B(1:m-1, :) \in \mathbb{R}^{(m-1) \times M}`$ denote the matrix of already-converged orthonormal rows. To prevent $`b_m`$ from converging to any previously extracted source, $`b_m`$ is projected onto the orthogonal complement of $`\mathrm{row}(B_{1:m-1})`$:

```math
b_m \leftarrow \left(I_M - B_{1:m-1}^T B_{1:m-1}\right) b_m.
```

3. **Unit-norm normalization:**

```math
b_m \leftarrow \frac{b_m}{\|b_m\|_F}.
```

#### Step 5 — Permutation/Scale Alignment and Error Metric (`Section1.m` lines 74–87)

Once $`B`$ is obtained, the overall transformation from the true sources $`S`$ to the whitened-domain outputs is the $`3 \times 3`$ composite product (`Section1.m` line 87):

```math
\mathrm{permutation\_matrix} = B \left(D^{-1/2} U^T\right) A.
```

To resolve permutation, sign, and scaling ambiguities before computing the reconstruction error $`E`$ (`Section1.m` lines 74–84), each estimated row in `S_hatt` is matched greedily to the true source row $`S(\mathrm{ind}, :)`$ with maximum absolute inner product $`|\mathrm{corr}| = |\hat{s}_i S^T|`$, multiplied by $`\mathrm{sign}(\mathrm{corr}(\mathrm{ind}))`$ to fix sign ambiguity, and rescaled to match the exact Frobenius ($`\ell_2`$) norm of the true source.

### 4.2 Section 2: Equivariant (Relative / Natural Gradient) ICA (`Section2.m`)

#### Step 1 — Full-Matrix KLD Objective without Pre-Whitening

In the **equivariant method** (Page 6 of the report), no spatial whitening stage is required: the separating matrix $`B \in \mathbb{R}^{M \times M}`$ is applied directly to the noisy observations $`X`$ ($`Y = B X`$, Figure 2.1 of the report). The Kullback-Leibler Divergence objective is

```math
f(B) = -H(Y) + \sum_{m=1}^M H(y_m) = -H(X) - \log|\det(B)| + \sum_{m=1}^M H(y_m).
```

The standard Euclidean gradient of $`f(B)`$ with respect to $`B`$ is

```math
\frac{\partial f}{\partial B} = \mathbb{E}\!\left\{\psi_Y(Y) X^T\right\} - B^{-T}.
```

#### Step 2 — Serial Update Rule and Relative / Natural Gradient

Right-multiplying the standard gradient by $`B^T B`$ (or equivalently performing a first-order Taylor expansion under a relative serial perturbation $`B^{(k+1)} = (I - \mu D) B^{(k)}`$ in terms of the combined system matrix $`C = B A`$, as noted on Page 11 of the report) yields the **equivariant natural/relative gradient update**:

```math
\left(\mathbb{E}\!\left\{\psi_Y(Y) X^T\right\} - B^{-T}\right) B^T B
= \left(\mathbb{E}\!\left\{\psi_Y(Y) Y^T\right\} - I_M\right) B
= D B,
```

where the matrix $`D \in \mathbb{R}^{M \times M}`$ depends **only on the separated outputs $`Y`$** (and not on $`B^{-1}`$ or the conditioning of $`A`$):

- **Off-diagonal entries ($`i \neq j`$):** Measure non-linear cross-correlation between distinct outputs $`y_i`$ and $`y_j`$:

```math
D_{i,j} = \mathbb{E}\!\left\{\psi_{Y_i}(y_i)\, y_j\right\} \approx \frac{1}{T} \hat{\psi}_{Y_i}(y_i) y_j^T, \qquad i \neq j.
```

- **Diagonal entries ($`i = j`$):** In the unconstrained equivariant formulation (written on Page 6 of the report), the diagonal entries regulate output variance to unity via $`D_{i,i} = 1 - \mathbb{E}\{y_i^2\}`$ (or $`\mathbb{E}\{\psi_{Y_i}(y_i) y_i\} - 1`$), eliminating the need for explicit row normalization. In `Section2.m` (lines 28–31), an equivalent unit-row-norm projection variant is implemented by zeroing out the diagonal of $`D`$ (`D = D - diag(diag(D))`) and explicitly normalizing the rows of $`B`$ to unit Euclidean norm at each step (`B = B ./ vecnorm(B, 2, 2)`):

```math
\begin{aligned}
D &= \frac{1}{T} \hat{\Psi}_Y(Y) Y^T - \mathrm{diag}\!\left(\mathrm{diag}\!\left(\frac{1}{T} \hat{\Psi}_Y(Y) Y^T\right)\right), \\
\frac{\partial f}{\partial B} &= D B, \\
B^{(k+1)} &= B^{(k)} - \mu D B^{(k)}, \qquad B^{(k+1)}(i, :) \leftarrow \frac{B^{(k+1)}(i, :)}{\left\|B^{(k+1)}(i, :)\right\|_2},
\end{aligned}
```

with step size $`\mu = 0.01`$ and `iterNum = 500`. Because the global system matrix $`C = B A`$ updates according to $`C^{(k+1)} = (I - \mu D) C^{(k)}`$, the convergence trajectory in $`C`$-space is completely invariant to the conditioning of the mixing matrix $`A`$ (**equivariance**), and no matrix inversion $`B^{-T}`$ is ever computed.

### 4.3 Section 3: Theoretical & Computational Comparison of the Three KLD-Based ICA Algorithms

The three KLD-minimization ICA algorithms studied across Homework 8 and Homework 9 share the same polynomial MSE marginal score estimator $`\hat{\psi}_{Y_i}(y_i) = \hat{\theta}_i^T K(y_i)`$ but differ fundamentally in how they handle the volume-preserving term $`-\log|\det(B)|`$ and the parameter space geometry:

1. **Method 1 — Direct $`D_{\mathrm{KL}}`$ Minimization via Gradient Projection (Homework 8):**
   - Operates on unwhitened data $`X`$ using the Euclidean gradient $`\frac{\partial f}{\partial B} = \frac{1}{T}\hat{\Psi}_Y(Y) X^T - B^{-T}`$ followed by row-wise $`\ell_2`$-normalization.
   - Requires computing the matrix inverse transpose $`B^{-T}`$ at every iteration and suffers from slow or ill-conditioned convergence when $`A`$ is poorly conditioned.
2. **Method 2 — Deflationary Orthonormal ICA after Spatial Whitening (Homework 9, Section 1):**
   - Pre-whitens $`X \to Z = D^{-1/2} U^T X`$ once at the beginning so that $`B`$ is restricted to the orthogonal group $`O(M)`$.
   - Eliminates $`-B^{-T}`$ completely and extracts the rows of $`B`$ one by one via Gram-Schmidt projection $`\left(I - B_{1:m-1}^T B_{1:m-1}\right)`$.
   - Achieves rapid per-row convergence and the lowest runtime, though errors accumulated in earlier rows propagate into subsequent rows via the Gram-Schmidt subspace constraint.
3. **Method 3 — Equivariant / Natural-Gradient ICA (Homework 9, Section 2):**
   - Replaces the Euclidean gradient with the relative/natural gradient $`\frac{\partial f}{\partial B} = D B`$ without needing pre-whitening or matrix inversion $`B^{-T}`$.
   - Updates all rows simultaneously and symmetrically; although it requires slightly more iterations ($`\approx 350\text{–}400`$ iterations at $`\mu = 0.01`$) for $`\|D B\|_F`$ to decay to zero, it avoids sequential deflation error accumulation and achieves the lowest reconstruction error $`E`$.

## 5. Results & Discussion

### 5.1 Section 1 — Deflationary Orthonormal ICA Results (`Section1.m`)

Because `B = randn(size(A))` is initialized randomly without a fixed seed, numerical values vary slightly across runs. For the representative run documented in the Persian report (Pages 3–5):

- **Estimated Composite Permutation Matrix (`permutation_matrix = B * (D^(-1/2) * U.') * A`):**

```math
\mathrm{permutation\_matrix}_{\mathrm{Deflation}} = \begin{bmatrix}
-0.0296 & 0.1216 & 0.9134 \\
-0.9901 & -0.0142 & -0.0159 \\
-0.0185 & -1.1489 & \boxed{0.6844}
\end{bmatrix}.
```

  - Row 1 cleanly isolates **Source 3** ($`0.9134`$), and Row 2 cleanly isolates **Source 1** ($`-0.9901`$).
  - In Row 3, although **Source 2** has the largest magnitude coefficient ($`-1.1489`$), the boxed entry $`(3, 3)`$ retains a substantial residual leakage of $`0.6844`$ from Source 3, causing the matrix to deviate from an ideal permutation matrix and visibly superimposing spiky artifacts from $`s_3(t)`$ onto the reconstructed sinusoid $`\hat{s}_1(t)`$ in Figure 1.5 (Page 4 of the report).
- **Relative Reconstruction Error (Part 1.2, Page 4):**

```math
E_{\mathrm{Deflation}} = \frac{\left\|\hat{S} - S\right\|_F^2}{\left\|S\right\|_F^2} = 0.17911.
```

- **Convergence Trajectory (Part 1.3, Figures 1.6 & 1.7, Page 5):** Plotting the combined gradient norm `score_func` ($`\sqrt{\sum_{m=1}^3 \|\partial f / \partial b_m\|_F^2}`$) over both $`500`$ iterations (Figure 1.6) and $`5000`$ iterations (Figure 1.7) shows that the curve drops sharply within the first $`\sim 25`$ iterations from $`2.28`$ to a constant floor of $`\approx 1.732`$ and does not decay to zero (the exact mathematical reason for this plateau in `Section1.m` is explained in Section 5.4 below).

### 5.2 Section 2 — Equivariant ICA Results (`Section2.m`)

For the representative run of the Equivariant ICA algorithm documented in the Persian report (Pages 6–8):

- **Estimated Composite Permutation Matrix (`permutation_matrix = B * A`):**

```math
\mathrm{permutation\_matrix}_{\mathrm{Equivariant}} = \begin{bmatrix}
0.0134 & -0.0667 & -0.4679 \\
-0.4374 & -0.0420 & 0.0426 \\
0.0127 & -0.3192 & \boxed{0.2220}
\end{bmatrix}.
```

  - Row 1 isolates **Source 3** ($`-0.4679`$), and Row 2 isolates **Source 1** ($`-0.4374`$) with very small cross-talk ($`\le 0.067`$).
  - Row 3 primarily extracts **Source 2** ($`-0.3192`$) with a smaller residual coupling of $`0.2220`$ in entry $`(3, 3)`$, yielding a much cleaner waveform reconstruction across all three channels in Figure 2.2 (Page 7 of the report).
- **Relative Reconstruction Error (Part 2.2, Page 7):**

```math
E_{\mathrm{Equivariant}} = \frac{\left\|\hat{S} - S\right\|_F^2}{\left\|S\right\|_F^2} = 0.08237,
```

which reduces the squared reconstruction error by more than **54%** compared to Deflation ICA ($`0.08237`$ vs. $`0.17911`$).
- **Convergence Trajectory (Part 2.3, Figures 2.3 & 2.4, Page 8):** Unlike Deflation ICA, the Frobenius norm of the relative gradient $`\|\frac{\partial f}{\partial B}\|_F = \|D B\|_F`$ in Equivariant ICA decreases smoothly from $`\approx 3.4`$ down to $`\mathbf{0}`$ within $`\sim 350\text{–}400`$ iterations and remains flat at zero through $`5000`$ iterations.

### 5.3 Section 3 — Summary Comparison Table Across All Three ICA Algorithms

The table below summarizes the quantitative and qualitative comparison reported in Section 3 of the Persian report (Pages 9–11):

| Evaluation Criterion | Method 1: Direct $`D_{\mathrm{KL}}`$ Minimization (HW8) | Method 2: Deflationary Orthonormal ICA (`Section1.m`) | Method 3: Equivariant ICA (`Section2.m`) |
| --- | --- | --- | --- |
| Pre-whitening required? | No (operates on $`X`$) | Yes ($`Z = D^{-1/2} U^T X`$) | No (operates directly on $`X`$) |
| Gradient update expression | $`\frac{1}{T}\hat{\Psi}_Y(Y) X^T - B^{-T}`$ | $`\frac{1}{T}\hat{\psi}_{Y_m}(y_m) Z^T`$ (row-wise) | $`D B`$ where $`D_{i,j} = \frac{1}{T}\hat{\psi}_{Y_i}(y_i) y_j^T`$ ($`i \neq j`$) |
| Constraint enforcement | Row $`\ell_2`$-normalization | Gram-Schmidt projection + unit norm | Off-diagonal $`D`$ + row $`\ell_2`$-normalization |
| Matrix inversion per step | Yes ($`B^{-T}`$ every iteration) | Only $`6 \times 6`$ polynomial MSE system | Only $`6 \times 6`$ polynomial MSE system |
| Reported Error $`E = \|\hat{S}-S\|_F^2 / \|S\|_F^2`$ | High (highest among the three) | $`0.17911`$ (moderate) | $`\mathbf{0.08237}`$ (**best separation quality**) |
| Execution Time (`tic`/`toc`, 500 iters) | $`1.089326\ \mathrm{s}`$ (slowest) | $`\mathbf{0.818347\ \mathrm{s}}`$ (**fastest runtime**) | $`0.918447\ \mathrm{s}`$ (intermediate) |
| Iterations to reach steady state | $`\sim 100\text{–}150`$ iterations (Figure 3.1) | $`\sim 25\text{–}30`$ iterations (Figure 3.2, plateaus at $`\approx 1.73`$) | $`\sim 350\text{–}400`$ iterations (Figure 3.3, converges to $`0`$) |

### 5.4 Notes on the Submitted Solutions / Implementation Notes

- **Why `score_func` in `Section1.m` Plateaus Around $`\sqrt{3} \approx 1.732`$ and Row 3 Shows Leakage (Lines 62–72):**
  1. **Fixed channel index `y = Y(1, :)` on line 63:** Inside the outer loop `for row = 1:size(B, 1)` and inner loop `for iter = 1:iterNum`, line 63 writes `y = Y(1, :);` instead of `y = Y(row, :);`. Consequently, for `row = 2` and `row = 3`, the score function $`\hat{\psi}_Y(y)`$ and gradient `df_B` are still computed using the first separated signal `Y(1, :)` rather than the active row `Y(row, :)`, which leaves the third row imperfectly separated after Gram-Schmidt orthogonalization against rows 1 and 2.
  2. **Unconstrained gradient norm vs. projected Riemannian gradient norm:** Even for `row = 1` (and for all rows when `y = Y(row, :)` is used), at a stationary point on the unit sphere $`\|b_m\|_2 = 1`$, the unconstrained gradient $`\frac{\partial f}{\partial b_m} = \mathbb{E}\{\psi_{Y_m}(y_m) z\}`$ is **collinear with $`b_m`$** rather than zero: by integration by parts, its projection onto $`b_m`$ is $`b_m^T \mathbb{E}\{\psi_{Y_m}(y_m) z\} = \mathbb{E}\{\psi_{Y_m}(y_m) y_m\} = 1`$. Therefore, $`\|\frac{\partial f}{\partial b_m}\|_2 \approx 1`$ for each of the $`M = 3`$ rows, so the combined Euclidean norm $`\sqrt{1^2 + 1^2 + 1^2} = \sqrt{3} \approx 1.732`$ naturally converges to $`\approx 1.732`$ rather than $`0`$ (whereas in `Section2.m`, subtracting `diag(diag(D))` removes the radial component $`\mathbb{E}\{\psi_{Y_i}(y_i) y_i\}`$ before computing `norm(df_B, "fro")`, so `score_func` in Section 2 converges to $`0`$).
- **Reconstruction Line `S_hat = B * X` vs. `S_hat = B * Z` in `Section1.m` (Line 74):** On line 74 of `Section1.m`, the unaligned separated signals are computed as `S_hat = B * X;` instead of applying the full whitened separating matrix $`B_{\mathrm{total}} = B D^{-1/2} U^T`$ (`S_hat = B * Z;`), whereas line 87 correctly includes the whitening factor when evaluating `permutation_matrix = B * (D^(-1/2) * U.') * A;`. Replacing `y = Y(1, :)` with `y = Y(row, :)` on line 63 and `S_hat = B * X` with `S_hat = B * Z` on line 74 reduces the Deflation ICA error $`E`$ from $`\sim 0.18`$ to $`\sim 0.011`$.
- **Source Norm Scaling Index in the Ambiguity Removal Loop (`Section1.m` line 83 & `Section2.m` line 40):** Inside the greedy correlation matching loop `for i = 1:size(B, 1)`, line 83 of `Section1.m` (and line 40 of `Section2.m`) rescales the matched row via `S_hat(ind, :) = S_hat(ind, :) * norm(S(i, :), "fro") ./ norm(S_hat(ind, :), "fro");` using `norm(S(i, :))` instead of `norm(S(ind, :))`. Because all three ground-truth sources in `hw9.mat` have identical unit norm ($`\|S(1, :)\|_2 = \|S(2, :)\|_2 = \|S(3, :)\|_2 = 1`$), `norm(S(i, :))` and `norm(S(ind, :))` are both identically equal to $`1`$, so this indexing difference has zero numerical impact.
- **Minor Typographical Slips in the Persian Report (Pages 1, 3, and 6):**
  - On **Page 1 (Introduction)**, the third sentence refers to removing permutation and scale ambiguities in "Part 3 (`قسمت 3`)" and plotting the convergence curve in "Part 3 (`قسمت 3`)"; the first reference means **Part 2 (`قسمت 2`)**.
  - On **Page 3 and Page 6**, the derivative of the MSE score-matching cost $`g(\theta) = \mathbb{E}\{\theta^T K(y) K(y)^T \theta\} - 2\mathbb{E}\{\theta^T K'(y)\}`$ is written as $`\frac{\partial g}{\partial \theta} = 2\mathbb{E}\{K(y) K^T(y)\} - 2\mathbb{E}\{K'(y)\} = 0`$ (omitting the vector $`\theta`$ after $`2\mathbb{E}\{K(y) K^T(y)\}`$ and the transpose on $`\hat{\theta}`$ in $`\hat{\psi}_Y(y) = \hat{\theta} K(y)`$), while the next step immediately writes the exact inverse formula $`\hat{\theta} = (\mathbb{E}\{K(y) K^T(y)\})^{-1} \mathbb{E}\{K'(y)\}`$ and `Psi_hat = Theta_hat.' * K_y` in code.

## 6. How to Use

### 6.1 Running the MATLAB Scripts

Open MATLAB, navigate to the `codes` directory, and run `Section1` and `Section2`:

```matlab
cd codes
Section1
Section2
```

- **Dependencies:** Both `Section1.m` and `Section2.m` use only base MATLAB functions (`eig`, `orth`, `randn`, `norm`, `vecnorm`, `mean`, `plot`, `subplot`). No external toolboxes are required.
- **Generated Figures:**
  - `Section1.m` opens **5 figures**:
    1. Three-panel subplot of the ground-truth source waveforms $`s_1(t)`$, $`s_2(t)`$, and $`s_3(t)`$ (Figure 1.1 in the report).
    2. Three-panel subplot of the noiseless mixtures $`X_{\mathrm{noNoise}} = A S`$ (Figure 1.2 in the report).
    3. Three-panel subplot of the noisy mixtures $`X = A S + \mathrm{Noise}`$ (Figure 1.3 in the report).
    4. Three-panel overlay plot comparing the Deflation ICA reconstructed sources $`\hat{s}_m(t)`$ (magenta) against the true sources $`s_m(t)`$ (black) (Figure 1.5 in the report).
    5. Convergence plot of the gradient norm `score_func` over $`500`$ iterations (Figure 1.6 in the report; changing `iterNum = 5e2` on line 58 to `5e3` reproduces the $`5000`$-iteration plot in Figure 1.7).
  - `Section2.m` opens **2 figures**:
    1. Three-panel overlay plot comparing the Equivariant ICA reconstructed sources $`\hat{s}_m(t)`$ (magenta) against the true sources $`s_m(t)`$ (black) (Figure 2.2 in the report).
    2. Convergence plot of the relative gradient norm `score_func` over $`500`$ iterations (Figure 2.3 in the report; changing `iterNum = 5e2` on line 18 to `5e3` reproduces Figure 2.4).

### 6.2 Expected Console Output (`Section1.m` and `Section2.m`)

Because both scripts initialize the separating matrix randomly via `B = randn(size(A))`, the exact value of `E` varies slightly across runs. For the representative executions documented in the Persian report (Pages 4 and 7), the console outputs are:

```text
Part 2:
Error (E) = 0.17911
```

for `Section1.m`, and

```text
Part 2:
Error (E) = 0.08237
```

for `Section2.m`.

### 6.3 Reading the Theoretical Derivations and Report

1. Use **Section 3** for the complete English translation of the assignment problems and **Section 4** for the step-by-step mathematical derivations of spatial whitening, polynomial MSE score function estimation, Deflationary Orthonormal ICA, and Equivariant ICA.
2. Use **Section 5** to review the numerical results, comparative benchmarks, and implementation notes alongside the original Persian report [BSS\_HW#9\_810100205\_Arman\_Majidi.pdf](./BSS_HW%239_810100205_Arman_Majidi.pdf) and assignment sheet [hw9.pdf](./hw9.pdf).

## 7. References

1. Akhavan, S. *Blind Source Separation: Homework 9*. Course assignment sheet and lecture notes, University of Tehran.
2. Cardoso, J.-F., & Laheld, B. H. (1996). Equivariant adaptive source separation. *IEEE Transactions on Signal Processing*, 44(12), 3017–3030.
3. Hyvärinen, A., Karhunen, J., & Oja, E. (2001). *Independent Component Analysis*. John Wiley & Sons.
4. Comon, P., & Jutten, C. (Eds.). (2010). *Handbook of Blind Source Separation: Independent Component Analysis and Applications*. Academic Press.
