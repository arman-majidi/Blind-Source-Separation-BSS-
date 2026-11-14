# Homework 8: Blind Source Separation via Mutual Information (KL Divergence) Minimization and Polynomial Score Function Estimation

**Course:** Blind Source Separation  
**Instructor:** Dr. Saeed Akhavan  
**Author:** Arman Majidi  
**Student ID:** 810100205  
**University:** University of Tehran  
**Semester:** Spring 2024-2025

## 1. Overview & Assignment Structure

This folder contains the **English problem statements, analytical derivations, MATLAB implementations, and simulation results for Homework 8**. The assignment investigates **Independent Component Analysis (ICA)** for noisy linear instantaneous mixtures by minimizing the **Kullback-Leibler (KL) divergence** (mutual information) between the separated outputs using **Mean Square Error (MSE) polynomial score function estimation**:

1. **Part 0 — Source and Observation Waveform Visualization:** Loads the $3 \times 3$ mixing matrix $A$, the $3 \times 1001$ source matrix $S$, and the $3 \times 1001$ additive noise matrix $\mathrm{Noise}$ from `hw8.mat`, constructs both the noiseless mixtures $`X_{\mathrm{noNoise}} = A S`$ and the noisy observations $X = A S + \mathrm{Noise}$ ($`\mathrm{SNR} \approx 22.11\ \mathrm{dB}`$), and visualizes the three time-domain source waveforms alongside their noiseless and noisy sensor mixtures.
2. **Part 1 — Mutual Information ($`D_{\mathrm{KL}}`$) Minimization via Linear MSE Score Function Estimation:** Formulates the ICA objective $`f(B) = -H(Y) + \sum_{m=1}^M H(y_m)`$ for the separating system $Y = B X$, derives its matrix gradient $`\frac{\partial f}{\partial B} = \mathbb{E}\!\left\{\psi_Y(y) x^T\right\} - B^{-T}`$ in terms of the marginal score functions $`\psi_{y_m}(y_m) = -p'_{y_m}(y_m)/p_{y_m}(y_m)`$, approximates each marginal score function linearly via $`\hat{\psi}_Y(y) = \theta^T K(y)`$ using the $5$-th degree polynomial kernel $K(y) = [1, y, y^2, y^3, y^4, y^5]^T$ under the integration-by-parts MSE identity $`\hat{\theta} = \left(\mathbb{E}\{K(y)K^T(y)\}\right)^{-1}\mathbb{E}\{K'(y)\}`$, optimizes $B$ via Steepest Descent with row-wise unit-norm projection, and evaluates the global system matrix $P = B A$ (`permutation_matrix`).
3. **Part 2 — Permutation, Sign, and Scale Disambiguation & Reconstruction Error Evaluation:** Resolves the inherent permutation, sign, and scaling ambiguities of BSS by matching each separated row of $\hat{S} = B X$ to the ground-truth source matrix $S$ via maximum absolute cross-correlation and energy normalization ($`\Vert{}\hat{s}_m\Vert{}_2 = \Vert{}s_m\Vert{}_2`$), overlays the recovered sources $`\hat{s}_m(t)`$ on the true sources $`s_m(t)`$, and computes the normalized Frobenius reconstruction error $`E = \Vert{}\hat{S} - S\Vert{}_F^2 / \Vert{}S\Vert{}_F^2`$.
4. **Part 3 — Convergence Analysis Across Iterations:** Tracks the Frobenius norm of the cost gradient $`\left\Vert{}\frac{\partial f}{\partial B}\right\Vert{}_F`$ (`score_func`) over $500$ and $5000$ Steepest Descent iterations to verify convergence toward a stationary point ($\frac{\partial f}{\partial B} \to 0$).

All parts of the assignment are mandatory; no optional or bonus problems were assigned. Because both the original assignment sheet and the submitted solution report are written in Persian, this `README.md` provides a **complete, self-contained English reference for every problem statement, mathematical derivation, algorithm, and numerical result**.

## 2. Directory Contents

| File | Description |
| --- | --- |
| [codes/Section1.m](./codes/Section1.m) | MATLAB script for Parts 0–3: waveform plotting of $S$, $`X_{\mathrm{noNoise}}`$, and $X$; polynomial kernel definitions `K_Y` and `dK_Y`; Steepest Descent $`D_{\mathrm{KL}}`$ minimization with MSE score function estimation; greedy correlation-based permutation/sign/energy alignment; evaluation of `permutation_matrix = B * A` and relative error $E$; and convergence plotting over `iterNum = 5000` iterations. |
| [codes/hw8.mat](./codes/hw8.mat) | Dataset containing the $3 \times 3$ unit-column-norm mixing matrix `A`, the $3 \times 1001$ unit-energy source matrix `S` (sinusoidal, triangular, and periodic pulse-train signals), and the $3 \times 1001$ additive zero-mean Gaussian noise matrix `Noise`. |
| [BSS\_HW#8\_810100205\_Arman\_Majidi.pdf](./BSS_HW%238_810100205_Arman_Majidi.pdf) | Submitted report and analytical solutions in Persian (including source and observation plots, MSE score function and $`D_{\mathrm{KL}}`$ gradient derivations, BSS block diagram, reported `permutation_matrix`, overlaid source reconstructions, and $500$- and $5000$-iteration convergence curves). |
| [hw8.pdf](./hw8.pdf) | Original assignment problem sheet in Persian. |
| `README.md` | Self-contained English problem statements, mathematical derivations, algorithm descriptions, numerical results, and usage guide. |

Keep the `codes` subfolder inside the homework directory.

## 3. Assignment Problem Statement

### 3.1 Problem Description and Dataset (`hw8.mat`)

In this assignment, we solve the **Blind Source Separation (BSS)** problem under the assumption that the latent source signals are **statistically independent**.

The mixing matrix `A`, the source matrix `S`, and the additive noise matrix `Noise` are provided in the file `hw8.mat` with dimensions $M = 3$ channels and $T = 1001$ time samples:

- **Mixing Matrix (`A` $\in \mathbb{R}^{3 \times 3}$):** Each column has unit Euclidean norm ($`\Vert{}a_j\Vert{}_2 = 1`$):

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
\end{bmatrix}.
```

- **Source Matrix (`S` $\in \mathbb{R}^{3 \times 1001}$):** Three zero-mean, unit-energy ($`\Vert{}s_m\Vert{}_2 = 1`$, $`\Vert{}S\Vert{}_F^2 = 3`$) deterministic/periodic waveforms:
  - $`s_1(t)`$: Sinusoidal wave ($\approx 3.5$ periods across $T = 1001$ samples, amplitude range $[-0.0447, 0.0446]$).
  - $`s_2(t)`$: Symmetric triangular wave ($10$ periods across $T = 1001$ samples, amplitude range $[-0.0546, 0.0537]$).
  - $`s_3(t)`$: Periodic sharp pulse train ($9$ positive pulses across $T = 1001$ samples, amplitude range $[-0.0230, 0.0987]$).
- **Additive Noise (`Noise` $\in \mathbb{R}^{3 \times 1001}$):** Zero-mean white Gaussian sensor noise with per-channel standard deviation $\sigma \approx 0.0025$, corresponding to an overall input Signal-to-Noise Ratio of

```math
\mathrm{SNR} = 10 \log_{10} \frac{\|A S\|_F^2}{\|\mathrm{Noise}\|_F^2} \approx 22.11\ \mathrm{dB}.
```

First, construct the noisy observation matrix $X$ according to the linear instantaneous mixing model:

```math
X = A S + \mathrm{Noise},
```

and plot the original sources $S$, the noiseless observations $`X_{\mathrm{noNoise}} = A S`$, and the noisy observations $X$ to inspect their time-domain waveforms.

Next, approach the problem from the perspective of **Blind Source Separation**: assume that only the noisy observation matrix $X \in \mathbb{R}^{3 \times 1001}$ is available and that the number of sources ($M = 3$) is known. Our strategy is to find a $3 \times 3$ separating matrix $B$ such that the separated outputs

```math
Y = B X
```

are as statistically independent of one another as possible. Implement the first class method based on **minimizing the Kullback-Leibler divergence ($`D_{\mathrm{KL}}`$)**. To estimate the marginal **score function** of each output channel, use the **linear Mean Square Error (MSE)** method with the $5$-th degree polynomial basis (kernel) vector:

```math
k(y) = \begin{bmatrix} 1 & y & y^2 & y^3 & y^4 & y^5 \end{bmatrix}^T.
```

### 3.2 Required Deliverables (Parts 1–3)

1. **Part 1 — Global System / Permutation Matrix ($B A$):** Multiply the final estimated separating matrix $B$ by the true mixing matrix $A$ and report the product $B A$. The resulting matrix should be close to a generalized **permutation matrix**, meaning that each row and each column should contain only one dominant non-zero entry.
2. **Part 2 — Ambiguity Resolution, Waveform Comparison, and Relative Error ($E$):** Resolve both the **permutation (ordering) ambiguity** and the **scaling (and sign) ambiguity** of the estimated sources so that the energy of each estimated source equals the energy of the corresponding true source. Plot the estimated sources $\hat{S}$ overlaid on the true sources $S$, and report the normalized reconstruction error:

```math
E = \frac{\|\hat{S} - S\|_F^2}{\|S\|_F^2}.
```

3. **Part 3 — Convergence Curve:** Plot the convergence curve of the optimization algorithm versus the iteration index.

## 4. Mathematical Formulation & Solutions

### 4.1 Part 0: Signal Mixing Model and Time-Domain Visualization (`Section1.m`)

Given the $3 \times 3$ mixing matrix $A$, the $3 \times 1001$ source matrix $S$, and the $3 \times 1001$ additive noise matrix $\mathrm{Noise}$, `codes/Section1.m` (lines 8–9) forms both the ideal noiseless mixtures and the observed noisy mixtures:

```math
\begin{aligned}
X_{\mathrm{noNoise}} &= A S \in \mathbb{R}^{3 \times 1001}, \\
X &= A S + \mathrm{Noise} \in \mathbb{R}^{3 \times 1001}.
\end{aligned}
```

As shown in Figures 1.1–1.3 of the report (Pages 1–2), each sensor observation $`x_m(t)`$ is a superposition of the sinusoid $`s_1(t)`$, the triangular wave $`s_2(t)`$, and the sharp pulse train $`s_3(t)`$, with small high-frequency ripples superimposed in the noisy observations $X(t)$.

### 4.2 Part 1: Mutual Information ($`D_{\mathrm{KL}}`$) Minimization and Polynomial Score Function Estimation

#### Mutual Information Objective Function and Matrix Gradient

Let $y = B x \in \mathbb{R}^M$ ($M = 3$) denote the output vector of the linear separating system $Y = B X$ (Figure 1.4 on Page 3 of the report). The statistical dependence among the components $`y_1, \dots, y_M`$ is measured by their **mutual information**, defined as the Kullback-Leibler divergence between the joint probability density function $`p_Y(y)`$ and the product of the marginal densities $`\prod_{m=1}^M p_{y_m}(y_m)`$:

```math
I(Y) = D_{\mathrm{KL}}\!\left( p_Y(y) \,\middle\|\, \prod_{m=1}^M p_{y_m}(y_m) \right)
= \int p_Y(y) \log \frac{p_Y(y)}{\prod_{m=1}^M p_{y_m}(y_m)} \, dy
= -H(y) + \sum_{m=1}^M H(y_m),
```

where $`H(y) = -\mathbb{E}\{\log p_Y(y)\}`$ is the joint differential entropy of $y$ and $`H(y_m) = -\mathbb{E}\{\log p_{y_m}(y_m)\}`$ is the marginal differential entropy of the $m$-th output channel. Therefore, the cost function to be minimized with respect to $B \in \mathbb{R}^{M \times M}$ is (Page 3 of the report):

```math
f(B) = -H(y) + \sum_{m=1}^M H(y_m).
```

Under the invertible linear transformation $y = B x$, the joint PDF satisfies $`p_Y(y) = \frac{p_X(x)}{\vert{}\det(B)\vert{}}`$, which yields the entropy transformation relation

```math
H(y) = H(x) + \log|\det(B)|.
```

Substituting $H(y)$ into $f(B)$ gives

```math
f(B) = -\log|\det(B)| - H(x) + \sum_{m=1}^M H(y_m).
```

Because the observation entropy $H(x)$ does not depend on the separating matrix $B$, differentiating $f(B)$ with respect to $B$ uses:

1. The matrix determinant derivative identity:

```math
\frac{\partial}{\partial B} \log|\det(B)| = \left(B^{-1}\right)^T = B^{-T}.
```

2. The marginal entropy derivative identity: defining the marginal **score function** $`\psi_{y_m}(y_m)`$ of the $m$-th output channel as the negative logarithmic derivative of its marginal PDF,

```math
\psi_{y_m}(y_m) = -\frac{d}{dy_m} \log p_{y_m}(y_m) = -\frac{p'_{y_m}(y_m)}{p_{y_m}(y_m)},
\qquad
\psi_Y(y) = \begin{bmatrix} \psi_{y_1}(y_1) \\ \vdots \\ \psi_{y_M}(y_M) \end{bmatrix} \in \mathbb{R}^{M \times 1},
```

differentiating $`\sum_{m=1}^M H(y_m) = -\sum_{m=1}^M \mathbb{E}\!\left\{\log p_{y_m}(b_m^T x)\right\}`$ with respect to the rows $`b_m^T`$ of $B$ yields

```math
\frac{\partial}{\partial B} \sum_{m=1}^M H(y_m) = \mathbb{E}\!\left\{ \psi_Y(y) x^T \right\}.
```

Combining both terms produces the exact matrix gradient of $f(B)$ (Page 3 of the report):

```math
\boxed{
\frac{\partial f}{\partial B} = \mathbb{E}\!\left\{ \psi_Y(y) x^T \right\} - B^{-T}.
}
```

#### Linear MSE Score Function Estimation via Integration by Parts

Because the marginal densities $`p_{y_m}(y_m)`$ are unknown during blind separation, each scalar score function $`\psi_Y(y)`$ (dropping the channel subscript $m$ for clarity) is approximated as a linear combination of the $5$-th degree polynomial basis functions:

```math
\hat{\psi}_Y(y) = \theta^T K(y) = K^T(y) \theta,
\qquad
K(y) = \begin{bmatrix} 1 \\ y \\ y^2 \\ y^3 \\ y^4 \\ y^5 \end{bmatrix} \in \mathbb{R}^{6 \times 1},
\qquad
K'(y) = \frac{dK(y)}{dy} = \begin{bmatrix} 0 \\ 1 \\ 2y \\ 3y^2 \\ 4y^3 \\ 5y^4 \end{bmatrix} \in \mathbb{R}^{6 \times 1}.
```

To find the optimal parameter vector $\theta \in \mathbb{R}^{6 \times 1}$ without knowing $`p_Y(y)`$ explicitly, we minimize the Mean Square Error (MSE):

```math
\begin{aligned}
\mathcal{J}_{\mathrm{MSE}}(\theta)
&= \mathbb{E}\!\left\{ \left( \psi_Y(y) - \theta^T K(y) \right)^2 \right\} \\
&= \mathbb{E}\!\left\{\psi_Y^2(y)\right\} + \mathbb{E}\!\left\{ \theta^T K(y) K^T(y) \theta \right\} - 2\mathbb{E}\!\left\{ \theta^T K(y) \psi_Y(y) \right\}.
\end{aligned}
```

Using the definition $`\psi_Y(y) = -p'_Y(y)/p_Y(y)`$ and applying integration by parts under the standard boundary condition $`\lim_{y \to \pm\infty} p_Y(y) K(y) = 0`$:

```math
\mathbb{E}\!\left\{ K(y) \psi_Y(y) \right\}
= \int_{-\infty}^{\infty} K(y) \left( -\frac{p'_Y(y)}{p_Y(y)} \right) p_Y(y) \, dy
= -\int_{-\infty}^{\infty} K(y) p'_Y(y) \, dy
= -\Big[ K(y) p_Y(y) \Big]_{-\infty}^{\infty} + \int_{-\infty}^{\infty} K'(y) p_Y(y) \, dy
= \mathbb{E}\!\left\{ K'(y) \right\}.
```

Dropping the $\theta$-independent constant $`\mathbb{E}\{\psi_Y^2(y)\}`$ yields the tractable quadratic objective $g(\theta)$ written on Page 3 of the report:

```math
g(\theta) = \mathbb{E}\!\left\{ \theta^T K(y) K^T(y) \theta \right\} - 2\mathbb{E}\!\left\{ \theta^T K'(y) \right\}.
```

Setting the gradient $\frac{\partial g}{\partial \theta}$ to zero gives the closed-form normal equations for $\hat{\theta}$:

```math
\frac{\partial g}{\partial \theta} = 2\mathbb{E}\!\left\{ K(y) K^T(y) \right\} \theta - 2\mathbb{E}\!\left\{ K'(y) \right\} = 0
\quad \Longrightarrow \quad
\boxed{
\hat{\theta} = \left( \mathbb{E}\!\left\{ K(y) K^T(y) \right\} \right)^{-1} \mathbb{E}\!\left\{ K'(y) \right\},
\qquad
\hat{\psi}_Y(y) = \hat{\theta}^T K(y).
}
```

#### Empirical Sample Implementation and Steepest Descent Loop (`Section1.m`)

In `codes/Section1.m` (lines 60–78), for a batch of $T = 1001$ samples, the polynomial basis matrix $`K_y \in \mathbb{R}^{6 \times T}`$ and derivative matrix $`dK_y \in \mathbb{R}^{6 \times T}`$ are evaluated for each row $Y(i, :)$ ($i = 1, 2, 3$) via anonymous functions:

```matlab
K_Y  = @(Y) [ones(1,T); Y; Y.^2; Y.^3; Y.^4; Y.^5];
dK_Y = @(Y) [zeros(1,T); ones(1,T); 2*Y; 3*Y.^2; 4*Y.^3; 5*Y.^4];
```

At each iteration $k = 1, \dots, \mathrm{iterNum}$ ($\mathrm{iterNum} = 5000$, step size $\mu = 0.005$):

1. **Compute separated outputs:** $Y = B^{(k)} X \in \mathbb{R}^{3 \times T}$.
2. **Estimate marginal score functions row by row ($i = 1, 2, 3$):**

```math
\hat{\theta}_i = \left( \frac{1}{T} K_y K_y^T \right)^{-1} \left( \frac{1}{T} \sum_{t=1}^T K'\big(y_i(t)\big) \right),
\qquad
\hat{\Psi}(i, :) = \hat{\theta}_i^T K_y \in \mathbb{R}^{1 \times T},
```

implemented in MATLAB (lines 72–74) as:

```matlab
K_y = K_Y(Y(i, :));
Theta_hat = (K_y * K_y.' / T) \ mean(dK_Y(Y(i,:)), 2);
Psi_hat(i, :) = Theta_hat.' * K_y;
```

3. **Evaluate sample gradient and update $B$ via Steepest Descent:**

```math
\frac{\partial f}{\partial B} = \frac{1}{T} \hat{\Psi} X^T - \left(B^{(k)}\right)^{-T},
\qquad
\tilde{B}^{(k+1)} = B^{(k)} - \mu \frac{\partial f}{\partial B}.
```

4. **Normalize rows of $B$ to prevent scale drift:** Because $`D_{\mathrm{KL}}`$ is scale-invariant, each row of $\tilde{B}^{(k+1)}$ is projected onto the unit $`\ell_2`$-sphere (`B = B ./ vecnorm(B, 2, 2)`):

```math
B^{(k+1)}(i, :) = \frac{\tilde{B}^{(k+1)}(i, :)}{\left\|\tilde{B}^{(k+1)}(i, :)\right\|_2},
\qquad i = 1, 2, 3.
```

After termination, the global transfer matrix `permutation_matrix = B * A` is computed (line 93). For the representative execution documented on Page 3 of the report:

```math
\mathrm{permutation\_matrix} = B A = \begin{bmatrix}
-0.4374 & -0.0420 & 0.0426 \\
0.0127 & -0.3192 & 0.2220 \\
0.0134 & -0.0667 & -0.4679
\end{bmatrix}.
```

### 4.3 Part 2: Resolving Permutation, Sign, and Scaling Ambiguities & Error Metric

In Blind Source Separation, even when $B$ perfectly inverts $A$ up to scaling and permutation ($B A = P \Lambda$, where $P$ is a permutation matrix and $\Lambda$ is a non-singular diagonal scaling/sign matrix), the raw output $Y = B X$ differs from $S$ in channel ordering, polarity ($\pm 1$), and amplitude scale.

To compare $\hat{S}$ directly against the ground-truth source matrix $S$, `codes/Section1.m` (lines 79–91) performs a greedy correlation-matching and energy-normalization procedure:

1. **Initial Frobenius normalization:** Normalize $\hat{S} = B X$ by its Frobenius norm (`S_hat = S_hat ./ norm(S_hat, "fro")`) and store a copy `S_hatt = S_hat` and `S_temp = S`.
2. **Greedy channel matching and sign alignment:** For each separated channel $i = 1, 2, 3$, compute the cross-correlation vector with the remaining unmatched rows of $`S_{\mathrm{temp}}`$:

```math
\mathrm{corr} = \hat{s}_i^{\mathrm{raw}} S_{\mathrm{temp}}^T \in \mathbb{R}^{1 \times 3},
\qquad
\mathrm{ind} = \arg\max_{j \in \{1, 2, 3\}} \left|\mathrm{corr}(j)\right|.
```

Zero out the matched row `S_temp(ind, :) = 0` so each ground-truth source is matched exactly once, and align the sign and energy of row `ind`:

```math
\hat{s}_{\mathrm{ind}} \leftarrow \mathrm{sign}\!\big(\mathrm{corr}(\mathrm{ind})\big) \, \hat{s}_i^{\mathrm{raw}},
\qquad
\hat{s}_{\mathrm{ind}} \leftarrow \hat{s}_{\mathrm{ind}} \frac{\|s_{\mathrm{ind}}\|_2}{\|\hat{s}_{\mathrm{ind}}\|_2}.
```

3. **Normalized Frobenius reconstruction error:** Compute the relative squared Frobenius error (line 91, Page 4 of the report):

```math
E = \frac{\|\hat{S} - S\|_F^2}{\|S\|_F^2}.
```

For the single-run realization documented on Page 4 of the Persian report, this error evaluates to:

```math
\boxed{
E = 0.22726.
}
```

### 4.4 Part 3: Convergence Tracking via Gradient Norm (`score_func`)

Although the theoretical cost function is the mutual information $`f(B) = -H(y) + \sum_{m=1}^M H(y_m)`$, evaluating differential entropies $`H(y_m)`$ directly requires density integration at each step. Instead, as explained in Part 3 of the report (Page 4), `codes/Section1.m` (line 90) monitors the **Frobenius norm of the objective gradient** at each iteration $k$:

```math
\mathrm{score\_func}(k) = \left\| \frac{\partial f}{\partial B}\!\left(B^{(k)}\right) \right\|_F
= \left\| \frac{1}{T} \hat{\Psi}^{(k)} X^T - \left(B^{(k)}\right)^{-T} \right\|_F.
```

As $B^{(k)}$ approaches a stationary point of $f(B)$, the gradient $\frac{\partial f}{\partial B}$ vanishes ($`\mathrm{score\_func}(k) \to 0`$). In the report (Page 5), this convergence curve is examined first over $500$ iterations and then over $5000$ iterations ($10\times$ longer), showing a rapid initial drop from $\sim 15\text{–}45$ down toward zero within the first $150\text{–}500$ iterations followed by a near-zero plateau.

## 5. Results & Discussion

### 5.1 Summary of Numerical Results and Visual Observations

Because `B = randn(size(A))` (line 61 of `Section1.m`) initializes the separating matrix randomly without a fixed random seed (`rng`), the exact trajectory and final local/stationary solution vary slightly across executions. The table below summarizes the dataset parameters, the representative single-run results documented in the Persian report, and the behavior across multi-seed executions:

| Problem Part & Quantity | Mathematical Expression | MATLAB Expression (`Section1.m`) | Reported Result & Observation |
| --- | --- | --- | --- |
| Part 0 — Dataset Dimensions & SNR | $X = A S + \mathrm{Noise}$ | `X_noNoise = A*S; X = A*S + Noise;` | $M = 3$ channels, $T = 1001$ samples, $`\Vert{}s_m\Vert{}_2 = 1`$, input $`\mathrm{SNR} = 22.11\ \mathrm{dB}`$ |
| Part 1 — Global System Matrix ($B A$) | $P = B A$ | `permutation_matrix = B * A` | $`\begin{bmatrix} -0.4374 & -0.0420 & 0.0426 \\ 0.0127 & -0.3192 & 0.2220 \\ 0.0134 & -0.0667 & -0.4679 \end{bmatrix}`$ |
| Part 1 — Off-Diagonal Cross-Talk Analysis | Ideal: $1$ non-zero entry per row/column | Highlighted entry `permutation_matrix(2, 3)` | Rows 1 and 3 isolate sources 1 and 3 cleanly ($-0.4374$ and $-0.4679$), whereas entry $(2,3) = \mathbf{0.2220}$ introduces residual leakage from source 3 into channel 2 |
| Part 2 — Waveform Recovery (Figure 1.5) | $\hat{S}$ vs. $S$ after sign/scale alignment | `plot(1:T, S_hat(i,:), "m", 1:T, S(i,:), "k")` | $`\hat{s}_1(t)`$ (sinusoid) and $`\hat{s}_2(t)`$ (triangle wave) closely track $`s_1(t)`$ and $`s_2(t)`$, while $`\hat{s}_3(t)`$ (pulse train) shows reduced peak amplitude in the reported run due to the $(2,3)$ leakage |
| Part 2 — Relative Frobenius Error $E$ | $`E = \frac{\Vert{}\hat{S} - S\Vert{}_F^2}{\Vert{}S\Vert{}_F^2}`$ | `E = norm(S_hat - S, "fro")^2 / norm(S, "fro")^2` | $\mathbf{E = 0.22726}$ in the reported run (across well-separated random initializations with `iterNum = 5000`, $E$ typically reaches $\mathbf{0.014\text{–}0.025}$) |
| Part 3 — Convergence Metric (`score_func`) | $`\left\Vert{}\frac{\partial f}{\partial B}\right\Vert{}_F`$ vs. iteration $k$ | `score_func(iter) = norm(df_B, "fro")` | Rapidly decays from initial values ($\sim 15\text{–}45$) to $< 0.5$ within $\sim 150$ iterations and approaches $0$ over $500$ and $5000$ iterations |

### 5.2 Notes on the Submitted Solutions / Implementation Notes

- **Sensitivity to Random Initialization $B^{(0)}$ and Ordinary vs. Natural Gradient:**
  - In `Section1.m` (line 61), $B^{(0)}$ is drawn from `randn(3, 3)` without spatial pre-whitening of $X$, and line 77 updates $B$ using standard Euclidean Steepest Descent (`B = B - mu * df_B`) followed by row-wise $`\ell_2`$ normalization (`B = B ./ vecnorm(B, 2, 2)`).
  - Because standard gradient descent involves $\left(B^{-1}\right)^T$ and does not post-multiply by $B^T B$ (Amari's natural gradient), when a random initialization $B^{(0)}$ passes near an ill-conditioned matrix during early iterations, the trajectory can experience a transient spike in $`\left\Vert{}B^{-T}\right\Vert{}_F`$ (visible as the small bump around iteration $4500$ in the $5000$-iteration plot on Page 5 of the report) or settle with moderate cross-talk on the third source ($0.2220$ in row 2, column 3 of `permutation_matrix`, giving $E = 0.22726$). For most random seeds over $5000$ iterations, all three sources—including the pulse train $`s_3(t)`$—are separated cleanly with $E \approx 0.014\text{–}0.025$.
- **Row-Energy Normalization Index in `Section1.m` (line 88):**
  - Inside the permutation-matching loop (`for i = 1:size(Y, 1)`), line 88 scales the matched source row `ind` using:

```matlab
S_hat(ind, :) = S_hat(ind, :) * norm(S(i, :), "fro") ./ norm(S_hat(ind, :), "fro");
```

  - Conceptually, the target norm for row `ind` is `norm(S(ind, :), "fro")` rather than `norm(S(i, :), "fro")`. However, in `hw8.mat`, every row of the ground-truth source matrix `S` is pre-normalized to exact unit energy ($`\Vert{}S(1, :)\Vert{}_2 = \Vert{}S(2, :)\Vert{}_2 = \Vert{}S(3, :)\Vert{}_2 = 1.000000`$), so `norm(S(i, :), "fro")` and `norm(S(ind, :), "fro")` are both identically equal to $1$, and the numerical result is completely unaffected.
- **Placement of Ambiguity Resolution Inside the Iteration Loop (`Section1.m` lines 79–91):**
  - Lines 79–91 compute `S_hat` and the reconstruction error `E` inside the `for iter = 1:iterNum` loop rather than after the loop finishes. Since `S_hat` and `E` are not fed back into the update of `B`, this does not alter the algorithm's output, though moving lines 79–91 outside the loop reduces execution time.
- **Minor Typographical Slips in the Persian Report (Pages 1, 3, 4, and 5):**
  - On **Page 1 (Introduction, line 3)**, the sentence states that in "Part 3 (`قسمت ۳`)" source and permutation ambiguities are removed and the error is computed, and in the next line again refers to "Part 3 (`قسمت ۳`)" for the convergence plot; the first reference refers to **Part 2 (`قسمت ۲`)**.
  - On **Page 3 (bottom paragraph)**, the variable name `permutation_matrix` is typed once without the `t` as `permutation_marix`.
  - On **Pages 4 and 5 (Part 3 figure numbering)**, the text on Page 4 refers to the $500$-iteration convergence plot as `تصویر ۱.۶` (Figure 1.6) and the text on Page 5 refers to the $5000$-iteration convergence plot as `تصویر ۱.۷` (Figure 1.7), whereas both figure captions beneath the plots on Page 5 are labeled `تصویر ۱.۵` (repeating the figure number of the source comparison plot on Page 4).

## 6. How to Use

### 6.1 Running the MATLAB Script

Open MATLAB, navigate to the `codes` directory, and run `Section1`:

```matlab
cd codes
Section1
```

- **Dependencies:** `Section1.m` uses only base MATLAB functions (`load`, `randn`, `vecnorm`, `inv`, `norm`, `mean`, `sign`, `max`, `subplot`, `plot`). Note that `vecnorm` requires MATLAB R2017b or later, and `Interpreter="latex"` name-value syntax requires MATLAB R2021a or later.
- **Generated Figures:** Running `Section1.m` opens **5 figures**:
  1. **Figure 1 (`S(1,:)` to `S(3,:)`):** 3-row subplot of the original source signals $`s_1(t)`$, $`s_2(t)`$, and $`s_3(t)`$ (Figure 1.1 in the report).
  2. **Figure 2 (`X(noNoise)(1,:)` to `X(noNoise)(3,:)`):** 3-row subplot of the noiseless mixtures $`X_{\mathrm{noNoise}} = A S`$ (Figure 1.2 in the report).
  3. **Figure 3 (`X(Noisy)(1,:)` to `X(Noisy)(3,:)`):** 3-row subplot of the noisy observations $X = A S + \mathrm{Noise}$ (Figure 1.3 in the report).
  4. **Figure 4 (`S_hat` vs. `S`):** 3-row subplot overlaying the separated, permutation- and scale-aligned sources $`\hat{s}_m(t)`$ (magenta) onto the ground-truth sources $`s_m(t)`$ (black) (Figure 1.5 in the report).
  5. **Figure 5 (`Convergence Plot`):** Plot of `score_func` ($`\Vert{}\partial f / \partial B\Vert{}_F`$) versus iteration index from $1$ to `iterNum = 5000` (Figures 1.6/1.7 on Page 5 of the report).

### 6.2 Expected Console Output (`Section1.m`)

Because `B = randn(size(A))` is randomly initialized on each run, the exact final error `E` printed to the MATLAB Command Window varies slightly across runs. For the representative run reported in `BSS_HW#8_810100205_Arman_Majidi.pdf`, the console output is:

```text
Part 2:
Error (E) = 0.22726
```

*(On typical random initializations where all three sources converge without cross-talk over $5000$ iterations, the console prints `Error (E) = 0.015` to `0.025`.)* To inspect the global system matrix after running `Section1.m`, type `permutation_matrix` in the Command Window:

```matlab
disp(permutation_matrix)
```

### 6.3 Reading the Theoretical Derivations and Report

1. Use **Section 3** for the complete English translation of the assignment sheet and dataset specifications, and **Section 4** for the step-by-step derivations of the $`D_{\mathrm{KL}}`$ mutual information gradient and the integration-by-parts linear MSE score function estimator.
2. Use **Section 5** to review the numerical results, permutation matrix analysis, and implementation notes alongside the original Persian report [BSS\_HW#8\_810100205\_Arman\_Majidi.pdf](./BSS_HW%238_810100205_Arman_Majidi.pdf) and assignment sheet [hw8.pdf](./hw8.pdf).

## 7. References

1. Akhavan, S. *Blind Source Separation: Homework 8*. Course assignment sheet and lecture notes, University of Tehran, Spring 2024-2025.
2. Babaie-Zadeh, M., Jutten, C., & Nayebi, K. (2002). Differential-based blind source separation. *Proceedings of the 4th International Symposium on Independent Component Analysis and Blind Signal Separation (ICA 2003)*, Nara, Japan.
3. Talebi, S. P., & Babaie-Zadeh, M. (2010). Score function estimation for blind source separation based on minimum mean square error. *Handbook of Blind Source Separation: Independent Component Analysis and Applications*, Comon, P., & Jutten, C. (Eds.), Academic Press.
4. Hyvärinen, A., Karhunen, J., & Oja, E. (2001). *Independent Component Analysis*. John Wiley & Sons.
