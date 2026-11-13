# Homework 6: Sparse Signal Recovery Under Noise (BP, MP, and LASSO), Low-Coherence Frame Design, and Dictionary Learning via MOD & K-SVD

**Course:** Blind Source Separation

**Instructor:** Dr. Saeed Akhavan

**Author:** Arman Majidi

**Student ID:** 810100205

**University:** University of Tehran

**Semester:** Spring 2024-2025

## 1. Overview & Assignment Structure

This folder contains the **English problem statements, analytical derivations, MATLAB implementations, and simulation results for Homework 6**. The assignment is organized into three main sections covering **noisy sparse signal recovery**, **low-coherence frame (overcomplete dictionary) design**, and **blind dictionary learning (sparse blind source separation)**:

1. **Section 1 — Sparse Signal Recovery in Noiseless and Noisy Regimes via Basis Pursuit (BP), Matching Pursuit (MP), and LASSO (Parts الف–د / Parts 1–4):** Evaluates sparse recovery on an underdetermined system with dictionary $D \in \mathbb{R}^{5 \times 15}$ ($M = 5$, $N = 15$ unit-norm atoms), noiseless observation $x \in \mathbb{R}^{5 \times 1}$, and noisy observation $x_{\mathrm{noisy}} \in \mathbb{R}^{5 \times 1}$ (`hw6-part1.mat`). Formulates and solves **Basis Pursuit (BP)** via Linear Programming (`linprog`) and **Matching Pursuit (MP)** in the noiseless regime, demonstrates how additive noise degrades exact equality-constrained BP into a non-sparse $5$-atom fit, and formulates the unconstrained **LASSO** problem solved via iterative **Coordinate Descent (Soft-Thresholding)** across varying regularization parameters $\lambda$ to identify the optimal sparsity-error trade-off range.

2. **Section 2 — Overcomplete Frame Construction and Mutual Coherence Minimization in 2D and 3D Spaces (Parts 1–2):** Designs an analytical equiangular frame of size $2 \times N$ in $\mathbb{R}^2$ by uniformly distributing $N$ unit vectors across the upper half-plane $[0, \pi)$ to avoid antipodal collinearity for even $N$, plotting the resulting mutual coherence $\mu(D) = \cos(\pi/N)$ for $N \in \{2, \dots, 35\}$. Proposes and implements an iterative stochastic single-atom replacement algorithm on the unit sphere in $\mathbb{R}^3$ ($50\,000$ iterations) to construct a $3 \times 10$ frame with minimized mutual coherence $\mu(D) \approx 0.6947$.

3. **Section 3 — Blind Dictionary Learning and Sparse Source Separation via MOD and K-SVD (Parts 1–6):** Computes the mutual coherence of a ground-truth $10 \times 40$ dictionary (`hw6-part3.mat`, $\mu(D) = 0.8680$), and implements both the **Method of Optimal Directions (MOD)** and **K-SVD** over $50$ alternating minimization iterations using **Orthogonal Matching Pursuit (OMP)** (`OMP.m`, with known sparsity level $N_0 = 2$) for the sparse coding stage on $T = 1500$ observation vectors $X \in \mathbb{R}^{10 \times 1500}$. Compares MOD and K-SVD in terms of representation error convergence trajectories, execution time (`tic`/`toc`), dictionary atom **Successful Recovery Rate (SRR)** under the $\vert{}d_i^T \hat{d}_j\vert{} > 0.98$ threshold, and normalized Frobenius source estimation error $E = \Vert{}\hat{S} - S\Vert{}_F^2 / \Vert{}S\Vert{}_F^2$ after resolving permutation and scaling/sign ambiguities.

All parts of all three sections are mandatory; no optional or bonus problems were assigned. Because both the original assignment sheet and the submitted solution report are written in Persian, this `README.md` provides a **complete, self-contained English reference for every problem statement, mathematical derivation, algorithm, and numerical result**.

## 2. Directory Contents

| File | Description | 
| ----- | ----- | 
| [codes/Section1.m](./codes/Section1.m) | MATLAB script for Section 1 (Parts 1–4 / الف–د): noiseless Basis Pursuit (`linprog`), noiseless Matching Pursuit (MP), noisy Basis Pursuit, and Coordinate-Descent LASSO (soft-thresholding) on `hw6-part1.mat`. | 
| [codes/Section2.m](./codes/Section2.m) | MATLAB script for Section 2 (Parts 1–2): analytical $2 \times N$ half-plane equiangular frame construction ($N = 2, \dots, 35$) and iterative $3 \times 10$ stochastic frame optimization ($50\,000$ iterations) minimizing mutual coherence $\mu(D)$. | 
| [codes/Section3.m](./codes/Section3.m) | MATLAB script for Section 3 (Parts 1–6): ground-truth dictionary mutual coherence calculation, 50-iteration **MOD** and **K-SVD** dictionary learning with `OMP`, representation error convergence plotting, Successful Recovery Rate (SRR) evaluation, and normalized source reconstruction error $E$. | 
| [codes/OMP.m](./codes/OMP.m) | Helper MATLAB function `s_hat = OMP(x, D, N, N0)` implementing $N_0$-step Orthogonal Matching Pursuit for the sparse coding stage of MOD and K-SVD in Section 3. | 
| [codes/hw6-part1.mat](./codes/hw6-part1.mat) | Dataset for Section 1 containing the dictionary matrix `D` ($5 \times 15$ double, unit-norm columns), noiseless observation vector `x` ($5 \times 1$ double), and noisy observation vector `x_noisy` ($5 \times 1$ double). | 
| [codes/hw6-part3.mat](./codes/hw6-part3.mat) | Dataset for Section 3 containing the ground-truth dictionary `D` ($10 \times 40$ double, unit-norm columns), sparse source matrix `S` ($40 \times 1500$ double), noiseless observation matrix `X` ($10 \times 1500$ double), and sparsity level `N0` ($= 2$, `uint8`). | 
| [BSS_HW#6_810100205_Arman_Majidi.pdf](./BSS_HW%236_810100205_Arman_Majidi.pdf) | Submitted report and analytical solutions in Persian (including BP/MP/LASSO recovered vectors, soft-thresholding diagram, 2D and 3D mutual coherence curves, MOD vs. K-SVD convergence plots across two runs, SRR percentages, and source errors). | 
| [hw6.pdf](./hw6.pdf) | Original assignment problem sheet in Persian. | 
| `README.md` | Self-contained English problem statements, mathematical derivations, algorithm descriptions, numerical results, and usage guide. | 

Keep the `codes` subfolder inside the homework directory.

## 3. Assignment Problem Statement

### 3.1 General Context from the Assignment Sheet

*(Note: Although the title header of `hw6.pdf` was inadvertently labeled "Homework Series 4" when distributed, this sheet is the official Homework 6 problem set covering LASSO, Frame Design, MOD, and K-SVD, as documented in `BSS_HW#6_810100205_Arman_Majidi.pdf`.)*

### 3.2 Section 1: Sparse Recovery on Noiseless and Noisy Observations (`hw6-part1.mat`)

The file `hw6-part1.mat` contains the dictionary matrix $D \in \mathbb{R}^{5 \times 15}$, the noiseless observation vector $x \in \mathbb{R}^{5 \times 1}$, and the noisy observation vector `x_noisy` $\in \mathbb{R}^{5 \times 1}$.

#### Part الف (Part 1 in Code) — Basis Pursuit (BP) on Noiseless Data

Apply the **Basis Pursuit (BP)** method to the noiseless data $x$ and extract the sparse source vector $s$.

#### Part ب (Part 2 in Code) — Matching Pursuit (MP) on Noiseless Data

Does implementing the **Matching Pursuit (MP)** method on the noiseless data $x$ also lead to the true solution of the problem?

#### Part ج (Part 3 in Code) — Basis Pursuit (BP) on Noisy Observations

Now consider the noisy observation vector `x_noisy` and apply the **BP** method. Is the obtained solution identical to the true sparse source vector?

#### Part د (Part 4 in Code) — LASSO Formulation and Regularization Parameter Tuning on Noisy Data

Formulate the **LASSO** problem using the noisy observation data `x_noisy`. Solve the problem for different values of the regularization parameter $\lambda$, and report a value of $\lambda$ (or a range of $\lambda$) that leads to the correct sparse solution.

### 3.3 Section 2: Low-Coherence Frame Construction in 2D and 3D Spaces

#### Part 1 — Designing a $2 \times N$ Frame in $\mathbb{R}^2$

Propose a method that generates a **frame** (overcomplete unit-norm dictionary) of dimensions $2 \times N$. Implement your proposed method in code and report/plot the **mutual coherence** $\mu(D)$ achieved by your algorithm as a function of $N$.

#### Part 2 — Designing a $3 \times 10$ Frame in $\mathbb{R}^3$

Propose a method that generates a **frame** of dimensions $3 \times 10$. Implement your proposed method in code and report/plot the **mutual coherence** $\mu(D)$ that you obtain.

### 3.4 Section 3: Blind Dictionary Learning and Sparse Source Separation via MOD and K-SVD (`hw6-part3.mat`)

In this section, we implement and compare two dictionary learning algorithms—**MOD (Method of Optimal Directions)** and **K-SVD**—for blind separation of sparse sources (or equivalently, learning an overcomplete dictionary for sparse signal representation).

The file `hw6-part3.mat` contains:

* **Ground-truth dictionary matrix** $D$**:** dimensions $10 \times 40$ ($M = 10$, $N = 40$).

* **Ground-truth sparse source matrix** $S$**:** dimensions $40 \times 1500$ ($N = 40$, $T = 1500$).

* **Observation matrix** $X$**:** dimensions $10 \times 1500$ ($M = 10$, $T = 1500$).

* **Sparsity level** $N_0$**:** $N_0 = 2$.

#### Part 1 — Mutual Coherence of the True Dictionary $D$

Compute and report the **mutual coherence** $\mu(D)$ of the ground-truth dictionary $D$.

#### Part 2 — Blind Estimation of $\hat{D}$ and $\hat{S}$ via MOD and K-SVD

Assuming that only the observation matrix $X$ is known (together with the sparsity level $N_0 = 2$), apply both the **MOD** and **K-SVD** algorithms to $X$ and obtain the estimated dictionary matrix $\hat{D}$ and the estimated source matrix $\hat{S}$ for each method. Wherever a sparse recovery algorithm is required inside the alternating loop, use **Orthogonal Matching Pursuit (OMP)**.

#### Part 3 — Representation Error Convergence Curves

Plot the convergence curve—defined as the objective function value or **Representation Error** versus iteration index—for both MOD and K-SVD on a single figure. Which method converges to a lower error value?

#### Part 4 — Convergence Execution Time (`tic` / `toc`)

Using MATLAB's `tic` and `toc` commands, measure and report the convergence runtime of each of the two algorithms.

#### Part 5 — Dictionary Atom Successful Recovery Rate (SRR)

To evaluate and compare the quality of dictionary learning for each method separately, iterate sequentially from the first column (atom) to the last column (atom) of $\hat{D}$ and compute the absolute correlation of each estimated atom $\hat{d}_i$ with all atoms of the ground-truth dictionary $D$. If the maximum of these absolute correlations exceeds $0.98$, consider the atom of $D$ that achieved the maximum correlation to be **successfully recovered**. Note that for subsequent steps and calculations, you should remove the already-recovered atom of $D$. For each method, what percentage of the columns of $D$ did you recover correctly? (This percentage is referred to as the **Successful Recovery Rate**.)

#### Part 6 — Normalized Source Reconstruction Error After Resolving Ambiguities

Resolve both the **permutation (ordering)** ambiguity and the **scaling/sign** ambiguity of the estimated sources $\hat{S}$, and then compute and report the following normalized Frobenius error criterion for both MOD and K-SVD:

```
E = \frac{\left\|\hat{S} - S\right\|_F^2}{\|S\|_F^2}.

```

## 4. Mathematical Formulation & Solutions

### 4.1 Section 1: Sparse Recovery via BP, MP, and LASSO (`Section1.m`)

#### Ground-Truth Structure of `hw6-part1.mat`

In `codes/hw6-part1.mat`, the dictionary $D \in \mathbb{R}^{5 \times 15}$ ($M = 5$, $N = 15$) has unit-norm columns ($\Vert{}d_j\Vert{}_2 = 1$ for $j = 1, \dots, 15$) with mutual coherence $\mu(D) = \max_{i \neq j} \vert{}d_i^T d_j\vert{} \approx 0.8839$. The noiseless observation vector $x \in \mathbb{R}^{5 \times 1}$ ($\Vert{}x\Vert{}_2 \approx 2.9216$) is generated by an exact $2$-sparse source vector $s^* \in \mathbb{R}^{15 \times 1}$ supported on atoms $2$ and $10$:

```
x = D s^* = 1 \cdot d_2 - 2 \cdot d_{10},
\qquad
s_n^* = \begin{cases}
1, & n = 2, \\
-2, & n = 10, \\
0, & \text{otherwise}.
\end{cases}

```

The noisy observation vector $x_{\mathrm{noisy}} = x + v \in \mathbb{R}^{5 \times 1}$ contains additive perturbation of Euclidean norm $\Vert{}x_{\mathrm{noisy}} - x\Vert{}_2 = 0.2$, corresponding to a relative observation noise level of:

```
\frac{\left\|x - x_{\mathrm{noisy}}\right\|_2}{\|x\|_2} = \frac{0.2}{2.921572} \approx 0.068456 \quad (\mathrm{SNR} \approx 23.29\ \mathrm{dB}).

```

#### Part الف (Part 1 in Code) — Basis Pursuit on Noiseless Data (`Section1.m` lines 6–17)

**Basis Pursuit (BP)** seeks the minimum $\ell_1$-norm solution to the noiseless linear system $x = D s$:

```
\hat{s} = \arg\min_{s \in \mathbb{R}^N} \|s\|_1 = \sum_{n=1}^{N} |s_n|
\qquad \text{subject to} \qquad
x = D s.

```

Decomposing $s = s^+ - s^-$ into non-negative parts $s_n^+ = \max(s_n, 0) \ge 0$ and $s_n^- = \max(-s_n, 0) \ge 0$ so that $\vert{}s_n\vert{} = s_n^+ + s_n^-$ (Page 1 of the report), and defining the augmented matrix $D_1 = \begin{bmatrix} D & -D \end{bmatrix} \in \mathbb{R}^{5 \times 30}$ and variable vector $y = \begin{bmatrix} s^+ \\ s^- \end{bmatrix} \in \mathbb{R}_+^{30 \times 1}$, converts BP into the standard **Linear Program (LP)**:

```
\hat{y} = \arg\min_{y \in \mathbb{R}^{2N}} \sum_{i=1}^{2N} y_i
\qquad \text{subject to} \qquad
D_1 y = x, \quad y_i \ge 0 \quad (\forall i = 1, \dots, 2N).

```

Solving this LP via `y = linprog(ones(1, 2*N), [], [], D_1, x, zeros(1, 2*N))` and forming $s = y(1:N) - y(N+1:2N)$ recovers the exact ground-truth $15 \times 1$ sparse vector (Page 2 of the report):

```
s_{\mathrm{BP}} = \begin{bmatrix}
0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & -2 & 0 & 0 & 0 & 0 & 0
\end{bmatrix}^T,

```

with relative reconstruction error $\mathrm{err} = 7.6002 \times 10^{-17} \approx 0$.

#### Part ب (Part 2 in Code) — Matching Pursuit on Noiseless Data (`Section1.m` lines 19–37)

In `codes/Section1.m` (lines 20–36), **Matching Pursuit (MP)** is implemented with stopping tolerance `err_thr = 1e-2`, maximum atom count `max_N_0 = M = 5`, and a full-size $5 \times 15$ sub-dictionary buffer `D_hat = zeros(M, N)`:

1. At iteration $i = 1, \dots, M$, find the dictionary column maximizing the inner product with the current residual: `[innerprodMax, ind] = max(x_r.' * D)`.

2. Insert the selected column into its original position `D_hat(:, ind) = D(:, ind)` and update the residual: $x_r \leftarrow x_r - \mathrm{innerprodMax} \cdot d_{\mathrm{ind}}$.

3. Solve the least-squares problem over all columns placed in `D_hat` so far via `S_hat = pinv(D_hat) * x`, and stop when $\|x - \hat{D}\hat{S}\|_2 / \|x\|_2 < 10^{-2}$.

Because line 26 uses signed maximization `max(x_r.' * D)` rather than magnitude maximization `max(abs(x_r.' * D))`:

* **Iteration 1:** Selects atom $\mathrm{ind} = 2$ ($s_2^* = +1$), leaving residual $x_r^{(1)} \approx -2 d_{10}$ whose inner product with $d_{10}$ is $-2 < 0$.

* **Iterations 2 & 3:** Because $x_r^T d_{10} < 0$, signed `max` skips column $10$ and picks columns $12$ and $4$, reducing the residual until the sign of the projection onto column $10$ flips positive.

* **Iteration 4 (**$N_0 = 4$**):** Selects column $10$. Since both true support columns $\{2, 10\}$ are now present inside `D_hat` alongside two extra columns, the pseudoinverse step `S_hat = pinv(D_hat) * x` assigns exact weights $s_2 = 1$ and $s_{10} = -2$ to the true atoms and machine-precision zeros ($\sim 10^{-16}\text{–}10^{-15}$) to the spurious atoms, terminating at $N_0 = 4$ with relative error $\mathrm{err} = 5.0851 \times 10^{-16}$ (Page 2 of the report):

```
s_{\mathrm{MP}} = \begin{bmatrix}
-6.9927 \times 10^{-16} \\ 1 \\ 5.3055 \times 10^{-17} \\ 9.7751 \times 10^{-16} \\ 0 \\ 0 \\ 0 \\ 0 \\ 0 \\ -2 \\ 0 \\ 3.0328 \times 10^{-15} \\ 0 \\ 0 \\ 0
\end{bmatrix}.

```

Thus, **yes** — because of the least-squares refit `pinv(D_hat) * x` once atoms $2$ and $10$ have been included by Iteration 4, MP recovers the same true non-zero coefficients ($s_2 = 1$ and $s_{10} = -2$) up to floating-point roundoff ($\sim 10^{-15}$).

#### Part ج (Part 3 in Code) — Basis Pursuit on Noisy Data (`Section1.m` lines 39–49)

When standard equality-constrained BP ($D_1 y = x_{\mathrm{noisy}}$) is applied directly to the noisy observation $x_{\mathrm{noisy}} \in \mathbb{R}^{5 \times 1}$, the LP solver is forced to satisfy all $M = 5$ noisy scalar equations exactly. By the fundamental theorem of linear programming, a basic feasible solution in $\mathbb{R}^5$ generically activates $M = 5$ columns of $D$, yielding (Page 3 of the report):

```
s_{\mathrm{BP, noisy}} = \begin{bmatrix}
0 & 0.9013 & 0.1304 & 0 & -0.2066 & 0 & 0 & 0 & 0 & -1.9396 & 0.0148 & 0 & 0 & 0 & 0
\end{bmatrix}^T,

```

with relative error against the noiseless signal $x$:

```
\mathrm{err} = \frac{\left\|x - \hat{D}\hat{S}\right\|_2}{\|x\|_2} = \frac{\left\|x - x_{\mathrm{noisy}}\right\|_2}{\|x\|_2} = 0.068456.

```

**Answer:** **No — the solution is not identical to the exact noiseless vector** because equality-constrained BP overfits the additive noise by introducing three spurious non-zero coefficients ($s_3 = 0.1304$, $s_5 = -0.2066$, $s_{11} = 0.0148$, so $N_0 = 5$ instead of $2$) and perturbing the two true coefficients ($s_2 = 0.9013$ vs. $1$, $s_{10} = -1.9396$ vs. $-2$). Nevertheless, it provides a reasonable approximation with small relative error ($6.85\%$) whose two largest-magnitude entries correctly point to support indices $2$ and $10$.

#### Part د (Part 4 in Code) — LASSO via Coordinate Descent and Soft-Thresholding (`Section1.m` lines 51–90)

To handle noisy observations without overfitting noise into extra dictionary atoms, we relax the hard equality constraint $x_{\mathrm{noisy}} = D s$ into the unconstrained **LASSO (Least Absolute Shrinkage and Selection Operator)** optimization problem:

```
\hat{s}_{\mathrm{LASSO}} = \arg\min_{s \in \mathbb{R}^N} \left\{ \left\|x - D s\right\|_2^2 + \lambda \|s\|_1 \right\},

```

where the regularization parameter $\lambda \ge 0$ controls the trade-off between data fidelity ($\Vert{}x - D s\Vert{}_2^2$) and sparsity ($\Vert{}s\Vert{}_1$).

Because the columns of $D$ have unit Euclidean norm ($\Vert{}d_i\Vert{}_2 = 1$), minimizing the LASSO objective with respect to a single coordinate $s_i$ while holding all other coordinates $\{s_j\}_{j \neq i}$ fixed yields the closed-form **Soft-Thresholding** update rule (`Section1.m` lines 58–81, Figure 1.1 on Page 4 of the report):

```
r_i = x - \sum_{\substack{j=1 \\ j \neq i}}^{N} d_j s_j,
\qquad
\rho_i = r_i^T d_i,
\qquad
\hat{s}_i = \mathrm{soft}_{\lambda / 2}(\rho_i) = \begin{cases}
\rho_i - \dfrac{\lambda}{2}, & \rho_i > \dfrac{\lambda}{2}, \\[8pt]
0, & -\dfrac{\lambda}{2} \le \rho_i \le \dfrac{\lambda}{2}, \\[8pt]
\rho_i + \dfrac{\lambda}{2}, & \rho_i < -\dfrac{\lambda}{2}.
\end{cases}

```

Running Coordinate Descent for `iterNum = 100` full sweeps across $i = 1, \dots, 15$ (Pages 4–5 of the report):

1. **Noiseless calibration (**$x$**):** Setting $\lambda = \frac{1}{8} = 0.125$ eliminates all $13$ inactive coordinates and recovers the exact $2$-sparse support $\{2, 10\}$:

```
s_{\mathrm{LASSO, noiseless}}\left(\lambda = 0.125\right) = \begin{bmatrix}
0 & 0.9747 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & -1.9598 & 0 & 0 & 0 & 0 & 0
\end{bmatrix}^T.

```

2. **Noisy observations (**$x_{\mathrm{noisy}}$**):** Increasing $\lambda$ to suppress the noise floor shows that $\lambda = 0.4$ (or a narrow interval around $\lambda \approx 0.4$) recovers the exact $2$-sparse support $\{2, 10\}$ while zeroing out all noise-induced atoms (Page 5 of the report):

```
s_{\mathrm{LASSO, noisy}}\left(\lambda = 0.4\right) = \begin{bmatrix}
0 & 0.8916 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & -1.8488 & 0 & 0 & 0 & 0 & 0
\end{bmatrix}^T,
\qquad
\mathrm{err} = 0.068456.

```

3. **Effect of varying** $\lambda$ **across regimes (Page 5 of the report):**

   * $0 \le \lambda < 0.4$ **(Under-regularized regime):** The sparsity penalty is too weak to reject the noise projections, so extra non-zero coefficients appear ($N_0 > 2$).

   * $\lambda \approx 0.4$ **(Optimal support-recovery regime):** Both true atoms ($s_2 \approx 0.8916$, $s_{10} \approx -1.8488$) are recovered cleanly with exact sparsity $N_0 = 2$.

   * $2.1 < \lambda \le 4.2$ **(Single-atom regime):** The soft-threshold deadzone $[-\lambda/2, \lambda/2]$ exceeds the projection of the weaker source $s_2^* = 1$, shrinking $s_2$ to zero and leaving only $N_0 = 1$ active coefficient ($s_{10} \neq 0$).

   * $\lambda > 4.2$ **(Zero-solution regime):** The threshold $\lambda / 2 > 2.1$ exceeds even the stronger source projection $|\rho_{10}|$, forcing the entire vector to zero ($s = \mathbf{0}_{15 \times 1}$, $N_0 = 0$) and causing the relative error in `Section1.m` to jump from $0.068456$ to $1$.

### 4.2 Section 2: Frame Design and Mutual Coherence Minimization (`Section2.m`)

The **mutual coherence** $\mu(D)$ of a unit-norm frame (dictionary) $D = \begin{bmatrix} d_1 & d_2 & \cdots & d_N \end{bmatrix} \in \mathbb{R}^{M \times N}$ ($\Vert{}d_i\Vert{}_2 = 1$) measures the maximum absolute cosine similarity between any two distinct atoms:

```
\mu(D) = \max_{1 \le i \neq j \le N} \left| d_i^T d_j \right|.

```

In sparse recovery, a smaller $\mu(D)$ guarantees that dictionary atoms are as close to mutually orthogonal as possible, improving uniqueness and recovery bounds (such as $\|s\|_0 < \frac{1}{2}\left(1 + \frac{1}{\mu(D)}\right)$).

#### Part 1 — Analytical Construction of a $2 \times N$ Frame in $\mathbb{R}^2$ (`Section2.m` lines 3–20)

Every unit-norm vector $d_i \in \mathbb{R}^2$ lies on the unit circle and is parameterized by its polar angle $\varphi_i$:

```
d_i = \begin{bmatrix} \cos(\varphi_i) \\ \sin(\varphi_i) \end{bmatrix},
\qquad
\left| d_i^T d_j \right| = \left| \cos(\varphi_i)\cos(\varphi_j) + \sin(\varphi_i)\sin(\varphi_j) \right| = \left| \cos(\varphi_i - \varphi_j) \right|.

```

To minimize the maximum absolute inner product $\max_{i \neq j} |\cos(\varphi_i - \varphi_j)|$, the angular separation between distinct axes (lines through the origin) must be maximized:

* **Why naive full-circle spacing** $[0, 2\pi)$ **fails for even** $N$**:** If $N$ is odd, placing $N$ vectors uniformly around $[0, 2\pi)$ at $\varphi_i = 2\pi(i-1)/N$ produces $N$ distinct lines spaced by $\pi/N$. However, if $N$ is **even**, spacing vectors by $2\pi/N$ around $[0, 2\pi)$ places every vector $d_{i + N/2} = -d_i$ directly opposite to $d_i$ ($\varphi_{i+N/2} - \varphi_i = \pi$), yielding $|d_i^T d_{i+N/2}| = |\cos(\pi)| = 1$ and the worst possible mutual coherence $\mu(D) = 1$.

* **Unified half-circle construction on** $[0, \pi)$ **for all** $N \ge 2$**:** Because $|d_i^T d_j|$ is invariant to sign flips ($d_i \leftrightarrow -d_i$), restricting all $N$ angles to the upper half-plane $[0, \pi)$ with uniform step $\Delta\varphi = \pi / N$ works simultaneously for **both even and odd** $N$ (`Section2.m` lines 7–9):

```
\varphi_i = (i - 1)\frac{\pi}{N}, \qquad i = 1, 2, \dots, N,
\qquad
D_{2 \times N} = \begin{bmatrix}
\cos(0) & \cos\!\left(\frac{\pi}{N}\right) & \cdots & \cos\!\left(\frac{(N-1)\pi}{N}\right) \\[6pt]
\sin(0) & \sin\!\left(\frac{\pi}{N}\right) & \cdots & \sin\!\left(\frac{(N-1)\pi}{N}\right)
\end{bmatrix}.

```

Under this construction, the minimum angular separation between any two lines is $\pi / N$ (occurring both between adjacent vectors $\varphi_{i+1} - \varphi_i = \pi/N$ and between the first and last vectors $\pi - (\varphi_N - \varphi_1) = \pi - \frac{N-1}{N}\pi = \frac{\pi}{N}$), so the exact minimum achievable mutual coherence in $\mathbb{R}^2$ is:

```
\boxed{
\mu\left(D_{2 \times N}\right) = \cos\!\left(\frac{\pi}{N}\right), \qquad N \ge 2.
}

```

As plotted in Figure 2.1 of the report (Page 7) for $N = 2, \dots, 35$, $\mu(D) = 0$ for $N = 2$ (orthogonal basis), rises steeply to $\mu(D) \approx 0.9010$ at $N = 7$ and $\mu(D) \approx 0.9239$ at $N = 8$ ($\approx 0.9511$ at $N = 10$), and approaches $1$ as $N$ grows ($0.9960$ at $N = 35$), demonstrating that a 2D space rapidly saturates and cannot host many low-coherence atoms.

#### Part 2 — Stochastic Optimization of a $3 \times 10$ Frame in $\mathbb{R}^3$ (`Section2.m` lines 21–51)

In $\mathbb{R}^3$ ($M = 3$, $N = 10$), uniformly distributing $N$ lines through the origin (a Grassmannian line packing problem) does not admit a simple 1D angular spacing formula as in $\mathbb{R}^2$. In `codes/Section2.m` (lines 22–46), an iterative **stochastic coordinate-descent / random single-atom replacement algorithm** on the unit sphere $S^2 \subset \mathbb{R}^3$ is proposed and implemented:

1. **Initialization (`lines 23–28`):** Draw a random Gaussian matrix `vecs = randn(3, 10)`, normalize each column to unit $\ell_2$-norm (`vecs = vecs ./ sqrt(sum(vecs.^2))`), and compute its initial mutual coherence:

```
\mu(D) = \max_{1 \le i \neq j \le 10} \left| \left(D^T D - \mathrm{diag}(\mathrm{diag}(D^T D))\right)_{ij} \right|.

```

2. **Iterative greedy atom replacement (`lines 29–46`, `iterNum = 50000`):** At each outer iteration `iter = 1:50000`:

   * Set `final_vecs = vecs` and `final_mu_D = mu_D`.

   * Loop over each column index `col = 1:10`:

     * Construct a trial dictionary `test_vecs = vecs` by replacing column `col` with a newly drawn unit-norm Gaussian vector $v_{\mathrm{new}} / \|v_{\mathrm{new}}\|_2 \in S^2$.

     * Compute the trial mutual coherence `test_mu_D`.

     * If `test_mu_D < final_mu_D`, accept the improved dictionary (`final_vecs = test_vecs`, `final_mu_D = test_mu_D`).

   * Update `vecs = final_vecs`, `mu_D = final_mu_D`, and record `list_mu_D(iter) = mu_D`.

As shown on Pages 7–8 and Figure 2.2 of the report, $\mu(D)$ drops rapidly from $\approx 0.93$ down to $\mu(D) = 0.6947$ over the $50\,000$ iterations—a substantial reduction compared to $\mu(D_{2 \times 10}) = \cos(\pi/10) \approx 0.9511$ ($\approx 0.9$ around $N = 8\text{–}10$) in 2D space, because the extra spatial dimension in $\mathbb{R}^3$ allows the $10$ unit vectors to spread out much further apart on the unit sphere.

### 4.3 Section 3: Dictionary Learning via MOD and K-SVD (`Section3.m` and `OMP.m`)

#### Part 1 — Mutual Coherence of the Ground-Truth Dictionary $D \in \mathbb{R}^{10 \times 40}$

In `codes/Section3.m` (lines 7–10), the mutual coherence of the unit-norm ground-truth dictionary $D \in \mathbb{R}^{10 \times 40}$ from `hw6-part3.mat` is computed via:

```
\mu(D) = \max_{1 \le i \neq j \le 40} \left| d_i^T d_j \right|
= \max_{i, j} \left| \left(D^T D - I_{40}\right)_{ij} \right|
= \mathbf{0.8680} \quad (\text{exact: } 0.868030,\ \text{attained between columns } 13 \text{ and } 38).

```

#### Part 2 — Alternating Minimization Framework for MOD and K-SVD

Given only the observation matrix $X = \begin{bmatrix} x_1 & x_2 & \cdots & x_T \end{bmatrix} \in \mathbb{R}^{M \times T}$ ($M = 10$, $T = 1500$) and the target column sparsity level $N_0 = 2$, blind dictionary learning seeks both a unit-norm dictionary $\hat{D} = \begin{bmatrix} \hat{d}_1 & \cdots & \hat{d}_N \end{bmatrix} \in \mathbb{R}^{M \times N}$ ($N = 40$) and a column-wise $N_0$-sparse representation matrix $\hat{S} = \begin{bmatrix} \hat{s}_1 & \cdots & \hat{s}_T \end{bmatrix} \in \mathbb{R}^{N \times T}$ by solving the joint optimization problem:

```
\left(\hat{D}, \hat{S}\right) = \arg\min_{D, S} \left\|X - D S\right\|_F^2
\qquad \text{subject to} \qquad
\begin{cases}
\left\|d_n\right\|_2 = 1, & \forall n = 1, \dots, N, \\
\left\|s_t\right\|_0 \le N_0, & \forall t = 1, \dots, T.
\end{cases}

```

Both **MOD** and **K-SVD** initialize $\hat{D}^{(0)}$ from the **same** normalized random Gaussian matrix (`D_hat_init = randn(M, N)` followed by column normalization) and alternate between two stages over `iterNum = 50` iterations:

##### Stage 1: Sparse Coding ($\hat{D}$ Fixed, Update $\hat{S}$ via `OMP.m`)

Holding the current dictionary $\hat{D}$ fixed, the objective $\|X - \hat{D} S\|_F^2 = \sum_{t=1}^T \|x_t - \hat{D} s_t\|_2^2$ decouples into $T = 1500$ independent sparse recovery subproblems:

```
\hat{s}_t = \arg\min_{s_t} \left\|x_t - \hat{D} s_t\right\|_2^2
\qquad \text{subject to} \qquad
\left\|s_t\right\|_0 \le N_0 = 2,
\qquad t = 1, 2, \dots, T.

```

Each subproblem is solved by calling the helper function `s_hat = OMP(X(:, i), D_hat, N, N0)` in `codes/OMP.m`:

1. Initialize residual $x_r = x_t$ and support vector $\mathrm{ind} \in \mathbb{Z}^{1 \times N_0}$.

2. For step $k = 1, \dots, N_0$:

   * Compute inner products $\mathrm{innerprod} = x_r^T \hat{D}$ and select the atom with maximum magnitude correlation:

```
\mathrm{ind}(k) = \arg\max_{1 \le j \le N} \left| x_r^T \hat{d}_j \right|.

```

* For $k = 1$, update the residual via single-atom projection: $x_r \leftarrow x_r - \left(x_r^T \hat{d}_{\mathrm{ind}(1)}\right) \hat{d}_{\mathrm{ind}(1)}$.

* For $k > 1$ (specifically $k = N_0 = 2$), form the active sub-dictionary $\hat{D}_{\mathrm{pr}} = \hat{D}(:, \mathrm{ind}(1:k))$, compute the orthogonal least-squares coefficients on the active support:

```
\hat{s}_t\big(\mathrm{ind}(1:k)\big) = \hat{D}_{\mathrm{pr}}^\dagger x_t,

```

```
 and update the residual $x_r = x_t - \hat{D} \hat{s}_t$.

```

##### Stage 2A: Dictionary Update in MOD ($\hat{S}$ Fixed, Global Least-Squares Update)

In the **Method of Optimal Directions (MOD)** (`Section3.m` lines 31–32), holding $\hat{S}$ fixed reduces $\min_D \|X - D \hat{S}\|_F^2$ to an unconstrained matrix least-squares problem whose closed-form minimizer is given by the right Moore-Penrose pseudoinverse of $\hat{S}$, followed by column-wise normalization onto the unit sphere:

```
\tilde{D}_{\mathrm{MOD}} = X \hat{S}^\dagger = X \hat{S}^T \left(\hat{S} \hat{S}^T\right)^{-1},
\qquad
\hat{d}_n = \frac{\tilde{d}_n}{\left\|\tilde{d}_n\right\|_2}, \quad n = 1, \dots, N.

```

##### Stage 2B: Dictionary Update in K-SVD ($\hat{S}$ Support Fixed, Sequential Rank-1 SVD Update)

Unlike MOD (which keeps all entries of $\hat{S}$ frozen during the dictionary update and then re-normalizes $\hat{D}$ without rescaling $\hat{S}$), **K-SVD** (`Section3.m` lines 52–64) updates one dictionary column $\hat{d}_i$ ($i = 1, \dots, N$) at a time while **simultaneously updating the non-zero coefficients in the** $i$**-th row of** $\hat{S}$, strictly preserving the sparsity pattern (support) of $\hat{S}$:

1. Isolate the contribution of the $i$-th atom $\hat{d}_i$ by computing the residual matrix $X_r^{(i)}$ over all other $N - 1$ atoms $n \in \{1, \dots, N\} \setminus \{i\}$:

```
X - \hat{D} \hat{S} = \underbrace{\left(X - \sum_{j \neq i} \hat{d}_j \hat{S}_{j, :}\right)}_{X_r^{(i)} \in \mathbb{R}^{M \times T}} - \hat{d}_i \hat{S}_{i, :}.

```

2. To prevent the rank-1 update from filling in zero entries of row $\hat{S}_{i, :}$, identify the active sample indices $\omega_i$ where atom $i$ is currently used (`ind = (S_hat_KSVD(i, :) ~= 0)`) and restrict $X_r^{(i)}$ to those columns:

```
\omega_i = \left\{ t \in \{1, \dots, T\} : \hat{S}_{i, t} \neq 0 \right\},
\qquad
M_{X_r}^{(i)} = X_r^{(i)}(:, \omega_i) \in \mathbb{R}^{M \times |\omega_i|}.

```

3. Compute the Singular Value Decomposition of the restricted residual matrix (`[U, G, V] = svd(m_Xr)`):

```
M_{X_r}^{(i)} = U G V^T.

```

4. By the Eckart-Young theorem, the best rank-1 Frobenius approximation to $M_{X_r}^{(i)}$ is given by the largest singular value $\sigma_{\max} = G(\mathrm{idx}, \mathrm{idx})$ and its corresponding left and right singular vectors $u_{\max} = U(:, \mathrm{idx})$ and $v_{\max} = V(:, \mathrm{idx})$. Because $U$ is orthogonal, $u_{\max}$ automatically has unit $\ell_2$-norm ($\Vert{}u_{\max}\Vert{}_2 = 1$), yielding the joint atom and coefficient update (`Section3.m` lines 60–63):

```
\hat{d}_i \leftarrow u_{\max},
\qquad
\hat{S}\left(i, \omega_i\right) \leftarrow \sigma_{\max} v_{\max}^T.

```

#### Parts 3 & 4 — Representation Error Metric and Convergence Comparison

At the end of each iteration `iter = 1:50`, both algorithms evaluate the **normalized squared Frobenius Representation Error** (`Section3.m` lines 33 and 65):

```
\mathrm{RErr}(\mathrm{iter}) = \frac{\left\|X - \hat{D}\hat{S}\right\|_F^2}{\|X\|_F^2}
= \frac{\mathrm{tr}\!\left(\left(X - \hat{D}\hat{S}\right)\left(X - \hat{D}\hat{S}\right)^T\right)}{\mathrm{tr}\!\left(X X^T\right)}.

```

* **Accuracy comparison (Part 3):** As shown in Figure 3.1 (Page 10 of the report), both algorithms reduce the representation error steeply during the first $15\text{–}25$ iterations and plateau by Iteration 50. In the primary run (Figure 3.1), **K-SVD converges to a lower representation error (**$\approx 0.022$**) than MOD (**$\approx 0.030$**)** because K-SVD jointly updates each atom $\hat{d}_i$ together with its active coefficients $\hat{S}(i, \omega_i)$ via exact rank-1 SVD steps. As noted on Pages 10–11 (Figure 3.2), because both methods are non-convex and depend on the random Gaussian initialization `D_hat_init`, in some random realizations MOD can reach a slightly lower plateau ($\approx 0.026$ vs. $\approx 0.030$).

* **Runtime comparison (Part 4):** Across all runs, **MOD converges faster in wall-clock time** ($\approx 2.79\text{–}2.80\ \mathrm{s}$) than **K-SVD** ($\approx 3.44\text{–}3.58\ \mathrm{s}$), because MOD updates $\hat{D}$ in a single vectorized pseudoinverse `X * pinv(S_hat_MOD)`, whereas K-SVD executes $N = 40$ sequential SVDs inside an inner loop at every iteration.

#### Part 5 — Dictionary Successful Recovery Rate (SRR) (`Section3.m` lines 36–38 & 68–70)

To measure how many of the $N = 40$ unit-norm ground-truth atoms $\{d_1, \dots, d_{40}\}$ were accurately learned by $\hat{D}$, `Section3.m` forms the $40 \times 40$ cross-correlation matrix between the estimated and true dictionary columns:

```
C_{\mathrm{corr}} = \hat{D}^T D \in \mathbb{R}^{N \times N},
\qquad
\left(C_{\mathrm{corr}}\right)_{ij} = \hat{d}_i^T d_j,

```

and counts the fraction of atom pairs whose absolute correlation exceeds the threshold $0.98$:

```
\mathrm{SRR} = \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{N} \mathbb{I}\!\left\{\left|\hat{d}_i^T d_j\right| > 0.98\right\} \times 100\%.

```

*(Note: Because the ground-truth dictionary has mutual coherence* $\mu(D) = 0.8680 < 0.98$*, each estimated atom* $\hat{d}_i$ *can have correlation* $> 0.98$ *with at most one ground-truth atom* $d_j$*; see Section 5.4 for a note on greedy without-replacement matching vs. `sum(abs(corr) > 0.98, 'all') / N`.)*

#### Part 6 — Resolving Permutation & Sign Ambiguities and Source Error $E$ (`Section3.m` lines 80–93)

Even when $\hat{D}$ recovers the columns of $D$ accurately, the rows of the estimated source matrix $\hat{S} \in \mathbb{R}^{40 \times 1500}$ are subject to:

1. **Permutation ambiguity:** The $j$-th true atom $d_j$ may appear at some row/column index $\pi(j)$ in $\hat{D}$.

2. **Sign / scale ambiguity:** Because $\|\hat{d}_i\|_2 = \|d_j\|_2 = 1$, each recovered atom satisfies $\hat{d}_{\pi(j)} \approx \pm d_j$, so the corresponding row of $\hat{S}$ has unit scale factor up to a $\pm 1$ sign flip.

In `codes/Section3.m` (lines 82–89):

1. **Resolve row permutation:** For each true atom $j \in \{1, \dots, 40\}$ (each column of `corr = D_hat.' * D`), find the row index $\mathrm{ind}(j)$ of $\hat{D}$ having maximum absolute correlation with $d_j$:

```
\mathrm{ind}(j) = \arg\max_{1 \le i \le N} \left| \hat{d}_i^T d_j \right|, \qquad j = 1, \dots, N,

```

and reorder the rows of $\hat{S}$ via `S_sort = S_hat(ind, :)`. 2. **Resolve sign ambiguity:** Take element-wise magnitudes `abs(S_sort)` and `abs(S)` (`diff = abs(S_sort) - abs(S)`). 3. **Evaluate normalized squared Frobenius source error:** Using the trace identity $\|A\|_F^2 = \mathrm{tr}(A^T A)$, compute:

```
E = \frac{\left\|\hat{S} - S\right\|_F^2}{\|S\|_F^2}
= \frac{\mathrm{tr}\!\left(\left(|\hat{S}_{\mathrm{sort}}| - |S|\right)^T \left(|\hat{S}_{\mathrm{sort}}| - |S|\right)\right)}{\mathrm{tr}\!\left(S^T S\right)}.

```

In the reported run (Pages 11–12), **K-SVD** achieves a substantially lower source estimation error ($E_{\mathrm{K\text{-}SVD}} = \mathbf{0.32125}$) than **MOD** ($E_{\mathrm{MOD}} = \mathbf{0.47698}$).

## 5. Results & Discussion

### 5.1 Section 1 — Summary of Noiseless and Noisy Sparse Recovery (`Section1.m`)

The table below summarizes the results of all four parts of Section 1 on `hw6-part1.mat` ($D \in \mathbb{R}^{5 \times 15}$, true support $\{2, 10\}$ with $s_2^* = 1$ and $s_{10}^* = -2$):

| Part & Method | Observation Input | Regularization / Stopping Rule | Recovered Sparsity $N_0$ | Non-Zero Entries of Recovered $\hat{s}$ | Reported Relative Error $\mathrm{err}$ | Recovery Assessment | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **Part الف (Part 1)** — Basis Pursuit (`linprog`) | Noiseless $x$ | Equality LP ($D s = x$) | $2$ | $s_2 = 1,\ s_{10} = -2$ (exact) | $7.6002 \times 10^{-17}$ | Exact machine-precision recovery of $s^*$. | 
| **Part ب (Part 2)** — Matching Pursuit (MP) | Noiseless $x$ | `err_thr = 1e-2`, `max_N_0 = 5` | $4$ atoms selected ($2$ dominant) | $s_2 = 1,\ s_{10} = -2$ (plus $4$ machine-$\epsilon$ entries $\sim 10^{-16}\text{–}10^{-15}$) | $5.0851 \times 10^{-16}$ | Recovers exact dominant weights after 4 steps via `pinv(D_hat) * x`. | 
| **Part ج (Part 3)** — Basis Pursuit (`linprog`) | Noisy `x_noisy` | Equality LP ($D s = x_{\mathrm{noisy}}$) | $5$ | $s_2 = 0.9013,\ s_3 = 0.1304,\ s_5 = -0.2066,\ s_{10} = -1.9396,\ s_{11} = 0.0148$ | $0.068456$ | Overfits additive noise into $3$ extra atoms, though close to $s^*$. | 
| **Part د (Part 4)** — LASSO (Coordinate Descent) | Noiseless $x$ | $\lambda = \frac{1}{8} = 0.125$ ($100$ sweeps) | $2$ | $s_2 = 0.9747,\ s_{10} = -1.9598$ | $2.5877 \times 10^{-16}$ | Exact 2-sparse support recovery with slight soft-threshold shrinkage. | 
| **Part د (Part 4)** — LASSO (Coordinate Descent) | Noisy `x_noisy` | $\lambda = 0.4$ ($100$ sweeps) | $2$ | $s_2 = 0.8916,\ s_{10} = -1.8488$ | $0.068456$ | **Optimal noisy recovery:** eliminates all spurious noise atoms ($N_0 = 2$). | 
| **Part د (Part 4)** — LASSO $\lambda$ Sweep | Noisy `x_noisy` | $0 \le \lambda < 0.4$  $2.1 < \lambda \le 4.2$  $\lambda > 4.2$ | $N_0 > 2$  $N_0 = 1$  $N_0 = 0$ | Extra noise atoms active ($0 \le \lambda < 0.4$)  Only $s_{10} \neq 0$ ($2.1 < \lambda \le 4.2$)  $s = \mathbf{0}_{15 \times 1}$ ($\lambda > 4.2$) | $0.068456$ ($\lambda \le 4.2$)  $1.0$ ($\lambda > 4.2$) | Best trade-off between exact sparsity ($N_0=2$) and low error is around $\lambda \approx 0.4$. | 

### 5.2 Section 2 — Summary of 2D and 3D Frame Mutual Coherence (`Section2.m`)

| Part & Frame Dimension | Construction Algorithm | Number of Atoms $N$ | Analytical / Reported Mutual Coherence $\mu(D)$ | Key Geometric Insight | 
| ----- | ----- | ----- | ----- | ----- | 
| **Part 1** — $2 \times N$ Frame in $\mathbb{R}^2$ | Uniform angular spacing on upper half-circle $\varphi_i = (i-1)\pi/N \in [0, \pi)$ | $N = 2, 3, \dots, 35$ | $\mu(D) = \cos(\pi/N)$:  $N=2 \to 0$  $N=3 \to 0.5000$  $N=4 \to 0.7071$  $N=8 \to 0.9239$ ($\approx 0.9$)  $N=10 \to 0.9511$  $N=35 \to 0.9960$ | Restricting angles to $[0, \pi)$ avoids antipodal pairs ($d_i = -d_j$, $\mu=1$) for even $N$, but 2D coherence exceeds $0.9$ by $N = 7\text{–}8$ (Figure 2.1). | 
| **Part 2** — $3 \times 10$ Frame in $\mathbb{R}^3$ | Stochastic greedy single-atom replacement on $S^2 \subset \mathbb{R}^3$ ($50\,000$ iterations) | $N = 10$ | $\mathbf{0.6947}$ | Adding a 3rd spatial dimension reduces mutual coherence at $N = 10$ from $\approx 0.95$ (in 2D) down to $\approx 0.6947$ (Figure 2.2). | 

### 5.3 Section 3 — Summary of MOD vs. K-SVD Dictionary Learning (`Section3.m`)

In `codes/hw6-part3.mat` ($D \in \mathbb{R}^{10 \times 40}$, $S \in \mathbb{R}^{40 \times 1500}$, $X \in \mathbb{R}^{10 \times 1500}$, $N_0 = 2$), the ground-truth dictionary mutual coherence is $\mu(D) = 0.8680$ (`0.868` in the report). The table below summarizes the two 50-iteration comparative runs documented in `BSS_HW#6_810100205_Arman_Majidi.pdf`:

| Metric / Criterion | MOD (Run 1 — Fig. 3.1) | K-SVD (Run 1 — Fig. 3.1) | MOD (Run 2 — Fig. 3.2) | K-SVD (Run 2 — Fig. 3.2) | Comparative Conclusion | 
| ----- | ----- | ----- | ----- | ----- | ----- | 
| Dictionary Update Rule | $\hat{D} = X \hat{S}^\dagger$ + column normalization | Sequential atom-by-atom rank-1 SVD on restricted residuals $M_{X_r}^{(i)}$ | $\hat{D} = X \hat{S}^\dagger$ + column normalization | Sequential atom-by-atom rank-1 SVD on restricted residuals $M_{X_r}^{(i)}$ | MOD updates all atoms globally; K-SVD jointly updates $\hat{d}_i$ and $\hat{S}(i, \omega_i)$. | 
| Convergence Time (`tic`/`toc`) | $2.797860\ \mathrm{s}$ | $3.576926\ \mathrm{s}$ | $2.793242\ \mathrm{s}$ | $3.441024\ \mathrm{s}$ | **MOD is faster** ($\sim 2.8\ \mathrm{s}$ vs. $\sim 3.5\ \mathrm{s}$) due to single matrix `pinv` vs. $40$ SVDs per iteration. | 
| Final Representation Error $\mathrm{RErr}(50)$ | $\approx 0.030$ | $\approx 0.022$ | $\approx 0.026$ | $\approx 0.030$ | K-SVD typically reaches a lower representation error (Fig. 3.1), though MOD can occasionally edge it out on specific random initializations (Fig. 3.2). | 
| Successful Recovery Rate ($\mathrm{SRR}$, $\Vert{}\hat{d}_i^T d_j\Vert{} > 0.98$) | $82\%$ | $90\%$ | $88\%$ | $82\%$ | Both methods successfully recover $82\%\text{–}90\%$ ($33\text{–}36$ out of $40$) of the true atoms. | 
| Normalized Source Error $E = \Vert{}\hat{S}-S\Vert{}_F^2 / \Vert{}S\Vert{}_F^2$ | $0.47698$ | $0.32125$ | — | — | **K-SVD achieves significantly lower source estimation error** ($0.32125$ vs. $0.47698$). | 

### 5.4 Notes on the Submitted Solutions / Implementation Notes

* **First-Step Coefficient Assignment in `codes/OMP.m` (lines 8–14):**

  * Inside `codes/OMP.m`, when `i == 1`, line 13 updates the residual via `x_r = x_r - innerprod(ind(i)) * D(:, ind(i))` without assigning `s_hat(ind(1)) = innerprod(ind(1))`, whereas the `if i > 1` branch on line 10 assigns `s_hat(ind(1:i)) = pinv(D_pr) * x` for all `1:i`. Because `N0 = 2 > 1` in Section 3, the `i = 2` iteration always executes `s_hat(ind(1:2)) = pinv(D_pr) * x` and populates both non-zero entries of `s_hat` properly. (If `OMP.m` were called with `N0 = 1`, `s_hat` would remain all zeros unless `s_hat(ind(1)) = innerprod(ind(1))` is added to the `else` branch.)

* **Residual Vector Variable in Noisy LASSO (`codes/Section1.m` line 63) and Saved Parameter `lambda_noisy = 4.2` (line 54):**

  * On line 63 of `Section1.m`, the noisy partial residual is written as `r_noisy = x_noisy - D_filtered * s_filtered;` (using the noiseless iterate `s_filtered` instead of `s_noisy_filtered = [s_noisy(1:i-1, :); s_noisy(i+1:end, :)]`). Because `s` is updated with $\lambda = 0.125$ on the noiseless vector $x$, $D_{\mathrm{filtered}} s_{\mathrm{filtered}} \approx x - d_i s_i$, so $\rho_{\mathrm{noisy}} = r_{\mathrm{noisy}}^T d_i \approx (x_{\mathrm{noisy}} - x + d_i s_i)^T d_i$. Moreover, line 54 of the saved script has `lambda_noisy = 4.2` from the final boundary test discussed at the bottom of Page 5 of the report; setting `lambda_noisy = 0.4` (and using `s_noisy_filtered`) reproduces the $2$-sparse noisy LASSO estimate reported on Page 5.

* **Error Evaluation Using `D_hat = x_noisy * pinv(s_noisy)` in `codes/Section1.m` (lines 13–15, 45–47, 82–87):**

  * As in Homework 5, `Section1.m` computes `D_hat = x * pinv(S_hat)` in Part 1 and `D_hat_noisy = x_noisy * pinv(s_noisy)` in Parts 3 and 4, followed by `err_func_noisy = x - D_hat_noisy * s_noisy`. For any non-zero vector $s_{\mathrm{noisy}} \neq \mathbf{0}$, the outer-product pseudoinverse identity $(x_{\mathrm{noisy}} s_{\mathrm{noisy}}^\dagger) s_{\mathrm{noisy}} = x_{\mathrm{noisy}} \frac{s_{\mathrm{noisy}}^T s_{\mathrm{noisy}}}{s_{\mathrm{noisy}}^T s_{\mathrm{noisy}}} = x_{\mathrm{noisy}}$ holds identically whenever $s_{\mathrm{noisy}} \neq \mathbf{0}$, and collapses to $\mathbf{0}$ when $s_{\mathrm{noisy}} = \mathbf{0}$. This explains why the script reports the constant value $\|x - x_{\mathrm{noisy}}\|_2 / \|x\|_2 = 0.068456$ for all $\lambda \in [0, 4.2]$ where $s_{\mathrm{noisy}} \neq \mathbf{0}$, and jumps to $\|x - \mathbf{0}\|_2 / \|x\|_2 = 1$ for $\lambda > 4.2$ where $s_{\mathrm{noisy}} = \mathbf{0}$. If evaluated directly with the true dictionary $D$ (`norm(x - D * s_noisy) / norm(x)`), the relative error against $x$ at $\lambda = 0.4$ is $\approx 0.0695$.

* **Successful Recovery Rate (SRR) Calculation in `codes/Section3.m` (lines 36–37 & 68–69):**

  * In `Section3.m`, SRR is computed in one vectorized line via `corr = D_hat.' * D; SRR = sum(abs(corr) > 0.98, 'all') / N;`. Because the true dictionary $D$ has mutual coherence $\mu(D) = 0.8680 < 0.98$, a single estimated unit-norm atom $\hat{d}_i$ cannot simultaneously have $|\hat{d}_i^T d_{j_1}| > 0.98$ and $|\hat{d}_i^T d_{j_2}| > 0.98$ for two distinct true atoms $j_1 \neq j_2$ (since $\cos(\arccos(0.98) + \arccos(0.98)) = 2(0.98)^2 - 1 = 0.9208 > 0.8680$). However, if two estimated atoms in $\hat{D}$ converge to the same true atom $d_j$, `sum(abs(corr) > 0.98, 'all')` counts both unless matched columns of $D$ are zeroed out sequentially without replacement as described in the Part 5 problem statement (or equivalently via `sum(max(abs(corr), [], 1) > 0.98) / N`).

* **Row Permutation and Sign Alignment in `codes/Section3.m` (lines 82–87):**

  * On lines 82–85, `[~, ind_MOD] = max(abs(corr_MOD))` finds the best-matching row of $\hat{D}$ for each column of $D$ (since `max` operates column-wise along dimension 1 of `corr_MOD = D_hat_MOD.' * D`), and reorders the rows of $\hat{S}$ via `S_sort_MOD = S_hat_MOD(ind_MOD, :)`.

  * On lines 86–87, sign ambiguity is removed by taking element-wise absolute values `diff_MOD = abs(S_sort_MOD) - abs(S)`. When `ind_MOD(j)` matches the true support row and $\hat{S}_{\mathrm{sort}}(j, t)$ has the same sign pattern as $S(j, t)$ (up to a global row sign $\pm 1$), `abs(S_sort) - abs(S)` cancels the global row sign flip without needing an explicit `sign(diag(D_hat(:, ind).' * D))` multiplier.

## 6. How to Use

### 6.1 Running the MATLAB Scripts

Open MATLAB, navigate to the `codes` directory (so that `hw6-part1.mat`, `hw6-part3.mat`, and `OMP.m` are on the active working path), and run the three scripts:

```
cd codes
Section1
Section2
Section3

```

* **Dependencies:**

  * `Section1.m` uses `linprog` from the **Optimization Toolbox** (Parts 1 and 3) and base MATLAB functions (`pinv`, `randn`) for Parts 2 and 4.

  * `Section2.m`, `Section3.m`, and `OMP.m` require **only base MATLAB** (`svd`, `pinv`, `randn`, `trace`, `ind2sub`, `sort`, `plot`).

* **Generated Figures:**

  * `Section1.m` prints all numerical results directly to the Command Window without opening figure windows (Figure 1.1 in the report is an analytical diagram of the soft-thresholding operator $\hat{s}_i = \mathrm{soft}_{\lambda/2}(\rho)$).

  * `Section2.m` opens **2 figures**:

    1. Plot of the 2D frame mutual coherence $\mu(D)$ versus atom count $N \in \{2, \dots, 35\}$ (Figure 2.1 in the report).

    2. Convergence trajectory of the 3D frame mutual coherence $\mu(D)$ over $50\,000$ random replacement iterations for $N = 10$ (Figure 2.2 in the report).

  * `Section3.m` opens **1 figure**:

    1. Overlay convergence plot of the normalized Representation Error versus iteration index ($1$ to $50$) for **MOD** and **K-SVD** (Figures 3.1 and 3.2 in the report).

### 6.2 Expected Console Output (`Section1.m` and `Section3.m`)

Running `Section1.m` produces the following deterministic console output (note that in the saved script `lambda_noisy = 4.2` is set to the upper boundary value from the Part 4 sweep, which prints `Error (with Noise) = 0.068456` when `s_noisy` is non-zero or `1` when all entries are thresholded to zero):

```
Part 1:
Optimization terminated.
Error (Without Noise) = 7.6002e-17



Part 2:
Error (without Noise) = 5.0851e-16



Part 3:
Optimization terminated.
Error (with Noise) = 0.068456



Part 4:
Error (without Noise) = 2.5877e-16
Error (with Noise) = 0.068456

```

In `Section3.m`, Part 1 (`mu_D`) is deterministic while Parts 2–6 depend on the random Gaussian initial dictionary `D_hat_init = randn(M, N)`. A representative console output matching Run 1 of the Persian report is:

```
Part 1:
mu_D = 0.86803



Part 2-5:
Elapsed time is 2.797860 seconds.
MOD : Successful Recovery Rate: 82%

Elapsed time is 3.576926 seconds.
K-SVD : Successful Recovery Rate: 90%



Part 6:
Error (MOD Algorithm) = 0.47698

Error (K-SVD Algorithm) = 0.32125

```

### 6.3 Reading the Theoretical Derivations and Report

1. Use **Section 3** for the complete English translation of the assignment problems and **Section 4** for the step-by-step mathematical derivations of Basis Pursuit, Matching Pursuit, Coordinate-Descent LASSO, 2D/3D low-coherence frame design, MOD, and K-SVD.

2. Use **Section 5** to review the quantitative results tables and implementation notes alongside the original Persian report [BSS_HW#6_810100205_Arman_Majidi.pdf](./BSS_HW%236_810100205_Arman_Majidi.pdf) and assignment sheet [hw6.pdf](./hw6.pdf).

## 7. References

1. Akhavan, S. *Blind Source Separation: Homework 6*. Course assignment sheet and lecture notes, University of Tehran.

2. Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B (Methodological)*, 58(1), 267–288.

3. Engan, K., Aase, S. O., & Hakon Husoy, J. (1999). Method of optimal directions for frame design. *Proceedings of IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)*, Vol. 5, 2443–2446.

4. Aharon, M., Elad, M., & Bruckstein, A. (2006). K-SVD: An algorithm for designing overcomplete dictionaries for sparse representation. *IEEE Transactions on Signal Processing*, 54(11), 4311–4322.

5. Strohmer, T., & Heath, R. W. (2003). Grassmannian frames with applications to coding and communication. *Applied and Computational Harmonic Analysis*, 14(3), 257–275.

6. Comon, P., & Jutten, C. (Eds.). (2010). *Handbook of Blind Source Separation: Independent Component Analysis and Applications*. Academic Press.
