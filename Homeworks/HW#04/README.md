# Homework 4: Non-Stationary Blind Source Separation via Joint Diagonalization (GEVD & Alternating Minimization) and Temporally Correlated BSS via Time-Lagged GEVD

**Course:** Blind Source Separation  
**Instructor:** Dr. Saeed Akhavan  
**Author:** Arman Majidi  
**Student ID:** 810100205  
**University:** University of Tehran  
**Semester:** Spring 2024-2025

## 1. Overview & Assignment Structure

This folder contains the **English problem statements, analytical derivations, MATLAB implementations, and simulation results for Homework 4**. The assignment is organized into two main sections that explore second-order Blind Source Separation (BSS) exploiting **non-stationarity** (piecewise time-varying source variances across multiple time windows) and **temporal correlation** (time-lagged spatial correlation matrices):

1. **Section 1 — Non-Stationary Uncorrelated Sources, Two-Window GEVD, Multi-Window Joint Diagonalization via Alternating Minimization, and Monte Carlo Noise Analysis (Parts الف–ی / Parts 1–5):** Synthesizes two statistically uncorrelated, piecewise non-stationary sinusoidal sources across $K = 5$ time windows of duration $1\ \mathrm{s}$ each ($f_s = 20\ \mathrm{Hz}$, $T = 100$ total samples) and mixes them via an orthogonal $2 \times 2$ mixing matrix $A$. Recovers the sources in the noiseless regime first from only the first two windows ($K = 2$) via **Generalized Eigenvalue Decomposition (GEVD)** and then across all $K = 5$ windows via **Simultaneous / Joint Diagonalization (JD)** using an iterative **Alternating Minimization (AM)** algorithm (`JD`). Evaluates noisy observations $Y = X + \sigma W$ over $100$-run Monte Carlo simulations as a function of the number of windows $K \in \{2, 3, 4, 5\}$ at $\mathrm{SNR} = 20\ \mathrm{dB}$ and across signal-to-noise ratios $\mathrm{SNR} \in \{5, 10, 15, 20\}\ \mathrm{dB}$ for $K = 5$.
2. **Section 2 — Temporally Correlated Stationary Sources, Time-Lagged GEVD (AMUSE / Second-Order BSS), and Spectral Separability Analysis (Parts الف–د / Parts 1–4):** Applies second-order BSS based on two time-lagged correlation matrices $R_X(\tau_1)$ and $R_X(\tau_2)$ ($\tau_1 = f_s / 4 = 25$ samples, $\tau_2 = f_s / 5 = 20$ samples at $f_s = 100\ \mathrm{Hz}$) to two noiseless $2 \times 500$ observation datasets (`hw4-X1.mat` and `hw4-X2.mat`). Analyzes the time-domain waveforms and Discrete Fourier Transform (DFT) magnitude spectra of the estimated sources and mixtures to determine when simple linear frequency-domain filtering suffices to separate the sources and when BSS spatial unmixing is required.

All parts of both sections are mandatory; no optional or bonus problems were assigned. Because both the original assignment sheet and the submitted solution report are written in Persian, this `README.md` provides a **complete, self-contained English reference for every problem statement, mathematical derivation, algorithm, and numerical result**.

## 2. Directory Contents

| File | Description |
| --- | --- |
| [codes/Section1.m](./codes/Section1.m) | MATLAB script for Section 1 (Parts 1–5 / الف–ی): non-stationary source and mixture generation, two-window GEVD separation, $K$-window Joint Diagonalization (`JD`) via Alternating Minimization, and 100-run Monte Carlo evaluations across $K \in \{2, 3, 4, 5\}$ and $\mathrm{SNR} \in \{5, 10, 15, 20\}\ \mathrm{dB}$. |
| [codes/Section2.m](./codes/Section2.m) | MATLAB script for Section 2 (Parts 1–4 / الف–د): time-lagged GEVD source separation on `hw4-X1.mat` ($X_1$) and `hw4-X2.mat` ($X_2$), and FFT magnitude spectrum analysis of the estimated sources and observations. |
| [codes/hw4-X1.mat](./codes/hw4-X1.mat) | First dataset for Section 2 containing the $2 \times 500$ noiseless observation matrix `X1` sampled at $f_s = 100\ \mathrm{Hz}$ ($5\ \mathrm{s}$ duration, composed of $2\ \mathrm{Hz}$ and $8\ \mathrm{Hz}$ sinusoidal components). |
| [codes/hw4-X2.mat](./codes/hw4-X2.mat) | Second dataset for Section 2 containing the $2 \times 500$ noiseless observation matrix `X2` sampled at $f_s = 100\ \mathrm{Hz}$ ($5\ \mathrm{s}$ duration, composed of $1\ \mathrm{Hz}$, $2\ \mathrm{Hz}$, $4\ \mathrm{Hz}$, and $8\ \mathrm{Hz}$ components). |
| [BSS\_HW#4\_810100205\_Arman\_Majidi.pdf](./BSS_HW%234_810100205_Arman_Majidi.pdf) | Submitted report and analytical solutions in Persian (including source/observation waveforms, GEVD matrices, Alternating Minimization cost trajectories, Monte Carlo error vectors, and FFT magnitude spectra). |
| [hw4.pdf](./hw4.pdf) | Original assignment problem sheet in Persian. |
| `README.md` | Self-contained English problem statements, mathematical derivations, algorithm descriptions, numerical results, and usage guide. |

Keep the `codes` subfolder inside the homework directory.

## 3. Assignment Problem Statement

### 3.1 General Instructions from the Assignment Sheet

- The exercises may be implemented in either **Python** or **MATLAB**.
- Upload both the typeset report and the MATLAB `.m` scripts (or Python notebook files).
- The report must be cleanly typeset in Persian (handwritten mathematical derivations may be scanned and inserted if desired).
- In MATLAB, write standard `.m` script files rather than Live Scripts (`.mlx`).

### 3.2 Section 1: Non-Stationary Blind Source Separation and Joint Diagonalization

In this section, we investigate Blind Source Separation when the sources are **non-stationary** and **mutually uncorrelated**. The signals of the two sources in the $k$-th time window ($k = 1, 2, \dots, 5$) are given by:

```math
\begin{aligned}
s_1^{(k)}(t) &= c_k \sin(2\pi t), \qquad k - 1 \le t < k, \\
s_2^{(k)}(t) &= d_k \sin(4\pi t), \qquad k - 1 \le t < k,
\end{aligned}
```

where the amplitude coefficients $c_k$ and $d_k$ in the $k$-th window are taken from the $k$-th entries of the vectors:

```math
\begin{aligned}
c &= \begin{bmatrix} 0.2 & 0.4 & 0.6 & -0.1 & -0.3 \end{bmatrix}, \\
d &= \begin{bmatrix} 0.1 & 0.3 & -0.2 & 0.5 & -0.3 \end{bmatrix}.
\end{aligned}
```

Assuming a sampling frequency of $f_s = 20\ \mathrm{Hz}$ ($20$ samples per $1\ \mathrm{s}$ window, for a total of $T = 100$ samples across $0 \le t < 5$) and the mixing matrix

```math
A = \begin{bmatrix} 0.8 & -0.6 \\ 0.6 & 0.8 \end{bmatrix},
```

generate the $2 \times 100$ observation matrix $X = A S$. Keep the matrix $X$ fixed throughout all parts of Section 1.

#### Part الف (Part 1 in Code/Report) — Source and Observation Waveforms

Plot the source signals $s_1(t), s_2(t)$ and the observed mixture signals $x_1(t), x_2(t)$ over the full $5\ \mathrm{s}$ interval.

#### Part ب (Part 2 in Code/Report) — Two-Window BSS via GEVD ($K = 2$)

Now approach the problem from a BSS perspective. Using **only the data from the first two time windows** ($k = 1, 2$), estimate the source signals $\hat{S}$. Before evaluating the reconstruction quality, resolve both the **scaling (amplitude/sign) ambiguity** and the **permutation ambiguity**. Report the quality of the estimated sources using the normalized squared Frobenius error criterion:

```math
E = \frac{\left\|\hat{S} - S\right\|_F^2}{\|S\|_F^2}.
```

#### Part ج (Part 3 in Code/Report) — Multi-Window BSS via Joint Diagonalization ($K = 5$)

Repeat Part ب using the data from **all five time windows** (the entire $2 \times 100$ observation matrix $X$, $K = 5$) by formulating and solving the multi-window Joint Diagonalization problem via **Alternating Minimization**. Plot the convergence curve of the optimization objective across iterations, plot the recovered sources against the ground-truth sources, and report the reconstruction error $E$.

#### Part د — Noisy Non-Stationary BSS at $\mathrm{SNR} = 20\ \mathrm{dB}$ ($K = 5$)

Now suppose additive noise enters the observation model:

```math
Y = X + \sigma W,
```

where the scalar $\sigma > 0$ denotes the noise standard deviation (scaling factor) and $W \in \mathbb{R}^{2 \times 100}$ is a random noise matrix whose entries are drawn independently from a zero-mean, unit-variance Gaussian distribution $\mathcal{N}(0, 1)$. After generating $W$, normalize it so that its Frobenius norm equals unity ($\|W\|_F = 1$). Choose the scalar $\sigma$ such that the ratio of desired signal power to noise power is

```math
\mathrm{SNR} = \frac{\|X\|_F^2}{\sigma^2} = 100 \quad \left(\text{i.e., } \mathrm{SNR} = 20\ \mathrm{dB}\right).
```

Using the noisy $2 \times 100$ observation matrix $Y$, estimate the sources and report the estimation error $E$.

#### Part ه (Part 4 in Code) — Effect of Window Count $K \in \{2, 3, 4, 5\}$ under Noise ($\mathrm{SNR} = 20\ \mathrm{dB}$)

Consider the noisy observation model $Y \in \mathbb{R}^{2 \times 100}$ from Part د at $\mathrm{SNR} = 20\ \mathrm{dB}$:

- Solve the BSS problem for different numbers of time windows $K \in \{2, 3, 4, 5\}$ and plot/report the average error $E$ as a function of $K$ (note that Part د corresponds to the case $K = 5$).
- To obtain statistically reliable results, generate $100$ independent realizations of the noise matrix $W$, construct $Y$ on each run, apply the $K$-window Joint Diagonalization algorithm for each $K \in \{2, 3, 4, 5\}$, and average the results across the $100$ Monte Carlo runs.

#### Part ی (Part 5 in Code) — Effect of $\mathrm{SNR} \in \{5, 10, 15, 20\}\ \mathrm{dB}$ for $K = 5$ Windows

Under the same noisy model as Part د, generate noisy observations $Y$ for $\mathrm{SNR} \in \{5, 10, 15, 20\}\ \mathrm{dB}$ using all $K = 5$ time windows (the entire observation duration):

- For each $\mathrm{SNR}$ value, generate $100$ independent realizations of the noise, solve the BSS problem on each realization using all $K = 5$ windows, and report/plot the average error across the $100$ Monte Carlo runs.

### 3.3 Section 2: BSS of Temporally Correlated Stationary Sources and Spectral Comparison

In this section, two noiseless $2 \times 500$ observation matrices `X1` (`hw4-X1.mat`) and `X2` (`hw4-X2.mat`) are provided, each generated by a linear instantaneous mixture of two **jointly uncorrelated** (temporally correlated) sources without noise at a sampling rate of $f_s = 100\ \mathrm{Hz}$ ($5\ \mathrm{s}$ duration).

#### Part الف (Part 1 in Code) — Source Separation from $X_1$

Estimate and plot the time-domain source signals $\hat{s}_1(t)$ and $\hat{s}_2(t)$ from the observation matrix $X_1$.

#### Part ب (Part 2 in Code) — Fourier Transform of Sources Estimated from $X_1$ and Filtering Analysis

Plot the magnitude of the Fourier transform of the estimated source signals from $X_1$ (as well as the observations $X_1$). Report your observations. Could we have estimated the sources simply by applying a basic linear filter (e.g., bandpass/lowpass/highpass filtering) directly to the observations $X_1$?

#### Part ج (Part 3 in Code) — Source Separation from $X_2$

Estimate and plot the time-domain source signals $\hat{s}_1(t)$ and $\hat{s}_2(t)$ from the second observation matrix $X_2$ ($f_s = 100\ \mathrm{Hz}$).

#### Part د (Part 4 in Code) — Fourier Transform of Sources Estimated from $X_2$ and Filtering Analysis

Plot the magnitude of the Fourier transform of the estimated source signals from $X_2$ (and compare with the observation spectrum). Report your observations. Could we have estimated the sources simply by applying a basic linear filter directly to the observations $X_2$? Why or why not?

## 4. Mathematical Formulation & Solutions

### 4.1 Section 1: Non-Stationary BSS via GEVD and Multi-Window Joint Diagonalization (`Section1.m`)

#### Part 1 (الف) — Signal Generation and Orthogonality Across Windows

Over each $1\ \mathrm{s}$ window $k \in \{1, 2, 3, 4, 5\}$ ($k - 1 \le t < k$), the discrete time grid with sampling rate $f_s = 20\ \mathrm{Hz}$ contains $N_w = 20$ samples:

```math
t_0 = \left[0,\, \frac{1}{20},\, \frac{2}{20},\, \dots,\, \frac{19}{20}\right] \in \mathbb{R}^{1 \times 20},
\qquad
t = (k - 1) + t_0 \in \mathbb{R}^{5 \times 20}.
```

In `codes/Section1.m` (lines 12–15), the $2 \times 100$ source matrix $S$ and observation matrix $X = A S$ are constructed by multiplying the $20 \times 5$ sinusoidal matrices by the window amplitude vectors $c$ and $d$ and reshaping column-wise (`reshape(..., 1, [])`):

```math
S = \begin{bmatrix} s_1(t) \\ s_2(t) \end{bmatrix} \in \mathbb{R}^{2 \times 100},
\qquad
A = \begin{bmatrix} 0.8 & -0.6 \\ 0.6 & 0.8 \end{bmatrix},
\qquad
X = A S \in \mathbb{R}^{2 \times 100}.
```

Note two fundamental structural properties of this model:

1. **Exact orthogonality of $s_1^{(k)}(t)$ and $s_2^{(k)}(t)$ within every window $k$:** Because $\sin(2\pi t)$ ($1\ \mathrm{Hz}$) and $\sin(4\pi t)$ ($2\ \mathrm{Hz}$) are harmonics with an integer number of periods on each unit interval $[k-1, k)$, their discrete inner product over the $N_w = 20$ samples of window $k$ is identically zero:

```math
\sum_{n=0}^{19} \sin\!\left(2\pi \frac{n}{20}\right) \sin\!\left(4\pi \frac{n}{20}\right) = 0,
\qquad
\sum_{n=0}^{19} \sin^2\!\left(2\pi \frac{n}{20}\right) = \sum_{n=0}^{19} \sin^2\!\left(4\pi \frac{n}{20}\right) = \frac{N_w}{2} = 10.
```

Consequently, in the $k$-th window, the unnormalized $2 \times 2$ source correlation matrix $R_S^{(k)} = S^{(k)} \left(S^{(k)}\right)^T$ is **strictly diagonal**:

```math
R_S^{(k)} = S^{(k)} \left(S^{(k)}\right)^T
= \begin{bmatrix} 10 c_k^2 & 0 \\ 0 & 10 d_k^2 \end{bmatrix},
\qquad k = 1, 2, 3, 4, 5.
```

2. **Orthogonality of the mixing matrix $A$:** The mixing matrix $A = \begin{bmatrix} 0.8 & -0.6 \\ 0.6 & 0.8 \end{bmatrix}$ is orthogonal ($A^T A = A A^T = I_2$), so the true unmixing matrix is $B^* = A^{-1} = A^T = \begin{bmatrix} 0.8 & 0.6 \\ -0.6 & 0.8 \end{bmatrix}$, and total signal energy is preserved ($\|X\|_F^2 = \|S\|_F^2 = 10 \sum_{k=1}^5 (c_k^2 + d_k^2) = 10(0.66 + 0.48) = 11.4$).

#### Part 2 (ب) — Two-Window Non-Stationary BSS via GEVD (`Section1.m` lines 41–70)

Let $X^{(1)} = X(:, 1:20)$ and $X^{(2)} = X(:, 21:40)$ denote the observation submatrices in windows $k = 1$ and $k = 2$. Their unnormalized spatial correlation matrices are:

```math
\begin{aligned}
R_X^{(1)} &= X^{(1)} \left(X^{(1)}\right)^T = A R_S^{(1)} A^T
= A \begin{bmatrix} 0.4 & 0 \\ 0 & 0.1 \end{bmatrix} A^T
= \begin{bmatrix} 0.292 & 0.144 \\ 0.144 & 0.208 \end{bmatrix}, \\
R_X^{(2)} &= X^{(2)} \left(X^{(2)}\right)^T = A R_S^{(2)} A^T
= A \begin{bmatrix} 1.6 & 0 \\ 0 & 0.9 \end{bmatrix} A^T
= \begin{bmatrix} 1.348 & 0.336 \\ 0.336 & 1.152 \end{bmatrix}.
\end{aligned}
```

Seeking an unmixing matrix $B$ such that $B R_X^{(1)} B^T = R_S^{(1)}$ and $B R_X^{(2)} B^T = R_S^{(2)}$ are simultaneously diagonal leads directly to the **Generalized Eigenvalue Decomposition (GEVD)** problem:

```math
R_X^{(1)} B^T = R_X^{(2)} B^T \Gamma
\quad \Longleftrightarrow \quad
\left[B^T, \Gamma\right] = \mathrm{GEVD}\!\left(R_X^{(1)}, R_X^{(2)}\right),
```

where the generalized eigenvalues on the diagonal of $\Gamma$ equal the ratios of the source energies between Window 1 and Window 2:

```math
\gamma_1 = \frac{10 c_1^2}{10 c_2^2} = \left(\frac{0.2}{0.4}\right)^2 = 0.25,
\qquad
\gamma_2 = \frac{10 d_1^2}{10 d_2^2} = \left(\frac{0.1}{0.3}\right)^2 = \frac{1}{9} \approx 0.1111.
```

Because $\gamma_1 \neq \gamma_2$ (the variance profiles of the two sources change at different rates between Window 1 and Window 2), the GEVD uniquely identifies the columns of $B^T$ up to scaling and permutation. Calling `[B_T, Gamma] = eig(window1_R_x, window2_R_x)` and sorting in descending order (`Section1.m` lines 46–49, Page 2 of the report) yields:

```math
\Gamma = \begin{bmatrix} 0.25 & 0 \\ 0 & 0.1111 \end{bmatrix},
\qquad
B^T = \begin{bmatrix} -0.6325 & 0.6325 \\ -0.4743 & -0.8433 \end{bmatrix}.
```

*(Here MATLAB scales each generalized eigenvector $b_i$ so that $b_i^T R_X^{(2)} b_i = 1$, i.e., $b_1 = -[0.8, 0.6]^T / \sqrt{1.6} = [-0.6325, -0.4743]^T$ and $b_2 = [0.6, -0.8]^T / \sqrt{0.9} = [0.6325, -0.8433]^T$, whose normalized directions are the exact columns of $A = (B^*)^T$.)*

**Resolving Scaling, Sign, and Permutation Ambiguities (`Section1.m` lines 50–54):**

1. Compute the raw source estimates $\hat{S} = B X = (B^T)^T X$.
2. Normalize each row of $S$ and $\hat{S}$ to unit Euclidean ($\ell_2$) norm so that every row has unit energy:

```math
S_{\mathrm{norm}}(i, :) = \frac{S(i, :)}{\|S(i, :)\|_2},
\qquad
\hat{S}_{\mathrm{norm}}(i, :) = \frac{\hat{S}(i, :)}{\|\hat{S}(i, :)\|_2},
\qquad i = 1, 2.
```

3. Compute the $2 \times 2$ cross-correlation matrix $R_{S, \hat{S}} = S_{\mathrm{norm}} \hat{S}_{\mathrm{norm}}^T$ (`R_s_sHat = S_norm * S_hat_norm.'`) and align the signs of the rows of $\hat{S}_{\mathrm{norm}}$ via `S_hat_norm = (2*(sum(R_s_sHat) > 0) - 1).' .* S_hat_norm`.
4. Evaluate the per-source squared $\ell_2$-error vector `E1 = sum((S_hat_norm - S_norm).^2, 2)` (Page 3 of the report):

```math
E_1 = \begin{bmatrix} \left\|\hat{s}_{1,\mathrm{norm}} - s_{1,\mathrm{norm}}\right\|_2^2 \\[4pt] \left\|\hat{s}_{2,\mathrm{norm}} - s_{2,\mathrm{norm}}\right\|_2^2 \end{bmatrix}
= \begin{bmatrix} 1.7704 \times 10^{-31} \\ 4.6125 \times 10^{-32} \end{bmatrix} \approx \begin{bmatrix} 0 \\ 0 \end{bmatrix},
```

confirming exact recovery of both non-stationary sources up to double-precision machine epsilon (Figure 1.3 of the report).

#### Part 3 (ج) — Multi-Window Joint Diagonalization via Alternating Minimization (`JD` in `Section1.m`)

When $K > 2$ time windows are available ($K = 5$ in Part ج), there are $K$ spatial correlation matrices $R_X^{(k)} = X^{(k)} \left(X^{(k)}\right)^T$ ($k = 1, \dots, K$). No single two-matrix GEVD uses all $K$ matrices simultaneously, and in the presence of noise no matrix $B$ can diagonalize all $K$ matrices exactly. Instead, we seek an orthonormal unmixing matrix $B = \begin{bmatrix} b_1^T \\ b_2^T \end{bmatrix} \in \mathbb{R}^{M \times M}$ ($M = 2$, $B B^T = I_M$) that minimizes the sum of squared off-diagonal entries across all $K$ correlation matrices $B R_X^{(k)} B^T$:

```math
\hat{B} = \arg\min_{B} f(B),
\qquad
f(B) = f(b_1, b_2, \dots, b_M) = \sum_{k=1}^{K} \sum_{i=1}^{M} \sum_{\substack{j=1 \\ j \neq i}}^{M} \left( b_i^T R_X^{(k)} b_j \right)^2.
```

*(Note: On Page 3 of the Persian report, the inner factor in the first display equation is typed as $(b_i^T R_X^{(k)} b_i)^2$, while the very next line writes the exact off-diagonal quadratic expansion in $b_j$ for $j \neq i$.)*

Isolating the dependence on the $i$-th row $b_i^T$ of $B$ while holding the remaining rows $\{b_j\}_{j \neq i}$ fixed allows rewriting the partial cost $f(b_i)$ as a quadratic form:

```math
f(b_i) = b_i^T \left( \sum_{k=1}^{K} \sum_{\substack{j=1 \\ j \neq i}}^{M} \left(R_X^{(k)} b_j\right) \left(R_X^{(k)} b_j\right)^T \right) b_i
= b_i^T R_i b_i,
```

where the $M \times M$ symmetric positive semi-definite matrix $R_i$ is defined as:

```math
R_i = \sum_{k=1}^{K} \sum_{\substack{j=1 \\ j \neq i}}^{M} \left(R_X^{(k)} b_j\right) \left(R_X^{(k)} b_j\right)^T.
```

To minimize $f(B)$ over orthonormal matrices $B$, the helper function `JD(Bhat, X, S_norm, k)` in `codes/Section1.m` (lines 183–218) executes `iterNum = 100` sweeps of **Alternating Minimization**:

1. **Random orthonormal initialization (`Section1.m` line 73):** Initialize `Bhat = orth(randn(observations_num))` $\in \mathbb{R}^{2 \times 2}$.
2. **3D tensor partitioning of windows (`lines 184–186`):** Reshape $X \in \mathbb{R}^{2 \times 100}$ into a $2 \times 20 \times 5$ tensor `X_paged = reshape(X, 2, [], 5)`, retain the first $k$ windows `X_paged = X_paged(:, :, 1:k)`, and compute all $k$ window correlation matrices in parallel via `R_x = pagemtimes(X_paged, pagetranspose(X_paged))` $\in \mathbb{R}^{2 \times 2 \times k}$.
3. **Alternating row updates (`lines 191–209`):** In each iteration `itr = 1:100`, loop over row index `i = 1:M`:
   - Select the complementary row indices $j \neq i$ (`b_j = Bhat(j, :).'`) and form $R_i$ via page-wise matrix multiplication:

```matlab
Rb  = pagemtimes(R_x, b_j);
R_i = sum(pagemtimes(Rb, pagetranspose(Rb)), 3);
```

   - Compute `[U, ~] = eig(R_i)` and set `Bhat(i, :) = U(:, 1).'`, which is the unit-norm eigenvector corresponding to the **smallest eigenvalue** $\lambda_{\min}(R_i)$ (since MATLAB's single-matrix `eig` sorts eigenvalues in ascending order for symmetric matrices).
   - For $i > 1$, project `Bhat(i, :)` onto the orthogonal complement of the previously updated rows `BB = Bhat(1:i-1, :).'` and re-normalize to unit norm (`lines 201–205`):

```math
\tilde{b}_i = \left(I_M - B_{1:i-1}^T B_{1:i-1}\right) b_i,
\qquad
b_i \leftarrow \frac{\tilde{b}_i}{\|\tilde{b}_i\|_2}.
```

   - Accumulate the objective value $f = \sum_{i=1}^M b_i^T R_i b_i$ into `func(itr)`.
4. **Permutation and sign alignment (`lines 210–217`):** Compute `S_hat = Bhat * X`, normalize rows to unit $\ell_2$-norm, swap the two rows of `S_hat_norm` if $|R_{S,\hat{S}}(1,1)| < |R_{S,\hat{S}}(1,2)|$, align row signs, and return `E = sum((S_hat_norm - S_norm).^2, 2)`.

In the noiseless $K = 5$ case (Part ج, Page 4 of the report), `func(itr)` drops from its initial value ($\approx 4.9$) to zero within the first $3$ iterations (Figure 1.5), and the final per-source squared error vector reaches machine precision:

```math
E_2 = \begin{bmatrix} 2.0096 \times 10^{-32} \\ 1.4412 \times 10^{-32} \end{bmatrix}.
```

#### Parts 4 & 5 (د، ه، ی) — Noisy Non-Stationary BSS and Monte Carlo Evaluation (`Section1.m` lines 95–180)

For a target signal-to-noise ratio $\mathrm{SNR}_{\mathrm{dB}}$, the linear SNR is

```math
\mathrm{SNR}_{\mathrm{linear}} = 10^{\mathrm{SNR}_{\mathrm{dB}} / 10}.
```

In `codes/Section1.m` (lines 98–99 and 103–105), the signal energy is computed row-wise (`power_X = sum(X.^2, 2)` $= [5.9520,\ 5.4480]^T$), each row of the Gaussian noise matrix $W \in \mathbb{R}^{2 \times 100}$ is normalized to unit $\ell_2$-norm (`W = W ./ sqrt(sum(W.^2, 2))`), and the row-wise noise standard deviation vector is set to

```math
\sigma_i = \sqrt{\frac{\|X(i, :)\|_2^2}{\mathrm{SNR}_{\mathrm{linear}}}}, \quad i = 1, 2,
\qquad
Y = X + \mathrm{diag}(\sigma_1, \sigma_2)\, W.
```

*(Note that because $\|W(i,:)\|_2^2 = 1$ for $i = 1, 2$, the total squared Frobenius norm of the additive noise is $\|\mathrm{diag}(\sigma_1, \sigma_2) W\|_F^2 = \sigma_1^2 + \sigma_2^2 = (\|X(1,:)\|_2^2 + \|X(2,:)\|_2^2)/\mathrm{SNR}_{\mathrm{linear}} = \|X\|_F^2 / \mathrm{SNR}_{\mathrm{linear}}$, so this construction simultaneously satisfies the overall Frobenius SNR relation $\mathrm{SNR} = \|X\|_F^2 / \|\text{Noise}\|_F^2$ as well as equal per-channel SNR.)*

- **Part 4 in Code (Parts د & ه on the Assignment Sheet, Pages 5 of the report):** Fixes $\mathrm{SNR} = 20\ \mathrm{dB}$ ($\mathrm{SNR}_{\mathrm{linear}} = 100$) and runs `JD(Bhat, Y, S_norm, k)` for $k \in \{2, 3, 4, 5\}$ across `runNum = 1:100` Monte Carlo realizations, plotting the 100-run average Alternating Minimization cost curves `func(:, :, 1:4)` in a $2 \times 2$ subplot grid (Figure 1.6) and recording the average error vector $E$.
- **Part 5 in Code (Part ی on the Assignment Sheet, Pages 5–6 of the report):** Sweeps $\mathrm{SNR}_{\mathrm{dB}} \in \{5, 10, 15, 20\}\ \mathrm{dB}$ by reshaping the $2 \times 4$ standard deviation matrix into a $2 \times 1 \times 4$ tensor (`reshape(sDeviation, 2, [], 4)`) to form the $2 \times 100 \times 4$ noisy observation tensor $Y$, runs `JD(Bhat, Y(:, :, m), S_norm, 5)` with $K = 5$ windows across $100$ Monte Carlo trials, and plots the 100-run average cost trajectories in Figure 1.7.

### 4.2 Section 2: Temporally Correlated BSS via Time-Lagged GEVD (`Section2.m`)

#### Theoretical Comparison: Non-Stationary BSS (Section 1) vs. Temporally Correlated BSS (Section 2)

As summarized in the introduction to Section 2 of the Persian report (Page 7):

- **Section 1 (Uncorrelated Non-Stationary Sources):** Assumes zero lag cross-correlation $\mathbb{E}\{s_1(t) s_2(t)\} = 0$ (and $\sum_t s_1(t) s_2(t) = 0$ within each window) while source variances change across windows $k = 1, \dots, K$, yielding diagonal instantaneous correlation matrices $R_S^{(k)}$ at lag $\tau = 0$.
- **Section 2 (Jointly Uncorrelated Stationary Sources with Temporal Structure):** Assumes the sources are mutually uncorrelated **at all time lags $\tau$**:

```math
\mathbb{E}\left\{s_1(t)\, s_2(t - \tau)\right\} = 0 \qquad \forall \tau,
```

so that the time-lagged source correlation matrix $R_S(\tau) = \mathbb{E}\{S(t) S^T(t - \tau)\}$ (approximated over finite samples by $S\, \mathrm{circshift}(S, \tau)^T$) is **diagonal for every lag $\tau$**. Under the noiseless mixing model $X(t) = A S(t)$ with unmixing matrix $B = A^{-1}$, the time-lagged observation correlation matrix satisfies:

```math
R_X(\tau) = A R_S(\tau) A^T
\quad \Longleftrightarrow \quad
R_S(\tau) = B R_X(\tau) B^T.
```

Evaluating $R_X(\tau)$ at two distinct time lags $\tau_1$ and $\tau_2$:

```math
\begin{cases}
R_S(\tau_1) = B R_X(\tau_1) B^T, \\
R_S(\tau_2) = B R_X(\tau_2) B^T,
\end{cases}
\quad \stackrel{\mathrm{GEVD}\left(R_X(\tau_1),\, R_X(\tau_2)\right)}{\Longrightarrow} \quad
\left[B^T, \Gamma\right] = \mathrm{eig}\!\left(R_X(\tau_1), R_X(\tau_2)\right),
```

recovers the unmixing matrix $B$ in closed form whenever the autocorrelation ratios $R_{s_1}(\tau_1)/R_{s_1}(\tau_2)$ and $R_{s_2}(\tau_1)/R_{s_2}(\tau_2)$ are distinct.

#### Parts 1 & 2 (الف & ب) — Source Separation and Spectral Analysis on $X_1$ (`Section2.m` lines 9–57)

For the first dataset `hw4-X1.mat` ($X_1 \in \mathbb{R}^{2 \times 500}$, $f_s = 100\ \mathrm{Hz}$, $T = 500$ samples over $5\ \mathrm{s}$), `codes/Section2.m` (lines 10–19) selects the two shift amounts $\tau_1 = f_s / 4 = 25$ and $\tau_2 = f_s / 5 = 20$, forms `Rx_11` and `Rx_12`, solves `[B_T, Gamma] = eig(Rx_11, Rx_12)`, sorts the eigenvalues in descending order, and normalizes the rows of $\hat{S}_1 = B X_1$ to unit $\ell_2$-norm:

```math
\Gamma = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix},
\qquad
B^T = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix},
\qquad
\hat{S}_1 = \mathrm{row\_normalize}\!\left(B X_1\right).
```

*(See Section 5.3 for a detailed note on `circshift(X1, fs/4, 1)` versus `circshift(X1, fs/4, 2)`: shifting along dimension 1 by $25$ and $20$ swaps/preserves the two rows of $X_1$, yielding the sum-and-difference matrix $B^T = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}$, whereas shifting along the time axis dimension 2 yields the exact unmixing eigenvectors that completely zero out the cross-tone.)*

Taking the 500-point centered Discrete Fourier Transform (`fftshift(fft(...))` over $f \in [-50, 50)\ \mathrm{Hz}$ with resolution $\Delta f = f_s / N = 0.2\ \mathrm{Hz}$):

- **Spectrum of the observations $X_1$ (Figure 2.3, Page 9):** Both sensor channels $X_1(1, :)$ and $X_1(2, :)$ consist exclusively of two sinusoidal tones at **$f = \pm 2\ \mathrm{Hz}$** and **$f = \pm 8\ \mathrm{Hz}$**:
  - $|X_1(1, f)|$: peaks of height $79.6913$ at $f = \pm 2\ \mathrm{Hz}$ and $108.398$ at $f = \pm 8\ \mathrm{Hz}$.
  - $|X_1(2, f)|$: peaks of height $326.922$ at $f = \pm 2\ \mathrm{Hz}$ and $85.6561$ at $f = \pm 8\ \mathrm{Hz}$.
- **Spectrum of the estimated sources $\hat{S}_1$ (Figure 2.2, Page 8):**
  - $|\hat{S}_1(1, f)|$: dominant peak of height $15.7449$ at $f = \pm 2\ \mathrm{Hz}$ (and residual peak $1.44832$ at $f = \pm 8\ \mathrm{Hz}$).
  - $|\hat{S}_1(2, f)|$: peaks of height $14.2696$ at $f = \pm 2\ \mathrm{Hz}$ and $6.81011$ at $f = \pm 8\ \mathrm{Hz}$.
- **Answer to the filtering question (Part ب):** Because the two underlying sources in $X_1$ are single-tone sinusoids at two distinct, non-overlapping frequencies ($2\ \mathrm{Hz}$ and $8\ \mathrm{Hz}$) and the mixing model $X_1 = A S_1$ is linear and time-invariant, **yes — the two sources in $X_1$ can also be separated directly by applying simple linear frequency-selective filters** (a lowpass/bandpass filter centered at $2\ \mathrm{Hz}$ to extract the $2\ \mathrm{Hz}$ source and a highpass/bandpass filter centered at $8\ \mathrm{Hz}$ to extract the $8\ \mathrm{Hz}$ source) or directly from the Dirac delta peak amplitudes in the Fourier domain.

#### Parts 3 & 4 (ج & د) — Source Separation and Spectral Analysis on $X_2$ (`Section2.m` lines 58–105)

Applying the same procedure in `codes/Section2.m` (lines 59–80) to the second dataset `hw4-X2.mat` ($X_2 \in \mathbb{R}^{2 \times 500}$, $f_s = 100\ \mathrm{Hz}$) again produces $\Gamma = \mathrm{diag}(1, -1)$ and $B^T = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}$, with the estimated source waveforms $\hat{S}_2$ shown in Figure 2.4 (Page 10 of the report).

Inspecting the DFT magnitude spectra in Figures 2.5 and 2.6 (Pages 10–11 of the report):

- **Spectrum of the estimated sources $\hat{S}_2$ (Figure 2.5, Page 10):** The estimated sources contain **four distinct frequency components** at **$f \in \{\pm 1,\ \pm 2,\ \pm 4,\ \pm 8\}\ \mathrm{Hz}$** (with data-tip callouts on Figure 2.5 showing $|\hat{S}_1(f)| = 9.6338$ at $\pm 2\ \mathrm{Hz}$ and $5.67361$ at $\pm 8\ \mathrm{Hz}$, and $|\hat{S}_2(f)| = 4.28827$ at $\pm 2\ \mathrm{Hz}$ and $10.3252$ at $\pm 8\ \mathrm{Hz}$, alongside the peaks at $\pm 1\ \mathrm{Hz}$ and $\pm 4\ \mathrm{Hz}$).
- **Why simple frequency filtering cannot generally replace BSS on multi-frequency / overlapping mixtures (Part د):**
  1. **Report analysis (Page 11):** In the Persian report, comparing Figure 2.5 ($1, 2, 4, 8\ \mathrm{Hz}$) against Figure 2.6 (which displayed only $2\ \mathrm{Hz}$ and $8\ \mathrm{Hz}$ because `Section2.m` line 94 plotted `X1_fft` a second time instead of `X2_fft`), the report notes that if a linear combination $A$ cancels or alters specific frequency components in a single sensor observation—or when each source itself comprises multiple frequency tones—one cannot simply read off individual source waveforms from a single observation's Fourier transform via naive single-band filtering.
  2. **True spectral structure of `hw4-X2.mat`:** In `hw4-X2.mat`, each true underlying source is a **two-tone signal** (one source composed of $1\ \mathrm{Hz} + 2\ \mathrm{Hz}$ and the other composed of $4\ \mathrm{Hz} + 8\ \mathrm{Hz}$). Without prior knowledge of which frequencies belong to the same physical source (or if the two sources shared overlapping frequency components), a simple bandpass filter in the Fourier domain cannot know how to group the four spectral peaks $\{1, 2, 4, 8\}\ \mathrm{Hz}$ into the two true sources nor what relative amplitude/phase weights to assign them, whereas time-lagged GEVD separates the two multi-tone sources blindly in a single spatial unmixing step.

## 5. Results & Discussion

### 5.1 Section 1 — Summary of Non-Stationary BSS Results (`Section1.m`)

The table below summarizes the analytical and numerical results across all five parts of Section 1 as documented in `BSS_HW#4_810100205_Arman_Majidi.pdf`:

| Problem Part | Method & Settings | Window Count $K$ | Noise Level ($\mathrm{SNR}$) | Reported Matrices / Error Vector $`E = \begin{bmatrix} E_1 \\ E_2 \end{bmatrix}`$ | Key Observation |
| --- | --- | --- | --- | --- | --- |
| Part 1 (الف) | Forward mixing $X = AS$ | $K = 5$ ($T = 100$) | Noiseless ($\infty$) | $`\|S\|_F^2 = \|X\|_F^2 = 11.4`$ (`power_X = [5.952, 5.448]^T`) | Piecewise sinusoidal bursts in $5$ windows of $20$ samples each (Figures 1.1 & 1.2). |
| Part 2 (ب) | Two-window GEVD (`eig(R_x1, R_x2)`) | $K = 2$ (Windows 1 & 2) | Noiseless ($\infty$) | $`\Gamma = \begin{bmatrix} 0.25 & 0 \\ 0 & 0.1111 \end{bmatrix}`$, $`B^T = \begin{bmatrix} -0.6325 & 0.6325 \\ -0.4743 & -0.8433 \end{bmatrix}`$, $`E = \begin{bmatrix} 1.7704 \times 10^{-31} \\ 4.6125 \times 10^{-32} \end{bmatrix}`$ | Exact closed-form recovery from two windows up to machine precision (Figure 1.3). |
| Part 3 (ج) | Joint Diagonalization via Alternating Minimization (`JD`) | $K = 5$ (All windows) | Noiseless ($\infty$) | $`E = \begin{bmatrix} 2.0096 \times 10^{-32} \\ 1.4412 \times 10^{-32} \end{bmatrix}`$ ($100$ AM iterations) | Off-diagonal cost $f(B)$ converges to $0$ in $\sim 3$ iterations (Figures 1.4 & 1.5). |
| Part 4 (د & ه) | Monte Carlo Joint Diagonalization ($100$ runs) | $K = 2$ | $20\ \mathrm{dB}$ | $`E = \begin{bmatrix} 0.0106 \\ 0.0145 \end{bmatrix}`$ | Rapid cost convergence within $< 5$ iterations (Figure 1.6, top-left). |
| Part 4 (د & ه) | Monte Carlo Joint Diagonalization ($100$ runs) | $K = 3, 4, 5$ | $20\ \mathrm{dB}$ | $`E = \begin{bmatrix} 0.0107 \\ 0.0146 \end{bmatrix}`$ (for $K = 3, 4, 5$) | Low reconstruction error ($\approx 1.07\%\text{–}1.46\%$) at $\mathrm{SNR} = 20\ \mathrm{dB}$ (Figure 1.6). |
| Part 5 (ی) | Monte Carlo Joint Diagonalization ($100$ runs) | $K = 5$ | $5\ \mathrm{dB}$ | $`E = \begin{bmatrix} 0.3748 \\ 0.4309 \end{bmatrix}`$ | Higher noise floor increases estimation error (Figure 1.7, top-left). |
| Part 5 (ی) | Monte Carlo Joint Diagonalization ($100$ runs) | $K = 5$ | $10, 15, 20\ \mathrm{dB}$ | $`E = \begin{bmatrix} 0.3758 \\ 0.4318 \end{bmatrix}`$ ($10\ \mathrm{dB}$), $`\begin{bmatrix} 0.3751 \\ 0.4314 \end{bmatrix}`$ ($15\ \mathrm{dB}$), $`\begin{bmatrix} 0.3749 \\ 0.4310 \end{bmatrix}`$ ($20\ \mathrm{dB}$) | See Section 5.3 for accumulator note on `E_list` in `Section1.m` lines 110–113 and 153–156. |

### 5.2 Section 2 — Summary of Temporally Correlated BSS & Spectral Analysis (`Section2.m`)

The table below summarizes the GEVD matrices and DFT spectral peaks ($N = 500$, $f_s = 100\ \mathrm{Hz}$, frequency resolution $\Delta f = 0.2\ \mathrm{Hz}$) reported in Section 2 of the Persian report:

| Dataset & Part | Signal Analyzed | Reported GEVD Matrices ($\Gamma$, $B^T$) | Active Frequencies ($\mathrm{Hz}$) | Reported FFT Peak Magnitudes at $f = \pm 2\ \mathrm{Hz}$ and $f = \pm 8\ \mathrm{Hz}$ | Separability by Simple Linear Filtering |
| --- | --- | --- | --- | --- | --- |
| `hw4-X1.mat` — Parts 1 & 2 (الف & ب) | Estimated sources $\hat{S}_1$ (Figures 2.1 & 2.2) | $`\Gamma = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}`$, $`B^T = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}`$ | $\pm 2\ \mathrm{Hz},\ \pm 8\ \mathrm{Hz}$ | $\|\hat{S}_1(1, f)\|$: $15.7449$ ($\pm 2\ \mathrm{Hz}$), $1.44832$ ($\pm 8\ \mathrm{Hz}$)<br>$\|\hat{S}_1(2, f)\|$: $14.2696$ ($\pm 2\ \mathrm{Hz}$), $6.81011$ ($\pm 8\ \mathrm{Hz}$) | **Yes** — Each source is a single sinusoid at a distinct frequency ($2\ \mathrm{Hz}$ vs. $8\ \mathrm{Hz}$). |
| `hw4-X1.mat` — Part 2 (ب) | Observations $X_1$ (Figure 2.3) | — | $\pm 2\ \mathrm{Hz},\ \pm 8\ \mathrm{Hz}$ | $\|X_1(1, f)\|$: $79.6913$ ($\pm 2\ \mathrm{Hz}$), $108.398$ ($\pm 8\ \mathrm{Hz}$)<br>$\|X_1(2, f)\|$: $326.922$ ($\pm 2\ \mathrm{Hz}$), $85.6561$ ($\pm 8\ \mathrm{Hz}$) | Same two frequencies ($\pm 2, \pm 8\ \mathrm{Hz}$) appear in $X_1$. |
| `hw4-X2.mat` — Parts 3 & 4 (ج & د) | Estimated sources $\hat{S}_2$ (Figures 2.4 & 2.5) | $`\Gamma = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}`$, $`B^T = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}`$ | $\pm 1,\ \pm 2,\ \pm 4,\ \pm 8\ \mathrm{Hz}$ | $\|\hat{S}_2(1, f)\|$: $9.6338$ ($\pm 2\ \mathrm{Hz}$), $5.67361$ ($\pm 8\ \mathrm{Hz}$)<br>$\|\hat{S}_2(2, f)\|$: $4.28827$ ($\pm 2\ \mathrm{Hz}$), $10.3252$ ($\pm 8\ \mathrm{Hz}$) (plus peaks at $\pm 1, \pm 4\ \mathrm{Hz}$) | **No** — Sources contain multiple tones ($\{1, 2\}\ \mathrm{Hz}$ and $\{4, 8\}\ \mathrm{Hz}$); naive filtering cannot group unknown multi-tone components. |
| `hw4-X2.mat` — Part 4 (د) | Observation spectrum plot (Figure 2.6) | — | Plotted `X1_fft` ($\pm 2, \pm 8\ \mathrm{Hz}$); true `X2` has $\pm 1, \pm 2, \pm 4, \pm 8\ \mathrm{Hz}$ | Figure 2.6 repeats $\|X_1(f)\|$; true $\|X_2(1, f)\|$ has $134.42$ ($\pm 1, \pm 2\ \mathrm{Hz}$) & $564.71$ ($\pm 4, \pm 8\ \mathrm{Hz}$), and $\|X_2(2, f)\|$ has $458.47$ ($\pm 1, \pm 2\ \mathrm{Hz}$) & $215.54$ ($\pm 4, \pm 8\ \mathrm{Hz}$) | See Section 5.3 for note on `Section2.m` line 94. |

### 5.3 Notes on the Submitted Solutions / Implementation Notes

- **Correspondence Between Assignment Sheet Lettering (الف–ی) and Script Sections (`Part 1`–`Part 5`) in `Section1.m`:**
  - The Persian assignment sheet (`hw4.pdf`) divides Section 1 into six lettered parts: **الف**, **ب**, **ج**, **د**, **ه**, and **ی**. Because Part **د** asks for the noisy reconstruction error at $\mathrm{SNR} = 20\ \mathrm{dB}$ with $K = 5$ windows and Part **ه** immediately asks to evaluate $\mathrm{SNR} = 20\ \mathrm{dB}$ across $K \in \{2, 3, 4, 5\}$ (explicitly noting on the sheet that Part د is the $K = 5$ case of Part ه), `codes/Section1.m` combines Parts د and ه into `%% Part 4` (labeled `قسمت ه` on Page 5 of the report) and implements Part ی as `%% Part 5` (labeled `قسمت ی` on Page 5 of the report).
- **Per-Source Unit-Norm Error Vector $E$ vs. Overall Scalar Frobenius Error:**
  - In `Section1.m` (lines 51–52, 69, 211, and 217), each row of $S$ and $\hat{S}$ is normalized to unit $\ell_2$-norm (`S_norm = S ./ sqrt(sum(S.^2, 2))`), and `E` is computed row-wise as the $2 \times 1$ vector `E = sum((S_hat_norm - S_norm).^2, 2)`. Because $\|S_{\mathrm{norm}}\|_F^2 = \|s_{1,\mathrm{norm}}\|_2^2 + \|s_{2,\mathrm{norm}}\|_2^2 = 1 + 1 = 2$, the scalar relative Frobenius error defined on the assignment sheet equals the average of the two components of `E`: $E_{\mathrm{scalar}} = (E(1) + E(2)) / 2$.
- **Monte Carlo Accumulator Indexing (`E_list`) and Plotted Metric (`func`) in `Section1.m` (Parts 4 & 5):**
  - In `Section1.m` lines 110–113 (Part 4) and lines 153–156 (Part 5), the Monte Carlo averaging lines for indices $2, 3, 4$ are written as:

```matlab
E_list(:, :, 1) = E_list(:, :, 1) + E_1 ./ 100;
E_list(:, :, 2) = E_list(:, :, 1) + E_2 ./ 100;
E_list(:, :, 3) = E_list(:, :, 1) + E_3 ./ 100;
E_list(:, :, 4) = E_list(:, :, 1) + E_4 ./ 100;
```

  - Because the right-hand side of slices `2`, `3`, and `4` references `E_list(:, :, 1)` instead of `E_list(:, :, 2)`, `E_list(:, :, 3)`, and `E_list(:, :, 4)`, at the end of the $100$-run loop (`runNum = 100`), `E_list(:, :, m)` ($m = 2, 3, 4$) equals the 99-run cumulative sum of `E_1 ./ 100` plus the 100th run's `(E_1 + E_m) ./ 100`. This is why the four reported $2 \times 1$ vectors in Part 4 ($[0.0106, 0.0145]^T$ for $k=2$ and $[0.0107, 0.0146]^T$ for $k=3,4,5$) and in Part 5 ($\approx [0.375, 0.431]^T$, anchored by the $\mathrm{SNR} = 5\ \mathrm{dB}$ slice `E_1`) are nearly identical across all four cases.
  - In addition, inside `JD` (lines 212–216), when a permutation swap is triggered on line 214 (`S_hat_norm = [S_hat_norm(2, :); S_hat_norm(1, :)]`), `R_s_sHat` is not recomputed before the sign alignment on line 216 (`S_hat_norm = (2*(sum(R_s_sHat) > 0) - 1).' .* S_hat_norm`), so in noisy runs where the dominant and secondary column signs of `R_s_sHat` differ, an occasional sign flip can contribute $4.0$ to that run's squared error.
  - Meanwhile, the subplot figures in Parts 4 and 5 (Figures 1.6 and 1.7 of the report) plot the 100-run average of the Alternating Minimization off-diagonal cost trajectory `func(itr)` ($f(B) = \sum_{i=1}^2 b_i^T R_i b_i$) versus iteration index `itr = 1:100` for each $k \in \{2, 3, 4, 5\}$ and each $\mathrm{SNR} \in \{5, 10, 15, 20\}\ \mathrm{dB}$.
- **Dimension Argument of `circshift` and Variable Reuse in `Section2.m`:**
  - **Dimension argument in `circshift` (lines 10–11 and 59–60):** Since `X1` and `X2` have size $2 \times 500$ ($2$ sensor rows by $500$ time columns), calling `circshift(X1, fs/4, 1)` and `circshift(X1, fs/5, 1)` applies the circular shift along dimension `1` (the $2$-element row dimension) rather than dimension `2` (the $500$-sample time dimension). Because $\tau_1 = f_s/4 = 25$ is odd and $\tau_2 = f_s/5 = 20$ is even, shifting a 2-row matrix along dimension `1` by $25$ swaps its two rows (`X1_shifted1 = [X1(2,:); X1(1,:)]`), whereas shifting by $20$ leaves `X1_shifted2 = X1` unchanged. Consequently, `Rx_11` and `Rx_12` are row-permuted versions of $X_1 X_1^T$, whose generalized eigenvectors in `[B_T, Gamma] = eig(Rx_11, Rx_12)` are always $B^T = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}$ with eigenvalues $\Gamma = \mathrm{diag}(1, -1)$. Also on lines 61–62, `Rx_21 = X1 * X2_shifted1.'` and `Rx_22 = X1 * X2_shifted2.'` multiply `X1` on the left instead of `X2`.
  - **Behavior when shifting along the time axis (`circshift(..., 2)`):** If `circshift(..., 2)` is used along the temporal dimension (and `X2` is used on lines 61–62), time-lagged GEVD yields $\Gamma = \mathrm{diag}(1.2361, -1.2361)$ and unit-norm columns $B^T = \begin{bmatrix} -0.6200 & 0.9716 \\ -0.7846 & 0.2368 \end{bmatrix}$ for $X_1$ (separating the pure $2\ \mathrm{Hz}$ tone and the pure $8\ \mathrm{Hz}$ tone with zero cross-talk), and $\Gamma = \mathrm{diag}(2, -4)$ for $X_2$ (cleanly separating the two-tone $\{1, 2\}\ \mathrm{Hz}$ source from the two-tone $\{4, 8\}\ \mathrm{Hz}$ source).
  - **Duplicate `X1_fft` plot in Section 2, Part 4 (lines 94–105):** On line 94 of `Section2.m`, `X1_fft = [fftshift(fft(X1(1,:))) ; fftshift(fft(X1(2,:)))]` is recomputed and plotted from `X1` instead of `X2`. As a result, Figure 2.6 on Page 11 of the report displays the spectrum of $X_1$ (peaks only at $\pm 2\ \mathrm{Hz}$ and $\pm 8\ \mathrm{Hz}$) rather than the spectrum of $X_2$ (which actually contains peaks at all four frequencies $\pm 1, \pm 2, \pm 4, \pm 8\ \mathrm{Hz}$).

## 6. How to Use

### 6.1 Running the MATLAB Scripts

Open MATLAB, navigate to the `codes` directory (so that `hw4-X1.mat` and `hw4-X2.mat` are in the current working directory), and run `Section1` and `Section2`:

```matlab
cd codes
Section1
Section2
```

- **Dependencies:**
  - `Section1.m` and `Section2.m` require only base MATLAB functions (`eig`, `sort`, `orth`, `randn`, `circshift`, `fft`, `fftshift`, `reshape`, `plot`, `subplot`) together with MATLAB's built-in page-wise tensor operations `pagemtimes` and `pagetranspose` (introduced in MATLAB R2020b). No external toolboxes are required.
- **Generated Figures:**
  - `Section1.m` opens **7 figures**:
    1. Two-panel subplot of the noiseless non-stationary source signals $s_1(t)$ and $s_2(t)$ across $t \in [0, 5)\ \mathrm{s}$ (Figure 1.1 in the report).
    2. Two-panel subplot of the noiseless observed mixtures $x_1(t)$ and $x_2(t)$ (Figure 1.2 in the report).
    3. Two-panel overlay of the normalized true sources $S_{\mathrm{norm}}$ (magenta solid line) and two-window GEVD estimates $\hat{S}_{\mathrm{norm}}$ (black dots) for Part 2 (Figure 1.3 in the report).
    4. Convergence trajectory of the Alternating Minimization objective `func` over $100$ iterations for Part 3 (Figure 1.5 in the report).
    5. Two-panel overlay of $S_{\mathrm{norm}}$ and the 5-window Joint Diagonalization estimates $\hat{S}_{\mathrm{norm}}$ for Part 3 (Figure 1.4 in the report).
    6. $2 \times 2$ subplot grid of the 100-run average Alternating Minimization cost curves for $k \in \{2, 3, 4, 5\}$ at $\mathrm{SNR} = 20\ \mathrm{dB}$ in Part 4 (Figure 1.6 in the report).
    7. $2 \times 2$ subplot grid of the 100-run average Alternating Minimization cost curves for $\mathrm{SNR} \in \{5, 10, 15, 20\}\ \mathrm{dB}$ with $K = 5$ in Part 5 (Figure 1.7 in the report).
  - `Section2.m` opens **6 figures**:
    1. Two-panel time-domain plot of the estimated sources $\hat{s}_1(t), \hat{s}_2(t)$ from `X1` (Figure 2.1 in the report).
    2. Two-panel FFT magnitude spectrum $|\hat{S}_1(f)|, |\hat{S}_2(f)|$ of the sources estimated from `X1` over $f \in [-50, 50)\ \mathrm{Hz}$ (Figure 2.2 in the report).
    3. Two-panel FFT magnitude spectrum $|X_1(f)|$ of the observations in `X1` (Figure 2.3 in the report).
    4. Two-panel time-domain plot of the estimated sources $\hat{s}_1(t), \hat{s}_2(t)$ from `X2` (Figure 2.4 in the report).
    5. Two-panel FFT magnitude spectrum $|\hat{S}_1(f)|, |\hat{S}_2(f)|$ of the sources estimated from `X2` (Figure 2.5 in the report).
    6. Two-panel FFT magnitude spectrum of the observations (Figure 2.6 in the report).

### 6.2 Expected Workspace Outputs (`Section1.m` and `Section2.m`)

Because all statements in `Section1.m` and `Section2.m` are terminated with semicolons, neither script prints text directly to the Command Window. After running `Section1.m`, inspect the deterministic Part 2 variables and the stochastic Part 3–5 variables in the MATLAB Command Window:

```text
>> Gamma
Gamma =
    0.2500         0
         0    0.1111

>> B_T
B_T =
   -0.6325    0.6325
   -0.4743   -0.8433

>> E1
E1 =
   1.0e-30 *
    0.1770
    0.0461
```

After running `Section2.m`, the deterministic workspace variables `Gamma` and `B_T` evaluate to:

```text
>> Gamma
Gamma =
     1     0
     0    -1

>> B_T
B_T =
     1     1
     1    -1
```

### 6.3 Reading the Theoretical Derivations and Report

1. Use **Section 3** for the complete English translation of the assignment problems and **Section 4** for the step-by-step mathematical derivations of non-stationary GEVD, multi-window Joint Diagonalization via Alternating Minimization, and time-lagged GEVD.
2. Use **Section 5** to review the quantitative results, spectral peak tables, and implementation notes alongside the original Persian report [BSS\_HW#4\_810100205\_Arman\_Majidi.pdf](./BSS_HW%234_810100205_Arman_Majidi.pdf) and assignment sheet [hw4.pdf](./hw4.pdf).

## 7. References

1. Akhavan, S. *Blind Source Separation: Homework 4*. Course assignment sheet and lecture notes, University of Tehran.
2. Souloumiac, A. (1995). Blind source detection and separation using second order non-stationarity. *Proceedings of IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)*, 1912–1915.
3. Tong, L., Soon, V. C., Huang, Y. F., & Liu, R. (1990). AMUSE: a new blind identification algorithm. *Proceedings of IEEE International Symposium on Circuits and Systems (ISCAS)*, 1784–1787.
4. Belouchrani, A., Abed-Meraim, K., Cardoso, J.-F., & Moulines, E. (1997). A blind source separation technique using second-order statistics. *IEEE Transactions on Signal Processing*, 45(2), 434–444.
5. Comon, P., & Jutten, C. (Eds.). (2010). *Handbook of Blind Source Separation: Independent Component Analysis and Applications*. Academic Press.
