# Homework 7: Single-Channel and Multi-Channel Sparse Blind Deconvolution in Time and Frequency Domains

**Course:** Blind Source Separation

**Instructor:** Dr. Saeed Akhavan

**Author:** Arman Majidi

**Student ID:** 810100205

**University:** University of Tehran

**Semester:** Spring 2024-2025

## 1. Overview & Assignment Structure

This folder contains the **English problem statements, mathematical derivations, MATLAB implementations, and simulation results for Homework 7**. The assignment consists of one comprehensive four-part section (`Section1.m`) investigating **Single-Channel Sparse Blind Deconvolution** and **Multi-Channel Sparse Blind Deconvolution** for spiky/impulsive time series in both the **time domain** and the **frequency domain**:

1. **Section 1, Part 1 — Time-Domain Single-Channel Sparse Blind Deconvolution on Noisy Observation `x1`:** Visually inspects the single-channel noisy observation $x_1(t)$ ($T = 2000$ samples, `hw7.mat`) and applies an alternating minimization algorithm (`Single_Channel__Blind_Deconvolution.m`) over $100$ iterations to jointly estimate the unit-norm spike waveform $\hat{s}(t)$ of length $L = 100$ samples and the sparse $K = 5$-impulse comb time series $\hat{\Psi}(t)$ (non-negative amplitudes $\alpha_k > 0$ separated by at least $\vert{}\tau_i - \tau_j\vert{} > L$), recovering the underlying sinusoidal-like action potential waveform and all $5$ occurrence times.

2. **Section 1, Part 2 — Time-Domain Single-Channel Sparse Blind Deconvolution on Noisy Observation `x2`:** Repeats the single-channel sparse blind deconvolution procedure on the second single-channel noisy signal $x_2(t)$ ($T = 2000$, $L = 100$, $K = 5$), recovering a positive triangular waveform $\hat{s}(t)$ and its $5$ sparse impulse locations and amplitudes, and comparing the reconstructed noiseless convolution $\hat{x}_2(t) = (\hat{s} * \hat{\Psi})(t)$ against $x_2(t)$.

3. **Section 1, Part 3 — Time-Domain Multi-Channel Sparse Blind Deconvolution on Two-Channel Mixture `X`:** Extends the single-channel algorithm to a two-channel instantaneous linear mixture $X = A B \in \mathbb{R}^{2 \times 2000}$ where each underlying source $b_n(t) = (s_n * \Psi_n)(t)$ ($n = 1, 2$) is the convolution of a distinct waveform $s_n(t)$ ($L = 100$) with a $K = 5$-spike comb train $\Psi_n(t)$. Uses a nested alternating minimization loop (`iterNum = 100`) that alternates between demixing $B$ via the pseudoinverse $\hat{B} = \hat{A}^\dagger X$, projecting each row of $\hat{B}$ onto the convolutive sparse model via single-channel blind deconvolution, and updating the mixing matrix via $\hat{A} = X \hat{B}^\dagger$ followed by normalization, successfully recovering $\hat{A}$, both spike waveforms $\hat{s}_1, \hat{s}_2$, and both impulse trains $\hat{\Psi}_1, \hat{\Psi}_2$.

4. **Section 1, Part 4 — Frequency-Domain Single-Channel Sparse Blind Deconvolution on `x1`:** Reformulates the single-channel blind deconvolution problem for $x_1(t)$ in the Discrete Fourier Transform (DFT / `fft`) domain by replacing time-domain convolution with element-wise spectral multiplication and division, while enforcing the non-negativity ($\alpha_k > 0$), $K$-sparsity ($K = 5$), and minimum refractory separation ($\vert{}\tau_i - \tau_j\vert{} > L$) constraints in the time domain via `ifft`/`fft`, and analyzes why direct spectral division under noise and repeated unnormalized FFT operations degrade estimation accuracy and scale down the recovered impulse amplitudes.

All four parts of the assignment are mandatory; no optional or bonus problems were assigned. Because both the original assignment sheet and the submitted solution report are written in Persian, this `README.md` provides a **complete, self-contained English reference for every problem statement, mathematical formulation, algorithm, and numerical/visual result**.

## 2. Directory Contents

| File | Description |
| ----- | ----- |
| [codes/Section1.m](https://www.google.com/search?q=./codes/Section1.m) | Main MATLAB script for Homework 7 (Parts 1–4): time-domain single-channel sparse blind deconvolution on `x1` and `x2`, multi-channel sparse blind deconvolution on the $2 \times 2000$ mixture `X`, and frequency-domain single-channel blind deconvolution on `x1`. |
| [codes/Single\_Channel\_\_Blind\_Deconvolution.m](https://www.google.com/search?q=./codes/Single_Channel__Blind_Deconvolution.m) | Helper MATLAB function `[s_hat, comb, x_hat] = Single_Channel__Blind_Deconvolution(x, L, K, T, iterNum)` implementing $100$-iteration time-domain alternating minimization for single-channel sparse blind deconvolution. |
| [codes/Not Used/Frequency\_Normalization.m](https://www.google.com/search?q=./codes/Not%2520Used/Frequency_Normalization.m) | Auxiliary MATLAB function `[signal_normalized_f, norm_sig] = Frequency_Normalization(signal_f)` for normalizing a frequency-domain vector via its time-domain $\ell_2$-norm (retained in `codes/Not Used/` for reference; not called by `Section1.m`). |
| [codes/hw7.mat](https://www.google.com/search?q=./codes/hw7.mat) | Dataset for Homework 7 containing two noisy single-channel observations `x1` ($1 \times 2000$ double) and `x2` ($1 \times 2000$ double), and one noiseless two-channel mixture matrix `X` ($2 \times 2000$ double). |
| [BSS\_HW#7\_810100205\_Arman\_Majidi.pdf](https://www.google.com/search?q=./BSS_HW%25237_810100205_Arman_Majidi.pdf) | Submitted report and analytical/simulation solutions in Persian (including mathematical formulations, estimated mixing matrix $\hat{A}$, and Figures 1.1–1.9). |
| [hw7.pdf](https://www.google.com/search?q=./hw7.pdf) | Original assignment problem sheet in Persian. |
| `README.md` | Self-contained English problem statements, mathematical derivations, algorithm descriptions, numerical results, and usage guide. |

Keep the `codes` subfolder (and its `codes/Not Used` subdirectory) inside the homework directory.

## 3. Assignment Problem Statement

### 3.1 General Context and Dataset (`hw7.mat`)

*(Submission deadline on the original assignment sheet: Sunday, Khordad 6, 17:00.)*

In this assignment, we implement the **Single-Channel Sparse Blind Deconvolution** method. As discussed in the course lectures, this method also forms the core building block for **Multi-Channel Sparse Blind Deconvolution**.

Two noisy single-channel signals `x1` and `x2` ($1 \times 2000$) and one two-channel signal matrix `X` ($2 \times 2000$) are provided in the file `hw7.mat`.

### 3.2 Section 1: Single-Channel and Multi-Channel Sparse Blind Deconvolution (`hw7.mat`)

#### Part 1 — Single-Channel Sparse Blind Deconvolution on `x1`

Plot the noisy single-channel signal $x_1(t)$ and inspect it visually. Using the time-domain single-channel sparse blind deconvolution method taught in class, estimate and plot both the **spike waveform** $s(t)$ and the **sparse impulse time series (comb function)** $\Psi(t)$ that generated this observation. Assume a waveform length of $L = 100$ samples and a total of $K = 5$ spike occurrences.

#### Part 2 — Single-Channel Sparse Blind Deconvolution on `x2`

Repeat Part 1 using the second noisy single-channel observation $x_2(t)$ (with $L = 100$ samples and $K = 5$ occurrences).

#### Part 3 — Multi-Channel Sparse Blind Deconvolution on Two-Channel Mixture `X`

First, plot the two-channel observation matrix $X \in \mathbb{R}^{2 \times 2000}$ and inspect its channels visually. Apply the **Multi-Channel Sparse Blind Deconvolution** algorithm to $X$ and recover the **two spike waveforms** ($s_1(t)$ and $s_2(t)$) and the **two sparse impulse time series** ($\Psi_1(t)$ and $\Psi_2(t)$) that generated these observations. Assume the length of each waveform is $L = 100$ samples and the number of occurrences for each source is $K = 5$.

#### Part 4 — Frequency-Domain Single-Channel Sparse Blind Deconvolution on `x1`

Reconsider Part 1. Suppose we transform the single-channel observation $x_1(t)$ into the **frequency domain**. By performing the deconvolution processing in the frequency domain, solve the problem of Part 1 again and compare the result.

## 4. Mathematical Formulation & Solutions

### 4.1 Ground-Truth Structure of the Dataset (`hw7.mat`)

The dataset `codes/hw7.mat` contains three variables over a discrete time horizon of $T = 2000$ samples:

1. **Noisy single-channel signal** `x1` $\in \mathbb{R}^{1 \times 2000}$ ($\Vert{}x_1\Vert{}_2 \approx 3.2253$): Consists of $K = 5$ non-overlapping occurrences of a zero-mean, full-cycle **sinusoidal waveform** $s^{(1)}(t)$ of duration $L = 100$ samples ($\Vert{}s^{(1)}\Vert{}_2 = 1$), located at true start indices $\tau^* = \begin{bmatrix} 101 & 301 & 801 & 1201 & 1401 \end{bmatrix}$ with positive amplitudes $\alpha^* \approx \begin{bmatrix} 1.4281 & 1.2700 & 1.2625 & 1.3294 & 1.7887 \end{bmatrix}^T$, corrupted by additive zero-mean white noise of standard deviation $\sigma_v \approx 0.00965$ ($\Vert{}v\Vert{}_2 \approx 0.4314$, $\mathrm{SNR} \approx 17.47\ \mathrm{dB}$).

2. **Noisy single-channel signal** `x2` $\in \mathbb{R}^{1 \times 2000}$ ($\Vert{}x_2\Vert{}_2 \approx 3.0836$): Consists of $K = 5$ non-overlapping occurrences of a positive, symmetric **triangular waveform** $s^{(2)}(t)$ of duration $L = 100$ samples ($\Vert{}s^{(2)}\Vert{}_2 = 1$), located at true start indices $\tau^* = \begin{bmatrix} 201 & 601 & 951 & 1301 & 1601 \end{bmatrix}$ with positive amplitudes $\alpha^* \approx \begin{bmatrix} 1.4610 & 1.0588 & 1.6908 & 1.2346 & 1.2951 \end{bmatrix}^T$, corrupted by additive zero-mean white noise of standard deviation $\sigma_v \approx 0.00974$ ($\Vert{}v\Vert{}_2 \approx 0.4356$, $\mathrm{SNR} \approx 17.00\ \mathrm{dB}$).

3. **Noiseless two-channel mixture** `X` $\in \mathbb{R}^{2 \times 2000}$ ($\Vert{}X\Vert{}_F \approx 5.0099$): Generated by an orthogonal $2 \times 2$ mixing matrix $A^*$ multiplying two noiseless convolutive source rows $B^* = \begin{bmatrix} b_1^*(t) \\ b_2^*(t) \end{bmatrix} \in \mathbb{R}^{2 \times 2000}$:

```math
X = A^* B^*,
\qquad
A^* = \begin{bmatrix}
0.6 & -0.8 \\
0.8 & 0.6
\end{bmatrix},
\qquad
b_n^*(t) = \left(s_n^* * \Psi_n^*\right)(t), \quad n \in \{1, 2\},
```

where:

* Source $b_1^*(t)$ uses the sinusoidal waveform $s_1^*(t) = s^{(1)}(t)$ ($L = 100$) convolved with $K = 5$ impulses at $\tau_1^* = \begin{bmatrix} 101 & 301 & 801 & 1201 & 1401 \end{bmatrix}$ with amplitudes $\alpha_1^* \approx \begin{bmatrix} 1.5918 & 1.1107 & 1.7858 & 1.6133 & 1.3526 \end{bmatrix}^T$.

* Source $b_2^*(t)$ uses the positive triangular waveform $s_2^*(t) = s^{(2)}(t)$ ($L = 100$) convolved with $K = 5$ impulses at $\tau_2^* = \begin{bmatrix} 111 & 401 & 751 & 1301 & 1451 \end{bmatrix}$ with amplitudes $\alpha_2^* \approx \begin{bmatrix} 1.6233 & 1.4745 & 1.6259 & 1.6412 & 1.8877 \end{bmatrix}^T$.

### 4.2 Time-Domain Single-Channel Sparse Blind Deconvolution (Parts 1 & 2, `Single_Channel__Blind_Deconvolution.m`)

#### Generative Model and Constrained Optimization Problem

As formulated on Page 2 of the Persian report, a single-channel spiky observation $x(t)$ ($t = 1, 2, \dots, T$) is modeled as the discrete linear convolution of a short, unit-energy morphological waveform (kernel) $s(t)$ of support length $L$ with a sparse impulse train (comb function) $\Psi(t)$ comprising $K$ positive Dirac impulses at occurrence times $\{\tau_k\}_{k=1}^K$ with amplitudes $\{\alpha_k\}_{k=1}^K$:

```math
x(t) = \left(s * \Psi\right)(t) + v(t) = \sum_{k=1}^{K} \alpha_k s\left(t - \tau_k\right) + v(t),
\qquad
\Psi(t) = \sum_{k=1}^{K} \alpha_k \delta\left(t - \tau_k\right),
\qquad t = 1, 2, \dots, T.
```

To prevent overlapping waveforms and resolve the scale and sign ambiguities inherent in blind deconvolution ($s * \Psi = (\gamma s) * (\gamma^{-1} \Psi)$), we impose three structural constraints:

1. **Unit-energy waveform normalization:** $\Vert{}s\Vert{}_2 = 1$ (resolves scaling ambiguity).

2. **Strictly positive spike amplitudes:** $\alpha_k > 0$ for all $k = 1, \dots, K$ (resolves sign ambiguity).

3. **Minimum refractory separation (non-overlapping occurrences):** $\vert{}\tau_i - \tau_j\vert{} > L$ for all $i \neq j$, ensuring that the $K$ copies of $s(t - \tau_k)$ of length $L$ do not overlap in time.

This yields the constrained least-squares optimization problem (Page 2 of the report):

```math
\left\{\hat{s}, \hat{\Psi}\right\} = \arg\min_{s, \Psi} \sum_{t=1}^{T} \left(x(t) - \left(s * \Psi\right)(t)\right)^2
\qquad \text{subject to} \qquad
\begin{cases}
\Psi(t) = \displaystyle\sum_{k=1}^{K} \alpha_k \delta\left(t - \tau_k\right), \\[10pt]
\|s\|_2 = 1, \\[4pt]
\left|\tau_i - \tau_j\right| > L, \quad \forall i \neq j, \\[4pt]
\alpha_k > 0, \quad \forall k = 1, \dots, K.
\end{cases}
```

#### Alternating Minimization Algorithm (`Single_Channel__Blind_Deconvolution.m`)

Because the objective is non-convex jointly in $(s, \{\tau_k, \alpha_k\})$ but tractable when one variable block is held fixed, `Single_Channel__Blind_Deconvolution.m` alternates between updating the waveform $\hat{s}$ and the comb parameters $\{\tau_k, \alpha_k\}$ for `iterNum = 100` iterations:

* **Initialization (`lines 2–6`):** Reshape the length-$T$ observation $x$ into $K = 5$ rows via `x1_resahpe = reshape(x, K, [])`, extract the initial $K \times L$ segment matrix $Y = \text{x1\_resahpe}(:, 1:L) \in \mathbb{R}^{K \times L}$, and initialize uniform weights $\alpha = \mathbf{1}_{K \times 1}$.

* **Step 1 — Waveform Update ($\Psi$ Fixed, Update $\hat{s}$, `lines 8–10`):** When the spike locations $\tau_1 < \tau_2 < \dots < \tau_K$ (with $\vert{}\tau_i - \tau_j\vert{} > L$) and amplitudes $\alpha = \begin{bmatrix} \alpha_1 & \cdots & \alpha_K \end{bmatrix}^T$ are fixed, the $K$ non-overlapping $L$-sample windows of $x(t)$ starting at $\tau_k$ form the rows of the segment matrix $Y \in \mathbb{R}^{K \times L}$:

```math
Y = \begin{bmatrix}
y_1^T \\
y_2^T \\
\vdots \\
y_K^T
\end{bmatrix} \in \mathbb{R}^{K \times L},
\qquad
Y_{k, \ell} = x\left(\tau_k + \ell - 1\right) \approx \alpha_k s(\ell),
\qquad k = 1, \dots, K, \quad \ell = 1, \dots, L.
```

Minimizing the rank-1 Frobenius fitting error $\Vert{}Y - \alpha s^T\Vert{}_F^2 = \sum_{k=1}^K \Vert{}y_k - \alpha_k s\Vert{}_2^2$ with respect to $s \in \mathbb{R}^{L \times 1}$, followed by projection onto the unit sphere $\Vert{}\hat{s}\Vert{}_2 = 1$, gives the closed-form coherent weighted average (Page 2 of the report):

```math
\tilde{s} = \frac{Y^T \alpha}{\alpha^T \alpha} = \frac{\sum_{k=1}^{K} \alpha_k y_k}{\sum_{k=1}^{K} \alpha_k^2},
\qquad
\hat{s} = \frac{\tilde{s}}{\left\|\tilde{s}\right\|_2}.
```

* **Step 2 — Comb Function Update ($\hat{s}$ Fixed, Update $\{\tau_k, \alpha_k\}$ and $Y$, `lines 11–26`):** Holding the unit-norm template $\hat{s} \in \mathbb{R}^{L \times 1}$ fixed, construct the $L \times (T - L + 1)$ Hankel/Toeplitz sliding-window matrix $Z$ via `Z = buffer(x, L, L-1, 'nodelay')`, whose $t$-th column is the $L$-sample window $z_t = x(t : t + L - 1)^T$ for $t = 1, \dots, T - L + 1$:

```math
Z = \begin{bmatrix}
x(1) & x(2) & \cdots & x(T - L + 1) \\
x(2) & x(3) & \cdots & x(T - L + 2) \\
\vdots & \vdots & \ddots & \vdots \\
x(L) & x(L + 1) & \cdots & x(T)
\end{bmatrix} \in \mathbb{R}^{L \times (T - L + 1)}.
```

Fitting $Z \approx \hat{s} b^T$ in the least-squares sense yields the matched-filter sliding correlation vector $b^T \in \mathbb{R}^{1 \times (T - L + 1)}$ (`line 13`, where $\hat{s}^\dagger = \hat{s}^T$ since $\Vert{}\hat{s}\Vert{}_2 = 1$):

```math
b^T = \hat{s}^\dagger Z = \hat{s}^T Z,
\qquad
b(t) = \sum_{\ell=1}^{L} \hat{s}(\ell) x(t + \ell - 1), \quad t = 1, \dots, T - L + 1.
```

To enforce $\alpha_k > 0$, negative correlations are rectified to zero via `b_T(b_T < 0) = 0` (`line 14`). Then, for $i = 1, \dots, K$, the algorithm greedily extracts the $K$ largest peaks subject to the refractory exclusion zone $\vert{}\tau_i - \tau_j\vert{} > L$ (`lines 15–20`):

```math
\hat{\tau}_i = \arg\max_{t} b(t),
\qquad
\hat{\alpha}_i = b\left(\hat{\tau}_i\right),
\qquad
b(t) \leftarrow 0 \quad \forall t \in \left[\max\left(\hat{\tau}_i - L, 1\right),\ \min\left(\hat{\tau}_i + L, T\right)\right].
```

The extracted spike indices $\{\hat{\tau}_i\}_{i=1}^K$ are sorted in ascending chronological order $\tau_1 < \tau_2 < \dots < \tau_K$ (`[tau_k, sort_ind] = sort(ind_corr_max, 'ascend')`), the corresponding amplitudes $\alpha = \text{corr\_max}(\text{sort\_ind})^T$ are aligned, and the segment matrix $Y$ is refreshed via `Y(i, :) = x(tau_k(i) : tau_k(i) + L - 1)` (`lines 21–26`).

* **Signal Reconstruction (`lines 28–30`):** After `iterNum = 100` iterations, the sparse comb vector `comb` $\in \mathbb{R}^{1 \times (T - L + 1)}$ is populated with $\hat{\alpha}_k$ at indices $\hat{\tau}_k$, and the denoised observation is reconstructed by discrete convolution (`conv(s_hat, comb)` of length $L + (T - L + 1) - 1 = T$):

```math
\hat{\Psi}(t) = \sum_{k=1}^{K} \hat{\alpha}_k \delta\left(t - \hat{\tau}_k\right),
\qquad
\hat{x}(t) = \left(\hat{s} * \hat{\Psi}\right)(t), \quad t = 1, \dots, T.
```

### 4.3 Time-Domain Multi-Channel Sparse Blind Deconvolution (Part 3, `Section1.m` lines 55–125)

#### Multi-Channel Convolutive Generative Model

In Part 3, we observe $M = 2$ sensor channels over $T = 2000$ time samples, collected in the matrix $X \in \mathbb{R}^{N \times T}$ ($N = 2$ in `Section1.m`). Each latent source row $b_n^T \in \mathbb{R}^{1 \times T}$ ($n = 1, \dots, N$) is itself a single-channel sparse convolutive process $b_n(t) = (s_n * \Psi_n)(t)$, and the sensors observe an instantaneous linear mixture $X = A B$ (Page 4 of the report):

```math
X = \sum_{n=1}^{N} a_n b_n^T = A B \in \mathbb{R}^{N \times T},
\qquad
B = \begin{bmatrix}
b_1^T \\
\vdots \\
b_N^T
\end{bmatrix} \in \mathbb{R}^{N \times T},
\qquad
b_n(t) = \left(s_n * \Psi_n\right)(t), \quad n = 1, \dots, N.
```

The joint multi-channel objective minimizes the squared Frobenius norm between the multi-channel observation $X$ and the structured model $A B$:

```math
\left\{\hat{A}, \left\{\hat{s}_n, \hat{\Psi}_n\right\}_{n=1}^{N}\right\} = \arg\min_{A, \{s_n, \Psi_n\}} \left\|X - A B\right\|_F^2
\qquad \text{subject to} \qquad
\begin{cases}
b_n(t) = \left(s_n * \Psi_n\right)(t), \quad \|s_n\|_2 = 1, \\[6pt]
\Psi_n(t) = \displaystyle\sum_{k=1}^{K} \alpha_{n,k} \delta\left(t - \tau_{n,k}\right), \quad \alpha_{n,k} > 0, \quad \left|\tau_{n,i} - \tau_{n,j}\right| > L.
\end{cases}
```

#### Two-Stage Alternating Minimization (`Section1.m` lines 69–85)

Starting from a random Gaussian mixing matrix `A_hat = randn(N, N)` normalized via `A_hat = A_hat ./ sqrt(sum(A_hat .^ 2, 2))`, the outer loop (`for iter = 1:iterNum`, $100$ iterations) alternates between:

1. **Stage 1 — Source Demixing and Single-Channel Deconvolution ($\hat{A}$ Fixed, Update $\hat{B}$, `lines 75–81`):**

   * First compute the unconstrained least-squares demixed source estimates via the Moore-Penrose pseudoinverse of $\hat{A}$:

```math
\tilde{B} = \hat{A}^\dagger X \in \mathbb{R}^{N \times T}.
```

   * Next, project each row $\tilde{b}_i^T = \tilde{B}(i, :)$ ($i = 1, \dots, N$) onto the single-channel sparse convolutive manifold by calling `Single_Channel__Blind_Deconvolution(b_hat, L, K, T, iterNum)` (`line 79`), and overwrite the $i$-th row of $\hat{B}$ with its denoised convolutive reconstruction $\hat{x}_i(t) = (\hat{s}_i * \hat{\Psi}_i)(t)$ (`line 80`):

```math
\left(\hat{s}_i, \hat{\Psi}_i, \hat{x}_i\right) = \mathrm{SCBD}\left(\tilde{B}_{i, :}, L, K, T\right),
\qquad
\hat{B}_{i, :} \leftarrow \hat{x}_i^T = \left(\hat{s}_i * \hat{\Psi}_i\right)^T, \quad i = 1, \dots, N.
```

2. **Stage 2 — Mixing Matrix Update ($\hat{B}$ Fixed, Update $\hat{A}$, `lines 82–84`):**

   * Holding the structured source matrix $\hat{B}$ fixed, update the mixing matrix via right pseudoinverse least squares followed by normalization:

```math
\tilde{A} = X \hat{B}^\dagger = X \hat{B}^T \left(\hat{B} \hat{B}^T\right)^{-1},
\qquad
\hat{A}_{i, :} \leftarrow \frac{\tilde{A}_{i, :}}{\left\|\tilde{A}_{i, :}\right\|_2}.
```

*(Note: Although the report diagram on Page 4 writes `Normalize Columns of A`, line 70 and line 84 of `Section1.m` use `sum(A_hat .^ 2, 2)` which normalizes the rows of $\hat{A}$. Because the ground-truth mixing matrix $A^* = \begin{bmatrix} 0.6 & -0.8 \\ 0.8 & 0.6 \end{bmatrix}$ is orthogonal, both its rows and columns have unit $\ell_2$-norm; see Section 5.4 for details.)*

As reported on Page 4 of the Persian report, this alternating algorithm converges to the estimated mixing matrix:

```math
\hat{A} = \begin{bmatrix}
0.6127 & -0.7903 \\
0.8125 & 0.5830
\end{bmatrix},
```

which closely matches the true orthogonal mixing matrix $A^* = \begin{bmatrix} 0.6 & -0.8 \\ 0.8 & 0.6 \end{bmatrix}$ (column cosine similarities $0.99999$ and $0.99998$).

### 4.4 Frequency-Domain Single-Channel Sparse Blind Deconvolution (Part 4, `Section1.m` lines 126–166)

#### Frequency-Domain Reformulation

By the convolution property of the Discrete Fourier Transform (approximating linear convolution over $T = 2000 \gg L = 100$ by circular convolution of length $T$), time-domain convolution $x_1(t) = (s * \Psi)(t)$ maps to element-wise multiplication in the $T$-point DFT domain:

```math
X_1(f) = \mathrm{FFT}\left\{x_1(t)\right\} \approx S(f) \odot \Psi(f), \qquad f = 1, 2, \dots, T.
```

In `codes/Section1.m` (lines 127–156), a $100$-iteration alternating loop in the frequency domain is implemented as follows:

1. **Initialization (`lines 128–133`):** Compute the $T$-point FFT of the observation `x1_f = fft(x1)`, draw a random length-$T$ initialization `S_hat_rand = rand(1, T)`, transform to the frequency domain `S_hat_f = fft(S_hat_rand)`, and normalize to unit Euclidean norm in the frequency domain:

```math
\hat{S}_f^{(0)} = \frac{\mathrm{FFT}\left\{\hat{s}_{\mathrm{rand}}\right\}}{\left\|\mathrm{FFT}\left\{\hat{s}_{\mathrm{rand}}\right\}\right\|_2}.
```

2. **Step 1 — Comb Update in Frequency and Time Domains ($\hat{S}_f$ Fixed, `lines 135–150`):**

   * Perform element-wise spectral division (`line 136`, written in code as `comb_f = S_hat_f ./ x1_f`) and return to the time domain via `comb_t = real(ifft(comb_f))` (`line 137`).

   * Enforce positivity `comb_t(comb_t < 0) = 0` (`line 138`) and greedily extract the $K = 5$ largest peaks $\{\tau_k, \alpha_k\}_{k=1}^K$ separated by at least $\vert{}\tau_i - \tau_j\vert{} > L$ (`lines 139–146`, with safeguard `alpha(alpha < 1e-5) = 1e2` on `line 147`).

   * Reconstruct the sparse length-$T$ comb signal $\hat{\Psi}(t)$ (`comb_t`) and transform it back to the frequency domain via `comb_f = fft(comb_t)` (`lines 148–150`).

3. **Step 2 — Waveform Spectrum Update ($\hat{\Psi}_f$ Fixed, `lines 151–153`):**

   * Update the waveform spectrum via element-wise spectral division and normalize to unit $\ell_2$-norm in the frequency domain:

```math
\tilde{S}_f = X_{1, f} \oslash \hat{\Psi}_f = \frac{X_1(f)}{\hat{\Psi}(f)},
\qquad
\hat{S}_f = \frac{\tilde{S}_f}{\left\|\tilde{S}_f\right\|_2}.
```

4. **Time-Domain Waveform Extraction (`lines 155–156`):** Compute `s_hat = ifft(ifftshift(S_hat_f))` and normalize `s_hat = s_hat ./ sqrt(sum(s_hat.^2))`.

#### Why the Frequency-Domain Implementation Exhibits Reduced Accuracy and Small Comb Amplitudes

As observed and explained on Page 7 of the Persian report (Figure 1.9), the frequency-domain pipeline in Part 4 does not achieve good spike localization accuracy (finding spikes in the reported run at $t \in \{68, 499, 815, 1281, 1677\}$ instead of $\{101, 301, 801, 1201, 1401\}$) and produces very small comb amplitudes ($\hat{\alpha}_k \in [0.72 \times 10^{-3},\ 3.20 \times 10^{-3}]$) for three mathematical reasons:

1. **Repeated $1/T$ Scaling in Unnormalized `fft`/`ifft` Pairs (Parseval Scaling):** In MATLAB, `fft` is unscaled while `ifft` divides by $T = 2000$, so Parseval's identity reads $\|\mathrm{FFT}\{u\}\|_2 = \sqrt{T} \|u\|_2$. Because line 133 and line 153 normalize $\hat{S}_f$ to unit norm in the **frequency domain** ($\|\hat{S}_f\|_2 = 1$) instead of normalizing $\hat{s}(t)$ to unit norm in the time domain ($\|\hat{s}\|_2 = 1 \iff \|\hat{S}_f\|_2 = \sqrt{T} \approx 44.72$), the implied time-domain waveform energy is reduced by $\frac{1}{\sqrt{T}} = \frac{1}{\sqrt{2000}} \approx 0.02236$. Combined with the $1/T$ factor in `ifft(comb_f)` on line 137 and the large frequency-domain norm $\|X_{1,f}\|_2 = \sqrt{T}\|x_1\|_2 \approx 144.24$, the resulting time-domain comb peaks `comb_t` are scaled down by approximately $\frac{1}{\sqrt{T} \|X_{1,f}\|_2} \sim 10^{-3}$.

2. **Inverted Spectral Division and Ill-Conditioning Under Noise:** On line 136, `comb_f = S_hat_f ./ x1_f` computes $\hat{S}(f) / X_1(f)$ rather than $X_1(f) / \hat{S}(f)$ (or the noise-robust frequency-domain matched filter $\hat{S}^*(f) X_1(f)$ corresponding to cross-correlation `ifft(conj(S_hat_f) .* x1_f)`). Dividing by $X_1(f)$ or $\hat{S}(f)$ near spectral zeros strongly amplifies high-frequency noise.

3. **Lack of Finite Support Projection ($L = 100$) on $\hat{s}(t)$:** In `Section1.m` lines 131–154, $\hat{S}_f \in \mathbb{C}^{1 \times 2000}$ is updated across all $T = 2000$ frequency bins without truncating its inverse DFT $\hat{s}(t)$ to support $[1, L]$ ($L = 100$) inside the loop, allowing $\hat{s}(t)$ to spread across all $2000$ time samples.

## 5. Results & Discussion

### 5.1 Quantitative Summary of Single-Channel and Multi-Channel Time-Domain Results (Parts 1–3)

The table below summarizes the ground-truth parameters and the exact recovered spike locations $\hat{\tau}_k$, spike amplitudes $\hat{\alpha}_k$, waveform correlations, and relative reconstruction errors for Parts 1, 2, and 3 (`Section1.m` and `Single_Channel__Blind_Deconvolution.m`):

| Part & Signal | True Spike Indices $\tau_k^*$ | Recovered Spike Indices $\hat{\tau}_k$ | True Amplitudes $\alpha_k^*$ | Recovered Amplitudes $\hat{\alpha}_k$ | Waveform Correlation $\vert{}\langle \hat{s}, s^*\rangle\vert{}$ | Relative Error $\frac{\Vert{}x - \hat{x}\Vert{}_2}{\Vert{}x\Vert{}_2}$ |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| **Part 1** — Noisy Single-Channel `x1` (Sinusoidal, Fig. 1.1–1.2) | $[101, 301, 801, 1201, 1401]$ | $[103, 303, 804, 1203, 1403]$ | $[1.4281, 1.2700, 1.2625, 1.3294, 1.7887]$ | $[1.4285, 1.2708, 1.2617, 1.3296, 1.7891]$ | $0.9912$ (up to $2$-sample shift) | $0.1339$ (equals noise floor $\frac{\Vert{}v_1\Vert{}_2}{\Vert{}x_1\Vert{}_2} \approx 0.1338$) |
| **Part 2** — Noisy Single-Channel `x2` (Triangular, Fig. 1.3–1.4) | $[201, 601, 951, 1301, 1601]$ | $[201, 600, 950, 1302, 1601]$ | $[1.4610, 1.0588, 1.6908, 1.2346, 1.2951]$ | $[1.4616, 1.0589, 1.6918, 1.2346, 1.2954]$ | $0.9993$ | $0.1401$ (equals noise floor $\frac{\Vert{}v_2\Vert{}_2}{\Vert{}x_2\Vert{}_2} \approx 0.1413$) |
| **Part 3** — Multi-Channel `X`, Source 1 ($\hat{s}_1, \hat{\Psi}_1$, Fig. 1.6) | $[101, 301, 801, 1201, 1401]$ | $[107, 307, 807, 1207, 1407]$ | $[1.5918, 1.1107, 1.7858, 1.6133, 1.3526]$ | $[1.5626, 1.0876, 1.7442, 1.5799, 1.3303]$ | $0.9809$ (shifted by $\Delta\tau = +6$ samples) | $\frac{\Vert{}b_1^* - \hat{b}_1\Vert{}_2}{\Vert{}b_1^*\Vert{}_2} = 0.1945$ (due to $6$-sample window edge truncation) |
| **Part 3** — Multi-Channel `X`, Source 2 ($\hat{s}_2, \hat{\Psi}_2$, Fig. 1.7) | $[111, 401, 751, 1301, 1451]$ | $[120, 410, 760, 1310, 1460]$ | $[1.6233, 1.4745, 1.6259, 1.6412, 1.8877]$ | $[1.6515, 1.4989, 1.6510, 1.6684, 1.9203]$ | $0.9990$ (shifted by $\Delta\tau = +9$ samples) | $\frac{\Vert{}b_2^* - \hat{b}_2\Vert{}_2}{\Vert{}b_2^*\Vert{}_2} = 0.0446$ (due to $9$-sample window edge truncation) |
| **Part 3** — Multi-Channel Reconstructed Mixture $\hat{X} = \hat{A}\hat{B}$ (Fig. 1.8) | Combined intervals of $b_1^*$ and $b_2^*$ | Combined intervals of $\hat{b}_1$ and $\hat{b}_2$ | Ground-truth $A^* = \begin{bmatrix} 0.6 & -0.8 \\ 0.8 & 0.6 \end{bmatrix}$ | Estimated $\hat{A} = \begin{bmatrix} 0.6127 & -0.7903 \\ 0.8125 & 0.5830 \end{bmatrix}$ | Column cosines: $0.99999$ and $0.99998$ | $\frac{\Vert{}X - \hat{A}\hat{B}\Vert{}_F}{\Vert{}X\Vert{}_F} = 0.1353$ |

### 5.2 Summary of Frequency-Domain Results (Part 4, Figure 1.9)

In Part 4 (`Section1.m` lines 126–166), the recovered comb vector depends on the random initialization `S_hat_rand = rand(1, T)`. The table below records the five impulse peaks marked by MATLAB data cursors in **Figure 1.9** (Page 7 of `BSS_HW#7_810100205_Arman_Majidi.pdf`):

| Spike Index $k$ | True Location $\tau_k^*$ in `x1` | True Amplitude $\alpha_k^*$ in `x1` | Frequency-Domain Estimated Location $\hat{\tau}_k$ (Fig. 1.9) | Frequency-Domain Estimated Amplitude $\hat{\alpha}_k$ (Fig. 1.9) | Observation |
| ----- | ----- | ----- | ----- | ----- | ----- |
| $k = 1$ | $101$ | $1.4281$ | $68$ | $0.0030307$ ($3.0307 \times 10^{-3}$) | Shifted by $-33$ samples; amplitude scaled by $\sim 2 \times 10^{-3}$. |
| $k = 2$ | $301$ | $1.2700$ | $499$ | $0.00219117$ ($2.1912 \times 10^{-3}$) | Spurious noise peak selected in $[300, 600]$. |
| $k = 3$ | $801$ | $1.2625$ | $815$ | $0.000721128$ ($7.2113 \times 10^{-4}$) | Near true spike at $801$ ($\Delta\tau = +14$ samples). |
| $k = 4$ | $1201$ | $1.3294$ | $1281$ | $0.0015879$ ($1.5879 \times 10^{-3}$) | Near end of the 4th spike window $[1201, 1300]$. |
| $k = 5$ | $1401$ | $1.7887$ | $1677$ | $0.00319956$ ($3.1996 \times 10^{-3}$) | Spurious noise peak selected after $t = 1500$. |

### 5.3 Visual Analysis of Report Figures (Figures 1.1–1.9)

* **Figures 1.1 & 1.2 (Part 1 — `x1`, Pages 1–2):** Figure 1.1 shows $x_1(t)$ in the time domain with $5$ clear noisy sinusoidal bursts. Figure 1.2 shows the recovered smooth single-cycle sine waveform $\hat{s}(t)$ over $t \in [1, 100]$ (top), the $5$ recovered Dirac impulses in $\hat{\Psi}(t)$ (middle), and the overlay of $x_1(t)$ (magenta) and $\hat{x}_1(t)$ (black) (bottom). As noted at the bottom of Page 2, the difference between $x_1(t)$ and $\hat{x}_1(t)$ outside and along the pulses is solely due to the additive white observation noise in $x_1(t)$, which the rank-1 convolutive model effectively filters out.

* **Figures 1.3 & 1.4 (Part 2 — `x2`, Page 3):** Figure 1.3 displays $x_2(t)$ with $5$ positive triangular pulses corrupted by baseline noise. Figure 1.4 shows that the algorithm cleanly recovers the positive triangular template $\hat{s}(t)$ (top), the $5$ impulse occurrences $\hat{\Psi}(t)$ at $\hat{\tau} = [201, 600, 950, 1302, 1601]$ (middle), and a noise-free reconstruction $\hat{x}_2(t)$ that tracks the true triangular envelope of $x_2(t)$ (bottom).

* **Figures 1.5–1.8 (Part 3 — Multi-Channel `X`, Pages 4–6):**

  * **Figure 1.5 (Page 4):** Shows the two noiseless sensor channels $X_1(t)$ and $X_2(t)$, where overlapping and closely spaced sinusoidal and triangular pulses are linearly mixed with opposite polarities on the second source.

  * **Figures 1.6 & 1.7 (Page 5):** Show the separated sinusoidal waveform $\hat{s}_1(t)$ and comb $\hat{\Psi}_1(t)$ (Figure 1.6) and the separated triangular waveform $\hat{s}_2(t)$ and comb $\hat{\Psi}_2(t)$ (Figure 1.7).

  * **Figure 1.8 (Page 6) — Discussion of Boundary/Truncation Distortion on Noiseless Data:** As discussed on Page 6 of the report, even though the two-channel observation $X$ is **noiseless**, comparing $X$ and $\hat{X} = \hat{A}\hat{B}$ in Figure 1.8 reveals a slight distortion at the leading edge of each pulse. Specifically, because the initial segment matrix `Y` in `Single_Channel__Blind_Deconvolution.m` locks onto the waveform with a small delay offset ($\Delta\tau_1 = +6$ samples for source 1 and $\Delta\tau_2 = +9$ samples for source 2) and the finite convolution window length is strictly truncated to $L = 100$ samples (`[tau_k : tau_k + L - 1]`), the first $6$ samples of each sinusoidal pulse and the first $9$ samples of each triangular pulse fall just before $\hat{\tau}_k$ and are zero-padded at the end of $\hat{s}_1(t)$ ($t \in [95, 100]$) and $\hat{s}_2(t)$ ($t \in [92, 100]$).

* **Figure 1.9 (Part 4 — Frequency-Domain Comb $\hat{\Psi}$, Page 7):** Displays the 5-spike comb vector `comb_t` obtained from the frequency-domain loop, illustrating both the shifted peak positions and the $\sim 10^{-3}$ amplitude attenuation caused by repeated unnormalized FFT operations and direct spectral division.

### 5.4 Notes on the Submitted Solutions / Implementation Notes

* **Row vs. Column Normalization of $\hat{A}$ in Part 3 (`codes/Section1.m` lines 70 & 84):**

  * On lines 70 and 84 of `Section1.m`, the mixing matrix is normalized via `A_hat = A_hat ./ sqrt(sum(A_hat .^ 2, 2))`. In MATLAB, `sum(..., 2)` sums along dimension 2 (across columns), which normalizes each **row** of $\hat{A}$ to unit Euclidean norm ($\|\hat{A}_{1,:}\|_2 = \sqrt{0.6127^2 + (-0.7903)^2} = 1.0000$, $\|\hat{A}_{2,:}\|_2 = \sqrt{0.8125^2 + 0.5830^2} = 1.0000$), whereas the theoretical summary on Page 4 states `Normalize Columns of A` (`sum(A_hat .^ 2, 1)`). Because the true mixing matrix $A^* = \begin{bmatrix} 0.6 & -0.8 \\ 0.8 & 0.6 \end{bmatrix}$ is **orthogonal**, both its rows and its columns have unit $\ell_2$-norm, so row normalization still converges accurately to $A^*$ with column norms $1.0176$ and $0.9821$.

* **Bottom Subplot Overlay in Figures 1.6 and 1.7 (`codes/Section1.m` lines 97–99 & 111–113):**

  * In `Section1.m`, line 97 plots `plot(1:T, X(1, :), 'm', 1:T, x_hat(1, :), 'k')` and line 111 plots `plot(1:T, X(2, :), 'm', 1:T, x_hat(2, :), 'k')` (with legend `"$X_1$", "$\hat{X_1}$"` in Figure 1.7 of the report and `"$X_2$", "$\hat{X_2}$"` in the updated script). Here `x_hat(1, :)` and `x_hat(2, :)` are the separated unmixed convolutive sources $\hat{b}_1(t) = (\hat{s}_1 * \hat{\Psi}_1)(t)$ and $\hat{b}_2(t) = (\hat{s}_2 * \hat{\Psi}_2)(t)$ rather than the sensor mixtures $\hat{X}_1(t), \hat{X}_2(t)$; the actual sensor mixture comparison $X$ vs. $\hat{X} = \hat{A}\hat{B}$ is plotted immediately afterward in **Figure 1.8** (`lines 115–124`). Also note that in the report PDF (Page 5, Figure 1.7), the y-axis labels of the top two subplots were labeled $\hat{s}_1$ and $\Psi_1$, which were already corrected to `$\hat{s}_2$` and `$\Psi_2$` on lines 104 and 109 of `codes/Section1.m`.

* **Boundary Index in `Single_Channel__Blind_Deconvolution.m` (line 25):**

  * On line 18 of `Single_Channel__Blind_Deconvolution.m`, `max_L = min(ind_corr_max(i) + L, T)` allows `b_T` (which initially has length $T - L + 1 = 1901$ from `buffer(x, L, L-1, 'nodelay')`) to be zero-extended up to length $T$ if a peak occurs near $T - L + 1$. On line 25, `Y(i, :) = x(tau_k(i) : tau_k(i) + L - 1)` requires $\tau_k(i) \le T - L + 1 = 1901$; since all true spikes in `x1`, `x2`, and `X` occur at $\tau_k \le 1601 \ll 1901$, `tau_k(i) + L - 1` never exceeds $T = 2000$.

* **Frequency-Domain Ratio and Unused Helper `codes/Not Used/Frequency_Normalization.m` (Part 4):**

  * As detailed in Section 4.4, `Section1.m` line 136 computes `comb_f = S_hat_f ./ x1_f` (the reciprocal of `x1_f ./ S_hat_f`) and normalizes `S_hat_f` directly in the frequency domain (`S_hat_f ./ vecnorm(S_hat_f)`). The helper function `codes/Not Used/Frequency_Normalization.m` was written during experimentation to normalize a frequency-domain signal by its time-domain $\ell_2$-norm (`ifft` $\to$ normalize $\to$ `fft`), which avoids the $\sqrt{T}$ Parseval scaling factor, and was archived in `codes/Not Used/` when finalizing the script.

## 6. How to Use

### 6.1 Running the MATLAB Script

Open MATLAB, navigate to the `codes` directory (so that `hw7.mat` and `Single_Channel__Blind_Deconvolution.m` are on the active working path), and run `Section1`:

```matlab
cd codes
Section1
```

* **Dependencies:**

  * `Single_Channel__Blind_Deconvolution.m` uses `buffer` from the **Signal Processing Toolbox** (line 12: `Z = buffer(x, L, L-1, 'nodelay')`).

  * All other operations in `Section1.m` and `Single_Channel__Blind_Deconvolution.m` use **base MATLAB** (`fft`, `ifft`, `ifftshift`, `conv`, `pinv`, `vecnorm`, `randn`, `rand`, `sort`, `reshape`, `plot`, `subplot`).

* **Generated Figures (`Section1.m` opens 8 figure windows by default, or 9 if lines 158–162 are uncommented):**

  1. **Figure 1 (Fig. 1.1 in report):** Time-domain plot of noisy observation $x_1(t)$ (Part 1).

  2. **Figure 2 (Fig. 1.2 in report):** 3-subplot figure for Part 1 showing recovered waveform $\hat{s}(t)$, recovered comb $\hat{\Psi}(t)$, and overlay of $x_1(t)$ vs. $\hat{x}_1(t)$.

  3. **Figure 3 (Fig. 1.3 in report):** Time-domain plot of noisy observation $x_2(t)$ (Part 2).

  4. **Figure 4 (Fig. 1.4 in report):** 3-subplot figure for Part 2 showing recovered triangular waveform $\hat{s}(t)$, recovered comb $\hat{\Psi}(t)$, and overlay of $x_2(t)$ vs. $\hat{x}_2(t)$.

  5. **Figure 5 (Fig. 1.5 in report):** 2-subplot figure for Part 3 showing the two-channel observation signals $X_1(t)$ and $X_2(t)$.

  6. **Figure 6 (Fig. 1.6 in report):** 3-subplot figure for Part 3 (Source 1) showing $\hat{s}_1(t)$, $\hat{\Psi}_1(t)$, and $X_1(t)$ vs. $\hat{x}_1(t)$.

  7. **Figure 7 (Fig. 1.7 in report):** 3-subplot figure for Part 3 (Source 2) showing $\hat{s}_2(t)$, $\hat{\Psi}_2(t)$, and $X_2(t)$ vs. $\hat{x}_2(t)$.

  8. **Figure 8 (Fig. 1.8 in report):** 2-subplot figure for Part 3 comparing the true sensor observations $X_1(t), X_2(t)$ against the reconstructed multi-channel observations $\hat{X}_1(t), \hat{X}_2(t)$ from $\hat{X} = \hat{A}\hat{B}$.

  9. **Figure 9 (Fig. 1.9 in report):** Plot of the frequency-domain estimated comb vector `comb_t` ($\hat{\Psi}(t)$) for Part 4 (uncommenting lines 158–162 also plots the time-domain waveform `s_hat` in a top subplot).

### 6.2 Inspecting Deterministic and Representative Workspace Outputs

Because all lines in `Section1.m` end with a semicolon `;`, running `Section1` produces the 9 figures above without printing text to the Command Window. Parts 1 and 2 are **completely deterministic**, while Part 3 (`A_hat = randn(N, N)`) and Part 4 (`S_hat_rand = rand(1, T)`) use random initialization. After running `Section1`, inspecting the recovered variables in the Command Window yields (matching the Persian report):

```text
>> % Part 1 (Deterministic): spike indices and amplitudes on x1
>> find(comb_x1 > 0)
ans =
   103   303   804  1203  1403

>> comb_x1(comb_x1 > 0)
ans =
    1.4285    1.2708    1.2617    1.3296    1.7891

>> % Part 2 (Deterministic): spike indices and amplitudes on x2
>> find(comb_x2 > 0)
ans =
   201   600   950  1302  1601

>> comb_x2(comb_x2 > 0)
ans =
    1.4616    1.0589    1.6918    1.2346    1.2954

>> % Part 3 (Representative run from Page 4 of report): estimated mixing matrix A_hat
>> A_hat
A_hat =
    0.6127   -0.7903
    0.8125    0.5830

>> % Part 4 (Representative run from Page 7 / Fig. 1.9 of report):
>> tau_k
tau_k =
    68   499   815  1281  1677

>> alpha
alpha =
    0.0030307   0.0021912   0.0007211   0.0015879   0.0031996
```

### 6.3 Reading the Theoretical Derivations and Report

1. Use **Section 3** for the complete English translation of the assignment problems and **Section 4** for the step-by-step mathematical formulations of single-channel time-domain sparse blind deconvolution, multi-channel alternating blind deconvolution, and frequency-domain blind deconvolution.

2. Use **Section 5** to review the quantitative results tables, figure-by-figure discussions, and implementation notes alongside the original Persian report [BSS\_HW#7\_810100205\_Arman\_Majidi.pdf](https://www.google.com/search?q=./BSS_HW%25237_810100205_Arman_Majidi.pdf) and assignment sheet [hw7.pdf](https://www.google.com/search?q=./hw7.pdf).

## 7. References

1. Akhavan, S. *Blind Source Separation: Homework 7 — Single-Channel and Multi-Channel Sparse Blind Deconvolution*. Course assignment sheet and lecture notes, University of Tehran.

2. Lewicki, M. S. (1998). A review of methods for spike sorting: the detection and classification of neural action potentials. *Network: Computation in Neural Systems*, 9(4), R53–R78.

3. Ekanadham, C., Tranchina, D., & Simoncelli, E. P. (2011). A blind sparse deconvolution method for neural spike identification. *Advances in Neural Information Processing Systems (NeurIPS)*, 24, 1440–1448.

4. Cheung, V., & Tresch, M. (2005). Non-negative matrix factorization and blind deconvolution for spike sorting and motor unit identification. *IEEE Transactions on Biomedical Engineering*.

5. Comon, P., & Jutten, C. (Eds.). (2010). *Handbook of Blind Source Separation: Independent Component Analysis and Applications*. Academic Press.
