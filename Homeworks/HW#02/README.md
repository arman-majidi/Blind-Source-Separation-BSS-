# Homework 2: Dimensionality Reduction, Spatial Whitening, SVD Subspace Analysis, and DOA & Frequency Estimation via Beamforming and MUSIC

**Course:** Blind Source Separation  
**Instructor:** Dr. Saeed Akhavan  
**Author:** Arman Majidi  
**Student ID:** 810100205  
**University:** University of Tehran  
**Semester:** Spring 2024-2025

## 1. Overview & Assignment Structure

This folder contains the **English problem statements, analytical derivations, MATLAB implementations, and simulation results for Homework 2**. The assignment is organized into two main sections that develop subspace-based Blind Source Separation (BSS) techniques using Principal Component Analysis (PCA), spatial whitening, Singular Value Decomposition (SVD), and array signal processing:

1. **Section 1 — Overdetermined Instantaneous Mixing, PCA, Spatial Whitening, and SVD Subspace Analysis (Parts 1–6):** Generates two statistically independent uniformly distributed sources and maps them into a three-sensor observation space via a $3 \times 2$ mixing matrix $A$, visualizes the 2D planar embedding of the 3D observations, performs eigen-decomposition (PCA) of the observation correlation matrix $R_X$, verifies the orthogonality of the null eigenvector $u_3$ to the observations and mixing columns, derives the $2 \times 3$ dimensionality-reduction and spatial-whitening matrix $B_{2\times 3}$, establishes the analytical and numerical equivalence between PCA/whitening and SVD ($X = Q G V^T$), recovers the $2 \times 2$ transformation $F$ linking the whitened signals to the original sources, and performs lossy dimensionality reduction to 1D under a $90\%$ energy preservation criterion.
2. **Section 2 — Narrowband Antenna Array Modeling and DOA & Frequency Estimation via Beamforming and MUSIC (Parts 1–5):** Formulates the reception of two narrowband far-field complex sinusoidal sources by a $10$-element Uniform Linear Array (ULA) in additive white Gaussian noise as a BSS model, decomposes the noisy spatio-temporal observation matrix via SVD into signal and noise (null) subspaces, estimates the source directions of arrival (DOAs, $\theta_1 = 10^\circ$ and $\theta_2 = 20^\circ$) using spatial **Beamforming** and **MUSIC**, and estimates the source baseband frequencies ($f_1 = 20\ \mathrm{kHz}$ and $f_2 = 10\ \mathrm{kHz}$) using temporal **Beamforming** and **MUSIC**.

All parts of both sections are mandatory; no optional or bonus problems were assigned. Because both the original assignment sheet and the submitted solution report are written in Persian, this `README.md` provides a **complete, self-contained English reference for every problem statement, mathematical proof, algorithm, and numerical result**.

## 2. Directory Contents

| File | Description |
| --- | --- |
| [codes/Section1.m](./codes/Section1.m) | MATLAB script for Section 1 (Parts 1–6): uniform source generation, 3D scatter plot, PCA eigen-decomposition of $R_X$, subspace projection matrix $C$, 2D spatial whitening ($B_{2d}$, $Z_{2d}$), SVD verification, source-to-whitened matrix $F$, and 1D dimensionality reduction ($B_{1d}$, $Z_{1d}$) with energy coverage computation. |
| [codes/Section2.m](./codes/Section2.m) | MATLAB script for Section 2 (Parts 1–5): $10$-element ULA signal generation in AWGN, SVD signal/null subspace partitioning, spatial DOA estimation via Beamforming and MUSIC, and temporal frequency estimation via Beamforming and MUSIC. |
| [BSS\_HW#2\_810100205\_Arman\_Majidi.pdf](./BSS_HW%232_810100205_Arman_Majidi.pdf) | Submitted report and analytical solutions in Persian (including 3D/2D scatter plots, whitened time-series waveforms, eigenvalue/SVD numerical matrices, and Beamforming/MUSIC spatial and temporal spectra). |
| [hw2.pdf](./hw2.pdf) | Original assignment problem sheet in Persian. |
| `README.md` | Self-contained English problem statements, mathematical derivations, algorithm descriptions, numerical results, and usage guide. |

Keep the `codes` subfolder inside the homework directory.

## 3. Assignment Problem Statement

### 3.1 General Instructions from the Assignment Sheet

- The exercises may be implemented in either **MATLAB** or **Python**.
- Upload both the typeset report and the MATLAB `.m` scripts (or Python notebook files).
- In MATLAB, write standard `.m` script files rather than Live Scripts (`.mlx`).
- Submit a clean, well-organized Persian report containing all plots, mathematical derivations, and numerical results.

### 3.2 Section 1: PCA, Spatial Whitening, SVD, and Dimensionality Reduction

Generate $T = 1000$ samples of source $s_1$, a random process uniformly distributed on $[-3, 3]$, and $T = 1000$ samples of source $s_2$, a random process uniformly distributed on $[-2, 2]$. If the generated sources have a non-zero mean, remove their mean so that they are strictly zero-mean. Combine the two sources via a linear instantaneous mixing model using the mixing matrix

```math
A = \begin{bmatrix} 1 & -2 \\ 2 & -1 \\ 3 & -2 \end{bmatrix},
\qquad
X_{3 \times T} = A_{3 \times 2} S_{2 \times T},
```

to produce the three observed mixtures $x_1$, $x_2$, and $x_3$.

#### Part 1 — 3D Scatter Plot and PCA Eigen-Decomposition

Plot the scatter diagram of the observations in three-dimensional space using the `scatter3` command. As can be observed, although the observations are three-dimensional, they are effectively distributed within a two-dimensional subspace. By computing the correlation matrix $R_X$ and applying Principal Component Analysis (PCA, using the `eig` command), obtain the eigenvector matrix $U$ and the diagonal eigenvalue matrix $D$.

#### Part 2 — Conceptual and Mathematical Subspace Properties

Understand each of the following three statements conceptually and verify them mathematically. Denote the eigenvectors ordered from largest to smallest eigenvalue by $u_1$, $u_2$, and $u_3$:

1. **Zero projection along $u_3$:** One of the eigenvalues is zero (corresponding to $u_3$). This means that the observed data have no projection—or in other words, zero dispersion/variance—along the direction of $u_3$. The mathematical equivalent of this statement is

```math
u_3^T X = 0.
```

2. **Orthogonality of $u_3$ to the mixing matrix $A$:** The eigenvector $u_3$ is orthogonal to the columns of the mixing matrix $A$ because those columns generated the data, and therefore the data lie entirely within the column space of $A$. The mathematical equivalent of this statement is

```math
u_3^T A = 0.
```

3. **Equivalence of the signal subspace and column space of $A$:** The eigenvectors $u_1$ and $u_2$ lie in the same two-dimensional subspace as the columns of $A$ (namely $a_1$ and $a_2$). The mathematical equivalent of this statement is

```math
A = \begin{bmatrix} u_1 & u_2 \end{bmatrix} C_{2 \times 2}.
```

Compute the entries of the $2 \times 2$ matrix $C_{2 \times 2}$.

#### Part 3 — Dimensionality Reduction to 2D and Spatial Whitening

Because one of the eigenvalues is zero, the observations can be projected into a two-dimensional subspace without losing any information. Find the matrix $B_{2 \times 3}$ that eliminates the redundant dimension of the data while simultaneously whitening the data in the new two-dimensional space:

```math
Z_{2 \times T} = B_{2 \times 3} X_{3 \times T}.
```

Plot the whitened signals $z_1(t)$ and $z_2(t)$ (as well as their 2D scatter diagram).

#### Part 4 — Singular Value Decomposition (SVD) of the Observation Matrix

Apply the Singular Value Decomposition ($X = Q G V^T$, written on the assignment sheet in factor notation as `[Q, G, V^T] = svd(X)`) to the initial observation matrix $X$.

1. What is the rank of the matrix $X$?
2. What are the mathematical relationships between the matrix $Q$ and $U$, between the matrix $G$ and $D$, and between the matrix $V^T$ (or $V$) and $Z$?

#### Part 5 — Subspace Relationship Between the Sources $S$ and the Right Singular Vectors / Whitened Signals

In Part 2, we saw that $u_1$ and $u_2$ span the column space of $A$ (namely $a_1$ and $a_2$). Now show that the first and second rows of $V^T$ (namely $v_1^T$ and $v_2^T$), or equivalently the whitened signals $z_1(t)$ and $z_2(t)$, lie in the row space of the source matrix $S$ (namely $s_1^T$ and $s_2^T$). The mathematical equivalent of this statement is

```math
S_{2 \times T} = F_{2 \times 2} Z_{2 \times T}.
```

Compute the entries of the matrix $F_{2 \times 2}$. Based on these results, explain why the source signals are orthogonal to the 3rd through $T$-th rows of $V^T$.

#### Part 6 — Dimensionality Reduction Under a 90% Energy Preservation Criterion

Suppose we are asked to reduce the dimensionality of the original observation data $X$ as much as possible while preserving at least $90\%$ of the total energy of the observations ($E_{\mathrm{tot}} = E_1 + E_2 + E_3$). How do you perform this reduction? Plot the reduced-dimension data in the resulting space.

### 3.3 Section 2: Array Signal Processing, Beamforming, and MUSIC for DOA & Frequency Estimation

A receiver employs a vertical Uniform Linear Array (ULA) with $M = 10$ antenna elements and inter-element spacing $d = 1\ \mathrm{m}$ that downconverts incoming radio-frequency signals with carrier frequency $f_c = 150\ \mathrm{MHz}$ to baseband. Two far-field sources are located at elevation angles $\theta_1 = 10^\circ$ and $\theta_2 = 20^\circ$. The baseband time-domain signals of these two sources are given by

```math
\begin{aligned}
s_1(t) &= \exp(j 2\pi f_1 t), &\qquad f_1 &= 20\ \mathrm{kHz}, \\
s_2(t) &= \exp(j 2\pi f_2 t), &\qquad f_2 &= 10\ \mathrm{kHz}.
\end{aligned}
```

Assume that the receiver records the sum of the downconverted source signals and additive noise over a duration of $1\ \mathrm{ms}$ ($10^{-3}\ \mathrm{s}$) at a sampling rate of $f_s = 1\ \mathrm{MHz}$ ($T = 1000$ samples). Furthermore, assume that the additive noise at each antenna element is zero-mean Gaussian with unit variance ($\sigma^2 = 1$) and is statistically independent of the sources.

#### Part 1 — BSS Mathematical Formulation and Subspace Decomposition

Write the mathematical formulation of this array reception problem from the perspective of Blind Source Separation (BSS), and explain how the SVD of the observation matrix $X$ partitions the observation space into signal and noise (null) subspaces.

#### Part 2 — DOA Estimation via Beamforming

From the noisy observation matrix $X$, estimate the source directions of arrival ($\theta_1$ and $\theta_2$) using the **Beamforming** method.

#### Part 3 — DOA Estimation via MUSIC

From the noisy observation matrix $X$, estimate the source directions of arrival ($\theta_1$ and $\theta_2$) using the **MUSIC** (MUltiple SIgnal Classification) method, and compare the resulting spatial spectrum with that of Beamforming.

#### Part 4 — Source Frequency Estimation via Temporal Beamforming

Suppose it is known a priori that the source signals are single-tone complex exponentials of the form $\exp(j 2\pi f t)$. From the observed data $X$, estimate the frequencies of the source signals using the **Beamforming** method.

#### Part 5 — Source Frequency Estimation via Temporal MUSIC

Suppose again that the source signals are known to be single-tone complex exponentials of the form $\exp(j 2\pi f t)$. From the observed data $X$, estimate the frequencies of the source signals using the **MUSIC** method, and compare its frequency resolution with Temporal Beamforming.

## 4. Mathematical Formulation & Solutions

### 4.1 Section 1: PCA, Spatial Whitening, SVD, and Dimensionality Reduction (`Section1.m`)

#### Part 1 — Planar Embedding of 3D Mixtures and PCA Eigen-Decomposition

The two independent sources $s_1(t) \sim \mathcal{U}(-3, 3)$ and $s_2(t) \sim \mathcal{U}(-2, 2)$ have theoretical variances $\sigma_{s_1}^2 = \frac{(3 - (-3))^2}{12} = 3$ and $\sigma_{s_2}^2 = \frac{(2 - (-2))^2}{12} = \frac{4}{3} \approx 1.3333$. Under thenoiseless linear instantaneous mixing model $X_{3 \times T} = A_{3 \times 2} S_{2 \times T}$ with

```math
A = \begin{bmatrix} a_1 & a_2 \end{bmatrix}
= \begin{bmatrix} 1 & -2 \\ 2 & -1 \\ 3 & -2 \end{bmatrix},
\qquad
X(t) = a_1 s_1(t) + a_2 s_2(t) \in \mathbb{R}^3,
```

every observation vector $X(t)$ is a linear combination of only two linearly independent $3 \times 1$ spatial mixing vectors $a_1 = [1, 2, 3]^T$ and $a_2 = [-2, -1, -2]^T$. Consequently, although $X(t)$ lives in $\mathbb{R}^3$, all $T = 1000$ samples lie strictly on a two-dimensional plane passing through the origin with normal vector

```math
n = a_1 \times a_2 = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} \times \begin{bmatrix} -2 \\ -1 \\ -2 \end{bmatrix}
= \begin{bmatrix} -1 \\ -4 \\ 3 \end{bmatrix}.
```

In a noisy three-sensor, two-source system, any energy observed along the normal direction $n$ would be purely due to sensor noise; projecting the 3D observations onto the 2D principal subspace therefore eliminates the noise-only dimension (denoising via PCA).

In `codes/Section1.m` (lines 19–23), the unnormalized sample correlation matrix $R_X \in \mathbb{R}^{3 \times 3}$ is formed as

```math
R_X = X X^T = U D U^T,
```

where `[U, D] = eig(R_x)` computes the orthonormal eigenvectors and eigenvalues, which are then sorted in descending order ($\lambda_1 \ge \lambda_2 \ge \lambda_3 \ge 0$) via `sort(diag(D), 'descend')`:

```math
U = \begin{bmatrix} u_1 & u_2 & u_3 \end{bmatrix}
= \begin{bmatrix}
0.3590 & 0.9125 & 0.1961 \\
0.5009 & -0.3657 & 0.7845 \\
0.7875 & -0.1834 & -0.5883
\end{bmatrix},
\qquad
D = \begin{bmatrix}
5.0195 \times 10^4 & 0 & 0 \\
0 & 1.9496 \times 10^3 & 0 \\
0 & 0 & 1.2836 \times 10^{-11}
\end{bmatrix}.
```

*(Note that using the unnormalized correlation matrix $R_X = X X^T$ rather than $\frac{1}{T} X X^T$ scales the eigenvalues by $T = 1000$ while leaving the eigenvectors $U$ and energy ratios unchanged, and makes $R_X = X X^T = Q G G^T Q^T$ match the SVD singular values directly via $D = G G^T$ in Part 4.)*

#### Part 2 — Proof of the Three Subspace Propositions and Calculation of $C_{2 \times 2}$

1. **Why $\lambda_3 = 0 \Longleftrightarrow u_3^T X = 0$:** The quadratic form $u_3^T R_X u_3$ measures the total energy (unnormalized sample variance) of the projection of $X$ onto the unit direction vector $u_3$:

```math
\lambda_3 = u_3^T R_X u_3 = u_3^T (X X^T) u_3 = \left\| u_3^T X \right\|_2^2.
```

Since $\lambda_3 \approx 1.2836 \times 10^{-11} \approx 0$ (zero up to double-precision floating-point roundoff), we have $\|u_3^T X\|_2^2 = 0$, which implies

```math
u_3^T X = \mathbf{0}_{1 \times T}.
```

Statistically and geometrically, the observations have zero variance and no component along $u_3$.

2. **Why $u_3^T A = 0$:** Substituting $X = A S$ into $u_3^T X = 0$ gives $u_3^T A S = \mathbf{0}_{1 \times T}$. Since the two independent sources have full row rank ($\mathrm{rank}(S) = 2$, so $S S^T$ is invertible), right-multiplying by $S^T (S S^T)^{-1}$ yields

```math
u_3^T A = \mathbf{0}_{1 \times 2}.
```

Indeed, comparing $u_3 = [0.1961, 0.7845, -0.5883]^T$ with the analytical normal vector $n = [-1, -4, 3]^T$ shows that

```math
u_3 = -\frac{n}{\|n\|_2} = \frac{1}{\sqrt{1^2 + 4^2 + (-3)^2}} \begin{bmatrix} 1 \\ 4 \\ -3 \end{bmatrix}
= \frac{1}{\sqrt{26}} \begin{bmatrix} 1 \\ 4 \\ -3 \end{bmatrix}
\approx \begin{bmatrix} 0.1961 \\ 0.7845 \\ -0.5883 \end{bmatrix},
```

and directly checking the inner products with the columns of $A$ confirms $u_3^T a_1 = (1 + 8 - 9)/\sqrt{26} = 0$ and $u_3^T a_2 = (-2 - 4 + 6)/\sqrt{26} = 0$.

3. **Why $A = U_{2d} C_{2 \times 2}$ and closed-form derivations of $C_{2 \times 2}$:** Because $U = [u_1, u_2, u_3]$ is an orthonormal basis of $\mathbb{R}^3$ ($U U^T = I_3 = u_1 u_1^T + u_2 u_2^T + u_3 u_3^T$) and $u_3^T A = 0$, expanding $A = I_3 A$ gives

```math
A = \left( u_1 u_1^T + u_2 u_2^T + u_3 u_3^T \right) A
= \begin{bmatrix} u_1 & u_2 \end{bmatrix}
\begin{bmatrix} u_1^T A \\ u_2^T A \end{bmatrix}
= U_{2d} C_{2 \times 2},
\qquad
U_{2d} = \begin{bmatrix} u_1 & u_2 \end{bmatrix} \in \mathbb{R}^{3 \times 2}.
```

In the Persian report (Pages 2–3), $C_{2 \times 2}$ is derived and cross-verified in two equivalent ways:

- **Method A (Direct projection via orthonormality of $U_{2d}$, implemented in `Section1.m` line 28 as `C = U_2d.' * A`):** Left-multiplying $A = U_{2d} C_{2 \times 2}$ by $U_{2d}^T$ and using $U_{2d}^T U_{2d} = I_2$ yields

```math
C_{2 \times 2} = U_{2d}^T A_{3 \times 2}
= \begin{bmatrix} 3.7234 & -2.7940 \\ -0.3690 & -1.0925 \end{bmatrix}.
```

- **Method B (Via the whitened signals $Z_{2d}$ and right pseudoinverse of $S$):** In the full/reduced PCA whitening model $X = A S$ and $Z_{2d} = B_{2d} X = D_{2d}^{-1/2} U_{2d}^T X$, the pseudo-inverse reconstruction is $X = U_{2d} D_{2d}^{1/2} Z_{2d}$. Equating $A S = U_{2d} D_{2d}^{1/2} Z_{2d}$ and substituting $A = U_{2d} C_{2 \times 2}$ gives

```math
U_{2d} C_{2 \times 2} S = U_{2d} D_{2d}^{1/2} Z_{2d}
\quad \Longrightarrow \quad
C_{2 \times 2} = D_{2d}^{1/2} Z_{2d} S^\dagger,
```

where $S^\dagger = S^T (S S^T)^{-1}$ is the right Moore-Penrose pseudoinverse of the non-square matrix $S_{2 \times T}$, computed in MATLAB via the right-matrix division operator `C = D_2d^(1/2) * Z_2d / S`. As shown on Page 3 of the report, both expressions yield the exact same $2 \times 2$ matrix $C$.

#### Part 3 — Dimensionality Reduction to 2D and Spatial Whitening

First, we check the fraction of total observation energy retained when discarding $u_3$:

```math
\text{Energy Coverage (2D)} = \frac{\lambda_1 + \lambda_2}{\lambda_1 + \lambda_2 + \lambda_3} \times 100\%
= \frac{5.0195 \times 10^4 + 1.9496 \times 10^3}{5.0195 \times 10^4 + 1.9496 \times 10^3 + 1.2836 \times 10^{-11}} \times 100\% = 100\%.
```

Thus, projecting $X$ onto the 2D subspace spanned by $U_{2d} = [u_1, u_2]$ preserves $100\%$ of the signal energy. Simply setting $B = U_{2d}^T$ would decorrelate the channels ($U_{2d}^T R_X U_{2d} = D_{2d}$) without normalizing their variances to unity. To simultaneously reduce the dimension from $3$ to $2$ and **whiten** the projected observations so that $R_Z = Z_{2d} Z_{2d}^T = I_2$, we left-multiply by $D_{2d}^{-1/2}$:

```math
B_{2d} = D_{2d}^{-1/2} U_{2d}^T \in \mathbb{R}^{2 \times 3},
\qquad
Z_{2d} = B_{2d} X = D_{2d}^{-1/2} U_{2d}^T X \in \mathbb{R}^{2 \times T}.
```

Verifying the correlation matrix of $Z_{2d}$:

```math
Z_{2d} Z_{2d}^T = D_{2d}^{-1/2} U_{2d}^T \left( X X^T \right) U_{2d} D_{2d}^{-1/2}
= D_{2d}^{-1/2} D_{2d} D_{2d}^{-1/2} = I_2.
```

For the reported run on Page 3 of the report, the $2 \times 3$ whitening matrix is

```math
B_{2d} = \begin{bmatrix}
0.0016 & 0.0022 & 0.0035 \\
0.0207 & -0.0083 & -0.0042
\end{bmatrix}.
```

Because $Z_{2d} Z_{2d}^T = I_2$ (rather than $T I_2$), each row $z_i(t)$ has unit Euclidean norm ($\sum_{t=1}^T z_i(t)^2 = 1$), so the sample amplitudes of $z_1(t)$ and $z_2(t)$ lie in the range $[-0.08, 0.08]$ (Figures 1.2 and 1.3 of the report), and the 2D scatter plot of $Z_{2d}$ is a rectified (orthogonal-edged) rotated rectangle.

#### Part 4 — SVD of the Observation Matrix and Equivalence to PCA & Whitening

Let the Singular Value Decomposition of $X_{3 \times T}$ be

```math
X_{3 \times T} = Q_{3 \times 3} G_{3 \times T} V_{T \times T}^T,
```

where $Q \in \mathbb{R}^{3 \times 3}$ and $V \in \mathbb{R}^{T \times T}$ are orthogonal matrices ($Q^T Q = Q Q^T = I_3$, $V^T V = V V^T = I_T$) and $G \in \mathbb{R}^{3 \times T}$ contains the non-negative singular values $\sigma_1 \ge \sigma_2 \ge \sigma_3 \ge 0$ on its main diagonal. *(In the Persian report, $V^T$ is written symbolically as $V$ in $X = Q G V$, while in `Section1.m` line 52 it is called `[Q, G, V_T] = svd(X)`.)*

1. **Rank of $X$:** Because $\sigma_1 > 0$, $\sigma_2 > 0$, and $\sigma_3 \approx 0$ (reflecting $\mathrm{rank}(A) = 2$ and $\mathrm{rank}(S) = 2$), the matrix $X$ has **rank $2$**:

```math
\mathrm{rank}(X) = 2.
```

2. **Relationship between $G$ and $D$, and between $Q$ and $U$:** Substituting $X = Q G V^T$ into $R_X = X X^T$ and using the orthonormality of the rows of $V^T$ ($V^T V = I_T$) gives

```math
R_X = X X^T = \left( Q G V^T \right) \left( Q G V^T \right)^T
= Q G \left( V^T V \right) G^T Q^T
= Q \left( G G^T \right) Q^T.
```

Comparing $R_X = Q (G G^T) Q^T$ with the ordered eigen-decomposition $R_X = U D U^T$ yields:

```math
\boxed{
G G^T = D \quad \left(\lambda_i = \sigma_i^2\right),
\qquad
Q = \pm U,
}
```

where $\pm$ indicates that each normalized eigenvector column in $U$ is unique up to a $\pm 1$ sign flip ($q_i = \pm u_i$).

3. **Relationship between $V^T$ and the whitened matrix $Z$:** From $Z_{2d} = D_{2d}^{-1/2} U_{2d}^T X$, left-multiplying by $U_{2d} D_{2d}^{1/2}$ reconstructs $X = U_{2d} D_{2d}^{1/2} Z_{2d}$. Comparing this with the rank-2 truncated SVD $X = Q_{3 \times 2} G_{2 \times 2} V_{2 \times T}^T$ (where $V_{2 \times T}^T$ denotes the first two rows of $V^T$, i.e., the first two columns of $V$ transposed) and using $Q_{3 \times 2} = \pm U_{2d}$ and $G_{2 \times 2} = D_{2d}^{1/2}$ yields

```math
\boxed{
V_{2 \times T}^T = \pm Z_{2 \times T}.
}
```

In the numerical run documented on Page 5 of the report, $Q = -U$ and $V_{2 \times T}^T = -Z_{2 \times T}$, so the two minus signs cancel identically in the product $X = Q G V^T = (-U_{2d}) D_{2d}^{1/2} (-Z_{2d})$.

#### Part 5 — Relationship Between Sources $S$ and Whitened Signals $Z_{2d}$ (`F` Matrix)

Since $G$ has only two non-zero singular values ($\sigma_3 \approx 0$), only the first two columns of $Q$ and the first two rows of $V^T$ contribute to $X$:

```math
A_{3 \times 2} S_{2 \times T} = X_{3 \times T} = Q_{3 \times 3} G_{3 \times 2} V_{2 \times T}^T.
```

Left-multiplying both sides by the left Moore-Penrose pseudoinverse $A^\dagger = (A^T A)^{-1} A^T \in \mathbb{R}^{2 \times 3}$ (denoted symbolically as $(A^{-1})_{2 \times 3}$ in the report) gives

```math
S_{2 \times T} = A^\dagger Q_{3 \times 3} G_{3 \times 2} V_{2 \times T}^T = F_{2 \times 2} Z_{2 \times T}.
```

Because the rows of $Z_{2 \times T}$ are orthonormal ($Z_{2 \times T} Z_{2 \times T}^T = I_2$), right-multiplying $S_{2 \times T} = F_{2 \times 2} Z_{2 \times T}$ by $Z_{2 \times T}^T$ provides the direct closed-form expression implemented in `Section1.m` (line 55: `F = S * Z_2d.'`):

```math
\boxed{
F_{2 \times 2} = S_{2 \times T} Z_{2 \times T}^T = \pm A^\dagger Q_{3 \times 2} G_{2 \times 2} = C_{2 \times 2}^{-1} D_{2 \times 2}^{1/2}.
}
```

In the Persian report (Page 6), both methods are evaluated and compared:

- **Direct correlation with whitened signals (`F = S * Z_2d.'`):**

```math
F = \begin{bmatrix} 48.0041 & -29.1940 \\ -16.2129 & -32.2421 \end{bmatrix}.
```

- **Pseudoinverse SVD formula (`F = A \ Q * G(:, 1:2)`):**

```math
F_{\mathrm{SVD}} = \begin{bmatrix} -48.0041 & 29.1940 \\ 16.2129 & 32.2421 \end{bmatrix} = -F,
```

where the overall minus sign arises directly from $V_{2 \times T}^T = -Z_{2 \times T}$ ($Q = -U$) in that run.

**Orthogonality of $S$ to the 3rd through $T$-th rows of $V^T$:** Because $S_{2 \times T} = F_{2 \times 2} Z_{2 \times T} = \pm F_{2 \times 2} V_{2 \times T}^T$, the rows of $S$ are linear combinations exclusively of the first two rows $v_1^T$ and $v_2^T$ of the orthogonal matrix $V^T$. Since the rows of $V^T$ are mutually orthogonal ($v_i^T v_k = 0$ for $i \neq k$), it follows immediately that

```math
S_{2 \times T} v_k = \mathbf{0}_{2 \times 1}
\qquad \forall k \in \{3, 4, \dots, T\}.
```

#### Part 6 — Lossy Dimensionality Reduction Under the 90% Energy Threshold

The total observation energy is the trace of $R_X$:

```math
E_{\mathrm{tot}} = \mathrm{tr}(R_X) = \lambda_1 + \lambda_2 + \lambda_3.
```

Testing whether a **single** principal component ($1\text{D}$ subspace spanned by $u_1$) suffices to preserve at least $90\%$ of the total energy (`Section1.m` lines 58–60):

```math
\frac{E_1}{E_{\mathrm{tot}}} = \frac{\lambda_1}{\lambda_1 + \lambda_2 + \lambda_3}
= \frac{5.0195 \times 10^4}{5.0195 \times 10^4 + 1.9496 \times 10^3 + 1.2836 \times 10^{-11}}
\approx 0.9626 \quad (96\%).
```

Because $96\% > 90\%$, we can reduce the 3D observations all the way down to **1 dimension** by keeping only the dominant eigenpair $(u_1, \lambda_1)$:

```math
U_{1d} = u_1 = \begin{bmatrix} 0.3590 \\ 0.5009 \\ 0.7875 \end{bmatrix},
\qquad
D_{1d} = \lambda_1 = 5.0195 \times 10^4,
\qquad
B_{1d} = D_{1d}^{-1/2} U_{1d}^T \in \mathbb{R}^{1 \times 3},
```

which produces the 1D whitened signal (`Section1.m` lines 61–62, plotted in Figure 1.4 of the report):

```math
Z_{1d}(t) = B_{1d} X(t) = \lambda_1^{-1/2} u_1^T X(t) = \pm v_1^T(t) \in \mathbb{R}^{1 \times T}.
```

### 4.2 Section 2: DOA and Frequency Estimation via Beamforming and MUSIC (`Section2.m`)

#### Part 1 — Narrowband Array BSS Model and SVD Subspace Decomposition

Consider a Uniform Linear Array (ULA) of $M = 10$ sensors spaced by $d = 1\ \mathrm{m}$ along a vertical axis. Taking the first antenna element ($m = 1$) as the spatial phase reference ($d_{1,1} = 0$), the distance of the $m$-th element from the first element is

```math
d_{m,1} = (m - 1) d,
\qquad m = 1, 2, \dots, M.
```

Under the far-field plane-wave assumption, a narrowband signal arriving from elevation angle $\theta$ reaches the $m$-th element with propagation time delay

```math
\tau_{m,1}(\theta) = \frac{d_{m,1} \sin(\theta)}{c},
```

where $c = 3 \times 10^8\ \mathrm{m/s}$ is the speed of light. After downconversion by the carrier frequency $f_c = 150\ \mathrm{MHz}$, this propagation delay manifests as a narrowband phase shift $-2\pi f_c \tau_{m,1}(\theta) = -k d_{m,1} \sin(\theta)$, where the carrier wavenumber is

```math
k = \frac{2\pi f_c}{c} = \frac{2\pi \times 150 \times 10^6}{3 \times 10^8} = \pi\ \mathrm{rad/m}.
```

Defining the $M \times 1$ array steering vector (spatial signature) $a(\theta)$ for direction $\theta$ as

```math
a(\theta) = \begin{bmatrix}
1 \\
e^{-j k d_{2,1} \sin(\theta)} \\
\vdots \\
e^{-j k d_{M,1} \sin(\theta)}
\end{bmatrix}
= \begin{bmatrix}
1 \\
e^{-j k d \sin(\theta)} \\
\vdots \\
e^{-j k (M-1) d \sin(\theta)}
\end{bmatrix} \in \mathbb{C}^{10 \times 1},
```

the $10 \times 1$ array observation vector at discrete time instances $t \in \{0, 1/f_s, \dots, (T-1)/f_s\}$ ($f_s = 1\ \mathrm{MHz}$, $T = 1000$) is expressed in standard linear instantaneous BSS form as

```math
X(t) = \begin{bmatrix} a(\theta_1) & a(\theta_2) \end{bmatrix}
\begin{bmatrix} s_1(t) \\ s_2(t) \end{bmatrix} + \mathrm{Noise}(t)
\quad \Longleftrightarrow \quad
X_{10 \times T} = A(\theta)_{10 \times 2} S_{2 \times T} + N_{10 \times T}.
```

Applying the Singular Value Decomposition (`[Q, G, V] = svd(X)` in `Section2.m` line 21) gives

```math
X_{10 \times 1000} = Q_{10 \times 10} G_{10 \times 1000} V_{1000 \times 1000}^H.
```

In the presence of spatially and temporally white zero-mean Gaussian noise of variance $\sigma^2 = 1$, the $10$ singular values of $X$ split into two distinct groups (Page 9 of the report):

```math
G = \begin{bmatrix}
110.9968 & 0 & 0 & \cdots & 0 & 0 & 0 & \cdots & 0 \\
0 & 96.9326 & 0 & \cdots & 0 & 0 & 0 & \cdots & 0 \\
0 & 0 & 33.6621 & \cdots & 0 & 0 & 0 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & 29.5326 & 0 & 0 & \cdots & 0 \\
0 & 0 & 0 & \cdots & 0 & 29.2155 & 0 & \cdots & 0
\end{bmatrix}_{10 \times 1000}.
```

- **Signal subspace ($p = 2$):** The first two singular values ($\sigma_1 \approx 111.00$, $\sigma_2 \approx 96.93$) are significantly larger than the rest because they combine the coherent energy of the two sources with the background noise.
- **Noise (null) subspace ($M - p = 8$ spatial dimensions, $T - p = 998$ temporal dimensions):** Because white noise is distributed isotropically across all dimensions, the remaining singular values $\sigma_3, \dots, \sigma_{10}$ plateau around a nearly constant noise floor ($\approx \sqrt{T}\sigma = \sqrt{1000} \approx 31.62$, ranging from $33.66$ down to $29.22$).

Accordingly, the left singular matrix $Q \in \mathbb{C}^{10 \times 10}$ and right singular matrix $V \in \mathbb{C}^{1000 \times 1000}$ are partitioned as

```math
Q_{10 \times 10} = \begin{bmatrix} Q_{\mathrm{sig},\, 10 \times 2} & Q_{\mathrm{null},\, 10 \times 8} \end{bmatrix},
\qquad
V_{T \times T} = \begin{bmatrix} V_{\mathrm{sig},\, T \times 2} & V_{\mathrm{null},\, T \times (T-2)} \end{bmatrix}.
```

#### Part 2 — Spatial DOA Estimation via Beamforming

Since the columns of $Q_{\mathrm{sig}} = Q(:, 1:2)$ span the same 2D spatial signal subspace as the true steering vectors $\{a(\theta_1), a(\theta_2)\}$, the **Beamforming** method sweeps the candidate angle $\theta \in [0^\circ, 90^\circ]$ (`theta = linspace(0, 90, 451)`, step size $0.2^\circ$) and maximizes the $\ell_2$-norm of the projection of $a(\theta)$ onto $Q_{\mathrm{sig}}$ (`Section2.m` lines 24–38):

```math
P_{\mathrm{BF}}(\theta) = \left\| a(\theta)^H Q_{\mathrm{sig}} \right\|_2,
\qquad
\{\hat{\theta}_1, \hat{\theta}_2\} = \arg\max_{\theta} P_{\mathrm{BF}}(\theta).
```

*(Because $a(\theta)$ is complex-valued, the Hermitian transpose $a(\theta)^H$ is used instead of the ordinary transpose $a(\theta)^T$.)* As shown in Figure 2.1 of the report, $P_{\mathrm{BF}}(\theta)$ attains its two dominant peaks at the exact source angles $\hat{\theta}_1 = 10^\circ$ ($P_{\mathrm{BF}} = 3.23042$) and $\hat{\theta}_2 = 20^\circ$ ($P_{\mathrm{BF}} = 3.23153$).

#### Part 3 — Spatial DOA Estimation via MUSIC

Because the spatial noise subspace $Q_{\mathrm{null}} = Q(:, 3:10) \in \mathbb{C}^{10 \times 8}$ is orthogonal to the signal subspace $Q_{\mathrm{sig}}$, the true steering vectors satisfy $a(\theta_i)^H Q_{\mathrm{null}} \approx \mathbf{0}_{1 \times 8}$. The **MUSIC** algorithm searches for the angles $\theta$ that minimize $\|a(\theta)^H Q_{\mathrm{null}}\|_2$, or equivalently maximize the reciprocal spatial pseudo-spectrum (`Section2.m` lines 40–51):

```math
P_{\mathrm{MUSIC}}(\theta) = \frac{1}{\left\| a(\theta)^H Q_{\mathrm{null}} \right\|_2},
\qquad
\{\hat{\theta}_1, \hat{\theta}_2\} = \arg\max_{\theta} P_{\mathrm{MUSIC}}(\theta).
```

As shown in Figure 2.2 of the report, $P_{\mathrm{MUSIC}}(\theta)$ exhibits two sharp, isolated peaks at $\hat{\theta}_1 = 10^\circ$ ($P_{\mathrm{MUSIC}} = 11.9403$) and $\hat{\theta}_2 = 20^\circ$ ($P_{\mathrm{MUSIC}} = 12.825$).

**Comparison of Spatial MUSIC vs. Spatial Beamforming:** In Beamforming, the candidate vector $a(\theta)$ is projected onto only $2$ columns ($Q_{\mathrm{sig}} \in \mathbb{C}^{10 \times 2}$), producing wide main lobes and noticeable sidelobes across $\theta \in [30^\circ, 90^\circ]$. In MUSIC, $a(\theta)$ must be simultaneously orthogonal to all $8$ columns of $Q_{\mathrm{null}} \in \mathbb{C}^{10 \times 8}$; because $8 > 2$ independent orthogonality constraints are enforced, the denominator drops sharply near $\theta_1 = 10^\circ$ and $\theta_2 = 20^\circ$, yielding a much sharper, flat-baseline spectrum.

#### Part 4 — Source Frequency Estimation via Temporal Beamforming

Given the prior knowledge that each source is a single-tone complex exponential $s(f) = \exp(j 2\pi f t) \in \mathbb{C}^{1 \times T}$ in the VLF/LF range, `Section2.m` (lines 54–67) defines a candidate frequency grid from $300\ \mathrm{Hz}$ to $30\ \mathrm{kHz}$ in $10\ \mathrm{Hz}$ increments (`f_var = 3e2: 1e1 :3e4`, $2971$ frequencies) and constructs the candidate temporal dictionary matrix `s_f = exp(1j * 2*pi * f_var.' * t)`.

In the SVD $X = Q G V^H$, the first two rows of $V^H$ span the row space of $S(t)$, which means the first two columns $V_{\mathrm{sig}} = V(:, 1:2) \in \mathbb{C}^{1000 \times 2}$ span the conjugate source waveforms $s_i(t)^* = \exp(-j 2\pi f_i t)$. Consequently, the Hermitian inner product between the conjugate basis $V_{\mathrm{sig}}$ and a candidate tone $s(f)^*$ is given directly by the matrix product $s(f) V_{\mathrm{sig}}$, and the **Temporal Beamforming** objective is

```math
P_{\mathrm{BF}}(f) = \left\| s(f) V_{\mathrm{sig}} \right\|_2,
\qquad
\{\hat{f}_1, \hat{f}_2\} = \arg\max_{f} P_{\mathrm{BF}}(f).
```

As shown in Figure 2.3 of the report, $P_{\mathrm{BF}}(f)$ peaks at the exact source frequencies $\hat{f}_2 = 10\,000\ \mathrm{Hz}$ ($10\ \mathrm{kHz}$, value $30.7832$) and $\hat{f}_1 = 20\,000\ \mathrm{Hz}$ ($20\ \mathrm{kHz}$, value $30.8099$).

#### Part 5 — Source Frequency Estimation via Temporal MUSIC

Using the $998$-dimensional temporal noise subspace $V_{\mathrm{null}} = V(:, 3:1000) \in \mathbb{C}^{1000 \times 998}$, the **Temporal MUSIC** pseudo-spectrum (`Section2.m` lines 70–81) is defined as

```math
P_{\mathrm{MUSIC}}(f) = \frac{1}{\left\| s(f) V_{\mathrm{null}} \right\|_2},
\qquad
\{\hat{f}_1, \hat{f}_2\} = \arg\max_{f} P_{\mathrm{MUSIC}}(f).
```

As shown in Figure 2.4 of the report, $P_{\mathrm{MUSIC}}(f)$ produces two needle-sharp peaks at $\hat{f}_2 = 10\,000\ \mathrm{Hz}$ ($10\ \mathrm{kHz}$, value $0.104138$) and $\hat{f}_1 = 20\,000\ \mathrm{Hz}$ ($20\ \mathrm{kHz}$, value $0.105017$).

**Comparison of Temporal MUSIC vs. Temporal Beamforming:** Temporal Beamforming relies on projecting $s(f)$ onto only $2$ basis vectors in $V_{\mathrm{sig}} \in \mathbb{C}^{1000 \times 2}$, resulting in pronounced periodic sidelobes across the entire $[0.3, 30]\ \mathrm{kHz}$ band (Figure 2.3). By contrast, Temporal MUSIC enforces orthogonality against all $T - 2 = 998$ basis vectors of the temporal noise subspace $V_{\mathrm{null}} \in \mathbb{C}^{1000 \times 998}$, eliminating the sidelobe ripples and producing a clean, high-resolution frequency spectrum (Figure 2.4).

## 5. Results & Discussion

### 5.1 Section 1 — Summary of PCA, Spatial Whitening, SVD, and Dimensionality Reduction (`Section1.m`)

Because `unifrnd` draws a fresh random sample of $S_{2 \times 1000}$ on each execution, sample eigenvalues and transformation entries vary slightly across runs while all subspace identities, ranks, and energy coverage percentages remain exact. The table below summarizes the analytical relationships alongside the numerical results documented in the Persian report:

| Part & Quantity | Mathematical Relation | MATLAB Expression (`Section1.m`) | Reported Numerical Result / Verification |
| --- | --- | --- | --- |
| Part 1 — Eigenvalues of $R_X = X X^T$ | $R_X = U D U^T$ | `[U, D] = eig(R_x)` (sorted descending) | $\lambda_1 = 5.0195 \times 10^4$, $\lambda_2 = 1.9496 \times 10^3$, $\lambda_3 = 1.2836 \times 10^{-11} \approx 0$ |
| Part 2 — Null eigenvector $u_3$ | $u_3^T X = 0$, $u_3^T A = 0$ | `U(:, 3)` | $u_3 = [0.1961,\ 0.7845,\ -0.5883]^T = \frac{1}{\sqrt{26}}[1,\ 4,\ -3]^T$ |
| Part 2 — Subspace expansion matrix $C$ | $A = U_{2d} C \Longleftrightarrow C = U_{2d}^T A = D_{2d}^{1/2} Z_{2d} S^\dagger$ | `C = U_2d.' * A` and `D_2d^(1/2) * Z_2d / S` | $`\begin{bmatrix} 3.7234 & -2.7940 \\ -0.3690 & -1.0925 \end{bmatrix}`$ (identical via both methods) |
| Part 3 — 2D Energy Coverage | $\frac{\lambda_1 + \lambda_2}{\lambda_1 + \lambda_2 + \lambda_3} \times 100\%$ | `sum(D_2d(:)) / sum(D(:)) * 100` | $100\%$ (lossless 3D $\to$ 2D reduction) |
| Part 3 — 2D Spatial Whitening Matrix $B_{2d}$ | $B_{2d} = D_{2d}^{-1/2} U_{2d}^T$, $Z_{2d} = B_{2d} X$ | `B_2d = D_2d^(-1/2) * U_2d.'` | $`\begin{bmatrix} 0.0016 & 0.0022 & 0.0035 \\ 0.0207 & -0.0083 & -0.0042 \end{bmatrix}`$, $Z_{2d} Z_{2d}^T = I_2$ |
| Part 4 — SVD vs. PCA & Whitening | $\mathrm{rank}(X)=2$, $G G^T = D$, $Q = \pm U$, $V_{2\times T}^T = \pm Z_{2d}$ | `[Q, G, V_T] = svd(X)` | Verified: $G G^T = D$, $Q = -U$, and $V_{2\times T}^T = -Z_{2\times T}$ |
| Part 5 — Source Reconstruction Matrix $F$ | $S = F Z_{2d} \Longleftrightarrow F = S Z_{2d}^T = \pm A^\dagger Q_{3\times 2} G_{2\times 2}$ | `F = S * Z_2d.'` and `A \ Q * G(:, 1:2)` | $`\pm \begin{bmatrix} 48.0041 & -29.1940 \\ -16.2129 & -32.2421 \end{bmatrix}`$, $S v_k = 0\ (\forall k \ge 3)$ |
| Part 6 — 1D Energy Coverage ($\ge 90\%$ target) | $\frac{\lambda_1}{\lambda_1 + \lambda_2 + \lambda_3} \times 100\%$ | `sum(D_1d(:)) / sum(D(:)) * 100` | $96.26\%$ (prints `96%`; exceeds $90\%$ threshold with $1$ dimension) |

### 5.2 Section 2 — Summary of DOA and Frequency Estimation (`Section2.m`)

In `codes/Section2.m`, the true parameters are $\theta_1 = 10^\circ$, $\theta_2 = 20^\circ$, $f_1 = 20\ \mathrm{kHz}$, and $f_2 = 10\ \mathrm{kHz}$ with $M = 10$ antennas and $T = 1000$ samples:

| Problem Part & Method | Target Parameter | Subspace Used & Dimension | Objective Function | Ground-Truth Values | Estimated Peak Locations | Reported Peak Heights |
| --- | --- | --- | --- | --- | --- | --- |
| Part 2 — Spatial Beamforming | DOA ($\theta_1, \theta_2$) | $Q_{\mathrm{sig}} \in \mathbb{C}^{10 \times 2}$ ($2$ cols) | $\left\|a(\theta)^H Q_{\mathrm{sig}}\right\|_2$ | $10^\circ,\ 20^\circ$ | $10^\circ,\ 20^\circ$ | $3.23042$ (at $10^\circ$), $3.23153$ (at $20^\circ$) |
| Part 3 — Spatial MUSIC | DOA ($\theta_1, \theta_2$) | $Q_{\mathrm{null}} \in \mathbb{C}^{10 \times 8}$ ($8$ cols) | $1 / \left\|a(\theta)^H Q_{\mathrm{null}}\right\|_2$ | $10^\circ,\ 20^\circ$ | $10^\circ,\ 20^\circ$ | $11.9403$ (at $10^\circ$), $12.825$ (at $20^\circ$) |
| Part 4 — Temporal Beamforming | Frequency ($f_2, f_1$) | $V_{\mathrm{sig}} \in \mathbb{C}^{1000 \times 2}$ ($2$ cols) | $\left\|s(f) V_{\mathrm{sig}}\right\|_2$ | $10\ \mathrm{kHz},\ 20\ \mathrm{kHz}$ | $10\ \mathrm{kHz},\ 20\ \mathrm{kHz}$ | $30.7832$ (at $10\ \mathrm{kHz}$), $30.8099$ (at $20\ \mathrm{kHz}$) |
| Part 5 — Temporal MUSIC | Frequency ($f_2, f_1$) | $V_{\mathrm{null}} \in \mathbb{C}^{1000 \times 998}$ ($998$ cols) | $1 / \left\|s(f) V_{\mathrm{null}}\right\|_2$ | $10\ \mathrm{kHz},\ 20\ \mathrm{kHz}$ | $10\ \mathrm{kHz},\ 20\ \mathrm{kHz}$ | $0.104138$ (at $10\ \mathrm{kHz}$), $0.105017$ (at $20\ \mathrm{kHz}$) |

### 5.3 Notes on the Submitted Solutions / Implementation Notes

- **3D scatter plot coordinate indexing in `Section1.m` (line 13):** Line 13 of `codes/Section1.m` calls `scatter3(X(1, :), X(2, :), X(2, :), 'square')`, passing `X(2, :)` as both the second and third coordinates instead of `X(3, :)`. All subsequent calculations in `Section1.m` (`R_x = X * X.'`, `eig`, `U_2d`, `B_2d`, `svd(X)`, `F`, and `Z_1d`) use the full $3 \times T$ matrix `X` with the true third row `X(3, :)`.
- **Two separate random realizations in Section 1 of the Persian report:** Because `Section1.m` does not fix a random seed (`rng`), the matrices reported in Parts 1–3 and 6 (Pages 2–3 and 7: $\lambda_1 = 5.0195 \times 10^4$, $\lambda_2 = 1.9496 \times 10^3$) and the SVD comparison matrices reported in Part 4 (Page 5: $\lambda_1 = 5.2288 \times 10^4$, $\lambda_2 = 1.9567 \times 10^3$) were recorded from two separate executions of `Section1.m`. Within each run, $G G^T = D$, $Q = \pm U$, and $V_{2 \times T}^T = \pm Z_{2 \times T}$ hold to machine precision.
- **Right singular factor convention in `svd(X)`:** In MATLAB, `[Q, G, V] = svd(X)` returns $V$ such that $X = Q G V^H$. In `Section1.m` (line 52), the third output is named `V_T` (`[Q, G, V_T] = svd(X)`), so the first two rows of $Z_{2d}$ match the transpose of the first two columns of `V_T` (`V_T(:, 1:2).'` $= -Z_{2d}$). Similarly, in `Section2.m` (lines 57 and 71), because the columns of `V_sig` and `V_null` already contain the conjugated right singular vectors, multiplying the row vector `s_f = exp(1j * 2*pi * f_var.' * t)` directly by `V_sig` and `V_null` (`s_f * V_sig` and `s_f * V_null`) properly evaluates the Hermitian inner products at positive frequencies $+f$.
- **Minor typographical slips in the Persian report (Pages 1, 8, and 10):**
  - On **Page 1 (Section 1, Part 1)**, the introductory sentence mentions generating three uniform distributions $s_1, s_2, s_3$ with `unifrnd()`, whereas only two sources $s_1 \sim \mathcal{U}(-3, 3)$ and $s_2 \sim \mathcal{U}(-2, 2)$ are generated in `Section1.m` (lines 4–5) and mapped to the three observations $x_1, x_2, x_3$.
  - On **Page 8 (Section 2, Part 1)**, the last column vector expression for $a_i(\theta)$ writes the phase of the 2nd through 10th elements with indices $1$ to $10$ ($e^{-j k 1 \sin(\theta_i)}, \dots, e^{-j k 10 \sin(\theta_i)}$) instead of $d_{m,1} = (m-1)d \in \{1, \dots, 9\}\ \mathrm{m}$; in `Section2.m` (line 5), the array element positions are defined accurately as `D = 0: d: M-1` ($0$ to $9\ \mathrm{m}$).
  - On **Page 10 (Section 2, Part 3)**, the sentence immediately below Figure 2.2 refers to "Figure 2.1" instead of "Figure 2.2" when stating that the MUSIC spectrum peaks at $\theta_1 = 10^\circ$ and $\theta_2 = 20^\circ$.

## 6. How to Use

### 6.1 Running the MATLAB Scripts

Open MATLAB, navigate to the `codes` directory (or add it to the MATLAB path), and run the two scripts:

```matlab
cd codes
Section1
Section2
```

- **Dependencies:**
  - `Section1.m` uses `unifrnd` (Statistics and Machine Learning Toolbox) alongside base MATLAB matrix functions (`eig`, `svd`, `scatter3`, `scatter`, `plot`).
  - `Section2.m` uses `findpeaks` (Signal Processing Toolbox) and base MATLAB functions (`svd`, `randn`, `deg2rad`, `linspace`).
- **Generated Figures:**
  - `Section1.m` opens **4 figures**:
    1. 3D scatter plot of the observation matrix $X_{3 \times T}$.
    2. 2D scatter plot of the whitened observations $Z_{2 \times T}$.
    3. Two-panel subplot of the 2D whitened time series $z_1(t)$ and $z_2(t)$.
    4. Time-domain plot of the 1D reduced and whitened signal $z(t)$ ($Z_{1d}$).
  - `Section2.m` opens **4 figures** (with peak markers automatically annotated by `findpeaks`):
    1. Spatial Beamforming spectrum $\|a(\theta)^H Q_{\mathrm{sig}}\|_2$ versus $\theta \in [0^\circ, 90^\circ]$.
    2. Spatial MUSIC spectrum $1 / \|a(\theta)^H Q_{\mathrm{null}}\|_2$ versus $\theta \in [0^\circ, 90^\circ]$.
    3. Temporal Beamforming spectrum $\|s(f) V_{\mathrm{sig}}\|_2$ versus $f \in [300\ \mathrm{Hz}, 30\ \mathrm{kHz}]$.
    4. Temporal MUSIC spectrum $1 / \|s(f) V_{\mathrm{null}}\|_2$ versus $f \in [300\ \mathrm{Hz}, 30\ \mathrm{kHz}]$.

### 6.2 Expected Console Output (`Section1.m` and `Section2.m`)

Running `Section1.m` prints the 2D and 1D percentage energy coverage to the MATLAB Command Window (which round to `100%` and `96%` across random realizations):

```text
Energy coverage ratio for 2-dimension is 100%
Energy coverage ratio for 1-dimension is 96%
```

In `Section2.m`, the estimated DOA and frequency peak locations are stored in the workspace variables `beamforming_peaks`, `music_peaks`, `beamforming_f_peaks`, and `music_f_peaks`, which evaluate to:

```text
beamforming_peaks   = [10, 20]
music_peaks         = [10, 20]
beamforming_f_peaks = [10000, 20000]
music_f_peaks       = [10000, 20000]
```

### 6.3 Reading the Theoretical Derivations and Report

1. Use **Section 3** for the complete English translation of the assignment problems and **Section 4** for the step-by-step mathematical proofs, subspace derivations, and algorithm descriptions.
2. Use **Section 5** to review the numerical results and implementation notes alongside the original Persian report [BSS\_HW#2\_810100205\_Arman\_Majidi.pdf](./BSS_HW%232_810100205_Arman_Majidi.pdf) and assignment sheet [hw2.pdf](./hw2.pdf).

## 7. References

1. Akhavan, S. *Blind Source Separation: Homework 2*. Course assignment sheet and lecture notes, University of Tehran.
2. Comon, P., & Jutten, C. (Eds.). (2010). *Handbook of Blind Source Separation: Independent Component Analysis and Applications*. Academic Press.
3. Schmidt, R. O. (1986). Multiple emitter location and signal parameter estimation. *IEEE Transactions on Antennas and Propagation*, 34(3), 276–280.
4. Hyvärinen, A., Karhunen, J., & Oja, E. (2001). *Independent Component Analysis*. John Wiley & Sons.
