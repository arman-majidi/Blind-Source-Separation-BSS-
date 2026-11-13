# Homework 3: Two-Class EEG Classification via Common Spatial Patterns (CSP) & Linear Discriminant Analysis (LDA), and SSVEP Frequency Recognition via Canonical Correlation Analysis (CCA)

**Course:** Blind Source Separation  
**Instructor:** Dr. Saeed Akhavan  
**Author:** Arman Majidi  
**Student ID:** 810100205  
**University:** University of Tehran  
**Semester:** Spring 2024-2025

## 1. Overview & Assignment Structure

This folder contains the **English problem statements, analytical derivations, MATLAB implementations, and simulation results for Homework 3**. The assignment is organized into two main sections that apply generalized eigenvalue-based Blind Source Separation (BSS) and spatial filtering methods to multi-channel Electroencephalography (EEG) brain-computer interface (BCI) paradigms:

1. **Section 1 — Two-Class Motor Imagery & Mental Arithmetic EEG Classification via Common Spatial Patterns (CSP) and Linear Discriminant Analysis (LDA) (Parts 1–5):** Preprocesses $30$-channel EEG recordings from two mental imagery classes (foot motor imagery vs. mental serial subtraction), computes trial-averaged class spatial correlation matrices, solves the Generalized Eigenvalue Decomposition (GEVD) problem to obtain normalized **Common Spatial Patterns (CSP)** filters $W_{\mathrm{csp}}$, analyzes the variance contrast and spatial scalp topography of the extreme filters ($W_{\mathrm{csp}}(:, 1)$ and $W_{\mathrm{csp}}(:, 30)$), extracts $14$-dimensional log/variance feature vectors using the top $7$ and bottom $7$ CSP filters, trains a **Fisher Linear Discriminant Analysis (LDA)** projection vector $W_{\mathrm{lda}}$ and decision threshold $c$ via GEVD, and evaluates classification accuracy on $40$ held-out test trials.
2. **Section 2 — Steady-State Visually Evoked Potential (SSVEP) Frequency Recognition via Canonical Correlation Analysis (CCA):** Constructs multi-harmonic sinusoidal reference templates up to $40\ \mathrm{Hz}$ for five candidate visual stimulus frequencies, formulates **Canonical Correlation Analysis (CCA)** as a Rayleigh quotient maximization problem solved via GEVD, and identifies the stimulus frequency of $15$ multi-channel SSVEP EEG trials in an unsupervised, training-free manner.

In addition to the standard loop-based implementations (`codes/more for/`), the repository includes vectorized 3D tensor implementations (`codes/less for/`) utilizing MATLAB's page-wise matrix operations (`pagemtimes` and `pagetranspose`). All parts of both sections are mandatory; no optional or bonus problems were assigned. Because both the original assignment sheet and the submitted solution report are written in Persian, this `README.md` provides a **complete, self-contained English reference for every problem statement, mathematical derivation, algorithm, and numerical result**.

## 2. Directory Contents

| File | Description |
| --- | --- |
| [codes/less for/Section1.m](https://www.google.com/search?q=./codes/less%2520for/Section1.m) | Vectorized MATLAB script for Section 1 (Parts 1–5) using `pagemtimes` and `pagetranspose`: 30-channel EEG mean removal, CSP spatial filter estimation ($W_{\mathrm{csp}}$), Trial 49 waveform and variance comparison, stem plots of spatial filter magnitudes, 14-filter feature extraction, LDA training ($W_{\mathrm{lda}}$, $c$), and 40-trial test classification. |
| [codes/less for/Section2.m](https://www.google.com/search?q=./codes/less%2520for/Section2.m) | Vectorized MATLAB script for Section 2 using `pagemtimes` and `pagetranspose`: harmonic reference template tensor construction ($X_t$), GEVD-based Canonical Correlation Analysis (CCA), and SSVEP frequency classification across 15 trials. |
| [codes/less for/hw3-1.mat](https://www.google.com/search?q=./codes/less%2520for/hw3-1.mat) | Dataset for Section 1 containing `TrainData_class1` ($30 \times 256 \times 60$), `TrainData_class2` ($30 \times 256 \times 60$), `TestData` ($30 \times 256 \times 40$), and ground-truth `TestLabel` ($1 \times 40$). |
| [codes/less for/hw3-2.mat](https://www.google.com/search?q=./codes/less%2520for/hw3-2.mat) | Dataset for Section 2 containing SSVEP EEG recordings `data` ($6 \times 1250 \times 15$), stimulus frequencies `freq` ($1 \times 5$), and ground-truth trial labels `label` ($1 \times 15$). |
| [codes/more for/Section1.m](https://www.google.com/search?q=./codes/more%2520for/Section1.m) | Loop-based MATLAB script for Section 1 (Parts 1–5) computing trial-by-trial spatial correlation matrices and CSP feature variances via explicit `for` loops. |
| [codes/more for/Section2.m](https://www.google.com/search?q=./codes/more%2520for/Section2.m) | Loop-based MATLAB script for Section 2 computing trial-by-trial sample correlation and cross-correlation matrices inside nested `for` loops. |
| [codes/more for/hw3-1.mat](https://www.google.com/search?q=./codes/more%2520for/hw3-1.mat) | Copy of the Section 1 EEG dataset (`hw3-1.mat`) inside `codes/more for/` for direct standalone execution. |
| [codes/more for/hw3-2.mat](https://www.google.com/search?q=./codes/more%2520for/hw3-2.mat) | Copy of the Section 2 SSVEP dataset (`hw3-2.mat`) inside `codes/more for/` for direct standalone execution. |
| [BSS\_HW#3\_810100205\_Arman\_Majidi.pdf](https://www.google.com/search?q=./BSS_HW%25233_810100205_Arman_Majidi.pdf) | Submitted report and analytical solutions in Persian (including CSP filtered signals, spatial filter stem plots, GEVD eigenvalue matrices, LDA projection vector $W_{\mathrm{lda}}$ and threshold $c$, CCA block diagram, and test classification plots). |
| [hw3.pdf](https://www.google.com/search?q=./hw3.pdf) | Original assignment problem sheet in Persian. |
| `README.md` | Self-contained English problem statements, mathematical derivations, algorithm descriptions, numerical results, and usage guide. |

Keep the `codes/less for` and `codes/more for` subfolders inside the homework directory.

## 3. Assignment Problem Statement

### 3.1 General Instructions from the Assignment Sheet

- The exercises may be implemented in either **Python** or **MATLAB**.
- Upload both the typeset report and the MATLAB `.m` scripts (or Python notebook files).
- The report must be cleanly typeset in Persian (handwritten mathematical derivations may be scanned and inserted if desired).
- In MATLAB, write standard `.m` script files rather than Live Scripts (`.mlx`).

### 3.2 Section 1: Two-Class EEG Classification Using Common Spatial Patterns (CSP) and LDA

We wish to perform a two-class classification task on multi-channel EEG data using the **Common Spatial Patterns (CSP)** approach:

- **Class 1:** Imagining foot movement (motor imagery).
- **Class 2:** Imagining performing a sequence of mental subtraction operations.

Three 3D tensors and one label vector are provided in the file `hw3-1.mat` as follows:

1. **Class 1 Training Data (`TrainData_class1`):** A 3D array of dimensions $30 \times 256 \times 60$ ($N_c = 30$ EEG channels, $T = 256$ time samples per trial, and $N_r = 60$ trials). For all trials, remove the mean of each channel so that every channel is zero-mean, and store the centered data back in `TrainData_class1`.
2. **Class 2 Training Data (`TrainData_class2`):** A 3D array of dimensions $30 \times 256 \times 60$ ($30$ channels, $256$ time samples, and $60$ trials). For all trials, remove the mean of each channel and store the centered data back in `TrainData_class2`.
3. **Test Data (`TestData`):** A 3D array of dimensions $30 \times 256 \times 40$ ($30$ channels, $256$ time samples, and $40$ test trials). For all trials, remove the mean of each channel and store the centered data back in `TestData`.
4. **Test Labels (`TestLabel`):** A vector of length $40$ ($1 \times 40$ in the `.mat` file) containing the true class labels ($\in \{1, 2\}$) of the test trials.

*(Note from the assignment sheet: Parts 1 to 3 below must be performed strictly using the training data, without using the test data.)*

#### Part 1 (الف) — CSP Spatial Filters and Variance Analysis on Trial 49

Using the training data, compute the CSP spatial filters ($W_{\mathrm{csp}}$) and apply them to the training data. Be sure to normalize the columns of $W_{\mathrm{csp}}$ to unit norm. Plot the filtered signal corresponding to the **first spatial filter** ($W_{\mathrm{csp}}(:, 1)$) and the **last spatial filter** ($W_{\mathrm{csp}}(:, 30)$) for the **49th trial of Class 1** and the **49th trial of Class 2** (similar to the figure drawn in class). As can be seen, for each trial, the output of one filter exhibits high dispersion (variance) while the output of the other filter exhibits low dispersion. Report the variances quantitatively as well (strictly for these two trials).

#### Part 2 (ب) — Spatial Filter Magnitudes and Dominant Brain Channels

Plot the absolute values (magnitudes) of the entries of the first and last spatial filters obtained in Part 1 ($|W_{\mathrm{csp}}(:, 1)|$ and $|W_{\mathrm{csp}}(:, 30)|$). For each class, which EEG channel (or channels) has the greatest influence on the filter output? Physically, this reveals which region of the brain exhibits the strongest activity for each of the mental tasks in Class 1 and Class 2.

#### Part 3 (ج) — Feature Extraction with 14 CSP Filters and Fisher LDA Training

Extract the variances of the spatially filtered training signals using the **14 most important spatial filters**—corresponding to filters $1$ through $7$ ($W_{\mathrm{csp}}(:, 1:7)$) and filters $24$ through $30$ ($W_{\mathrm{csp}}(:, 24:30)$). In this step, each $30 \times 256$ training trial is compressed into a $14 \times 1$ feature vector. Next, using a **Linear Discriminant Analysis (LDA)** classifier, find the optimal linear projection vector ($W_{\mathrm{lda}}$) and the corresponding decision threshold ($c$) that best separate the two classes. Be sure to normalize $W_{\mathrm{lda}}$ to unit norm. Note that $W_{\mathrm{lda}}$ is a $14 \times 1$ column vector and $c$ is a scalar. Report the numerical values of $W_{\mathrm{lda}}$ and $c$.

#### Part 4 (د) — Unlabeled Test Data Classification

Now turn to the test data (`TestData`). First, assume that the true class labels of the test data are unknown. Using the $14$ most important CSP spatial filters, the LDA weight vector $W_{\mathrm{lda}}$, and the threshold $c$ extracted solely from the training data in Parts 1–3, decide the class label of each of the $40$ test trials.

#### Part 5 (ه) — Test Accuracy Evaluation and Label Visualization

Compare the predicted labels obtained in Part 4 against the true test labels (`TestLabel`). How many of the $40$ test trials were classified correctly? Plot the true labels and the predicted labels on a single figure where the horizontal axis ($x$-axis) represents the test trial index ($1$ to $40$) and the vertical axis ($y$-axis) shows the true and estimated class labels.

### 3.3 Section 2: SSVEP Frequency Recognition Using Canonical Correlation Analysis (CCA)

In this exercise, we use **Canonical Correlation Analysis (CCA)** to identify the visual stimulus frequency of **Steady-State Visually Evoked Potential (SSVEP)** EEG recordings. One 3D array and two vectors are provided in the file `hw3-2.mat` as follows:

- **Stimulus Frequencies (`freq`):** A $1 \times 5$ vector containing the five candidate visual flicker frequencies used in the experiment:

```math
\mathrm{freq} = \begin{bmatrix} 6.5 & 7.35 & 8.3 & 9.6 & 11.61 \end{bmatrix}\ \mathrm{Hz}.
```

- **Recorded EEG Data (`data`):** A 3D array of dimensions $6 \times 1250 \times 15$ ($6$ EEG channels, $1250$ time samples per trial, and $15$ trials). The sampling frequency is $f_s = 250\ \mathrm{Hz}$; equivalently, the duration of each trial is $5\ \mathrm{s}$.
- **True Stimulus Labels (`label`):** A $1 \times 15$ vector containing the ground-truth stimulus frequency (in $\mathrm{Hz}$) under which each of the $15$ trials was recorded.

Using the **CCA** method, estimate the stimulus frequency for each of the $15$ trials and compare the estimated frequencies against the true labels in `label`. When constructing the sinusoidal reference `template` matrices, include harmonics up to $40\ \mathrm{Hz}$.

*(Note from the assignment sheet: In this exercise, no machine learning / supervised training approach is used to estimate the trial labels!)*

## 4. Mathematical Formulation & Solutions

### 4.1 Section 1: Common Spatial Patterns (CSP) and Fisher LDA on EEG Data (`Section1.m`)

#### Preprocessing — Channel Mean Removal

Let $X_n^{(r)} \in \mathbb{R}^{N_c \times T}$ ($N_c = 30$, $T = 256$) denote the $n$-th EEG trial of class $r \in \{1, 2\}$. Before computing spatial correlation matrices, the DC offset of each channel is removed across dimensions $2$ and $3$ (`Section1.m` lines 5–7):

```math
\tilde{X}_n^{(r)} = X_n^{(r)} - \mu_{\mathrm{ch}}^{(r)},
```

implemented in MATLAB as `TrainData_class1 = TrainData_class1 - mean(TrainData_class1, [2 3])` (and identically for `TrainData_class2` and `TestData`).

#### Part 1 — Derivation of CSP Spatial Filters and Trial 49 Variance Analysis

The **Common Spatial Patterns (CSP)** algorithm seeks a spatial filter $w \in \mathbb{R}^{30 \times 1}$ that maximizes the variance (average energy) of the spatially filtered multi-channel EEG signal $y(t) = w^T X(t)$ in one class while simultaneously minimizing its variance in the other class. Because the data are stored as 3D tensors of $N_r = 60$ trials, the class-conditional $30 \times 30$ spatial correlation matrices $R_x^{(1)}$ and $R_x^{(2)}$ are estimated by averaging the sample outer products across all $N_r = 60$ trials:

```math
R_x^{(r)} = \frac{1}{N_r} \sum_{n=1}^{N_r} X_n^{(r)} \left(X_n^{(r)}\right)^T \in \mathbb{R}^{30 \times 30},
\qquad r \in \{1, 2\}.
```

- In `codes/more for/Section1.m` (lines 11–15), $R_x^{(1)}$ (`Rx_1`) and $R_x^{(2)}$ (`Rx_2`) are accumulated over $i = 1, \dots, 60$ via a `for` loop.
- In `codes/less for/Section1.m` (lines 11–12), they are computed without loops via 3D page-wise matrix multiplication:

```matlab
Rx_1 = mean(pagemtimes(TrainData_class1, pagetranspose(TrainData_class1)), 3);
Rx_2 = mean(pagemtimes(TrainData_class2, pagetranspose(TrainData_class2)), 3);
```

For any spatial filter $w \in \mathbb{R}^{30 \times 1}$, the ratio of the average filtered signal variance in Class 1 to that in Class 2 is the **Rayleigh quotient**:

```math
J_{\mathrm{CSP}}(w) = \frac{w^T R_x^{(1)} w}{w^T R_x^{(2)} w}.
```

Maximizing (and minimizing) $J_{\mathrm{CSP}}(w)$ leads to the Generalized Eigenvalue Decomposition (GEVD) problem

```math
R_x^{(1)} w_k = \lambda_k R_x^{(2)} w_k
\quad \Longleftrightarrow \quad
\mathrm{GEVD}\left(R_x^{(1)}, R_x^{(2)}\right),
```

which is solved in MATLAB via `[U, D] = eig(Rx_1, Rx_2)`, sorted in descending order $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_{30} > 0$, and normalized column-wise to unit Euclidean norm (`W_csp = U ./ sqrt(sum(U.^2))`). The resulting sorted generalized eigenvalue matrix $D_{\mathrm{csp}}$ is (Page 1 of the report):

```math
D_{\mathrm{csp}} = \begin{bmatrix}
2.2494 & 0 & 0 & \cdots & 0 & 0 \\
0 & 1.4902 & 0 & \cdots & 0 & 0 \\
0 & 0 & 1.3556 & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & \cdots & 0.3085 & 0 \\
0 & 0 & 0 & \cdots & 0 & 0.3010
\end{bmatrix}_{30 \times 30},
\qquad
W_{\mathrm{csp}} = \begin{bmatrix} w_1 & w_2 & \cdots & w_{30} \end{bmatrix} \in \mathbb{R}^{30 \times 30}.
```

- **First spatial filter ($w_1 = W_{\mathrm{csp}}(:, 1)$, $\lambda_1 = 2.2494$):** Maximizes the variance of **Class 1** relative to Class 2.
- **Last spatial filter ($w_{30} = W_{\mathrm{csp}}(:, 30)$, $\lambda_{30} = 0.3010$):** Minimizes the variance of Class 1 relative to Class 2, which equivalently maximizes the variance of **Class 2** relative to Class 1 (by symmetry, $1 / \lambda_{30} \approx 3.3220$).

Applying $W_{\mathrm{csp}}(:, 1)$ and $W_{\mathrm{csp}}(:, 30)$ to the $49$-th training trial of both classes (`Section1.m` lines 18–21 in `less for/`):

```math
\begin{aligned}
y_{1, 49}^{(1)} &= W_{\mathrm{csp}}(:, 1)^T X_{49}^{(1)}, &\qquad y_{1, 49}^{(2)} &= W_{\mathrm{csp}}(:, 1)^T X_{49}^{(2)}, \\
y_{30, 49}^{(1)} &= W_{\mathrm{csp}}(:, 30)^T X_{49}^{(1)}, &\qquad y_{30, 49}^{(2)} &= W_{\mathrm{csp}}(:, 30)^T X_{49}^{(2)},
\end{aligned}
```

yields the time-domain waveforms shown in Figure 1.1 of the report and the following sample variances (Page 2 of the report):

```math
\text{49th Trial Variances:} \quad
\begin{cases}
\text{1st Filter } W_{\mathrm{csp}}(:, 1): & \begin{cases} \text{Class 1:} & \mathrm{Var}\big(y_{1, 49}^{(1)}\big) = 1.406656, \\ \text{Class 2:} & \mathrm{Var}\big(y_{1, 49}^{(2)}\big) = 0.868848, \end{cases} \\[12pt]
\text{30th Filter } W_{\mathrm{csp}}(:, 30): & \begin{cases} \text{Class 1:} & \mathrm{Var}\big(y_{30, 49}^{(1)}\big) = 1.008859, \\ \text{Class 2:} & \mathrm{Var}\big(y_{30, 49}^{(2)}\big) = 7.579428. \end{cases}
\end{cases}
```

As expected, the first filter produces higher variance for Class 1 ($1.4067 > 0.8688$), whereas the 30th filter produces much higher variance for Class 2 ($7.5794 \gg 1.0089$).

#### Part 2 — Spatial Filter Topography and Dominant EEG Channels

Because each filtered output $y_k(t) = \sum_{m=1}^{30} W_{\mathrm{csp}}(m, k) X_m(t)$ is a linear combination of the $30$ EEG channels, the magnitude $|W_{\mathrm{csp}}(m, k)|$ measures the contribution of the $m$-th scalp electrode to the $k$-th spatial filter. Inspecting the discrete stem plots `stem(abs(W_csp(:, 1)))` and `stem(abs(W_csp(:, 30)))` in Figure 1.2 of the report (Page 2) reveals:

- **1st Spatial Filter ($W_{\mathrm{csp}}(:, 1)$, dominant for Class 1 — Foot Motor Imagery):**
  - **Channel 8:** $|W_{\mathrm{csp}}(8, 1)| = 0.511170$ (largest weight)
  - **Channel 12:** $|W_{\mathrm{csp}}(12, 1)| = 0.364258$ (second largest weight)
- **30th Spatial Filter ($W_{\mathrm{csp}}(:, 30)$, dominant for Class 2 — Mental Subtraction):**
  - **Channel 7:** $|W_{\mathrm{csp}}(7, 30)| = 0.728248$ (largest weight)
  - **Channel 11:** $|W_{\mathrm{csp}}(11, 30)| = 0.468695$ (second largest weight)

#### Part 3 — 14-Filter CSP Feature Extraction and Fisher LDA via GEVD

To compress each $30 \times 256$ trial into a discriminative low-dimensional feature vector, we retain the $7$ leading CSP filters (maximizing Class 1 variance) and the $7$ trailing CSP filters (maximizing Class 2 variance):

```math
W_{\mathrm{imp}} = \begin{bmatrix} W_{\mathrm{csp}}(:, 1:7) & W_{\mathrm{csp}}(:, 24:30) \end{bmatrix} \in \mathbb{R}^{30 \times 14}.
```

For each training trial $t \in \{1, 2, \dots, 60\}$ of class $i \in \{1, 2\}$, we project $X_t^{(i)} \in \mathbb{R}^{30 \times 256}$ onto $W_{\mathrm{imp}}$ and compute the temporal variance of each of the $14$ filtered rows (`var(..., 0, 2)`), yielding a $14 \times 1$ feature vector $x^{(i)}(t) \in \mathbb{R}^{14 \times 1}$:

```math
Z_t^{(i)} = W_{\mathrm{imp}}^T X_t^{(i)} \in \mathbb{R}^{14 \times 256},
\qquad
x_k^{(i)}(t) = \mathrm{Var}\!\left(Z_t^{(i)}(k, :)\right), \quad k = 1, \dots, 14.
```

From the $N_1 = N_2 = 60$ training feature vectors $\left\{x^{(i)}(t)\right\}_{t=1}^{60}$ of each class, we compute the $14 \times 1$ class sample means $\mu_1, \mu_2$ and the $14 \times 14$ within-class sample covariance matrices $\Sigma_1, \Sigma_2$:

```math
\mu_i = \frac{1}{N_i} \sum_{t=1}^{N_i} x^{(i)}(t) \in \mathbb{R}^{14 \times 1},
\qquad
\Sigma_i = \frac{1}{N_i} \sum_{t=1}^{N_i} \left(x^{(i)}(t) - \mu_i\right)\left(x^{(i)}(t) - \mu_i\right)^T \in \mathbb{R}^{14 \times 14},
\qquad i \in \{1, 2\}.
```

**Fisher's Linear Discriminant Analysis (LDA)** seeks a unit-norm linear projection vector $W_{\mathrm{lda}} \in \mathbb{R}^{14 \times 1}$ that maximizes the ratio of the between-class scatter to the within-class scatter:

```math
J_{\mathrm{LDA}}(w) = \frac{\left(w^T \mu_1 - w^T \mu_2\right)^2}{w^T \left(\Sigma_1 + \Sigma_2\right) w}
= \frac{w^T S_B w}{w^T S_W w},
```

where the between-class scatter matrix $S_B$ and within-class scatter matrix $S_W$ are

```math
S_B = (\mu_1 - \mu_2)(\mu_1 - \mu_2)^T \in \mathbb{R}^{14 \times 14},
\qquad
S_W = \Sigma_1 + \Sigma_2 \in \mathbb{R}^{14 \times 14}.
```

Maximizing $J_{\mathrm{LDA}}(w)$ is equivalent to the Generalized Eigenvalue Decomposition problem (`[U, D] = eig(one, two)` in `Section1.m`):

```math
S_B w = \lambda S_W w
\quad \Longleftrightarrow \quad
\mathrm{GEVD}\!\left((\mu_1 - \mu_2)(\mu_1 - \mu_2)^T,\, \Sigma_1 + \Sigma_2\right).
```

Because $S_B = (\mu_1 - \mu_2)(\mu_1 - \mu_2)^T$ is an outer product of a single vector with itself, $\mathrm{rank}(S_B) = 1$, so only the first generalized eigenvalue is non-zero ($\lambda_1 = 3.7805$) and the remaining $13$ eigenvalues are zero up to numerical precision (Page 3 of the report):

```math
D_{\mathrm{lda}} = \begin{bmatrix}
3.7805 & 0 & 0 & \cdots & 0 & 0 \\
0 & \approx 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \approx 0 & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & \cdots & \approx 0 & 0 \\
0 & 0 & 0 & \cdots & 0 & \approx 0
\end{bmatrix}_{14 \times 14}.
```

Selecting the principal eigenvector corresponding to $\lambda_1 = 3.7805$ and normalizing it to unit Euclidean norm ($\|W_{\mathrm{lda}}\|_2 = 1$) gives the $14 \times 1$ optimal LDA weight vector (Page 4 of the report):

```math
W_{\mathrm{lda}} = \begin{bmatrix}
0.0052 \\
0.5123 \\
-7.8671 \times 10^{-4} \\
0.3894 \\
-0.0075 \\
0.4989 \\
0.5197 \\
0.0295 \\
-0.0306 \\
-0.1383 \\
-0.0182 \\
-0.2058 \\
0.0462 \\
0.0316
\end{bmatrix} \in \mathbb{R}^{14 \times 1}.
```

Projecting the 14D class mean vectors onto $W_{\mathrm{lda}}$ gives the 1D projected class means and midpoint decision threshold $c$:

```math
\begin{aligned}
\mu^{(1)} &= W_{\mathrm{lda}}^T \mu_1 = 2.8995, \\
\mu^{(2)} &= W_{\mathrm{lda}}^T \mu_2 = 1.1926, \\
c &= \frac{\mu^{(1)} + \mu^{(2)}}{2} = \frac{2.899535 + 1.192601}{2} = 2.0461.
\end{aligned}
```

*(Note: In the Persian report text on Page 3, the scalar $c = 2.0461$ is written with two middle digits transposed as `2.4061`, while the MATLAB scripts `Section1.m` compute and apply the exact midpoint `c = 2.046068`.)*

#### Parts 4 & 5 — Test Set Classification Rule and Accuracy Evaluation

For each of the $N_{\mathrm{test}} = 40$ unlabeled test trials $X_{\mathrm{test}, m} \in \mathbb{R}^{30 \times 256}$ ($m = 1, \dots, 40$), we extract the $14 \times 1$ CSP variance vector $x_{\mathrm{test}}(m)$ using $W_{\mathrm{imp}}$ and compute the scalar LDA projection $z_m = W_{\mathrm{lda}}^T x_{\mathrm{test}}(m)$. Because $\mu^{(1)} = 2.8995 > c > \mu^{(2)} = 1.1926$, the minimum-distance linear classifier assigns:

```math
\hat{y}_m = \begin{cases}
1, & \text{if } W_{\mathrm{lda}}^T x_{\mathrm{test}}(m) \ge c, \\
2, & \text{if } W_{\mathrm{lda}}^T x_{\mathrm{test}}(m) < c,
\end{cases}
```

which is implemented compactly in `Section1.m` (line 82 in `less for/`, line 91 in `more for/`) as:

```matlab
MyLabel = W_lda.' * var_num;
MyLabel = 1 + (MyLabel < c);
```

Comparing `MyLabel` with the ground-truth vector `TestLabel` (where trials $1\text{–}20$ belong to Class 1 and trials $21\text{–}40$ belong to Class 2):

- **Correctly classified trials:** $34$ out of $40$ trials ($\mathbf{85\%}$ accuracy).
- **Misclassified trials:** $6$ out of $40$ trials ($\mathbf{15\%}$ error rate), occurring at test trial indices

```math
m_{\mathrm{err}} \in \{6, 16, 32, 37, 39, 40\},
```

specifically $2$ false positives in Class 1 (trials $6$ and $16$ predicted as Class 2) and $4$ false negatives in Class 2 (trials $32, 37, 39, 40$ predicted as Class 1), as displayed in Figure 1.3 of the report (Page 5).

### 4.2 Section 2: SSVEP Frequency Recognition via Canonical Correlation Analysis (`Section2.m`)

#### Harmonic Reference Template Construction

The five candidate SSVEP visual flicker frequencies in `hw3-2.mat` are

```math
f_1 = 6.5\ \mathrm{Hz}, \quad
f_2 = 7.35\ \mathrm{Hz}, \quad
f_3 = 8.3\ \mathrm{Hz}, \quad
f_4 = 9.6\ \mathrm{Hz}, \quad
f_5 = 11.61\ \mathrm{Hz}.
```

Over the $5\ \mathrm{s}$ recording window sampled at $f_s = 250\ \mathrm{Hz}$, the discrete time vector has $T = 1250$ samples:

```math
t = \left[0,\, \frac{1}{f_s},\, \frac{2}{f_s},\, \dots,\, 5 - \frac{1}{f_s}\right] \in \mathbb{R}^{1 \times 1250}.
```

Including $K = 6$ harmonics (since for the lowest frequency $f_1 = 6.5\ \mathrm{Hz}$, the $6$-th harmonic is $6 \times 6.5 = 39\ \mathrm{Hz} \le 40\ \mathrm{Hz}$, and fixing $K = 6$ harmonics across all five frequencies allows stacking the templates into a uniform 3D tensor), the reference template matrix $X_{t_i} \in \mathbb{R}^{12 \times 1250}$ for stimulus frequency $f_i$ ($i = 1, \dots, 5$) is constructed as (`Section2.m` lines 9–17):

```math
X_{t_i} = \begin{bmatrix}
\sin(2\pi f_i t) \\
\cos(2\pi f_i t) \\
\sin(2 \times 2\pi f_i t) \\
\cos(2 \times 2\pi f_i t) \\
\vdots \\
\sin(6 \times 2\pi f_i t) \\
\cos(6 \times 2\pi f_i t)
\end{bmatrix} \in \mathbb{R}^{12 \times 1250},
\qquad i = 1, 2, 3, 4, 5.
```

Concatenating $X_{t_1}, \dots, X_{t_5}$ along the third dimension (`Xt = cat(3, Xt1, Xt2, Xt3, Xt4, Xt5)`) yields the $12 \times 1250 \times 5$ template tensor $X_t$.

#### Mathematical Derivation of CCA via Rayleigh Quotient and GEVD

For a given test EEG trial $Y \in \mathbb{R}^{6 \times 1250}$ and a candidate reference template $X = X^{(f_j)} = X_{t_j} \in \mathbb{R}^{12 \times 1250}$, **Canonical Correlation Analysis (CCA)** seeks linear spatial/harmonic combination vectors $a \in \mathbb{R}^{12 \times 1}$ and $b \in \mathbb{R}^{6 \times 1}$ that maximize the absolute Pearson correlation coefficient $\rho$ between the 1D projected signals $u = a^T X \in \mathbb{R}^{1 \times 1250}$ and $v = b^T Y \in \mathbb{R}^{1 \times 1250}$ (Pages 7–8 of the report):

```math
(a^*, b^*) = \arg\max_{a, b} \left|\rho\left(a^T X, b^T Y\right)\right|,
\qquad
\rho = \frac{a^T X Y^T b}{\sqrt{a^T X X^T a}\,\sqrt{b^T Y Y^T b}}.
```

Define the unnormalized sample auto-correlation and cross-correlation matrices:

```math
\begin{aligned}
R_x &= X X^T \in \mathbb{R}^{12 \times 12}, &\qquad R_y &= Y Y^T \in \mathbb{R}^{6 \times 6}, \\
R_{xy} &= X Y^T \in \mathbb{R}^{12 \times 6}, &\qquad R_{yx} &= Y X^T = R_{xy}^T \in \mathbb{R}^{6 \times 12}.
\end{aligned}
```

Applying the whitening change of variables

```math
c = R_x^{1/2} a \quad \Longleftrightarrow \quad a = R_x^{-1/2} c,
\qquad
d = R_y^{1/2} b \quad \Longleftrightarrow \quad b = R_y^{-1/2} d,
```

transforms the correlation magnitude into

```math
|\rho| = \frac{\left| c^T R_x^{-1/2} R_{xy} R_y^{-1/2} d \right|}{\sqrt{c^T c}\,\sqrt{d^T d}}.
```

Defining the vector $e^T = c^T R_x^{-1/2} R_{xy} R_y^{-1/2}$ allows writing the numerator as the inner product $|e^T d|$. By the **Cauchy-Schwarz inequality**, $|e^T d| \le \sqrt{e^T e}\,\sqrt{d^T d}$ with equality if and only if $d$ is collinear with $e$. Substituting this upper bound cancels $\sqrt{d^T d}$ and reduces the optimization over $c$ to a **Rayleigh quotient**:

```math
|\rho| \le \frac{\sqrt{e^T e}\,\sqrt{d^T d}}{\sqrt{c^T c}\,\sqrt{d^T d}} = \frac{\sqrt{e^T e}}{\sqrt{c^T c}}
\quad \Longrightarrow \quad
|\rho|^2 = \frac{c^T \Sigma_1 c}{c^T I_{12} c},
```

where the symmetric positive semi-definite CCA matrices $\Sigma_1 \in \mathbb{R}^{12 \times 12}$ and $\Sigma_2 \in \mathbb{R}^{6 \times 6}$ are defined as:

```math
\begin{aligned}
\Sigma_1 &= R_x^{-1/2} R_{xy} R_y^{-1/2} R_y^{-1/2} R_{yx} R_x^{-1/2} = R_x^{-1/2} R_{xy} R_y^{-1} R_{yx} R_x^{-1/2} \in \mathbb{R}^{12 \times 12}, \\
\Sigma_2 &= R_y^{-1/2} R_{yx} R_x^{-1/2} R_x^{-1/2} R_{xy} R_y^{-1/2} = R_y^{-1/2} R_{yx} R_x^{-1} R_{xy} R_y^{-1/2} \in \mathbb{R}^{6 \times 6}.
\end{aligned}
```

Maximizing the Rayleigh quotients for $c$ and $d$ yields two Generalized Eigenvalue Decompositions (`Section2.m` lines 30–33 in `more for/`, lines 32–35 in `less for/`):

```math
\begin{cases}
c : \mathrm{GEVD}(\Sigma_1, I_{12}) \quad \Longrightarrow \quad a = R_x^{-1/2} c_1, \\[6pt]
d : \mathrm{GEVD}(\Sigma_2, I_{6}) \quad \Longrightarrow \quad b = R_y^{-1/2} d_1,
\end{cases}
```

where $c_1$ and $d_1$ are the dominant unit-norm eigenvectors of $\Sigma_1$ and $\Sigma_2$, respectively. Because $\|c_1\|_2 = 1$ and $\|d_1\|_2 = 1$, the canonical variates $a^T X$ and $b^T Y$ automatically satisfy unit-norm scaling:

```math
a^T R_x a = c_1^T c_1 = 1,
\qquad
b^T R_y b = d_1^T d_1 = 1,
```

so the inner product `(a.' * X) * (b.' * Y).'` in `Section2.m` (line 34 in `more for/`, line 36 in `less for/`) directly evaluates the canonical correlation $\rho_j$:

```math
\rho_j = \left(a^T X_{t_j}\right)\left(b^T Y_i\right)^T = a^T R_{xy} b = \pm \sqrt{\lambda_{\max}(\Sigma_1)}.
```

For each trial $i \in \{1, \dots, 15\}$, the algorithm evaluates $|\rho_j|$ across all five candidate templates $j \in \{1, 2, 3, 4, 5\}$ and selects the stimulus frequency that maximizes $|\rho_j|$:

```math
\hat{j}_i = \arg\max_{j \in \{1, \dots, 5\}} |\rho_j|,
\qquad
\hat{f}_{\mathrm{pred}}(i) = \mathrm{freq}(\hat{j}_i).
```

## 5. Results & Discussion

### 5.1 Section 1 — Summary of CSP and LDA EEG Classification Results (`Section1.m`)

All results in Section 1 are **completely deterministic** because the dataset `hw3-1.mat` is fixed and no random number generation is used:

| Problem Part & Quantity | Mathematical Expression | MATLAB Expression (`Section1.m`) | Exact Numerical Result / Observation |
| --- | --- | --- | --- |
| Part 1 — Extreme CSP Eigenvalues | $R_x^{(1)} w_k = \lambda_k R_x^{(2)} w_k$ | `[U, D] = eig(Rx_1, Rx_2)` (sorted descending) | $\lambda_1 = 2.2494,\ \lambda_2 = 1.4902,\ \lambda_3 = 1.3556,\ \dots,\ \lambda_{29} = 0.3085,\ \lambda_{30} = 0.3010$ |
| Part 1 — Trial 49 Variance with 1st Filter ($W_{\mathrm{csp}}(:, 1)$) | $\mathrm{Var}\!\left(W_{\mathrm{csp}}(:, 1)^T X_{49}^{(r)}\right)$ | `var(W1_exp49_r1)` and `var(W2_exp49_r1)` | Class 1: $\mathbf{1.406656}$ vs. Class 2: $\mathbf{0.868848}$ (higher variance for Class 1) |
| Part 1 — Trial 49 Variance with 30th Filter ($W_{\mathrm{csp}}(:, 30)$) | $\mathrm{Var}\!\left(W_{\mathrm{csp}}(:, 30)^T X_{49}^{(r)}\right)$ | `var(W1_exp49_r30)` and `var(W2_exp49_r30)` | Class 1: $\mathbf{1.008859}$ vs. Class 2: $\mathbf{7.579428}$ (higher variance for Class 2) |
| Part 2 — Dominant Channels in 1st Filter | $\arg\max_m |W_{\mathrm{csp}}(m, 1)|$ | `stem(abs(W_csp(:, 1)))` | **Channel 8** ($0.511170$) and **Channel 12** ($0.364258$) |
| Part 2 — Dominant Channels in 30th Filter | $\arg\max_m |W_{\mathrm{csp}}(m, 30)|$ | `stem(abs(W_csp(:, 30)))` | **Channel 7** ($0.728248$) and **Channel 11** ($0.468695$) |
| Part 3 — LDA Generalized Eigenvalues | $\mathrm{GEVD}(S_B, S_W)$ | `eig((mu_1-mu_2)*(mu_1-mu_2).', sigma_1+sigma_2)` | $\lambda_1 = \mathbf{3.7805}$, $\lambda_2 \approx \dots \approx \lambda_{14} \approx 0$ ($\mathrm{rank}(S_B) = 1$) |
| Part 3 — Projected Class Means & LDA Threshold $c$ | $\mu^{(i)} = W_{\mathrm{lda}}^T \mu_i,\ c = \frac{\mu^{(1)} + \mu^{(2)}}{2}$ | `c = (Mu1 + Mu2) / 2` | $\mu^{(1)} = 2.8995,\ \mu^{(2)} = 1.1926,\ c = \mathbf{2.0461}$ (typed as `2.4061` in report text) |
| Parts 4 & 5 — Test Set Classification Accuracy | $\hat{y}_m = 1 + \mathbb{I}\{W_{\mathrm{lda}}^T x_{\mathrm{test}}(m) < c\}$ | `sum(TestLabel == MyLabel) / 40 * 100` | $\mathbf{85\%}$ ($34 / 40$ correct; $6$ errors at trials $6, 16, 32, 37, 39, 40$) |

### 5.2 Section 2 — Summary of SSVEP Frequency Recognition via CCA (`Section2.m`)

The table below lists the canonical correlation magnitudes $|\rho_j|$ across all five candidate reference frequencies $\mathrm{freq} = [6.5, 7.35, 8.3, 9.6, 11.61]\ \mathrm{Hz}$ for each of the $15$ SSVEP trials in `hw3-2.mat`:

| Trial Index $i$ | $|\rho_1|$ ($6.5\ \mathrm{Hz}$) | $|\rho_2|$ ($7.35\ \mathrm{Hz}$) | $|\rho_3|$ ($8.3\ \mathrm{Hz}$) | $|\rho_4|$ ($9.6\ \mathrm{Hz}$) | $|\rho_5|$ ($11.61\ \mathrm{Hz}$) | Max Index $\hat{j}_i$ | Predicted Frequency `mylabel(i)` | True Frequency `label(i)` | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Trial 1 | **0.4396** | 0.1046 | 0.1130 | 0.1356 | 0.1200 | 1 | $6.5\ \mathrm{Hz}$ | $6.5\ \mathrm{Hz}$ | Correct |
| Trial 2 | **0.5292** | 0.1035 | 0.1542 | 0.1267 | 0.1399 | 1 | $6.5\ \mathrm{Hz}$ | $6.5\ \mathrm{Hz}$ | Correct |
| Trial 3 | **0.5616** | 0.1427 | 0.1052 | 0.1107 | 0.1575 | 1 | $6.5\ \mathrm{Hz}$ | $6.5\ \mathrm{Hz}$ | Correct |
| Trial 4 | 0.1153 | **0.5613** | 0.1883 | 0.1525 | 0.1059 | 2 | $7.35\ \mathrm{Hz}$ | $7.35\ \mathrm{Hz}$ | Correct |
| Trial 5 | 0.1700 | **0.4597** | 0.1922 | 0.0906 | 0.1321 | 2 | $7.35\ \mathrm{Hz}$ | $7.35\ \mathrm{Hz}$ | Correct |
| Trial 6 | 0.1452 | **0.5313** | 0.1657 | 0.1106 | 0.1798 | 2 | $7.35\ \mathrm{Hz}$ | $7.35\ \mathrm{Hz}$ | Correct |
| Trial 7 | 0.1598 | 0.1202 | **0.5438** | 0.1347 | 0.1606 | 3 | $8.3\ \mathrm{Hz}$ | $8.3\ \mathrm{Hz}$ | Correct |
| Trial 8 | 0.1294 | 0.1137 | **0.5653** | 0.1002 | 0.1090 | 3 | $8.3\ \mathrm{Hz}$ | $8.3\ \mathrm{Hz}$ | Correct |
| Trial 9 | 0.1067 | 0.1295 | **0.4015** | 0.1358 | 0.1410 | 3 | $8.3\ \mathrm{Hz}$ | $8.3\ \mathrm{Hz}$ | Correct |
| Trial 10 | 0.1436 | 0.1402 | 0.2082 | **0.3670** | 0.1579 | 4 | $9.6\ \mathrm{Hz}$ | $9.6\ \mathrm{Hz}$ | Correct |
| Trial 11 | 0.1535 | 0.1473 | 0.2000 | **0.3636** | 0.1272 | 4 | $9.6\ \mathrm{Hz}$ | $9.6\ \mathrm{Hz}$ | Correct |
| Trial 12 | 0.1556 | 0.1449 | 0.1491 | **0.4662** | 0.1316 | 4 | $9.6\ \mathrm{Hz}$ | $9.6\ \mathrm{Hz}$ | Correct |
| Trial 13 | 0.1429 | 0.1794 | 0.1826 | 0.1192 | **0.3689** | 5 | $11.61\ \mathrm{Hz}$ | $11.61\ \mathrm{Hz}$ | Correct |
| Trial 14 | 0.1513 | 0.1957 | 0.2010 | 0.1596 | **0.4587** | 5 | $11.61\ \mathrm{Hz}$ | $11.61\ \mathrm{Hz}$ | Correct |
| Trial 15 | 0.1827 | 0.1814 | 0.1873 | 0.1225 | **0.5004** | 5 | $11.61\ \mathrm{Hz}$ | $11.61\ \mathrm{Hz}$ | Correct |

Across all $15$ trials, the canonical correlation $|\rho_j|$ attains a clear global maximum at the true stimulus frequency, yielding a **$\mathbf{100\%}$ classification accuracy ($15 / 15$)** without any supervised training data (Figure 2.2 on Page 8 of the report).

### 5.3 Notes on the Submitted Solutions / Implementation Notes

- **Loop-based (`codes/more for/`) vs. Tensorized (`codes/less for/`) Implementations:**
  - In **Section 1**, `codes/more for/Section1.m` uses three `for` loops over the trial dimension (lines 12–15 for `Rx_1` and `Rx_2`, lines 59–66 for training feature variances, and lines 86–89 for test feature variances). In `codes/less for/Section1.m`, all three `for` loops are eliminated by using `pagemtimes` and `pagetranspose` (`Rx_1 = mean(pagemtimes(TrainData_class1, pagetranspose(TrainData_class1)), 3)` and `W1_imp = pagemtimes(important_filters.', TrainData_class1)` followed by `var(..., 0, 2)` and `reshape`). Both scripts produce identical numerical results to machine precision.
  - In **Section 2**, `codes/less for/Section2.m` precomputes the template and trial auto-correlation tensors `R_X = pagemtimes(Xt, pagetranspose(Xt))` and `R_Y = pagemtimes(data, pagetranspose(data))` outside the loops and computes `R_XY` and `R_YX` across all 5 templates simultaneously inside the outer trial loop, while keeping the inner loop over $j = 1, \dots, 5$ to evaluate fractional matrix powers (`R_x^(-1/2)`) and `eig`.
- **Global vs. Per-Trial Channel Mean Removal in `Section1.m` (lines 5–7):** Lines 5–7 of `Section1.m` call `mean(TrainData_class1, [2 3])`, which subtracts the $30 \times 1$ channel mean averaged across both time samples (dimension 2) and trials (dimension 3) rather than subtracting each trial's individual temporal mean `mean(TrainData_class1, 2)`. Because each raw trial in `hw3-1.mat` is already nearly zero-mean, both centering conventions yield the same $85\%$ ($34/40$) test accuracy.
- **Harmonic Count ($K = 6$) in `Section2.m` (line 10):** The assignment sheet asks for template harmonics up to $40\ \mathrm{Hz}$. For the lowest stimulus frequency $f_1 = 6.5\ \mathrm{Hz}$, $\lfloor 40 / 6.5 \rfloor = 6$ harmonics lie below $40\ \mathrm{Hz}$ ($6 \times 6.5 = 39\ \mathrm{Hz}$). In `Section2.m` (line 10: `for i = 1:6`), all five template matrices `Xt1` through `Xt5` are constructed with $K = 6$ harmonics ($12$ rows each) so that they have identical matrix dimensions ($12 \times 1250$) and can be concatenated into a single 3D tensor `Xt = cat(3, Xt1, Xt2, Xt3, Xt4, Xt5)` of size $12 \times 1250 \times 5$.
- **Eigenvalue Ordering in `Section2.m` (`eig(Sigma1, eye(12))` and `eig(Sigma2, eye(6))`):** In `Section2.m`, `[c, D1] = eig(Sigma1, eye(12))` and `[d, D2] = eig(Sigma2, eye(6))` are called without a subsequent `sort(..., 'descend')` before extracting `c(:, 1)` and `d(:, 1)`. Because `Sigma1` and `Sigma2` are formed via repeated matrix multiplications (`R_x^(-1/2) * R_xy * R_y^(-1/2) * R_y^(-1/2) * R_yx * R_x^(-1/2)`), tiny floating-point asymmetries ($\sim 10^{-16}$) cause MATLAB's two-argument `eig(A, B)` to dispatch to the non-symmetric QZ generalized solver, which orders the largest-magnitude eigenvalues first (`c(:, 1)` and `d(:, 1)`).
- **Minor Typographical Slips in the Persian Report (Pages 2, 3, 4, and 7):**
  - On **Page 2 (Figure 1.1 legend)** and in `Section1.m` (lines 30 and 36), the plot legend strings are labeled `$Var(Ch1)$` and `$Var(Ch2)$`, which refer to the filtered time-domain signals of **Class 1** and **Class 2** (not Channels 1 and 2).
  - On **Page 3 (Section 1, Part ج)**, the formula for the within-class covariance matrix $\Sigma_i$ writes the summation upper limit as $N_1$ and the superscript inside the summand as $x^{(1)}(t)$ instead of $N_i$ and $x^{(i)}(t) - \mu_i$ for $i = 1, 2$; in `Section1.m` (lines 69–70), `sigma_1` and `sigma_2` are computed properly using `x_1_60_exps - mu_1` and `x_2_60_exps - mu_2`.
  - On **Page 3 (bottom line)**, the LDA threshold computed by `Section1.m` is `c = 2.0461` (`2.046068`), which was typed with two digits transposed as `2.4061` in the report narrative.
  - On **Page 4 (Section 1, Part د)**, the last sentence of the paragraph states that if the projected test values are "less than (`کمتر`) the threshold $c$ they belong to Class 2 and if they are less than (`کمتر`) the threshold they belong to Class 1" (repeating "less than" twice instead of "greater than" for Class 1); in `Section1.m` (line 91: `MyLabel = 1 + (MyLabel < c)`), values $\ge c$ are properly assigned to Class 1 and values $< c$ to Class 2.
  - On **Page 7 (Section 2, Part الف)**, the fourth correlation matrix equation is written as $R_x = Y X^T$ instead of $R_{yx} = Y X^T$, the second square root in the denominator of $\rho$ is written as $\sqrt{b^T Y Y^T Y}$ instead of $\sqrt{b^T Y Y^T b}$, and the whitening change-of-variables brace writes $c = R_x^{-1/2} a$ and $y = R_y^{-1/2} b$ instead of $c = R_x^{1/2} a$ and $d = R_y^{1/2} b$ (whereas Page 8 writes the inverse relations $a = R_x^{-1/2} c$ and $b = R_y^{-1/2} d$, matching `Section2.m`).

## 6. How to Use

### 6.1 Running the MATLAB Scripts

Open MATLAB, navigate to either `codes/less for` (vectorized tensor version) or `codes/more for` (loop-based version), and run `Section1` and `Section2`:

```matlab
cd 'codes/less for'
Section1
Section2
```

or equivalently for the loop-based scripts:

```matlab
cd 'codes/more for'
Section1
Section2
```

- **Dependencies:**
  - Both `Section1.m` and `Section2.m` in `codes/more for/` use only base MATLAB functions (`eig`, `sort`, `var`, `mean`, `cat`, `plot`, `stem`).
  - `codes/less for/Section1.m` and `codes/less for/Section2.m` additionally use MATLAB's built-in page-wise tensor functions `pagemtimes` and `pagetranspose` (introduced in MATLAB R2020b).
- **Generated Figures:**
  - `Section1.m` opens **3 figures**:
    1. Two-panel subplot of the CSP-filtered time-domain signals for Trial 49 of Class 1 and Class 2 using the 1st spatial filter $W_{\mathrm{csp}}(:, 1)$ and the 30th spatial filter $W_{\mathrm{csp}}(:, 30)$ (Figure 1.1 in the report).
    2. Two-panel stem plot of the channel-wise magnitudes $|W_{\mathrm{csp}}(:, 1)|$ and $|W_{\mathrm{csp}}(:, 30)|$ across all $30$ EEG channels (Figure 1.2 in the report).
    3. Stem comparison plot of `TestLabel` (black squares) versus predicted `MyLabel` (magenta crosses) across the $40$ test trials (Figure 1.3 in the report).
  - `Section2.m` opens **1 figure**:
    1. Stem comparison plot of true stimulus frequencies `label` (black squares) versus CCA-predicted frequencies `mylabel` (magenta crosses) across the $15$ SSVEP trials (Figure 2.2 in the report).

### 6.2 Expected Console Output (`Section1.m` and `Section2.m`)

Because both datasets (`hw3-1.mat` and `hw3-2.mat`) and all algorithms are completely deterministic, running `Section1.m` prints the exact console output below:

```text
Variances of Filtered Signal of 1st Filter is:
	1.406656 for 1st Class
	0.868848 for 2nd Class

Variances of Filtered Signal of 30th Filter is:
	1.008859 for 1st Class
	7.579428 for 2nd Class

Prediction Rate is: 85%
```

Running `Section2.m` prints the exact console output below:

```text
Prediction Rate is: 100%
```

### 6.3 Reading the Theoretical Derivations and Report

1. Use **Section 3** for the complete English translation of the assignment problems and **Section 4** for the step-by-step mathematical derivations of CSP, Fisher LDA, and CCA.
2. Use **Section 5** to review the quantitative results, canonical correlation tables, and implementation notes alongside the original Persian report [BSS\_HW#3\_810100205\_Arman\_Majidi.pdf](https://www.google.com/search?q=./BSS_HW%25233_810100205_Arman_Majidi.pdf) and assignment sheet [hw3.pdf](https://www.google.com/search?q=./hw3.pdf).

## 7. References

1. Akhavan, S. *Blind Source Separation: Homework 3*. Course assignment sheet and lecture notes, University of Tehran.
2. Blankertz, B., Tomioka, R., Lemm, S., Kawanabe, M., & Müller, K.-R. (2008). Optimizing spatial filters for robust EEG single-trial analysis. *IEEE Signal Processing Magazine*, 25(1), 41–56.
3. Lin, Z., Zhang, C., Wu, W., & Gao, X. (2006). Frequency recognition based on canonical correlation analysis for SSVEP-based BCIs. *IEEE Transactions on Biomedical Engineering*, 53(12), 2610–2614.
4. Comon, P., & Jutten, C. (Eds.). (2010). *Handbook of Blind Source Separation: Independent Component Analysis and Applications*. Academic Press.
