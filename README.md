# Blind Source Separation (BSS)

**Course:** Blind Source Separation  
**Instructor:** Dr. Saeed Akhavan  
**Author:** Arman Majidi  
**Student ID:** 810100205  
**University:** University of Tehran  
**Semester:** Spring 2024-2025

## 1. Overview

This repository contains my **homework assignments (HW1–HW10)** and **final project** for the graduate course *Blind Source Separation*, including MATLAB implementations, analytical derivations, and simulation results. Each folder has its own detailed `README.md` with the full problem statements, mathematical solutions, and results in English.

## 2. Repository Structure

| Folder | Description |
| --- | --- |
| [Homeworks/](./Homeworks) | Ten homework assignments (`HW#01`–`HW#10`), each with a `README.md`, MATLAB `codes/`, the assignment sheet, and the submitted Persian report. |
| [Project/](./Project) | Final project: multi-class motor execution EEG classification with a `README.md`, MATLAB `codes/`, and the project and report PDFs. |

## 3. Course Summary

Blind Source Separation recovers unknown source signals $`\mathbf{s}(t)`$ from observed mixtures $`\mathbf{x}(t)`$ with little or no knowledge of the mixing process. The standard instantaneous linear model is:


```math
\mathbf{x}(t) = \mathbf{A}\,\mathbf{s}(t) + \mathbf{n}(t), \qquad \hat{\mathbf{s}}(t) = \mathbf{B}\,\mathbf{x}(t)
```


where $`\mathbf{A}`$ is the unknown mixing matrix and $`\mathbf{B} \approx \mathbf{A}^{-1}`$ is the separation matrix to be estimated. The course covers the main families of BSS methods, according to the assumption they exploit:

- **Geometry, preprocessing and subspace methods:** mixing matrix estimation from scatter-plot geometry, spatial whitening, SVD, beamforming, and MUSIC.
- **Second-order statistics:** non-stationarity and temporal correlation, via joint diagonalization with Generalized Eigenvalue Decomposition (GEVD); Common Spatial Patterns (CSP) and Linear Discriminant Analysis (LDA).
- **Sparsity:** sparse recovery ($`\ell_0 / \ell_1`$ methods such as MP, OMP, BP, IRLS, and LASSO), dictionary learning (MOD, K-SVD), and sparse blind deconvolution.
- **Independence and higher-order statistics:** Independent Component Analysis (ICA) through mutual information minimization, kurtosis and negentropy maximization, and the FastICA algorithm.

## 4. Homeworks

| Homework | Topic |
| --- | --- |
| [HW#01](./Homeworks/HW%2301) | Geometric mixing matrix estimation, marginal distributions of linear mixtures, and constrained and unconstrained optimization |
| [HW#02](./Homeworks/HW%2302) | Dimensionality reduction, spatial whitening, SVD, and DOA and frequency estimation via beamforming and MUSIC |
| [HW#03](./Homeworks/HW%2303) | Two-class EEG classification with CSP and LDA, and SSVEP frequency recognition with CCA |
| [HW#04](./Homeworks/HW%2304) | Non-stationary BSS via joint diagonalization (GEVD and alternating minimization), and time-lagged GEVD for temporally correlated sources |
| [HW#05](./Homeworks/HW%2305) | Sparse signal recovery: subset selection, minimum $\ell_2$-norm, MP, OMP, BP, and IRLS |
| [HW#06](./Homeworks/HW%2306) | Noisy sparse recovery (BP, MP, LASSO), low-coherence frame design, and dictionary learning with MOD and K-SVD |
| [HW#07](./Homeworks/HW%2307) | Single-channel and multi-channel sparse blind deconvolution in the time and frequency domains |
| [HW#08](./Homeworks/HW%2308) | BSS via mutual information (KL divergence) minimization and polynomial score function estimation |
| [HW#09](./Homeworks/HW%2309) | ICA via mutual information minimization: deflationary separation after whitening, and equivariant relative-gradient separation |
| [HW#10](./Homeworks/HW%2310) | ICA via kurtosis and outlier-robust negentropy maximization, using gradient projection and FastICA |

## 5. Project

[**Multi-Class Motor Execution EEG Classification via Hierarchical Binary Tree CSP and LDA**](./Project)

A four-class BCI classification problem (right arm, right thumb, right foot, and rest) on 63-channel EEG recordings from 15 subjects in the 2nd Iranian National BCI Competition dataset. The problem is decomposed into a three-stage binary decision tree. Each stage uses a subject-specific Butterworth bandpass filter, CSP spatial filters, and a Fisher LDA classifier. Performance is evaluated with 100 randomized Leave-One-Out Cross-Validation (LOOCV) iterations per subject, giving a mean test accuracy of 64.77% across all 15 subjects.

## 6. How to Use

All implementations are written in MATLAB. Open MATLAB, navigate to the `codes/` folder of the desired homework or the project, and run the scripts. For example:

```matlab
cd "Homeworks/HW#01/codes"
section_1
```

- **Requirements:** MATLAB R2020b or later. Some scripts also need the Signal Processing Toolbox and the Statistics and Machine Learning Toolbox.
- For the project, copy the dataset files `subj_1.mat` through `subj_15.mat` into `Project/codes/` before running `subj1.m`–`subj15.m`.

## 7. References

1. Akhavan, S. *Blind Source Separation*. Course lecture notes and assignment sheets, University of Tehran.
2. Comon, P., & Jutten, C. (Eds.). (2010). *Handbook of Blind Source Separation: Independent Component Analysis and Applications*. Academic Press.
3. Hyvärinen, A., Karhunen, J., & Oja, E. (2001). *Independent Component Analysis*. Wiley.
