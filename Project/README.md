# Project: Multi-Class Motor Execution EEG Classification via Hierarchical Binary Tree Common Spatial Patterns (CSP) and Linear Discriminant Analysis (LDA)

**Course:** Blind Source Separation  
**Instructor:** Dr. Saeed Akhavan  
**Author:** Arman Majidi  
**Student ID:** 810100205  
**University:** University of Tehran  
**Semester:** Spring 2024-2025

## 1. Overview & Assignment Structure

This folder contains the **English problem statements, dataset recording protocol, mathematical derivations, MATLAB implementations, hyperparameter tuning logs, and cross-validation results for the Blind Source Separation Final Project**. The project addresses a four-class Brain-Computer Interface (BCI) motor execution classification problem on $63$-channel Electroencephalography (EEG) recordings from $15$ subjects in the **2nd Iranian National BCI Competition** dataset:

1. **Dataset & Experimental Protocol Analysis (`project.pdf`, `Recording.pdf`):** Examines $63$-channel EEG signals (plus a $64$-th trigger channel) sampled at $f_s = 2400\ \mathrm{Hz}$ during a $3\ \mathrm{s}$ motor execution window ($T = 7200$ samples per trial) across four movement classes: **Class 1 (Right Arm Movement)**, **Class 2 (Right Thumb Movement)**, **Class 3 (Right Foot Movement)**, and **Class 4 (Resting / No Movement)** across $15$ healthy right-handed subjects (`subj_1.mat` through `subj_15.mat`).
2. **Hierarchical Three-Stage Binary Decision Tree (`Report.pdf`, `Train.m`, `Test.m`, `Filter.m`, `Label.m`):** Decomposes the four-class EEG classification problem into a cascade of three binary One-vs-Rest / One-vs-One stages:
   - **Stage 1 (Classifier 1):** Separates **Class 4 (No Movement)** from the three active movement classes **$\{1, 2, 3\}$**.
   - **Stage 2 (Classifier 2):** Separates **Class 3 (Right Foot)** from the two upper-limb movement classes **$\{1, 2\}$**.
   - **Stage 3 (Classifier 3):** Separates **Class 2 (Right Thumb)** from **Class 1 (Right Arm)**.
   Each binary stage applies a subject- and stage-specific 7th-order Butterworth bandpass filter (`Filter.m`), extracts discriminative spatial filters via **Common Spatial Patterns (CSP)** solved through Generalized Eigenvalue Decomposition (GEVD), computes log/sample variance features across the leading and trailing CSP filters, and trains a **Fisher Linear Discriminant Analysis (LDA)** linear projection vector, midpoint threshold, and automatic decision polarity rule (`Train.m`, `Label.m`).
3. **Subject-Specific Hyperparameter Optimization & Monte Carlo Leave-One-Out Evaluation (`subj1.m`–`subj15.m`):** Evaluates each of the $15$ subjects independently over $100$ randomized **Leave-One-Out Cross-Validation (LOOCV)** iterations (`runNum = 100`), documenting three intermediate hyperparameter tuning stages alongside the final selected bandpass filter cutoffs and CSP filter counts, and reporting the resulting $4 \times 4$ probabilistic training and test confusion matrices.

All components of the project are mandatory; as noted in the assignment sheet and reference report, movement-onset detection was part of the original competition but is not required in the course project (which evaluates only the $4 \times 4$ training and test class confusion matrices). Furthermore, all MATLAB functions and scripts are vectorized using 3D tensor operations (`pagemtimes`, `pagetranspose`, `sosfilt`, and `accumarray`) so that only the outer $100$-iteration Monte Carlo cross-validation loop requires a `for` statement. Because the project prompt, recording protocol, reference competition report, and submitted solution report are all written in Persian, this `README.md` serves as a **complete, self-contained English reference for the experimental protocol, problem statement, mathematical derivations, codebase architecture, and subject-by-subject numerical results**.

## 2. Directory Contents

| File | Description |
| --- | --- |
| [codes/Filter.m](./codes/Filter.m) | Vectorized temporal bandpass filter function applying a 7th-order digital Butterworth bandpass filter (`butter`, `zp2sos`, `sosfilt`) along the time dimension (dimension 2) at $f_s = 2400\ \mathrm{Hz}$. |
| [codes/Train.m](./codes/Train.m) | Binary training function implementing 3D tensor mean removal, trial-averaged spatial covariance estimation (`pagemtimes`, `pagetranspose`), GEVD-based Common Spatial Patterns (`W_csp`), variance feature extraction, and GEVD-based Fisher LDA (`W_lda`, `thr`, `rule`). |
| [codes/Test.m](./codes/Test.m) | Hierarchical three-stage decision-tree evaluation function that cascades Classifiers 1, 2, and 3 on multi-trial 3D EEG tensors and accumulates the $4 \times 4$ confusion matrix via `sub2ind` and `accumarray`. |
| [codes/Label.m](./codes/Label.m) | Binary decision helper function that maps 1D LDA projections to class labels (`sectionNum + 1` vs. `1`) based on the midpoint threshold `thr` and mean ordering flag `rule`. |
| [codes/subj1.m](./codes/subj1.m) | Main LOOCV training, testing, and confusion-matrix evaluation script for Subject 1 (`filtersNum = 2`, bands `[20 35]`, `[0.1 17]`, `[0.1 4]` Hz). |
| [codes/subj2.m](./codes/subj2.m) | Main LOOCV training, testing, and confusion-matrix evaluation script for Subject 2 (`filtersNum = 4`, bands `[20 45]`, `[3 15]`, `[0.1 2]` Hz). |
| [codes/subj3.m](./codes/subj3.m) | Main LOOCV training, testing, and confusion-matrix evaluation script for Subject 3 (`filtersNum = 2`, bands `[20 40]`, `[0.1 10]`, `[0.1 3.6]` Hz). |
| [codes/subj4.m](./codes/subj4.m) | Main LOOCV training, testing, and confusion-matrix evaluation script for Subject 4 (`filtersNum = 2`, bands `[20 40]`, `[11 45]`, `[0.1 6]` Hz). |
| [codes/subj5.m](./codes/subj5.m) | Main LOOCV training, testing, and confusion-matrix evaluation script for Subject 5 (`filtersNum = 3`, bands `[20 40]`, `[13 19]`, `[1 6]` Hz). |
| [codes/subj6.m](./codes/subj6.m) | Main LOOCV training, testing, and confusion-matrix evaluation script for Subject 6 (`filtersNum = 3`, bands `[20 35]`, `[10 17]`, `[0.1 7]` Hz). |
| [codes/subj7.m](./codes/subj7.m) | Main LOOCV training, testing, and confusion-matrix evaluation script for Subject 7 (`filtersNum = 3`, bands `[20 45]`, `[11 16]`, `[1 8]` Hz). |
| [codes/subj8.m](./codes/subj8.m) | Main LOOCV training, testing, and confusion-matrix evaluation script for Subject 8 (`filtersNum = 4`, bands `[25 50]`, `[10 35]`, `[1 17]` Hz). |
| [codes/subj9.m](./codes/subj9.m) | Main LOOCV training, testing, and confusion-matrix evaluation script for Subject 9 (`filtersNum = 4`, bands `[20 40]`, `[10 19]`, `[1 8]` Hz). |
| [codes/subj10.m](./codes/subj10.m) | Main LOOCV training, testing, and confusion-matrix evaluation script for Subject 10 (`filtersNum = 4` in script / `3` in final report, bands `[20 40]`, `[10 45]`/`[10 40]`, `[2 5]` Hz). |
| [codes/subj11.m](./codes/subj11.m) | Main LOOCV training, testing, and confusion-matrix evaluation script for Subject 11 (`filtersNum = 4`, bands `[20 45]`, `[11 17]`, `[1 4]` Hz). |
| [codes/subj12.m](./codes/subj12.m) | Main LOOCV training, testing, and confusion-matrix evaluation script for Subject 12 (`filtersNum = 2`, bands `[20 40]`, `[5 20]`, `[0.1 2]` Hz). |
| [codes/subj13.m](./codes/subj13.m) | Main LOOCV training, testing, and confusion-matrix evaluation script for Subject 13 (`filtersNum = 5`, bands `[12 30]`, `[25 40]`, `[1 15]` Hz). |
| [codes/subj14.m](./codes/subj14.m) | Main LOOCV training, testing, and confusion-matrix evaluation script for Subject 14 (`filtersNum = 3`, bands `[19 35]`, `[6 18]`, `[5 9]` Hz). |
| [codes/subj15.m](./codes/subj15.m) | Main LOOCV training, testing, and confusion-matrix evaluation script for Subject 15 (`filtersNum = 2`, bands `[20 35]`, `[10 50]`, `[0.1 6]` Hz). |
| [codes/unused functions/Concat.m](./codes/unused%20functions/Concat.m) | Legacy helper function for concatenating cell-array tensors along the 3rd dimension (superseded by MATLAB's built-in `cat(3, ...)` after `tic`/`toc` benchmarking). |
| [codes/unused functions/Filter.m](./codes/unused%20functions/Filter.m) | Legacy FFT-based ideal brick-wall bandpass filter (`fft`, `fftshift`, `ifftshift`, `ifft`) over channel and trial loops (superseded by the $2\times$ faster vectorized Butterworth filter in `codes/Filter.m`). |
| [BSS\_Project\_810100205\_Arman\_Majidi.pdf](./BSS_Project_810100205_Arman_Majidi.pdf) | Submitted 33-page Persian project report containing function descriptions, the hierarchical tree diagram, CSP and LDA mathematical derivations, and 4-stage hyperparameter tuning logs and confusion matrices for all 15 subjects. |
| [project.pdf](./project.pdf) | Original course final project assignment sheet in Persian. |
| [Recording.pdf](./Recording.pdf) | Official signal acquisition and experimental protocol documentation for the 2nd Iranian National BCI Competition dataset in Persian (including 63-channel scalp topography maps and trial timing diagram). |
| [Report.pdf](./Report.pdf) | Reference methodology summary in Persian by the competition-winning team (*Team Misagh*: Saeed Akhavan and Samaneh Esmaeili). |
| `README.md` | Self-contained English problem statement, recording protocol specification, mathematical derivations, algorithm architecture, complete 15-subject numerical results, and usage guide. |

*(Note: The raw EEG data files `subj_1.mat` through `subj_15.mat`, distributed separately in the course `dataset` folder due to file size, should be placed inside `codes/` or on the MATLAB path before running `subj1.m`–`subj15.m`.)*

## 3. Assignment Problem Statement

### 3.1 Final Project Prompt (`project.pdf`)

In this final project, we implement a multi-class Brain-Computer Interface (BCI) EEG classification system in **MATLAB** using the methods studied in the Blind Source Separation course:

1. **Dataset (`dataset/`):** The EEG dataset from the **2nd National BCI Competition** is provided in the `dataset` folder. The recordings of $15$ distinct human subjects are stored in the files `subj_n.mat` for $n = 1, 2, \dots, 15$.
2. **Signal Recording Protocol (`Recording.pdf`):** Full specifications of the EEG acquisition hardware, electrode montage, experimental paradigm, and data structure are documented in `Recording.pdf`. First read `Recording.pdf` carefully while loading one subject's `.mat` file in MATLAB to understand the exact structure of the data.
3. **Reference Winner's Report (`Report.pdf`):** A summary of the methodology proposed by the competition winner is provided in `Report.pdf`. You may implement this hierarchical structure, or you may use any other BSS/classification techniques studied in the course. Note that the filter parameters listed as examples in `Report.pdf` are not necessarily optimal for every subject, and you should tune and select appropriate parameters yourself.
4. **Evaluation Protocol (100-Run Leave-One-Out Cross-Validation):** Using the **Leave-One-Out (LOO)** cross-validation approach repeated **$100$ times**, compute the probabilistic $4 \times 4$ **confusion matrix** for each subject individually. For every subject, report both the **training confusion matrix** and the **testing confusion matrix** separately.

### 3.2 EEG Recording Protocol & Dataset Format (`Recording.pdf`)

#### 1. Subjects and Hardware Specifications

- **Participants:** $15$ healthy right-handed volunteers ($5$ female and $10$ male) with a mean age of $31$ years.
- **Acquisition System:** A $64$-channel EEG recording system operating at a sampling rate of $f_s = 2400\ \mathrm{Hz}$ with an active power-line notch filter ($50\ \mathrm{Hz}$ mains hum removal).
- **Reference, Ground, and Analog/Hardware Filtering:** The system ground electrode was placed on the forehead, and one channel served as the reference electrode attached to the right earlobe. After removing the reference channel from the dataset, **$N_c = 63$ active EEG channels** remain. All electrode signals were low-pass filtered with a cutoff frequency of $50\ \mathrm{Hz}$.
- **Electrode Montage (Figures 1 & 2 of `Recording.pdf`):** The $63$ scalp electrodes follow the extended international 10–20 system, mapped to matrix row indices $1$ through $63$ as follows:
  - **Row 1:** `AFz`
  - **Rows 2–3:** `Fp1` (2), `Fp2` (3)
  - **Rows 4–5:** `AF3` (4), `AF4` (5)
  - **Rows 6–10:** `F7` (6), `F3` (7), `Fz` (8), `F4` (9), `F8` (10)
  - **Rows 11–14:** `FC5` (11), `FC1` (12), `FC2` (13), `FC6` (14)
  - **Rows 15–19:** `T7` (15), `C3` (16), `Cz` (17), `C4` (18), `T8` (19)
  - **Rows 20–23:** `CP5` (20), `CP1` (21), `CP2` (22), `CP6` (23)
  - **Rows 24–28:** `P7` (24), `P3` (25), `Pz` (26), `P4` (27), `P8` (28)
  - **Rows 29–30:** `PO3` (29), `PO4` (30)
  - **Rows 31–32:** `O1` (31), `O2` (32)
  - **Rows 33–34:** `AF7` (33), `AF8` (34)
  - **Rows 35–38:** `F5` (35), `F1` (36), `F2` (37), `F6` (38)
  - **Rows 39–43:** `FT7` (39), `FC3` (40), `FCz` (41), `FC4` (42), `FT8` (43)
  - **Rows 44–47:** `C5` (44), `C1` (45), `C2` (46), `C6` (47)
  - **Rows 48–52:** `TP7` (48), `CP3` (49), `CPz` (50), `CP4` (51), `TP8` (52)
  - **Rows 53–56:** `P5` (53), `P1` (54), `P2` (55), `P6` (56)
  - **Rows 57–59:** `PO7` (57), `POz` (58), `PO8` (59)
  - **Row 60:** `Oz` (60)
  - **Rows 61–62:** `P9` (61), `P10` (62)
  - **Row 63:** `TP9` (63)

#### 2. Experimental Paradigm and Trial Timeline (Figure 3 of `Recording.pdf`)

Each subject sat comfortably in a chair located $0.5\ \mathrm{m}$ in front of a display monitor and performed actual motor execution of three right-sided body parts—**right arm**, **right thumb**, and **right foot**—or remained at rest (**no movement**). Each experimental trial lasted $9\ \mathrm{s}$ according to the following timeline:

1. **Fixation Cross ($t \in [0, 2]\ \mathrm{s}$, duration $2\ \mathrm{s}$):** A `+` symbol appears at the center of the screen; the subject rests quietly without thinking of any specific movement and prepares for the upcoming cue.
2. **Task Cue ($t \in [2, 4]\ \mathrm{s}$, duration $2\ \mathrm{s}$):** One of the visual task cues appears on the screen indicating which movement is to be executed.
3. **"Go" Prompt & Motor Execution Window ($t \in [4, 7]\ \mathrm{s}$, duration $3\ \mathrm{s}$):** At $t = 4\ \mathrm{s}$, the cue disappears and the word `"Go"` is displayed for $0.5\ \mathrm{s}$. Starting from the onset of `"Go"`, the subject has **$3\ \mathrm{s}$** ($t \in [4, 7]\ \mathrm{s}$) to execute the instructed limb movement (or remain motionless for Class 4).
4. **Inter-Trial Break ($t \in [7, 9]\ \mathrm{s}$, duration $2\ \mathrm{s}$):** Rest period before the next trial.

#### 3. Data Structure in `subj_i.mat`

For each subject $i \in \{1, 2, \dots, 15\}$, the file `subj_i.mat` contains a $1 \times 4$ MATLAB cell array named `data` covering the $3\ \mathrm{s}$ motor execution interval from the onset of `"Go"` ($T = 3 \times 2400 = 7200$ time samples per trial):

- **`data{1}` (Class 1 — Right Arm Movement):** 3D array of size $64 \times 7200 \times N_1$.
- **`data{2}` (Class 2 — Right Thumb Movement):** 3D array of size $64 \times 7200 \times N_2$.
- **`data{3}` (Class 3 — Right Foot Movement):** 3D array of size $64 \times 7200 \times N_3$.
- **`data{4}` (Class 4 — No Movement / Rest):** 3D array of size $64 \times 7200 \times N_4$.

In each 3D array `data{k}`:
- **Dimension 1 ($64$ rows):** Rows $1\text{–}63$ contain the $63$ EEG channels, and the last row (**Row 64**) is the **trigger channel** indicating the movement-onset sample with a value of `1` (and entirely `NaN` for Class 4, where no movement occurs). In the classification scripts (`subj1.m`–`subj15.m`), Row 64 is removed via `data{k}(1:end-1, :, :)`, leaving a $63 \times 7200 \times N_k$ EEG tensor.
- **Dimension 2 ($7200$ columns):** Discrete time sample points over the $3\ \mathrm{s}$ window at $f_s = 2400\ \mathrm{Hz}$.
- **Dimension 3 ($N_k$ pages):** Number of recorded trials for class $k$.

### 3.3 Summary of the Competition Winner's Methodology (`Report.pdf`)

The reference document `Report.pdf` (authored by *Team Misagh*: Saeed Akhavan and Samaneh Esmaeili) outlines the following hierarchical strategy for separating the four classes:

1. **Hierarchical Tree-Based Binary Decomposition:** Empirical inspection of the EEG data shows that **Class 4 (No Movement)** is easiest to distinguish from the three active movement classes $\{1, 2, 3\}$, and **Class 3 (Right Foot)** differs more strongly from the two upper-limb classes $\{1, 2\}$ than Classes 1 and 2 differ from each other. Therefore, a **three-stage binary classification tree** is constructed:
   - **Classifier 1:** Tests whether a trial belongs to **Class 4** or to **Classes $\{1, 2, 3\}$** (e.g., for Subject 1 in `Report.pdf`, bandpass filter $[20, 40]\ \mathrm{Hz}$). If classified as Class 4, the decision terminates; otherwise, the trial proceeds to Classifier 2.
   - **Classifier 2:** Tests whether a remaining trial belongs to **Class 3** or to **Classes $\{1, 2\}$** (e.g., for Subject 1 in `Report.pdf`, bandpass filter $[0.1, 7]\ \mathrm{Hz}$). If classified as Class 3, the decision terminates; otherwise, the trial proceeds to Classifier 3.
   - **Classifier 3:** Tests whether a remaining trial belongs to **Class 2** or to **Class 1** (e.g., for Subject 1 in `Report.pdf`, bandpass filter $[0.1, 4]\ \mathrm{Hz}$).
2. **Per-Stage Processing Pipeline:** Every binary classifier applies (i) a subject- and stage-specific bandpass filter, (ii) **Common Spatial Patterns (CSP)** dimensionality reduction, (iii) channel-wise **variance** feature extraction, and (iv) **Linear Discriminant Analysis (LDA)** linear projection and thresholding.
3. **Reference Leave-One-Out Benchmark:** Across all subjects in the competition, the reference report states average class recognition accuracies of **$0.92$, $0.73$, $0.62$, and $0.51$** for **Classes 4, 3, 2, and 1**, respectively (corresponding to an overall four-class average accuracy of $0.6950$, or $69.50\%$).

## 4. Mathematical Formulation & Solutions

### 4.1 Hierarchical Decision Tree Architecture

As illustrated in Figure 1 on Page 2 of the submitted Persian report (`BSS_Project_810100205_Arman_Majidi.pdf`), the four-class EEG classification problem is solved via a sequential three-node binary decision tree:

```text
Input EEG Trial X (63 x 7200)
  │
  ├──► [First Classification: Filter 1 + CSP 1 + LDA 1]
  │       ├──► Assigned Class 4 (No Movement) ──────────────────────► Output: Class 4
  │       └──► Assigned Class {1, 2, 3}
  │               │
  │               ├──► [Second Classification: Filter 2 + CSP 2 + LDA 2]
  │               │       ├──► Assigned Class 3 (Right Foot) ───────► Output: Class 3
  │               │       └──► Assigned Class {1, 2}
  │               │               │
  │               │               └──► [Third Classification: Filter 3 + CSP 3 + LDA 3]
  │               │                       ├──► Assigned Class 2 ────► Output: Class 2 (Right Thumb)
  │               │                       └──► Assigned Class 1 ────► Output: Class 1 (Right Arm)
```

In each Monte Carlo iteration (`runtimeNum = 1:100`), one random trial index $\mathrm{ind}_k \sim \mathcal{U}\{1, \dots, N_k\}$ is drawn from each class $k \in \{1, 2, 3, 4\}$ via `randi(size(data{k}, 3))` (`subj1.m` lines 17–24) to form a 4-trial test set (`testLabel = [1, 2, 3, 4]`), while the remaining $(N_1 - 1) + (N_2 - 1) + (N_3 - 1) + (N_4 - 1)$ trials form the training set:

- **First Classifier Training Split:** Positive group `one = train{4}` (Class 4, $N^{(1)} = N_4 - 1$ trials) versus composite group `all = cat(3, train{1:3})` (Classes 1, 2, and 3 combined, $N^{(2)} = (N_1 - 1) + (N_2 - 1) + (N_3 - 1)$ trials).
- **Second Classifier Training Split:** Positive group `one = train{3}` (Class 3, $N^{(1)} = N_3 - 1$ trials) versus composite group `all = cat(3, train{1:2})` (Classes 1 and 2 combined, $N^{(2)} = (N_1 - 1) + (N_2 - 1)$ trials).
- **Third Classifier Training Split:** Positive group `one = train{2}` (Class 2, $N^{(1)} = N_2 - 1$ trials) versus negative group `all = train{1}` (Class 1, $N^{(2)} = N_1 - 1$ trials).

### 4.2 Preprocessing and Temporal Bandpass Filtering (`Filter.m`)

#### 1. Channel-Wise DC Removal

Before spatial covariance computation in `Train.m` (lines 2–3) and before filtering in `Test.m` (line 4), the global mean of each EEG channel across time samples (dimension 2) and trials (dimension 3) is subtracted:

```math
\tilde{X}_n^{(r)} = X_n^{(r)} - \mu_{\mathrm{ch}}^{(r)},
\qquad
\mu_{\mathrm{ch}}^{(r)} = \frac{1}{T N_r} \sum_{n=1}^{N_r} \sum_{t=1}^{T} X_n^{(r)}(:, t) \in \mathbb{R}^{63 \times 1},
```

implemented via `one = one - mean(one, [2, 3])` and `all = all - mean(all, [2, 3])`.

#### 2. Vectorized 7th-Order Butterworth Bandpass Filter (`codes/Filter.m`)

As explained on Page 1 of the Persian report, an initial frequency-domain bandpass filter (`codes/unused functions/Filter.m`) transformed each trial and channel via `fft` and `fftshift`, zeroed out frequency bins outside $[f_{c,1}, f_{c,2}]$, and applied `ifft(ifftshift(...))` inside nested `for` loops:

```math
\hat{X}_{c, n}(f) = \mathcal{F}\!\left\{X_{c, n}(t)\right\} \cdot \mathbb{I}\!\left\{f_{c,1} \le |f| \le f_{c,2}\right\},
\qquad
X_{c, n}^{\mathrm{filt}}(t) = \mathrm{Re}\!\left\{\mathcal{F}^{-1}\!\left\{\hat{X}_{c, n}(f)\right\}\right\}.
```

To eliminate the nested channel/trial loops and improve numerical efficiency, `codes/Filter.m` replaces the FFT loop with a vectorized **7th-order digital Butterworth bandpass filter** (14th-order transfer function) in second-order section (SOS) form operating directly along dimension 2 of the 3D EEG tensor at $f_s = 2400\ \mathrm{Hz}$, cutting execution time roughly in half ($\approx 2\times$ speedup):

```matlab
function filteredData = Filter(data, fc)
    fs = 2400;
    [z, p, k] = butter(7, fc ./ (fs/2), "bandpass");
    sos = zp2sos(z, p, k);
    filteredData = sosfilt(sos, data, 2);
end
```

### 4.3 Common Spatial Patterns (CSP) Spatial Filtering (`Train.m`)

Let $X^{(1)} = \left\{X_n^{(1)}\right\}_{n=1}^{N_1}$ and $X^{(2)} = \left\{X_n^{(2)}\right\}_{n=1}^{N_2}$ denote the zero-mean, bandpass-filtered $63 \times 7200$ EEG trial sets for the two binary groups (`one` and `all`) at a given tree stage. The **Common Spatial Patterns (CSP)** algorithm seeks a spatial filter vector $w \in \mathbb{R}^{63 \times 1}$ that extremizes the ratio of average projected signal energy between the two groups (Page 3 of the Persian report):

```math
f(w) = \frac{\left\|w^T X^{(1)}\right\|_2^2}{\left\|w^T X^{(2)}\right\|_2^2}
= \frac{w^T R_x^{(1)} w}{w^T R_x^{(2)} w},
\qquad \text{s.t.} \quad w^T w = 1,
```

where the trial-averaged $63 \times 63$ spatial correlation matrices $R_x^{(1)}$ and $R_x^{(2)}$ are defined as

```math
\begin{aligned}
R_x^{(1)} &= \frac{1}{N_1} \sum_{n=1}^{N_1} X_n^{(1)} \left(X_n^{(1)}\right)^T \in \mathbb{R}^{63 \times 63}, \\
R_x^{(2)} &= \frac{1}{N_2} \sum_{n=1}^{N_2} X_n^{(2)} \left(X_n^{(2)}\right)^T \in \mathbb{R}^{63 \times 63},
\end{aligned}
```

and are computed without loops in `Train.m` (lines 6–7) via 3D page-wise matrix multiplication:

```matlab
Rx_1 = mean(pagemtimes(one, pagetranspose(one)), 3);
Rx_2 = mean(pagemtimes(all, pagetranspose(all)), 3);
```

Setting the gradient of the Rayleigh quotient $f(w)$ with respect to $w$ equal to zero yields the **Generalized Eigenvalue Decomposition (GEVD)** problem:

```math
\nabla_w f(w) = 0
\quad \Longrightarrow \quad
R_x^{(1)} w = \left(\frac{w^T R_x^{(1)} w}{w^T R_x^{(2)} w}\right) R_x^{(2)} w
\quad \Longleftrightarrow \quad
\left(R_x^{(2)}\right)^{-1} R_x^{(1)} w = \lambda w,
```

which is solved in MATLAB via `[U, D] = eig(Rx_1, Rx_2)` (`Train.m` lines 8–11), sorted in descending order of the generalized eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_{63} > 0$, and normalized by the Frobenius norm $\sqrt{\mathrm{tr}(U U^T)}$ (`W_csp = U ./ sqrt(trace(U*U.'))`).

#### Selection of Extreme Spatial Filters

The leading eigenvectors (largest $\lambda_k$) maximize the variance of Group 1 (`one`) relative to Group 2 (`all`), whereas the trailing eigenvectors (smallest $\lambda_k$) maximize the variance of Group 2 (`all`) relative to Group 1 (`one`). Given the subject-specific parameter $m = \mathrm{filtersNum} \in \{2, 3, 4, 5\}$, `Train.m` (line 12) extracts the top and bottom spatial filters:

```matlab
W_csp = [W_csp(:, 1:num) W_csp(:, end-(num+1):end)];
```

which retains the first $m$ columns (`1:num`) and the last $m + 2$ columns (`end-(num+1):end`), forming a reduced spatial filter matrix $W_{\mathrm{csp}} \in \mathbb{R}^{63 \times d}$ with $d = 2m + 2$ columns.

### 4.4 Variance Feature Extraction and Fisher LDA Classifier (`Train.m`, `Label.m`, `Test.m`)

#### 1. Spatial Projection and Variance Features

Projecting every $63 \times 7200$ trial $X_t^{(i)}$ ($i \in \{1, 2\}$, $t = 1, \dots, N_i$) onto $W_{\mathrm{csp}}^T \in \mathbb{R}^{d \times 63}$ via `pagemtimes(W_csp.', ...)` and computing the sample variance along the time dimension (`var(..., 0, 2)`) compresses each multi-channel EEG trial into a $d \times 1$ feature vector $`x^{(i)}(t) \in \mathbb{R}^{d \times 1}`$ (`Train.m` lines 13–18):

```math
Z_t^{(i)} = W_{\mathrm{csp}}^T X_t^{(i)} \in \mathbb{R}^{d \times 7200},
\qquad
x_k^{(i)}(t) = \mathrm{Var}\!\left(Z_t^{(i)}(k, :)\right), \quad k = 1, \dots, d.
```

#### 2. Fisher Linear Discriminant Analysis (LDA) via GEVD

Let $\mu_1, \mu_2 \in \mathbb{R}^{d \times 1}$ and $\Sigma_1, \Sigma_2 \in \mathbb{R}^{d \times d}$ denote the sample mean vectors and within-class covariance matrices of the feature vectors $`\left\{x^{(i)}(t)\right\}_{t=1}^{N_i}`$ for the two binary groups (Page 3 of the Persian report and `Train.m` lines 19–22):

```math
\begin{aligned}
\mu_1 &= \frac{1}{N_1} \sum_{t=1}^{N_1} x^{(1)}(t), &\qquad
\Sigma_1 &= \frac{1}{N_1} \sum_{t=1}^{N_1} \left(x^{(1)}(t) - \mu_1\right)\left(x^{(1)}(t) - \mu_1\right)^T, \\
\mu_2 &= \frac{1}{N_2} \sum_{t=1}^{N_2} x^{(2)}(t), &\qquad
\Sigma_2 &= \frac{1}{N_2} \sum_{t=1}^{N_2} \left(x^{(2)}(t) - \mu_2\right)\left(x^{(2)}(t) - \mu_2\right)^T.
\end{aligned}
```

Under a linear projection $w_{\mathrm{lda}} \in \mathbb{R}^{d \times 1}$, the projected 1D class means and variances are

```math
\mu^{(i)} = w_{\mathrm{lda}}^T \mu_i,
\qquad
\left(\sigma^{(i)}\right)^2 = w_{\mathrm{lda}}^T \Sigma_i w_{\mathrm{lda}},
\qquad i \in \{1, 2\}.
```

**Fisher's LDA** maximizes the Rayleigh quotient of between-class scatter $S_B = (\mu_1 - \mu_2)(\mu_1 - \mu_2)^T$ to total within-class scatter $S_W = \Sigma_1 + \Sigma_2$ (`Train.m` lines 23–28):

```math
f(w_{\mathrm{lda}}) = \frac{w_{\mathrm{lda}}^T (\mu_1 - \mu_2)(\mu_1 - \mu_2)^T w_{\mathrm{lda}}}{w_{\mathrm{lda}}^T (\Sigma_1 + \Sigma_2) w_{\mathrm{lda}}}
= \frac{w_{\mathrm{lda}}^T S_B w_{\mathrm{lda}}}{w_{\mathrm{lda}}^T S_W w_{\mathrm{lda}}}
\quad \Longrightarrow \quad
S_B w_{\mathrm{lda}} = \lambda S_W w_{\mathrm{lda}},
```

solved in `Train.m` via `[U, D] = eig(a, b)` where `a = (mu_1 - mu_2) * (mu_1 - mu_2).'` and `b = sigma_1 + sigma_2`, sorted in descending eigenvalue order, normalized via `W_lda = U ./ sqrt(trace(U*U.'))`, and reduced to the principal eigenvector `W_lda = W_lda(:, 1)` corresponding to the single non-zero eigenvalue of the rank-1 matrix $S_B$.

#### 3. Midpoint Threshold `thr` and Polarity Flag `rule` (`Train.m`, `Label.m`)

Because the eigenvector $w_{\mathrm{lda}}$ returned by `eig(a, b)` has an arbitrary $\pm 1$ sign orientation, the projected mean $\mu^{(1)} = w_{\mathrm{lda}}^T \mu_1$ of Group 1 (`one`) can be either larger or smaller than $\mu^{(2)} = w_{\mathrm{lda}}^T \mu_2$ of Group 2 (`all`). To handle both eigenvector signs automatically, `Train.m` (lines 29–32) computes the midpoint threshold `thr` and a binary polarity flag `rule` $\in \{1, 2\}$:

```math
\mathrm{thr} = \frac{\mu^{(1)} + \mu^{(2)}}{2},
\qquad
\mathrm{rule} = \mathbb{I}\!\left\{\mu^{(2)} < \mu^{(1)}\right\} + 1 = \begin{cases}
2, & \text{if } \mu^{(1)} > \mu^{(2)}, \\
1, & \text{if } \mu^{(1)} \le \mu^{(2)}.
\end{cases}
```

At stage $s \in \{1, 2, 3\}$ with $\mathrm{sectionNum} = 4 - s \in \{3, 2, 1\}$ (representing the target class offset so that $\mathrm{sectionNum} + 1 \in \{4, 3, 2\}$ is the isolated class at stage $s$, while $1$ is the continuation placeholder for the remaining classes), `Label.m` assigns the scalar LDA projection $z = w_{\mathrm{lda}}^T x_{\mathrm{test}}$ according to:

```math
\hat{y}_{\mathrm{stage}}(z) = \begin{cases}
\mathrm{sectionNum} \cdot \mathbb{I}\{z < \mathrm{thr}\} + 1, & \text{if } \mathrm{rule} = 1 \ \left(\mu^{(1)} \le \mu^{(2)}\right), \\
\mathrm{sectionNum} \cdot \mathbb{I}\{z > \mathrm{thr}\} + 1, & \text{if } \mathrm{rule} = 2 \ \left(\mu^{(1)} > \mu^{(2)}\right),
\end{cases}
```

so a trial whose projection $z$ falls on the $\mu^{(1)}$ side of $\mathrm{thr}$ receives label $\mathrm{sectionNum} + 1 \in \{4, 3, 2\}$, whereas a trial on the $\mu^{(2)}$ side receives temporary label `1` (`ind = (myLabel == 1)`) and is passed down to the next stage of `Test.m`.

#### 4. Vectorized Confusion Matrix Accumulation (`Test.m`)

Given ground-truth class labels `trueLabel` $\in \{1, 2, 3, 4\}^M$ and predicted labels `myLabel` $\in \{1, 2, 3, 4\}^M$, `Test.m` (lines 36–38) accumulates the $4 \times 4$ count matrix $C \in \mathbb{Z}_{\ge 0}^{4 \times 4}$ without a `for` loop using linear indexing (`sub2ind`) and `accumarray`:

```matlab
inds = sub2ind(size(confusionMatrix), trueLabel, myLabel);
cnts = accumarray(inds.', 1, size(confusionMatrix(:)));
confusionMatrix = confusionMatrix + reshape(cnts, size(confusionMatrix));
```

After summing `trainConfusionMatrix` and `testConfusionMatrix` across all $100$ Monte Carlo runs (`subj1.m` lines 40–51), each row is normalized by its row sum (`trainConfusionMatrix = trainConfusionMatrix ./ sum(trainConfusionMatrix, 2)`), producing the row-stochastic probabilistic confusion matrix where entry $(i, j)$ represents the empirical conditional probability $\hat{P}(\text{Predicted Class } j \mid \text{True Class } i)$.

## 5. Results & Discussion

### 5.1 Master Summary of Final Hyperparameters and Accuracies Across All 15 Subjects

The table below compiles the final tuned hyperparameters (`filtersNum`, Stage 1 bandpass `filter1`, Stage 2 bandpass `filter2`, and Stage 3 bandpass `filter3`), the per-class test recall (diagonal entries $C_{\mathrm{test}}(k, k)$ for Classes $k = 1, 2, 3, 4$), and the macro-averaged 4-class training and test accuracies ($\frac{1}{4}\mathrm{tr}(C)$) documented in the boxed final results of `BSS_Project_810100205_Arman_Majidi.pdf` (Pages 4–33):

| Subject | `filtersNum` ($m$) | Total CSP Filters ($2m+2$) | Stage 1 Band `filter1` (Hz) | Stage 2 Band `filter2` (Hz) | Stage 3 Band `filter3` (Hz) | Test Class 1 (Arm) | Test Class 2 (Thumb) | Test Class 3 (Foot) | Test Class 4 (Rest) | Mean Train Acc. | Mean Test Acc. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Subject 1** | $2$ | $6$ | $[20,\ 35]$ | $[0.1,\ 17]$ | $[0.1,\ 4]$ | $0.4800$ | $0.6000$ | $0.6100$ | $0.9400$ | $65.75\%$* | **$65.75\%$** |
| **Subject 2** | $4$ | $10$ | $[20,\ 45]$ | $[3,\ 15]$ | $[0.1,\ 2]$ | $0.8000$ | $0.3500$ | $0.7200$ | $0.7200$ | $88.13\%$ | **$64.75\%$** |
| **Subject 3** | $2$ | $6$ | $[20,\ 40]$ | $[0.1,\ 10]$ | $[0.1,\ 3.6]$ | $0.6300$ | $0.4800$ | $0.8000$ | $0.8500$ | $79.89\%$ | **$69.00\%$** |
| **Subject 4** | $2$ | $6$ | $[20,\ 40]$ | $[11,\ 45]$ | $[0.1,\ 6]$ | $0.5600$ | $0.6700$ | $0.7400$ | $1.0000$ | $96.80\%$ | **$74.25\%$** |
| **Subject 5** | $3$ | $8$ | $[20,\ 40]$ | $[13,\ 19]$ | $[1,\ 6]$ | $0.5300$ | $0.6000$ | $0.4900$ | $1.0000$ | $99.75\%$ | **$65.50\%$** |
| **Subject 6** | $3$ | $8$ | $[20,\ 35]$ | $[10,\ 17]$ | $[0.1,\ 7]$ | $0.5200$ | $0.7200$ | $0.6100$ | $1.0000$ | $99.90\%$ | **$71.25\%$** |
| **Subject 7** | $3$ | $8$ | $[20,\ 45]$ | $[11,\ 16]$ | $[1,\ 8]$ | $0.3700$ | $0.6200$ | $0.5000$ | $0.9200$ | $85.42\%$ | **$60.25\%$** |
| **Subject 8** | $4$ | $10$ | $[25,\ 50]$ | $[10,\ 35]$ | $[1,\ 17]$ | $0.5700$ | $0.5000$ | $0.6800$ | $0.9500$ | $92.05\%$ | **$67.50\%$** |
| **Subject 9** | $4$ | $10$ | $[20,\ 40]$ | $[10,\ 19]$ | $[1,\ 8]$ | $0.5300$ | $0.6500$ | $0.4900$ | $0.7200$ | $95.92\%$ | **$59.75\%$** |
| **Subject 10** | $3$ ($4$ in `.m`) | $8$ ($10$) | $[20,\ 40]$ | $[10,\ 40]$ ($[10,\ 45]$) | $[2,\ 5]$ | $0.6700$ | $0.4400$ | $0.4200$ | $0.8600$ | $93.09\%$ | **$59.75\%$** |
| **Subject 11** | $4$ | $10$ | $[20,\ 45]$ | $[11,\ 17]$ | $[1,\ 4]$ | $0.5900$ | $0.6200$ | $0.6300$ | $0.8200$ | $95.10\%$ | **$66.50\%$** |
| **Subject 12** | $2$ | $6$ | $[20,\ 40]$ | $[5,\ 20]$ | $[0.1,\ 2]$ | $0.4500$ | $0.7700$ | $0.4700$ | $0.9000$ | $80.47\%$ | **$64.75\%$** |
| **Subject 13** | $5$ | $12$ | $[12,\ 30]$ | $[25,\ 40]$ | $[1,\ 15]$ | $0.6900$ | $0.6700$ | $0.5200$ | $0.9800$ | $99.78\%$ | **$71.50\%$** |
| **Subject 14** | $3$ | $8$ | $[19,\ 35]$ | $[6,\ 18]$ | $[5,\ 9]$ | $0.2400$ | $0.2700$ | $0.3500$ | $0.6700$ | $84.11\%$ | **$38.25\%$** |
| **Subject 15** | $2$ | $6$ | $[20,\ 35]$ | $[10,\ 50]$ | $[0.1,\ 6]$ | $0.4400$ | $0.6600$ | $0.8100$ | $1.0000$ | $84.85\%$ | **$72.75\%$** |
| **15-Subject Mean** | — | — | — | — | — | **$0.5380$** | **$0.5747$** | **$0.5893$** | **$0.8887$** | **$89.40\%$** | **$64.77\%$** |
| **Winner Benchmark (`Report.pdf`)** | — | — | — | — | — | **$0.5100$** | **$0.6200$** | **$0.7300$** | **$0.9200$** | — | **$69.50\%$** |

*(*For Subject 1, as noted in Section 5.4, the `trainConfusionMatrix` box on Page 5 of the Persian report duplicates `testConfusionMatrix`.)*

### 5.2 Neurophysiological & Algorithmic Discussion

1. **Frequency-Band Specialization Across the Three Tree Stages:**
   - **Stage 1 (Class 4 Rest vs. Active Movement $\{1, 2, 3\}$):** Across $14$ of the $15$ subjects, the optimal bandpass filter lies in the **beta / low-gamma band** ($[19, 35]\ \mathrm{Hz}$ to $[25, 50]\ \mathrm{Hz}$, and $[12, 30]\ \mathrm{Hz}$ for Subject 13). Active motor execution induces pronounced beta Event-Related Desynchronization/Synchronization (ERD/ERS) over the sensorimotor cortex relative to the resting state, enabling Stage 1 to achieve the highest class recognition accuracy across all subjects ($88.87\%$ average recall on Class 4, reaching **$100\%$ test recall** on Subjects 4, 5, 6, and 15).
   - **Stage 2 (Class 3 Right Foot vs. Upper Limb $\{1, 2\}$):** Optimal passbands typically span the **alpha/mu and low-beta bands** ($[3, 15]\ \mathrm{Hz}$ to $[13, 19]\ \mathrm{Hz}$, or broader bands up to $40\text{–}50\ \mathrm{Hz}$), capturing somatotopic differences between the medial foot representation (near vertex electrodes `Cz`/`CPz`) and lateral hand/arm motor areas (`C3`/`CP3`), yielding a $58.93\%$ average test recall on Class 3 (up to $80\%$–$81\%$ on Subjects 3 and 15).
   - **Stage 3 (Class 2 Right Thumb vs. Class 1 Right Arm):** Because right thumb and right arm movements activate adjacent, overlapping regions of the contralateral left primary motor cortex, they are best separated in the **delta/theta Movement-Related Cortical Potential (MRCP) band** ($[0.1, 2]\ \mathrm{Hz}$ to $[1, 8]\ \mathrm{Hz}$). Even without subject-specific channel subset selection, the tuned classifier achieves **$53.80\%$ mean test recall on Class 1** (exceeding the $51.00\%$ reference benchmark in `Report.pdf`) and **$57.47\%$ mean test recall on Class 2**.
2. **Training vs. Test Generalization:** Because CSP and LDA estimate $63 \times 63$ spatial covariance matrices from relatively small single-subject trial counts in a $63$-channel space, the training confusion matrices reach $80\%\text{–}99.9\%$ accuracy while leave-one-out test accuracies average $64.77\%$ across all $15$ subjects ($66.66\%$ across Subjects 1–13 and 15, with Subject 14 being the most challenging noisy recording at $38.25\%$, still well above the $25\%$ four-class random chance level).

### 5.3 Complete Subject-by-Subject Hyperparameter Tuning Logs & Confusion Matrices (Pages 4–33 of Report)

For each subject, the Persian report documents three intermediate tuning configurations (**Parameter Sets 1, 2, and 3**) evaluated on the leave-one-out test set followed by the **Final Parameters** and their boxed $4 \times 4$ `trainConfusionMatrix` and `testConfusionMatrix`. In every $4 \times 4$ matrix below, rows $1, 2, 3, 4$ correspond to the true classes (Class 1: Right Arm, Class 2: Right Thumb, Class 3: Right Foot, Class 4: Rest) and columns $1, 2, 3, 4$ correspond to the predicted classes.

#### Subject 1 (`codes/subj1.m`, Report Pages 4–5)

| Tuning Stage | `filtersNum` | `filter1` (Hz) | `filter2` (Hz) | `filter3` (Hz) | Test Diagonal Recall $[C_{11}, C_{22}, C_{33}, C_{44}]$ | Mean Test Acc. |
| --- | --- | --- | --- | --- | --- | --- |
| Parameter Set 1 | $3$ | $[20,\ 35]$ | $[5,\ 20]$ | $[0.1,\ 4]$ | $[0.5300,\ 0.4900,\ 0.6000,\ 0.9100]$ | $63.25\%$ |
| Parameter Set 2 | $2$ | $[20,\ 35]$ | $[5,\ 20]$ | $[0.1,\ 4]$ | $[0.4300,\ 0.6500,\ 0.4300,\ 0.9700]$ | $62.00\%$ |
| Parameter Set 3 | $2$ | $[20,\ 35]$ | $[0.1,\ 7]$ | $[0.1,\ 4]$ | $[0.3000,\ 0.5300,\ 0.6200,\ 0.9200]$ | $59.25\%$ |
| **Final Parameters** | **$2$** | **$[20,\ 35]$** | **$[0.1,\ 17]$** | **$[0.1,\ 4]$** | **$[0.4800,\ 0.6000,\ 0.6100,\ 0.9400]$** | **$65.75\%$** |

- **Intermediate & Final Confusion Matrices (Subject 1):**

```math
\begin{aligned}
C_{\mathrm{test}}^{(1, \mathrm{set\,1})} &= \begin{bmatrix}
0.5300 & 0.1500 & 0.3100 & 0.0100 \\
0.0900 & 0.4900 & 0.3700 & 0.0500 \\
0.1800 & 0.2100 & 0.6000 & 0.0100 \\
0.0300 & 0.0200 & 0.0400 & 0.9100
\end{bmatrix}, &\quad
C_{\mathrm{test}}^{(1, \mathrm{set\,2})} &= \begin{bmatrix}
0.4300 & 0.3600 & 0.2000 & 0.0100 \\
0.1400 & 0.6500 & 0.1500 & 0.0600 \\
0.1700 & 0.3800 & 0.4300 & 0.0200 \\
0 & 0.0100 & 0.0200 & 0.9700
\end{bmatrix}, \\[8pt]
C_{\mathrm{test}}^{(1, \mathrm{set\,3})} &= \begin{bmatrix}
0.3000 & 0.1100 & 0.5600 & 0.3000 \\
0.1700 & 0.5300 & 0.2700 & 0.0300 \\
0.1100 & 0.2500 & 0.6200 & 0.0200 \\
0 & 0 & 0.0800 & 0.9200
\end{bmatrix}, &\quad
C_{\mathrm{train}}^{(1, \mathrm{final})} &= C_{\mathrm{test}}^{(1, \mathrm{final})} = \begin{bmatrix}
0.4800 & 0.1300 & 0.3700 & 0.0200 \\
0.0600 & 0.6000 & 0.2900 & 0.0500 \\
0.1400 & 0.2200 & 0.6100 & 0.0300 \\
0 & 0 & 0.0600 & 0.9400
\end{bmatrix}.
\end{aligned}
```

#### Subject 2 (`codes/subj2.m`, Report Pages 6–7)

| Tuning Stage | `filtersNum` | `filter1` (Hz) | `filter2` (Hz) | `filter3` (Hz) | Test Diagonal Recall $[C_{11}, C_{22}, C_{33}, C_{44}]$ | Mean Test Acc. |
| --- | --- | --- | --- | --- | --- | --- |
| Parameter Set 1 | $5$ | $[20,\ 45]$ | $[4,\ 20]$ | $[0.1,\ 3]$ | $[0.5400,\ 0.2000,\ 0.5400,\ 0.7300]$ | $50.25\%$ |
| Parameter Set 2 | $5$ | $[25,\ 45]$ | $[4,\ 20]$ | $[0.1,\ 5]$ | $[0.4600,\ 0.2600,\ 0.5100,\ 0.5400]$ | $44.25\%$ |
| Parameter Set 3 | $5$ | $[20,\ 40]$ | $[4,\ 15]$ | $[0.1,\ 7]$ | $[0.4000,\ 0.2300,\ 0.5900,\ 0.7000]$ | $48.00\%$ |
| **Final Parameters** | **$4$** | **$[20,\ 45]$** | **$[3,\ 15]$** | **$[0.1,\ 2]$** | **$[0.8000,\ 0.3500,\ 0.7200,\ 0.7200]$** | **$64.75\%$** |

- **Intermediate & Final Confusion Matrices (Subject 2):**

```math
\begin{aligned}
C_{\mathrm{test}}^{(2, \mathrm{set\,1})} &= \begin{bmatrix}
0.5400 & 0.2000 & 0.0900 & 0.1700 \\
0.7300 & 0.2000 & 0 & 0.0700 \\
0.1500 & 0.0500 & 0.5400 & 0.2600 \\
0.1800 & 0.0500 & 0.0400 & 0.7300
\end{bmatrix}, &\quad
C_{\mathrm{test}}^{(2, \mathrm{set\,2})} &= \begin{bmatrix}
0.4600 & 0.2800 & 0.0200 & 0.2400 \\
0.5800 & 0.2600 & 0 & 0.1600 \\
0.0800 & 0.0600 & 0.5100 & 0.3500 \\
0.1400 & 0.0400 & 0.2800 & 0.5400
\end{bmatrix}, \\[8pt]
C_{\mathrm{test}}^{(2, \mathrm{set\,3})} &= \begin{bmatrix}
0.4000 & 0.4300 & 0.0300 & 0.1900 \\
0.7200 & 0.2300 & 0 & 0.1000 \\
0.1200 & 0.0600 & 0.5900 & 0.3100 \\
0.1600 & 0.0200 & 0.1200 & 0.7000
\end{bmatrix}, \\[8pt]
C_{\mathrm{train}}^{(2, \mathrm{final})} &= \begin{bmatrix}
0.9426 & 0.0205 & 0.0016 & 0.0353 \\
0 & 0.9183 & 0 & 0.0817 \\
0 & 0 & 0.7963 & 0.2037 \\
0.0853 & 0 & 0.0237 & 0.8679
\end{bmatrix}, &\quad
C_{\mathrm{test}}^{(2, \mathrm{final})} &= \begin{bmatrix}
0.8000 & 0.0700 & 0.0500 & 0.0800 \\
0.5700 & 0.3500 & 0 & 0.0800 \\
0.0800 & 0.0200 & 0.7200 & 0.1800 \\
0.1000 & 0.0900 & 0.0900 & 0.7200
\end{bmatrix}.
\end{aligned}
```

#### Subject 3 (`codes/subj3.m`, Report Pages 8–9)

| Tuning Stage | `filtersNum` | `filter1` (Hz) | `filter2` (Hz) | `filter3` (Hz) | Test Diagonal Recall $[C_{11}, C_{22}, C_{33}, C_{44}]$ | Mean Test Acc. |
| --- | --- | --- | --- | --- | --- | --- |
| Parameter Set 1 | $2$ | $[20,\ 45]$ | $[0.1,\ 10]$ | $[0.1,\ 4]$ | $[0.4500,\ 0.4300,\ 0.8100,\ 0.8500]$ | $63.50\%$ |
| Parameter Set 2 | $2$ | $[20,\ 40]$ | $[0.1,\ 10]$ | $[0.1,\ 3]$ | $[0.5200,\ 0.4400,\ 0.7700,\ 0.8900]$ | $65.50\%$ |
| Parameter Set 3 | $2$ | $[20,\ 40]$ | $[0.1,\ 10]$ | $[0.1,\ 3.8]$ | $[0.4900,\ 0.3400,\ 0.7700,\ 0.8700]$ | $61.75\%$ |
| **Final Parameters** | **$2$** | **$[20,\ 40]$** | **$[0.1,\ 10]$** | **$[0.1,\ 3.6]$** | **$[0.6300,\ 0.4800,\ 0.8000,\ 0.8500]$** | **$69.00\%$** |

- **Intermediate & Final Confusion Matrices (Subject 3):**

```math
\begin{aligned}
C_{\mathrm{test}}^{(3, \mathrm{set\,1})} &= \begin{bmatrix}
0.4500 & 0.1800 & 0.1800 & 0.1900 \\
0 & 0.4300 & 0.2800 & 0.2900 \\
0.0300 & 0.0500 & 0.8100 & 0.1100 \\
0 & 0.0400 & 0.1100 & 0.8500
\end{bmatrix}, &\quad
C_{\mathrm{test}}^{(3, \mathrm{set\,2})} &= \begin{bmatrix}
0.5200 & 0.1800 & 0.0200 & 0.1700 \\
0.0100 & 0.4400 & 0.3200 & 0.2900 \\
0.0600 & 0.0900 & 0.7700 & 0.0800 \\
0.0100 & 0.0400 & 0.0600 & 0.8900
\end{bmatrix}, \\[8pt]
C_{\mathrm{test}}^{(3, \mathrm{set\,3})} &= \begin{bmatrix}
0.4900 & 0.2000 & 0.1100 & 0.2000 \\
0.0900 & 0.3400 & 0.4100 & 0.1600 \\
0.0300 & 0.0900 & 0.7700 & 0.1100 \\
0.0300 & 0.0100 & 0.0900 & 0.8700
\end{bmatrix}, \\[8pt]
C_{\mathrm{train}}^{(3, \mathrm{final})} &= \begin{bmatrix}
0.7922 & 0.0167 & 0.0356 & 0.1556 \\
0 & 0.6400 & 0.1489 & 0.2111 \\
0 & 0 & 0.8950 & 0.1050 \\
0.0205 & 0.0268 & 0.0842 & 0.8684
\end{bmatrix}, &\quad
C_{\mathrm{test}}^{(3, \mathrm{final})} &= \begin{bmatrix}
0.6300 & 0.1600 & 0.1100 & 0.1000 \\
0.0100 & 0.4800 & 0.3100 & 0.2000 \\
0.0200 & 0.0400 & 0.8000 & 0.1400 \\
0 & 0.0500 & 0.1000 & 0.8500
\end{bmatrix}.
\end{aligned}
```

#### Subject 4 (`codes/subj4.m`, Report Pages 10–11)

| Tuning Stage | `filtersNum` | `filter1` (Hz) | `filter2` (Hz) | `filter3` (Hz) | Test Diagonal Recall $[C_{11}, C_{22}, C_{33}, C_{44}]$ | Mean Test Acc. |
| --- | --- | --- | --- | --- | --- | --- |
| Parameter Set 1 | $2$ | $[20,\ 45]$ | $[0.1,\ 10]$ | $[0.1,\ 4]$ | $[0.2800,\ 0.5100,\ 0.5800,\ 1.0000]$ | $59.25\%$ |
| Parameter Set 2 | $2$ | $[20,\ 40]$ | $[9,\ 32]$ | $[0.1,\ 8]$ | $[0.5300,\ 0.6300,\ 0.6500,\ 1.0000]$ | $70.25\%$ |
| Parameter Set 3 | $2$ | $[20,\ 40]$ | $[11,\ 45]$ | $[0.1,\ 6]$ | $[0.3500,\ 0.5800,\ 0.7200,\ 1.0000]$ | $66.25\%$ |
| **Final Parameters** | **$2$** | **$[20,\ 40]$** | **$[11,\ 45]$** | **$[0.1,\ 6]$** | **$[0.5600,\ 0.6700,\ 0.7400,\ 1.0000]$** | **$74.25\%$** |

- **Intermediate & Final Confusion Matrices (Subject 4):**

```math
\begin{aligned}
C_{\mathrm{test}}^{(4, \mathrm{set\,1})} &= \begin{bmatrix}
0.2800 & 0.2200 & 0.5000 & 0 \\
0.1700 & 0.5100 & 0.3200 & 0 \\
0.2000 & 0.2200 & 0.5800 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}, &\quad
C_{\mathrm{test}}^{(4, \mathrm{set\,2})} &= \begin{bmatrix}
0.5300 & 0.3500 & 0.1200 & 0 \\
0.3700 & 0.6300 & 0 & 0 \\
0.0700 & 0.2800 & 0.6500 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}, \\[8pt]
C_{\mathrm{test}}^{(4, \mathrm{set\,3})} &= \begin{bmatrix}
0.3500 & 0.6100 & 0.0400 & 0 \\
0.4200 & 0.5800 & 0 & 0 \\
0.1100 & 0.1700 & 0.7200 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}, \\[8pt]
C_{\mathrm{train}}^{(4, \mathrm{final})} &= \begin{bmatrix}
0.9983 & 0 & 0.0017 & 0 \\
0.0871 & 0.9124 & 0.0006 & 0 \\
0.0056 & 0.0333 & 0.9611 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}, &\quad
C_{\mathrm{test}}^{(4, \mathrm{final})} &= \begin{bmatrix}
0.5600 & 0.4100 & 0.0300 & 0 \\
0.3000 & 0.6700 & 0.0300 & 0 \\
0.0900 & 0.1700 & 0.7400 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}.
\end{aligned}
```

#### Subject 5 (`codes/subj5.m`, Report Pages 12–13)

| Tuning Stage | `filtersNum` | `filter1` (Hz) | `filter2` (Hz) | `filter3` (Hz) | Test Diagonal Recall $[C_{11}, C_{22}, C_{33}, C_{44}]$ | Mean Test Acc. |
| --- | --- | --- | --- | --- | --- | --- |
| Parameter Set 1 | $2$ | $[20,\ 45]$ | $[11,\ 45]$ | $[0.1,\ 6]$ | $[0.4600,\ 0.4700,\ 0.1000,\ 1.0000]$ | $50.75\%$ |
| Parameter Set 2 | $3$ | $[20,\ 40]$ | $[9,\ 32]$ | $[0.1,\ 8]$ | $[0.5500,\ 0.5700,\ 0.4100,\ 1.0000]$ | $63.25\%$ |
| Parameter Set 3 | $3$ | $[20,\ 40]$ | $[13,\ 19]$ | $[0.7,\ 5]$ | $[0.6300,\ 0.5800,\ 0.3700,\ 1.0000]$ | $64.50\%$ |
| **Final Parameters** | **$3$** | **$[20,\ 40]$** | **$[13,\ 19]$** | **$[1,\ 6]$** | **$[0.5300,\ 0.6000,\ 0.4900,\ 1.0000]$** | **$65.50\%$** |

- **Intermediate & Final Confusion Matrices (Subject 5):**

```math
\begin{aligned}
C_{\mathrm{test}}^{(5, \mathrm{set\,1})} &= \begin{bmatrix}
0.4600 & 0.1500 & 0.3900 & 0 \\
0.2900 & 0.4700 & 0.2400 & 0 \\
0.4600 & 0.4400 & 0.1000 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}, &\quad
C_{\mathrm{test}}^{(5, \mathrm{set\,2})} &= \begin{bmatrix}
0.5500 & 0.2200 & 0.2300 & 0 \\
0.3200 & 0.5700 & 0.1100 & 0 \\
0.3000 & 0.2900 & 0.4100 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}, \\[8pt]
C_{\mathrm{test}}^{(5, \mathrm{set\,3})} &= \begin{bmatrix}
0.6300 & 0.2600 & 0.1100 & 0 \\
0.2800 & 0.5800 & 0.1400 & 0 \\
0.3800 & 0.2500 & 0.3700 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}, \\[8pt]
C_{\mathrm{train}}^{(5, \mathrm{final})} &= \begin{bmatrix}
0.9953 & 0.0032 & 0.0016 & 0 \\
0 & 0.9989 & 0.0011 & 0 \\
0.0011 & 0.0032 & 0.9958 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}, &\quad
C_{\mathrm{test}}^{(5, \mathrm{final})} &= \begin{bmatrix}
0.5300 & 0.3500 & 0.1200 & 0 \\
0.2500 & 0.6000 & 0.1500 & 0 \\
0.1400 & 0.3700 & 0.4900 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}.
\end{aligned}
```

#### Subject 6 (`codes/subj6.m`, Report Pages 14–15)

| Tuning Stage | `filtersNum` | `filter1` (Hz) | `filter2` (Hz) | `filter3` (Hz) | Test Diagonal Recall $[C_{11}, C_{22}, C_{33}, C_{44}]$ | Mean Test Acc. |
| --- | --- | --- | --- | --- | --- | --- |
| Parameter Set 1 | $2$ | $[20,\ 35]$ | $[12,\ 15]$ | $[1,\ 9]$ | $[0.4400,\ 0.5800,\ 0.2800,\ 1.0000]$ | $57.50\%$ |
| Parameter Set 2 | $3$ | $[20,\ 35]$ | $[8,\ 19]$ | $[0.1,\ 7]$ | $[0.4600,\ 0.6600,\ 0.4600,\ 1.0000]$ | $64.50\%$ |
| Parameter Set 3 | $3$ | $[20,\ 35]$ | $[10,\ 15]$ | $[0.1,\ 7]$ | $[0.4400,\ 0.7200,\ 0.5000,\ 1.0000]$ | $66.50\%$ |
| **Final Parameters** | **$3$** | **$[20,\ 35]$** | **$[10,\ 17]$** | **$[0.1,\ 7]$** | **$[0.5200,\ 0.7200,\ 0.6100,\ 1.0000]$** | **$71.25\%$** |

- **Intermediate & Final Confusion Matrices (Subject 6):**

```math
\begin{aligned}
C_{\mathrm{test}}^{(6, \mathrm{set\,1})} &= \begin{bmatrix}
0.4400 & 0.4200 & 0.1400 & 0 \\
0.3600 & 0.5800 & 0.0600 & 0 \\
0.3600 & 0.2000 & 0.2800 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}, &\quad
C_{\mathrm{test}}^{(6, \mathrm{set\,2})} &= \begin{bmatrix}
0.4600 & 0.4500 & 0.0900 & 0 \\
0.2600 & 0.6600 & 0.0800 & 0 \\
0.1800 & 0.3600 & 0.4600 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}, \\[8pt]
C_{\mathrm{test}}^{(6, \mathrm{set\,3})} &= \begin{bmatrix}
0.4400 & 0.4200 & 0.1400 & 0 \\
0.2800 & 0.7200 & 0 & 0 \\
0.1600 & 0.3400 & 0.5000 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}, \\[8pt]
C_{\mathrm{train}}^{(6, \mathrm{final})} &= \begin{bmatrix}
0.9963 & 0.3600 & 0.0032 & 0 \\
0 & 0.9995 & 0.0005 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}, &\quad
C_{\mathrm{test}}^{(6, \mathrm{final})} &= \begin{bmatrix}
0.5200 & 0.3600 & 0.1200 & 0 \\
0.2200 & 0.7200 & 0.0600 & 0 \\
0.1700 & 0.2200 & 0.6100 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}.
\end{aligned}
```

#### Subject 7 (`codes/subj7.m`, Report Pages 16–17)

| Tuning Stage | `filtersNum` | `filter1` (Hz) | `filter2` (Hz) | `filter3` (Hz) | Test Diagonal Recall $[C_{11}, C_{22}, C_{33}, C_{44}]$ | Mean Test Acc. |
| --- | --- | --- | --- | --- | --- | --- |
| Parameter Set 1 | $3$ | $[20,\ 45]$ | $[11,\ 16]$ | $[1.5,\ 8]$ | $[0.3800,\ 0.5600,\ 0.4600,\ 0.8800]$ | $57.00\%$ |
| Parameter Set 2 | $3$ | $[20,\ 45]$ | $[11,\ 19]$ | $[1.5,\ 8]$ | $[0.3400,\ 0.5800,\ 0.3400,\ 0.9400]$ | $55.00\%$ |
| Parameter Set 3 | $3$ | $[20,\ 45]$ | $[11,\ 14]$ | $[1.5,\ 8]$ | $[0.3300,\ 0.7000,\ 0.3400,\ 0.9200]$ | $57.25\%$ |
| **Final Parameters** | **$3$** | **$[20,\ 45]$** | **$[11,\ 16]$** | **$[1,\ 8]$** | **$[0.3700,\ 0.6200,\ 0.5000,\ 0.9200]$** | **$60.25\%$** |

- **Intermediate & Final Confusion Matrices (Subject 7):**

```math
\begin{aligned}
C_{\mathrm{test}}^{(7, \mathrm{set\,1})} &= \begin{bmatrix}
0.3800 & 0.2500 & 0.0900 & 0.2800 \\
0.1700 & 0.5600 & 0.1100 & 0.1600 \\
0.1200 & 0.2300 & 0.4600 & 0.1900 \\
0.0400 & 0.0800 & 0 & 0.8800
\end{bmatrix}, &\quad
C_{\mathrm{test}}^{(7, \mathrm{set\,2})} &= \begin{bmatrix}
0.3400 & 0.1700 & 0.0900 & 0.4000 \\
0.1200 & 0.5800 & 0.0900 & 0.2100 \\
0.1300 & 0.2600 & 0.3400 & 0.2700 \\
0.0300 & 0 & 0.0300 & 0.9400
\end{bmatrix}, \\[8pt]
C_{\mathrm{test}}^{(7, \mathrm{set\,3})} &= \begin{bmatrix}
0.3300 & 0.2800 & 0 & 0.4100 \\
0.1500 & 0.7000 & 0 & 0.1500 \\
0.0900 & 0.2400 & 0.3400 & 0.3300 \\
0.0400 & 0.0400 & 0 & 0.9200
\end{bmatrix}, \\[8pt]
C_{\mathrm{train}}^{(7, \mathrm{final})} &= \begin{bmatrix}
0.6575 & 0 & 0.0019 & 0.3406 \\
0 & 0.9337 & 0 & 0.0663 \\
0 & 0 & 0.8594 & 0.1406 \\
0 & 0.0337 & 0 & 0.9663
\end{bmatrix}, &\quad
C_{\mathrm{test}}^{(7, \mathrm{final})} &= \begin{bmatrix}
0.3700 & 0.2800 & 0.0200 & 0.3300 \\
0.1900 & 0.6200 & 0.0300 & 0.1600 \\
0.0500 & 0.1700 & 0.5000 & 0.2800 \\
0 & 0.0800 & 0 & 0.9200
\end{bmatrix}.
\end{aligned}
```

#### Subject 8 (`codes/subj8.m`, Report Pages 18–19)

| Tuning Stage | `filtersNum` | `filter1` (Hz) | `filter2` (Hz) | `filter3` (Hz) | Test Diagonal Recall $[C_{11}, C_{22}, C_{33}, C_{44}]$ | Mean Test Acc. |
| --- | --- | --- | --- | --- | --- | --- |
| Parameter Set 1 | $3$ | $[20,\ 45]$ | $[11,\ 16]$ | $[1.5,\ 8]$ | $[0.3800,\ 0.5600,\ 0.4600,\ 0.8800]$ | $57.00\%$ |
| Parameter Set 2 | $4$ | $[20,\ 50]$ | $[10,\ 35]$ | $[1,\ 11]$ | $[0.5400,\ 0.5000,\ 0.7000,\ 0.8600]$ | $65.00\%$ |
| Parameter Set 3 | $4$ | $[20,\ 50]$ | $[10,\ 35]$ | $[1,\ 11]$ | $[0.6000,\ 0.4500,\ 0.7600,\ 0.8800]$ | $67.25\%$ |
| **Final Parameters** | **$4$** | **$[25,\ 50]$** | **$[10,\ 35]$** | **$[1,\ 17]$** | **$[0.5700,\ 0.5000,\ 0.6800,\ 0.9500]$** | **$67.50\%$** |

- **Intermediate & Final Confusion Matrices (Subject 8):**

```math
\begin{aligned}
C_{\mathrm{test}}^{(8, \mathrm{set\,1})} &= \begin{bmatrix}
0.3800 & 0.2500 & 0.0900 & 0.2800 \\
0.1700 & 0.5600 & 0.1100 & 0.1600 \\
0.1200 & 0.2300 & 0.4600 & 0.1900 \\
0.0400 & 0.0800 & 0 & 0.8800
\end{bmatrix}, &\quad
C_{\mathrm{test}}^{(8, \mathrm{set\,2})} &= \begin{bmatrix}
0.5400 & 0.1800 & 0.0900 & 0.1900 \\
0.3300 & 0.5000 & 0.0800 & 0.0900 \\
0.0700 & 0.1700 & 0.7000 & 0.0600 \\
0.0700 & 0.0200 & 0.0500 & 0.8600
\end{bmatrix}, \\[8pt]
C_{\mathrm{test}}^{(8, \mathrm{set\,3})} &= \begin{bmatrix}
0.6000 & 0.1500 & 0.0700 & 0.1800 \\
0.2800 & 0.4500 & 0.0300 & 0.2400 \\
0.0600 & 0.0900 & 0.7600 & 0.0900 \\
0.0700 & 0 & 0 & 0.8800
\end{bmatrix}, \\[8pt]
C_{\mathrm{train}}^{(8, \mathrm{final})} &= \begin{bmatrix}
0.8426 & 0.0005 & 0.0053 & 0.1516 \\
0.0044 & 0.8911 & 0.0050 & 0.0994 \\
0.0263 & 0.0021 & 0.9484 & 0.0232 \\
0 & 0 & 0 & 1
\end{bmatrix}, &\quad
C_{\mathrm{test}}^{(8, \mathrm{final})} &= \begin{bmatrix}
0.5700 & 0.1900 & 0.0800 & 0.1600 \\
0.3100 & 0.5000 & 0.0800 & 0.1100 \\
0.1400 & 0.1100 & 0.6800 & 0.0700 \\
0.0100 & 0 & 0.0400 & 0.9500
\end{bmatrix}.
\end{aligned}
```

#### Subject 9 (`codes/subj9.m`, Report Pages 20–21)

| Tuning Stage | `filtersNum` | `filter1` (Hz) | `filter2` (Hz) | `filter3` (Hz) | Test Diagonal Recall $[C_{11}, C_{22}, C_{33}, C_{44}]$ | Mean Test Acc. |
| --- | --- | --- | --- | --- | --- | --- |
| Parameter Set 1 | $4$ | $[20,\ 45]$ | $[10,\ 19]$ | $[1,\ 8]$ | $[0.5400,\ 0.6900,\ 0.3900,\ 0.6600]$ | $57.00\%$ |
| Parameter Set 2 | $4$ | $[20,\ 50]$ | $[10,\ 19]$ | $[1,\ 8]$ | $[0.6100,\ 0.6500,\ 0.4600,\ 0.5100]$ | $55.75\%$ |
| Parameter Set 3 | $3$ | $[20,\ 35]$ | $[10,\ 19]$ | $[1,\ 8]$ | $[0.6600,\ 0.6000,\ 0.3900,\ 0.6900]$ | $58.50\%$ |
| **Final Parameters** | **$4$** | **$[20,\ 40]$** | **$[10,\ 19]$** | **$[1,\ 8]$** | **$[0.5300,\ 0.6500,\ 0.4900,\ 0.7200]$** | **$59.75\%$** |

- **Intermediate & Final Confusion Matrices (Subject 9):**

```math
\begin{aligned}
C_{\mathrm{test}}^{(9, \mathrm{set\,1})} &= \begin{bmatrix}
0.5400 & 0.2800 & 0.0300 & 0.1500 \\
0.2200 & 0.6900 & 0.0100 & 0.0800 \\
0.1100 & 0.2300 & 0.3900 & 0.2700 \\
0.1400 & 0.2000 & 0 & 0.6600
\end{bmatrix}, &\quad
C_{\mathrm{test}}^{(9, \mathrm{set\,2})} &= \begin{bmatrix}
0.6100 & 0.2800 & 0.0300 & 0.0800 \\
0.2900 & 0.6500 & 0.0200 & 0.0400 \\
0.0800 & 0.3400 & 0.4600 & 0.1200 \\
0.2000 & 0.2300 & 0.0600 & 0.5100
\end{bmatrix}, \\[8pt]
C_{\mathrm{test}}^{(9, \mathrm{set\,3})} &= \begin{bmatrix}
0.6600 & 0.2200 & 0.0500 & 0.0700 \\
0.3400 & 0.6000 & 0.0400 & 0.0200 \\
0.0600 & 0.3900 & 0.3900 & 0.1600 \\
0.1000 & 0.1700 & 0.0400 & 0.6900
\end{bmatrix}, \\[8pt]
C_{\mathrm{train}}^{(9, \mathrm{final})} &= \begin{bmatrix}
0.9033 & 0.0767 & 0.0017 & 0.0183 \\
0 & 0.9944 & 0 & 0.0056 \\
0 & 0.0017 & 0.9539 & 0.0444 \\
0.0063 & 0.0084 & 0 & 0.9853
\end{bmatrix}, &\quad
C_{\mathrm{test}}^{(9, \mathrm{final})} &= \begin{bmatrix}
0.5300 & 0.3000 & 0.0600 & 0.1100 \\
0.2600 & 0.6500 & 0.0300 & 0.0600 \\
0.0800 & 0.3100 & 0.4900 & 0.1200 \\
0.0700 & 0.2000 & 0.0100 & 0.7200
\end{bmatrix}.
\end{aligned}
```

#### Subject 10 (`codes/subj10.m`, Report Pages 22–23)

| Tuning Stage | `filtersNum` | `filter1` (Hz) | `filter2` (Hz) | `filter3` (Hz) | Test Diagonal Recall $[C_{11}, C_{22}, C_{33}, C_{44}]$ | Mean Test Acc. |
| --- | --- | --- | --- | --- | --- | --- |
| Parameter Set 1 | $4$ | $[20,\ 40]$ | $[10,\ 29]$ | $[1,\ 9]$ | $[0.5600,\ 0.4300,\ 0.3200,\ 0.8700]$ | $54.50\%$ |
| Parameter Set 2 | $4$ | $[20,\ 40]$ | $[10,\ 35]$ | $[2,\ 7]$ | $[0.6300,\ 0.3900,\ 0.4000,\ 0.9000]$ | $58.00\%$ |
| Parameter Set 3 (in `subj10.m`) | $4$ | $[20,\ 40]$ | $[10,\ 45]$ | $[2,\ 5]$ | $[0.6700,\ 0.2300,\ 0.3700,\ 0.9200]$ | $54.75\%$ |
| **Final Parameters (Report)** | **$3$** | **$[20,\ 40]$** | **$[10,\ 40]$** | **$[2,\ 5]$** | **$[0.6700,\ 0.4400,\ 0.4200,\ 0.8600]$** | **$59.75\%$** |

- **Intermediate & Final Confusion Matrices (Subject 10):**

```math
\begin{aligned}
C_{\mathrm{test}}^{(10, \mathrm{set\,1})} &= \begin{bmatrix}
0.5600 & 0.1600 & 0.2700 & 0.0100 \\
0.3900 & 0.4300 & 0.0700 & 0.1100 \\
0.1400 & 0.2000 & 0.3200 & 0.3400 \\
0.0800 & 0.0100 & 0.0400 & 0.8700
\end{bmatrix}, &\quad
C_{\mathrm{test}}^{(10, \mathrm{set\,2})} &= \begin{bmatrix}
0.6300 & 0.2600 & 0.0800 & 0.0300 \\
0.3800 & 0.3900 & 0.1300 & 0.1000 \\
0.2000 & 0.1000 & 0.4000 & 0.3000 \\
0.0600 & 0 & 0.0400 & 0.9000
\end{bmatrix}, \\[8pt]
C_{\mathrm{test}}^{(10, \mathrm{set\,3})} &= \begin{bmatrix}
0.6700 & 0.2100 & 0.0900 & 0.0300 \\
0.5700 & 0.2300 & 0.1200 & 0.0800 \\
0.1200 & 0.1500 & 0.3700 & 0.3600 \\
0.0100 & 0.0100 & 0.0600 & 0.9200
\end{bmatrix}, \\[8pt]
C_{\mathrm{train}}^{(10, \mathrm{final})} &= \begin{bmatrix}
0.9933 & 0 & 0.0056 & 0.0011 \\
0 & 0.9613 & 0.0200 & 0.0187 \\
0.0095 & 0.0105 & 0.7700 & 0.2100 \\
0 & 0.0011 & 0 & 0.9989
\end{bmatrix}, &\quad
C_{\mathrm{test}}^{(10, \mathrm{final})} &= \begin{bmatrix}
0.6700 & 0.1700 & 0.1500 & 0.0100 \\
0.3400 & 0.4400 & 0.1300 & 0.0900 \\
0.2300 & 0.1400 & 0.4200 & 0.2100 \\
0.0300 & 0.0200 & 0.0900 & 0.8600
\end{bmatrix}.
\end{aligned}
```

#### Subject 11 (`codes/subj11.m`, Report Pages 24–25)

| Tuning Stage | `filtersNum` | `filter1` (Hz) | `filter2` (Hz) | `filter3` (Hz) | Test Diagonal Recall $[C_{11}, C_{22}, C_{33}, C_{44}]$ | Mean Test Acc. |
| --- | --- | --- | --- | --- | --- | --- |
| Parameter Set 1 | $2$ | $[20,\ 35]$ | $[11,\ 17]$ | $[1,\ 5]$ | $[0.4000,\ 0.5300,\ 0.6500,\ 0.6100]$ | $54.75\%$ |
| Parameter Set 2 | $3$ | $[20,\ 35]$ | $[11,\ 17]$ | $[1,\ 5]$ | $[0.4000,\ 0.6000,\ 0.7000,\ 0.6900]$ | $59.75\%$ |
| Parameter Set 3 | $4$ | $[20,\ 35]$ | $[11,\ 17]$ | $[1,\ 5]$ | $[0.4700,\ 0.4900,\ 0.5600,\ 0.7400]$ | $56.50\%$ |
| **Final Parameters** | **$4$** | **$[20,\ 45]$** | **$[11,\ 17]$** | **$[1,\ 4]$** | **$[0.5900,\ 0.6200,\ 0.6300,\ 0.8200]$** | **$66.50\%$** |

- **Intermediate & Final Confusion Matrices (Subject 11):**

```math
\begin{aligned}
C_{\mathrm{test}}^{(11, \mathrm{set\,1})} &= \begin{bmatrix}
0.4000 & 0.4000 & 0 & 0.2000 \\
0.3100 & 0.5300 & 0.0100 & 0.1500 \\
0.0300 & 0.2200 & 0.6500 & 0.1000 \\
0.0400 & 0.1000 & 0.2500 & 0.6100
\end{bmatrix}, &\quad
C_{\mathrm{test}}^{(11, \mathrm{set\,2})} &= \begin{bmatrix}
0.4000 & 0.4200 & 0.0400 & 0.1400 \\
0.2900 & 0.6000 & 0.0200 & 0.0900 \\
0.0800 & 0.1700 & 0.7000 & 0.0500 \\
0.1000 & 0.1500 & 0.0600 & 0.6900
\end{bmatrix}, \\[8pt]
C_{\mathrm{test}}^{(11, \mathrm{set\,3})} &= \begin{bmatrix}
0.4700 & 0.3500 & 0.0500 & 0.1300 \\
0.3900 & 0.4900 & 0.0100 & 0.1100 \\
0.1000 & 0.2100 & 0.5600 & 0.1300 \\
0.1100 & 0.1500 & 0 & 0.7400
\end{bmatrix}, \\[8pt]
C_{\mathrm{train}}^{(11, \mathrm{final})} &= \begin{bmatrix}
0.9222 & 0.0461 & 0 & 0.0317 \\
0 & 0.9681 & 0 & 0.0319 \\
0 & 0 & 0.9233 & 0.0767 \\
0.0042 & 0.0053 & 0 & 0.9905
\end{bmatrix}, &\quad
C_{\mathrm{test}}^{(11, \mathrm{final})} &= \begin{bmatrix}
0.5900 & 0.2700 & 0.0500 & 0.0900 \\
0.2500 & 0.6200 & 0.0200 & 0.1100 \\
0.1100 & 0.1600 & 0.6300 & 0.1000 \\
0.0600 & 0.0900 & 0.0300 & 0.8200
\end{bmatrix}.
\end{aligned}
```

#### Subject 12 (`codes/subj12.m`, Report Pages 26–27)

| Tuning Stage | `filtersNum` | `filter1` (Hz) | `filter2` (Hz) | `filter3` (Hz) | Test Diagonal Recall $[C_{11}, C_{22}, C_{33}, C_{44}]$ | Mean Test Acc. |
| --- | --- | --- | --- | --- | --- | --- |
| Parameter Set 1 | $4$ | $[20,\ 40]$ | $[10,\ 20]$ | $[0.1,\ 3]$ | $[0.3200,\ 0.4900,\ 0.4800,\ 0.8000]$ | $52.25\%$ |
| Parameter Set 2 | $3$ | $[20,\ 40]$ | $[10,\ 20]$ | $[0.1,\ 3]$ | $[0.3800,\ 0.5100,\ 0.5600,\ 0.9000]$ | $58.75\%$ |
| Parameter Set 3 | $2$ | $[20,\ 40]$ | $[10,\ 20]$ | $[0.1,\ 3]$ | $[0.3000,\ 0.6300,\ 0.5500,\ 0.9500]$ | $60.75\%$ |
| **Final Parameters** | **$2$** | **$[20,\ 40]$** | **$[5,\ 20]$** | **$[0.1,\ 2]$** | **$[0.4500,\ 0.7700,\ 0.4700,\ 0.9000]$** | **$64.75\%$** |

- **Intermediate & Final Confusion Matrices (Subject 12):**

```math
\begin{aligned}
C_{\mathrm{test}}^{(12, \mathrm{set\,1})} &= \begin{bmatrix}
0.3200 & 0.2300 & 0.2000 & 0.2500 \\
0.3000 & 0.4900 & 0.0300 & 0.1800 \\
0.2300 & 0.1400 & 0.4800 & 0.1500 \\
0 & 0.0200 & 0.1800 & 0.8000
\end{bmatrix}, &\quad
C_{\mathrm{test}}^{(12, \mathrm{set\,2})} &= \begin{bmatrix}
0.3800 & 0.1800 & 0.0500 & 0.3900 \\
0.2900 & 0.5100 & 0.0200 & 0.1800 \\
0.2300 & 0.0600 & 0.5600 & 0.1500 \\
0 & 0.0100 & 0.0900 & 0.9000
\end{bmatrix}, \\[8pt]
C_{\mathrm{test}}^{(12, \mathrm{set\,3})} &= \begin{bmatrix}
0.3000 & 0.2700 & 0.1100 & 0.3200 \\
0.1100 & 0.6300 & 0 & 0.2600 \\
0.1700 & 0.0800 & 0.5500 & 0.2000 \\
0 & 0.0200 & 0.0300 & 0.9500
\end{bmatrix}, \\[8pt]
C_{\mathrm{train}}^{(12, \mathrm{final})} &= \begin{bmatrix}
0.6300 & 0.0376 & 0 & 0.3324 \\
0 & 0.8189 & 0 & 0.1811 \\
0.0042 & 0.0226 & 0.8316 & 0.1416 \\
0 & 0.0289 & 0.0326 & 0.9384
\end{bmatrix}, &\quad
C_{\mathrm{test}}^{(12, \mathrm{final})} &= \begin{bmatrix}
0.4500 & 0.2500 & 0 & 0.3000 \\
0.0400 & 0.7700 & 0.0500 & 0.1400 \\
0.1600 & 0.0900 & 0.4700 & 0.2800 \\
0 & 0.0200 & 0.0800 & 0.9000
\end{bmatrix}.
\end{aligned}
```

#### Subject 13 (`codes/subj13.m`, Report Pages 28–29)

| Tuning Stage | `filtersNum` | `filter1` (Hz) | `filter2` (Hz) | `filter3` (Hz) | Test Diagonal Recall $[C_{11}, C_{22}, C_{33}, C_{44}]$ | Mean Test Acc. |
| --- | --- | --- | --- | --- | --- | --- |
| Parameter Set 1 | $2$ | $[12,\ 30]$ | $[10,\ 30]$ | $[1,\ 6]$ | $[0.4500,\ 0.4800,\ 0.4200,\ 1.0000]$ | $58.75\%$ |
| Parameter Set 2 | $2$ | $[12,\ 30]$ | $[10,\ 30]$ | $[1,\ 4]$ | $[0.5000,\ 0.2900,\ 0.4400,\ 1.0000]$ | $55.75\%$ |
| Parameter Set 3 | $5$ | $[12,\ 30]$ | $[10,\ 30]$ | $[1,\ 4]$ | $[0.4100,\ 0.5300,\ 0.4400,\ 0.9500]$ | $58.25\%$ |
| **Final Parameters** | **$5$** | **$[12,\ 30]$** | **$[25,\ 40]$** | **$[1,\ 15]$** | **$[0.6900,\ 0.6700,\ 0.5200,\ 0.9800]$** | **$71.50\%$** |

- **Intermediate & Final Confusion Matrices (Subject 13):**

```math
\begin{aligned}
C_{\mathrm{test}}^{(13, \mathrm{set\,1})} &= \begin{bmatrix}
0.4500 & 0.3500 & 0.2000 & 0 \\
0.4600 & 0.4800 & 0 & 0.0600 \\
0.2100 & 0.3400 & 0.4200 & 0.0300 \\
0 & 0 & 0 & 1
\end{bmatrix}, &\quad
C_{\mathrm{test}}^{(13, \mathrm{set\,2})} &= \begin{bmatrix}
0.5000 & 0.3900 & 0.1100 & 0 \\
0.5700 & 0.2900 & 0.0700 & 0.0700 \\
0.3000 & 0.2100 & 0.4400 & 0.0500 \\
0 & 0 & 0 & 1
\end{bmatrix}, \\[8pt]
C_{\mathrm{test}}^{(13, \mathrm{set\,3})} &= \begin{bmatrix}
0.4100 & 0.4600 & 0.1300 & 0 \\
0.3200 & 0.5300 & 0.0600 & 0.0900 \\
0.3500 & 0.1800 & 0.4400 & 0.0300 \\
0.0100 & 0.0300 & 0.0100 & 0.9500
\end{bmatrix}, \\[8pt]
C_{\mathrm{train}}^{(13, \mathrm{final})} &= \begin{bmatrix}
0.9995 & 0 & 0.0005 & 0 \\
0 & 0.9942 & 0.0011 & 0.0047 \\
0 & 0 & 0.9974 & 0.0026 \\
0 & 0 & 0 & 1
\end{bmatrix}, &\quad
C_{\mathrm{test}}^{(13, \mathrm{final})} &= \begin{bmatrix}
0.6900 & 0.2700 & 0.0400 & 0 \\
0.1500 & 0.6700 & 0.1000 & 0.0800 \\
0.2300 & 0.2000 & 0.5200 & 0.0500 \\
0 & 0.0200 & 0 & 0.9800
\end{bmatrix}.
\end{aligned}
```

#### Subject 14 (`codes/subj14.m`, Report Pages 30–31)

| Tuning Stage | `filtersNum` | `filter1` (Hz) | `filter2` (Hz) | `filter3` (Hz) | Test Diagonal Recall $[C_{11}, C_{22}, C_{33}, C_{44}]$ | Mean Test Acc. |
| --- | --- | --- | --- | --- | --- | --- |
| Parameter Set 1 | $3$ | $[20,\ 35]$ | $[13,\ 20]$ | $[0.1,\ 1.5]$ | $[0.3800,\ 0.2100,\ 0.0000,\ 0.6500]$ | $31.00\%$ |
| Parameter Set 2 | $3$ | $[20,\ 35]$ | $[6,\ 20]$ | $[0.1,\ 0.8]$ | $[0.4100,\ 0.2000,\ 0.3600,\ 0.6300]$ | $40.00\%$ |
| Parameter Set 3 | $3$ | $[20,\ 35]$ | $[3,\ 20]$ | $[5,\ 9]$ | $[0.3700,\ 0.2200,\ 0.2300,\ 0.6300]$ | $36.25\%$ |
| **Final Parameters** | **$3$** | **$[19,\ 35]$** | **$[6,\ 18]$** | **$[5,\ 9]$** | **$[0.2400,\ 0.2700,\ 0.3500,\ 0.6700]$** | **$38.25\%$** |

- **Intermediate & Final Confusion Matrices (Subject 14):**

```math
\begin{aligned}
C_{\mathrm{test}}^{(14, \mathrm{set\,1})} &= \begin{bmatrix}
0.3800 & 0.2700 & 0.0700 & 0.2800 \\
0.3800 & 0.2100 & 0.1100 & 0.3000 \\
0.5300 & 0.2000 & 0 & 0.2700 \\
0.2000 & 0.1500 & 0 & 0.6500
\end{bmatrix}, &\quad
C_{\mathrm{test}}^{(14, \mathrm{set\,2})} &= \begin{bmatrix}
0.4100 & 0.1300 & 0.2800 & 0.1800 \\
0.2500 & 0.2000 & 0.2900 & 0.2600 \\
0.3300 & 0.1200 & 0.3600 & 0.1900 \\
0.2500 & 0.1000 & 0.0200 & 0.6300
\end{bmatrix}, \\[8pt]
C_{\mathrm{test}}^{(14, \mathrm{set\,3})} &= \begin{bmatrix}
0.3700 & 0.3400 & 0.0900 & 0.2000 \\
0.3000 & 0.2200 & 0.2200 & 0.2600 \\
0.2700 & 0.2100 & 0.2300 & 0.2900 \\
0.2600 & 0.1000 & 0.0100 & 0.6300
\end{bmatrix}, \\[8pt]
C_{\mathrm{train}}^{(14, \mathrm{final})} &= \begin{bmatrix}
0.7916 & 0 & 0.0284 & 0.1800 \\
0 & 0.8058 & 0.0353 & 0.1589 \\
0.0022 & 0.0450 & 0.7689 & 0.1839 \\
0.0021 & 0 & 0 & 0.9979
\end{bmatrix}, &\quad
C_{\mathrm{test}}^{(14, \mathrm{final})} &= \begin{bmatrix}
0.2400 & 0.3100 & 0.2000 & 0.2500 \\
0.4000 & 0.2700 & 0.1800 & 0.1500 \\
0.1700 & 0.1700 & 0.3500 & 0.3100 \\
0.1500 & 0.1800 & 0 & 0.6700
\end{bmatrix}.
\end{aligned}
```

#### Subject 15 (`codes/subj15.m`, Report Pages 32–33)

| Tuning Stage | `filtersNum` | `filter1` (Hz) | `filter2` (Hz) | `filter3` (Hz) | Test Diagonal Recall $[C_{11}, C_{22}, C_{33}, C_{44}]$ | Mean Test Acc. |
| --- | --- | --- | --- | --- | --- | --- |
| Parameter Set 1 | $2$ | $[20,\ 35]$ | $[10,\ 50]$ | $[5,\ 9]$ | $[0.5900,\ 0.5200,\ 0.7900,\ 1.0000]$ | $72.50\%$ |
| Parameter Set 2 | $3$ | $[20,\ 35]$ | $[10,\ 50]$ | $[0.1,\ 5]$ | $[0.4100,\ 0.6000,\ 0.7800,\ 1.0000]$ | $69.75\%$ |
| Parameter Set 3 | $2$ | $[20,\ 35]$ | $[10,\ 50]$ | $[0.1,\ 5]$ | $[0.3400,\ 0.6600,\ 0.8400,\ 1.0000]$ | $71.00\%$ |
| **Final Parameters** | **$2$** | **$[20,\ 35]$** | **$[10,\ 50]$** | **$[0.1,\ 6]$** | **$[0.4400,\ 0.6600,\ 0.8100,\ 1.0000]$** | **$72.75\%$** |

- **Intermediate & Final Confusion Matrices (Subject 15):**

```math
\begin{aligned}
C_{\mathrm{test}}^{(15, \mathrm{set\,1})} &= \begin{bmatrix}
0.5900 & 0.2700 & 0.1400 & 0 \\
0.2600 & 0.5200 & 0.2200 & 0 \\
0.1900 & 0.0200 & 0.7900 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}, &\quad
C_{\mathrm{test}}^{(15, \mathrm{set\,2})} &= \begin{bmatrix}
0.4100 & 0.4500 & 0.1400 & 0 \\
0.2200 & 0.6000 & 0.1700 & 0 \\
0.1300 & 0.0900 & 0.7800 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}, \\[8pt]
C_{\mathrm{test}}^{(15, \mathrm{set\,3})} &= \begin{bmatrix}
0.3400 & 0.5400 & 0.1200 & 0 \\
0.1300 & 0.6600 & 0.2100 & 0 \\
0.0400 & 0.1200 & 0.8400 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}, \\[8pt]
C_{\mathrm{train}}^{(15, \mathrm{final})} &= \begin{bmatrix}
0.5200 & 0.3789 & 0.1011 & 0 \\
0 & 0.9550 & 0.0450 & 0 \\
0.0742 & 0.0068 & 0.9189 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}, &\quad
C_{\mathrm{test}}^{(15, \mathrm{final})} &= \begin{bmatrix}
0.4400 & 0.3800 & 0.1800 & 0 \\
0.1800 & 0.6600 & 0.1600 & 0 \\
0.1200 & 0.0700 & 0.8100 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}.
\end{aligned}
```

### 5.4 Notes on the Submitted Solutions / Implementation Notes

- **CSP Column Indexing (`end-(num+1):end`) in `Train.m` (line 12):** Line 12 of `codes/Train.m` selects the extreme CSP filters via `W_csp = [W_csp(:, 1:num) W_csp(:, end-(num+1):end)]`. Because MATLAB indexing `end-(num+1):end` spans `num + 2` columns (for example, when `num = 2`, `end-3:end` selects the last $4$ columns), the total number of retained CSP filters at each tree stage is $2\,\mathrm{filtersNum} + 2$ ($6, 8, 10, 12$ filters for `filtersNum` $= 2, 3, 4, 5$).
- **Frobenius vs. Column-Wise Normalization in `Train.m` (lines 11 and 27):** In `Train.m`, `W_csp = U ./ sqrt(trace(U*U.'))` and `W_lda = U ./ sqrt(trace(U*U.'))` divide the eigenvector matrices by the scalar Frobenius norm $\|U\|_F = \sqrt{\mathrm{tr}(U U^T)}$ rather than normalizing each column separately. Because `eig(Rx_1, Rx_2)` in MATLAB already normalizes generalized eigenvectors uniformly and Fisher LDA projections and midpoint thresholds scale linearly with $\|W_{\mathrm{lda}}\|_2$ (`Mu1 = W_lda.' * mu_1`, `thr = (Mu1 + Mu2)/2`), this uniform scalar scaling leaves the classification boundary and decisions invariant.
- **Dataset Filename in `codes/subj4.m` (line 4) and Parameter Set in `codes/subj10.m` (lines 6–9):**
  - In `codes/subj4.m` (line 4), the `load` command is written as `load subj_3.mat` (a copy-paste carryover from `subj3.m`) instead of `load subj_4.mat`, while lines 6–9 contain Subject 4's tuned bandpass filters (`filter1 = [20 40]`, `filter2 = [11 45]`, `filter3 = [0.1 6]`). To reproduce Subject 4's results on `subj_4.mat`, change line 4 of `subj4.m` to `load subj_4.mat`.
  - In `codes/subj10.m` (lines 6–9), the script stores **Parameter Set 3** from Page 22–23 of the report (`filtersNum = 4`, `filter1 = [20 40]`, `filter2 = [10 45]`, `filter3 = [2 5]`), whereas the **Final Parameters** reported on Page 23 use `filtersNum = 3` and `filter2 = [10 40]`.
- **Order of GEVD Arguments on Page 3 of the Persian Report vs. `Train.m` (lines 23–25):** On Page 3 of `BSS_Project_810100205_Arman_Majidi.pdf`, the definitions $\circledast = \Sigma_1 + \Sigma_2$ and $\circledast\circledast = (\mu_1 - \mu_2)(\mu_1 - \mu_2)^T$ are followed by the boxed expression `[U, \Lambda] = eig(\circledast, \circledast\circledast)` (listing $S_W$ before $S_B$). In the actual code (`Train.m` lines 23–25), the matrices are passed in the standard order `a = (mu_1 - mu_2) * (mu_1 - mu_2).'`, `b = sigma_1 + sigma_2`, and `[U, D] = eig(a, b)` (i.e., $\mathrm{GEVD}(S_B, S_W)$).
- **Minor Typographical Slips in the Persian Report (`BSS_Project_810100205_Arman_Majidi.pdf`) and `Report.pdf`:**
  - **Page 5 (Subject 1 Final `trainConfusionMatrix`):** In the boxed final results for Subject 1 on Page 5, the $4 \times 4$ `testConfusionMatrix` (`diag = [0.4800, 0.6000, 0.6100, 0.9400]`) was accidentally pasted into both the `trainConfusionMatrix` and `testConfusionMatrix` boxes.
  - **Page 15 (Subject 6 Final `trainConfusionMatrix`):** In row 1 of `trainConfusionMatrix` for Subject 6 (`[0.9963, 0.3600, 0.0032, 0]`), entry $(1, 2)$ was typed as `0.3600` (copied from `testConfusionMatrix(1, 2) = 0.3600`) instead of `0.0005` ($1 - 0.9963 - 0.0032 = 0.0005$).
  - **Page 18 (Subject 8 Parameter Set 1):** The initial parameter listing and `testConfusionMatrix` under Parameter Set 1 of Subject 8 duplicate Parameter Set 1 of Subject 7 from Page 16 before transitioning to Subject 8's tuned filter sets in Parameter Sets 2, 3, and Final.
  - **`Report.pdf` Page 3 (Classifier 2 description):** The last sentence of Classifier 2 in `Report.pdf` writes "if the test datum is diagnosed as Class 2 we stop here, otherwise if it belongs to the other 2 classes we proceed to Classifier 3" (a typo for **Class 3**, as shown in the tree diagram on Page 1 of `Report.pdf`).

## 6. How to Use

### 6.1 Dataset Placement and Running the MATLAB Scripts

1. Copy the competition dataset files `subj_1.mat` through `subj_15.mat` (from the course `dataset/` directory) into the `codes/` directory (or add the `dataset/` folder to your MATLAB path via `addpath('/path/to/dataset')`).
2. Open MATLAB, navigate to the `codes/` folder, and run any subject script (`subj1` through `subj15`):

```matlab
cd codes
subj1
```

To run all $15$ subjects sequentially (after ensuring `subj4.m` loads `subj_4.mat`), execute:

```matlab
cd codes
for s = 1:15
    fprintf('=== Running Subject %d ===\n', s);
    run(sprintf('subj%d.m', s));
end
```

- **Dependencies:**
  - **Signal Processing Toolbox:** Required by `codes/Filter.m` for `butter`, `zp2sos`, and `sosfilt`.
  - **MATLAB R2020b or later:** Required by `codes/Train.m` and `codes/Test.m` for 3D page-wise tensor operations `pagemtimes` and `pagetranspose`.

### 6.2 Representative Console Output (`subj1.m`)

Because each script uses `randi` to draw $100$ random Leave-One-Out train/test splits without fixing a random seed (`rng`), the entries of `trainConfusionMatrix` and `testConfusionMatrix` vary slightly across runs around the 100-run Monte Carlo values documented in the Persian report. A representative console output for `subj1.m` (from Page 5 of the report) is:

```text
Training Confusion Matrix:
    0.4800    0.1300    0.3700    0.0200
    0.0600    0.6000    0.2900    0.0500
    0.1400    0.2200    0.6100    0.0300
         0         0    0.0600    0.9400

Test Confusion Matrix:
    0.4800    0.1300    0.3700    0.0200
    0.0600    0.6000    0.2900    0.0500
    0.1400    0.2200    0.6100    0.0300
         0         0    0.0600    0.9400
```

and for `subj4.m` (from Page 11 of the report):

```text
Training Confusion Matrix:
    0.9983         0    0.0017         0
    0.0871    0.9124    0.0006         0
    0.0056    0.0333    0.9611         0
         0         0         0    1.0000

Test Confusion Matrix:
    0.5600    0.4100    0.0300         0
    0.3000    0.6700    0.0300         0
    0.0900    0.1700    0.7400         0
         0         0         0    1.0000
```

### 6.3 Reading the Theoretical Derivations and Report

1. Use **Section 3** for the complete English translation of the final project sheet ([project.pdf](./project.pdf)), the 63-channel EEG recording protocol ([Recording.pdf](./Recording.pdf)), and the reference competition report ([Report.pdf](./Report.pdf)).
2. Use **Section 4** for the mathematical derivations of the 3-stage binary tree, Butterworth bandpass filtering, Common Spatial Patterns (CSP), and Fisher LDA.
3. Use **Section 5** to inspect the master 15-subject summary table, the complete 4-stage hyperparameter tuning logs and confusion matrices for Subjects 1–15, and the implementation notes alongside the original Persian report [BSS\_Project\_810100205\_Arman\_Majidi.pdf](./BSS_Project_810100205_Arman_Majidi.pdf).

## 7. References

1. Akhavan, S. *Blind Source Separation: Final Project*. Course assignment sheet and lecture notes, University of Tehran.
2. Akhavan, S., & Esmaeili, S. (*Team Misagh*). *Report on EEG Data Classification in the 2nd Iranian National BCI Competition* (`Report.pdf` & `Recording.pdf`).
3. Blankertz, B., Müller, K.-R., Curio, G., Vaughan, T. M., Schalk, G., Wolpaw, J. R., Schlögl, A., Neuper, C., Pfurtscheller, G., Hinterberger, T., Schröder, M., & Birbaumer, N. (2004). The BCI Competition 2003: Progress and perspectives in detection and discrimination of EEG single trials. *IEEE Transactions on Biomedical Engineering*, 51(6), 1044–1051.
4. Blankertz, B., Müller, K.-R., Krusienski, D. J., Schalk, G., Wolpaw, J. R., Schlögl, A., Pfurtscheller, G., Millán, J. d. R., Schröder, M., & Birbaumer, N. (2006). The BCI Competition III: Validating alternative approaches to actual BCI problems. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 14(2), 153–159.
5. Blankertz, B., Tomioka, R., Lemm, S., Kawanabe, M., & Müller, K.-R. (2008). Optimizing spatial filters for robust EEG single-trial analysis. *IEEE Signal Processing Magazine*, 25(1), 41–56.
