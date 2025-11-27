# TDA Diagnostics of CNN Predictors on Shock Tube Dynamics

---

## 📌 Overview

This project investigates whether **Topological Data Analysis (TDA)** can detect — and quantify —  the **failure modes of neural temporal predictors** of PDE simulations.

We study a 1D **Brio–Wu MHD shock tube**–like Riemann problem and evaluate:

- Ground-truth fields (density & pressure)

- Baseline CNN next-step predictor

- Iterated multi-step forecasting

- TDA diagnostics (Betti curves, persistence diagrams)

This repository contains the full pipeline: simulation preprocessing, temporal ML prediction, and TDA-based evaluation.

---

## 🔍 Project Goals

1. Generate and preprocess 1D hydrodynamic/MHD simulation output

2. **Train a CNN to predict the next timestep** from $(\rho_t,p_t)$

3. **Evaluate prediction quality** (pointwise error, multi-step drift)

4. **Compute topological features** of:
   
   - Ground truth evolution
   
   - CNN predictions

5. **Compare topological signatures** in $\beta_{0}(\varepsilon,t)$, and $\beta_{1}(\varepsilon,t)$ to assess model failure (shock smearing, incorrect wave speeds, missing rarefaction branches) 

---

## 📁 Repository Structure

```git
project/
│
├── data/                          # Raw & normalized simulation arrays
├── results/
│     ├── tda/                     # Betti curves, persistence diagrams
│     ├── models/                  # Trained CNN weights + predictions
│
├── src/
│   ├── io/
│   │   └── load_brio_wu.py        # Loads raw simulation
│   ├── tda/
│   │   ├── spatial_embedding.py   # Takens spatial embedding
│   │   └── persistence_tools.py   # VR filtration, Betti curves, landscapes
│   ├── models/
│   │   └── cnn1d_temporal.py      # Baseline CNN
│
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_tda_ground_truth.ipynb
│   ├── 03_models_baseline.ipynb
│   └── 04_tda_predictions.ipynb
│
├── LICENSE
├── .gitignore
└── README.md
```

---

## 🧠 Baseline 1D CNN

  We use a lightweight CNN with 3 convolutional layers:

```python
Conv1D(2 → 32, kernel = 5, padding = 2) + ReLU
Conv1D(32 → 32, kernel = 5, padding = 2) + ReLU
Conv1D(32 → 2, kernel = 5, padding = 2)
```

- Training samples: 40 time-pairs  

- Loss: MSE on normalized fields

---

## 🧮 TDA Pipeline

For each time $t_{k}$:

1. Spatial Takens embedding $\Phi_{m,\tau}(\rho_{t})$

2. Vietoris–Rips persistence ($H_0$ and $H_1$)

3. Betti curves over radius $\beta_{j}(\varepsilon,t)$

4. Visual comparison with predicted fields

---

## 📊 Key Findings

- The CNN predicts smooth fields but **fails to reproduce shocks and rarefaction waves**.

- Iterated forecasting diverges quickly.

- TDA successfully identifies:
  
  - collapse of shock topology in predictions
  
  - missing rarefaction structures
  
  - smoothed/shrunk H1 features

- Betti-number heatmaps clearly show **topological mismatch** between simulation and CNN.

This provides a clean proof-of-principle: **TDA can diagnose ML surrogate failures in PDE prediction**.

---

## 🪞 Poster Reference

The results of this repository were presented at **“XXXIX EFNNE”** as a poster titled:  

> **Topological Machine Learning for Predicting Spatiotemporal Evolution in 1D Magnetohydrodynamics**

Poster and figures: [`docs/XXXIX_EFNNE_Poster.pdf`](https://github.com/GabrielWendell/TDA-Brio-Wu_MHD_CNN-XXXIX_EFNNE/blob/main/docs/XXXIX_EFNNE_Poster.pdf)

You can also check out a 15-minute seminar I gave at IIP on the topic of [*Topological Data Analysis Applied to Complex Systems*](https://www.youtube.com/watch?v=KZo9ciaaBKI&list=PLqTLz9G2bGs2-ToEsAuUg3u6ZASY1UbAk&index=23&t=625s) for a more basic reference on the fundamentals of TDA.

<img src = "docs/IIP_Seminar_Img.png" width="700" height="300">

---

## 📜 License

This project is released under the **MIT License**.  
See `LICENSE` for details.

---

## 🙌 Acknowledgments

Supported by the Pos-Graduation Program in Physics (UFRN).  
Special thanks to the XXXIX EFNNE organizers.
