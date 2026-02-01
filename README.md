<div align="center">

# 📉 Model Compression using Low-Rank Approximation (SVD)

**Notebook:** `CDM_6_Final.ipynb`  
**Dataset:** CIFAR-10  
**Author:** Danial Nadafi  
**University:** Amir Kabir University of Technology (Tehran Polytechnic)  
**Supervisor:** Dr. Mehdi Ghatee  
**TA:** Dr. Behnam Yousefimehr  

</div>

## 📖 Project Overview

This project implements a **model compression pipeline** using **Singular Value Decomposition (SVD)** to approximate dense layers in deep neural networks. The study focuses on the **ResNet-18** architecture trained on the **CIFAR-10** dataset. The primary goal is to evaluate the trade-offs between **parameter reduction**, **inference latency**, and **classification accuracy** before and after fine-tuning.

The workflow consists of:

1. **Spectral Analysis:** Analyzing the singular values of weight matrices to identify redundancy.  
2. **Low-Rank Approximation:** Replacing the fully connected (linear) layer with truncated SVD components.  
3. **Fine-Tuning:** Retraining the compressed model to recover lost accuracy.  

---

## 🧠 Techniques & Methodology

### 1. Baseline Training
A standard ResNet-18 model is trained on CIFAR-10 to establish a performance baseline.

### 2. SVD Compression
The weight matrix \( W \) of the final linear layer is decomposed using SVD:

$$
W \approx U_k \Sigma_k V_k^T
$$

The single layer is replaced by two smaller linear layers representing $U_k$ and $\Sigma_k V_k^T$.

### 3. Evaluation Metrics

- **Compression Rate:** Percentage of singular values discarded (0%, 50%, 80%)  
- **Parameter Count:** Total trainable parameters  
- **Inference Latency:** Average CPU time per batch  
- **Accuracy:** Test accuracy before and after fine-tuning  

---

## 📊 Experimental Results

| Compression Rate | Total Parameters | Initial Accuracy | Final Accuracy (Fine-tuned) | CPU Latency (ms) |
|------------------|------------------|------------------|-----------------------------|------------------|
| 0% (Baseline)    | **11,181,642**       | **74.13%**           | 74.13%                      | ~106.3           |
| 50%              | 11,178,610       | 60.75%           | **80.60%**                      | **~101.7**           |
| 80%              | 11,177,044       | 19.68%           | 41.25%                      | ~115.4           |

---

## 🔍 Key Findings

- **Accuracy Boost at 50% Compression:**  
  Moderate low-rank approximation combined with fine-tuning improved generalization, likely due to a regularization effect.

- **Critical Information Loss at 80%:**  
  Aggressive compression removed essential representational capacity, leading to irreversible accuracy degradation.

- **Minimal Parameter Reduction:**  
  Compressing only the final fully connected layer yields limited savings (~3k parameters).  
  Meaningful compression requires extending SVD to convolutional kernels.

---

## 📈 Visualizations

The notebook includes:

- **Singular Value Spectrum:** Decay behavior of singular values  
- **Accuracy Recovery Curves:** After fine-tuning  
