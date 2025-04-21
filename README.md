# HGCL-DR: A Heterogeneous Graph Contrastive Learning Framework for Drug Repositioning

Drug repositioning aims to identify new therapeutic uses for existing drugs, providing a cost-effective alternative to traditional drug development. This repository contains the official implementation of **HGCL-DR**, a novel **heterogeneous graph contrastive learning** framework that effectively integrates both **global** and **local** feature representations for drug repositioning.

---

## 🌟 Highlights

- **Improved Heterogeneous Graph Contrastive Learning**: Captures semantic consistency across heterogeneous relations.
- **Subgraph-Based Local Feature Learning**: Uses a bidirectional GCN and graph diffusion to model long-range dependencies.
- **Global-Local Fusion**: Contrastive learning enhances representation quality across different spaces.
- **Strong Performance**: Outperforms state-of-the-art baselines on four benchmark datasets in AUPR, AUROC, and F1-score.

---

## 🧠 Model Architecture

<!-- Insert your model diagram here -->
<!-- For example: ![HGCL-DR Architecture](./figures/hgcl_dr_model.png) -->
<p align="center"><img src="model (2).jpg" width="700"></p>

---

## 📁 Project Structure

## ▶️ How to Use

To train the HGCL-DR model on your dataset, simply run the following command:

```bash
python main.py
