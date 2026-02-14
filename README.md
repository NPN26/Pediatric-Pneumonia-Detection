# Pediatric Pneumonia Detection using Lightweight CNN

Implementation and evaluation of a lightweight convolutional neural network architecture (Chen et al., 2024) for automated binary pneumonia detection in pediatric chest X-rays.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## Project Overview

This project implements a **lightweight CNN architecture** originally designed for multi-class chest X-ray classification (COVID-19/Pneumonia/Normal) and adapts it for **binary pediatric pneumonia detection**. The architecture achieves **92% accuracy** with only ~2 million parameters through efficient Ghost Convolution modules.

### Key Highlights

- [x] **92% test accuracy** on binary pneumonia classification
- [x] **~2.07M parameters** (50% reduction via Ghost Convolutions)
- [x] Balanced detection across bacterial and viral pneumonia subtypes
- [x] Comprehensive **Grad-CAM interpretability analysis**

---

## Dataset

**Source**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) - Kaggle Dataset

**Dataset Statistics**:

- **Total Images**: 5,863 pediatric chest X-rays
- **Training Set**: 5,216 images
- **Validation Set**: 16 images
- **Test Set**: 624 images
- **Class Distribution**:
  - Normal: 1,583 images (27%)
  - Pneumonia: 4,280 images (73%) - includes bacterial and viral subtypes

**Data Source**: Guangzhou Women and Children's Medical Center, Guangzhou  
**Citation**: Kermany et al. (2018), _Cell_

**Note**: Due to size constraints, the dataset is not included in this repository. Download from Kaggle and place in `chest_xray/` directory with structure:

```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

---

## Architecture

Based on **Chen et al. (2024)** lightweight CNN architecture with the following components:

### Core Components

1. **Ghost Convolution Modules**
   - 50% parameter reduction compared to standard convolutions
   - Generates feature maps from cheap linear operations
2. **Feature Extraction (FE) Modules**
   - Residual connections for gradient flow
   - Progressive feature learning across 3 stages
3. **Multi-scale Feature (MF) Module**
   - Spatial pyramid pooling
   - Ghost-Dilated Convolutions for multi-scale context
4. **Efficient Design**
   - Total parameters: ~2.07M
   - Suitable for resource-constrained environments

**Original Paper**: Chen et al. (2024). "Lightweight convolutional neural network for chest X-ray images classification." _Nature Scientific Reports_. [DOI: 10.1038/s41598-024-80826-z](https://doi.org/10.1038/s41598-024-80826-z)

---

## Experimental Results

### Performance Metrics

| Metric            | Value  |
| ----------------- | ------ |
| **Test Accuracy** | 92.00% |
| **Precision**     | TBD    |
| **Recall**        | TBD    |
| **F1-Score**      | TBD    |
| **Parameters**    | 2.07M  |

### Comparison with Original Paper

| Aspect             | Original Paper (Chen et al.)     | This Implementation              |
| ------------------ | -------------------------------- | -------------------------------- |
| Task               | 3-class (COVID/Pneumonia/Normal) | Binary (Normal/Pneumonia)        |
| Pneumonia Accuracy | 97.10%                           | 92.00%                           |
| Dataset            | Mixed age, balanced              | Pediatric only, imbalanced (3:1) |
| Architecture       | Lightweight CNN                  | Same architecture                |
| Parameters         | ~2M                              | ~2.07M                           |

**Performance Gap Analysis**:

- 5% difference explained by dataset characteristics
- Class imbalance (3:1 ratio) vs original balanced dataset
- Pediatric-only population may have different radiological patterns

### Bacterial vs Viral Pneumonia Analysis

- No significant detection bias between bacterial and viral subtypes
- Model learns general pneumonia patterns, not subtype-specific features
- Suggests robust feature learning across pneumonia types

---

## Interpretability Analysis (Grad-CAM)

Comprehensive Grad-CAM study reveals model attention patterns:

### Positive Findings (majority of cases)

**Appropriate Lung Field Focus**:

- Attention on intercostal spaces (gaps between ribs)
- Highlights areas of consolidation and infiltrates
- Clinically relevant anatomical regions

### Concerning Patterns (minority of cases)

**Indirect Sign Reliance**:

- Central mediastinum attention (heart/spine region)
- Diaphragm area highlighting
- May indicate learning secondary signs rather than direct pathology

**Potential Artifacts**:

- Occasional extra-thoracic focus
- Suggests possible overfitting to imaging protocols

**Clinical Implications**: Model shows mostly appropriate behavior but requires multi-institutional validation before deployment.

---

## Getting Started

### Prerequisites

```bash
Python 3.8+
PyTorch 2.0+
CUDA-capable GPU (recommended)
```

### Installation

1. **Clone the repository**:

```bash
git clone "https://github.com/NPN26/Pediatric-Pneumonia-Detection"
cd pneumonia_project
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Download the dataset**:

- Download from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Extract to `chest_xray/` directory

### Usage

**Training and Evaluation**:
Open `model.ipynb` in Jupyter Notebook or VS Code and run all cells sequentially.

**Key Sections**:

1. **Data Loading & Preprocessing** - Data augmentation and class balancing
2. **Architecture Implementation** - Ghost Convolution modules
3. **Training** - With early stopping and learning rate scheduling
4. **Evaluation** - Comprehensive metrics and confusion matrix
5. **Bacterial vs Viral Analysis** - Subtype detection patterns
6. **Grad-CAM Visualization** - Model interpretability study

**Loading Pre-trained Model**:

```python
import torch
model = LightweightCNN(num_classes=2)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
```

---

## Project Structure

```
pneumonia_project/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── model.ipynb                  # Main Jupyter notebook
├── best_model.pth              # Trained model weights
├── chest_xray/                 # Dataset (download separately)
│   ├── train/
│   ├── val/
│   └── test/
└── update_gradcam_v3.py        # Grad-CAM utility script
```

---

## Key Learnings & Insights

### Technical Achievements

1. Successfully implemented complex architecture from academic literature
2. Adapted multi-class model to binary classification task
3. Comprehensive evaluation beyond standard metrics
4. Critical analysis of model behavior through interpretability

### Limitations & Honest Assessment

**Model Limitations**:

- Single-institution dataset (generalization uncertain)
- Pediatric-only population (adult applicability unknown)
- Some reliance on indirect radiological signs
- Requires external validation before clinical use

**What This Project Is**:

- Faithful implementation of published architecture
- Comprehensive evaluation on different dataset
- Thoughtful analysis with transparent reporting

**What This Project Is NOT**:

- Novel architectural contribution
- Clinically validated diagnostic tool
- Production-ready system
- FDA-approved medical device

---

## Future Work

### Immediate Improvements

- [ ] Hyperparameter tuning for imbalanced datasets
- [ ] Ensemble methods with multiple architectures
- [ ] Attention mechanisms for lung region focus
- [ ] Additional data augmentation strategies

### Research Directions

- [ ] Multi-institutional validation
- [ ] Adult population testing
- [ ] Radiologist collaboration for Grad-CAM validation
- [ ] Integration with lung segmentation preprocessing
- [ ] Prospective clinical trial design

---

## References

1. **Chen et al. (2024)**. "Lightweight convolutional neural network for chest X-ray images classification." _Nature Scientific Reports_. DOI: 10.1038/s41598-024-80826-z

2. **Han et al. (2020)**. "GhostNet: More Features from Cheap Operations." _CVPR_.

3. **Kermany et al. (2018)**. "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning." _Cell_.

4. **Selvaraju et al. (2017)**. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." _ICCV_.

5. **He et al. (2016)**. "Deep Residual Learning for Image Recognition." _CVPR_.

---

## Author

**Nirup Nandish**  
_Applied Deep Learning Project - January 2026_

---

## Acknowledgments

- Chen et al. for the original lightweight CNN architecture
- Kermany et al. for the pediatric pneumonia dataset
- Guangzhou Women and Children's Medical Center for data collection
- Kaggle community for dataset hosting

---

## Disclaimer

This project is for **educational and research purposes only**. The model is **NOT intended for clinical use** without proper validation, regulatory approval, and medical professional oversight. Always consult qualified healthcare providers for medical diagnosis and treatment decisions.

---

## Contact

For questions or collaboration opportunities, please open an issue.

---

_Last Updated: January 2026_
