# Hierarchical-Cross-Modal-Attention-for-Multimodal-Multiclass-Alzheimer’s-Disease-Diagnosis

## Project Metadata
### Authors
- **Team:** Sabrina Jahan Maisha (g202427560)
- **Supervisor Name:** Dr. Muzammil Behzad
- **Affiliations:** King Fahd University of Petroleum & Minerals (KFUPM)

## Introduction
<div align="justify">
<strong>Alzheimer's Disease (AD)</strong> is a progressive neurodegenerative disorder and the most common cause of dementia, representing a critical global health challenge. According to <strong>WHO</strong>, <strong>57 million people</strong> lived with dementia worldwide in 2021, with Alzheimer's disease being the most common form. Its progression from a preclinical stage to <strong>Cognitively Normal (CN)</strong>, then to <strong>Mild Cognitive Impairment (MCI)</strong>, and finally to <strong>Alzheimer’s disease (AD)</strong> underscores the paramount significance of early and accurate diagnosis. Early detection of MCI and accurate differentiation among these groups is crucial for timely intervention. However, this early detection is difficult due to the subtle and often overlapping brain changes between healthy aging, MCI, and early AD, making robust automated diagnosis a key research frontier. Traditional diagnostic methods rely on a mix of MRI, PET scans, neuropsychological tests, and clinical assessments, but no single modality is sufficient due to overlapping biomarkers and heterogeneous disease presentations. Recent advances in deep learning have shown promise in integrating multimodal data to improve diagnostic accuracy, interpretability, and clinical utility. While these approaches show improved performance, they introduce new challenges, particularly in effectively aligning and fusing data from these diverse and complex modalities to create a coherent and discriminative feature set for classification. <br><br>
</div>
<div align="justify">
This project builds on recent multimodal deep learning frameworks that combine neuroimaging, cognitive scores, and clinical data for automated AD classification. By leveraging feature extraction, contrastive alignment between modalities, and modern fusion strategies, the goal is to advance multi-class Alzheimer’s detection using different modalities such as Brain MRI images and structured clinical cognitive assessments to enhance model performance, robustness and interpretability.
</div>
<img width="729" height="351" alt="Intro-1" src="https://github.com/user-attachments/assets/522a3036-81ce-4963-b965-8a492355af2c" />

## Problem Statement
<div align="justify">
Current state-of-the-art approaches achieve moderate performance in three-class AD diagnosis, indicating the need for more sophisticated multimodal integration strategies. The technical challenges include:
<ul>
<li><strong>Problem 1:</strong> Existing fusion strategies rely on simple concatenation or fixed tabular models, limiting the model’s ability to capture
complex cross-modal interactions between imaging, clinical and cognitive features.</li>
<li><strong>Problem 2:</strong> Current frameworks suffer from bias or robustness due to the inability to the sensitivity in class imbalance</li>
<li><strong>Problem 3:</strong> Most of the works deploy fragmented training pipelines for optimization which prevents true end-to-end learning.</li>
</ul>
</div>

## Application Area and Project Domain
<div align="justify">
The project lies at the intersection of medical imaging, clinical decision support, and deep learning within the healthcare AI domain. The application area is neurological disease diagnosis, specifically Alzheimer’s disease progression modelling. This work is directly relevant to computational neuroimaging, multimodal data fusion, and explainable artificial intelligence (XAI) in healthcare.<br><br>

The project domain extends beyond traditional diagnostic classification to encompass predictive modeling for disease progression, biomarker discovery, and personalized risk assessment. Clinical applications include integration with electronic health record systems, radiological reporting workflows, and clinical trial screening protocols.<br><br>

The project’s outcomes can be applied to -
<li><strong>Hospital memory clinics</strong> for assisting neurologists in the early detection of mild cognitive impairment (MCI);</li>
<li><strong>Research studies or clinical trials</strong> for identifying appropriate patient cohorts based on disease progression patterns, and</li>
<li><strong>Large-scale public health initiatives</strong> like improving ADNI or OASIS datasets, which are focused on large-scale cognitive impairment screening</li>

By improving the accuracy and interpretability of AD diagnosis, this research contributes to the broader objective of developing trustworthy AI systems for clinical neurology, with potential implications for personalised treatment planning and therapeutic development.
</div>

<img width="900" height="410" alt="1-f04beb8c" src="https://github.com/user-attachments/assets/37eaf897-409f-4ac9-8260-101ba9878d53" />


## What is the paper trying to do, and what are you planning to do?
<div align="justify">
The base reference paper <strong><i>(“Multistage Alignment and Fusion for Multimodal Multiclass Alzheimer’s Disease Diagnosis, MICCAI 2025”)</i></strong> proposes a novel multimodal framework that integrates T1-weighted MRI, tau PET, diffusion MRI-derived fiber orientation distributions, and Montreal Cognitive Assessment (MoCA) scores. Its primary contributions are: 1) A novel Swin-FOD model to extract order-balanced features from high-dimensional FOD data, 2) A two-stage contrastive learning framework to align MRI and PET features in a shared latent space, and 3) The use of a pre-trained TabPFN model to classify the fused multimodal features without needing fine-tuning. The model achieved ~73% accuracy on ADNI dataset (n=1147) for three-class classification, demonstrating significant improvement over existing methods and also provided Shapley analysis to quantify modality contributions.<br><br>

Building on this, my project plans to extend the work by implementing a dynamic feature gating mechanism, and interpretability components. 

Based on the findings, I am planning to do -

<strong><u>Proposed Solution 1:</u> To build up a cross-modal interactions model, not just to concatenate </strong> 
To capture complex cross-modal interactions, I will extend the fusion module by modality-specific attention gates, cross-modal transformer blocks, adaptive feature recalibration and multi-head cross-attention before feeding features into TabPFN. This will allow the model to dynamically weight MRI, PET and cognitive scores per patient and learns how much to trust each modality for each patient. Practically, this can be a simple attention gate (learned weights) or a tiny transformer over “tokens” for MRI–PET, cognitive scores, demographics, and genetics. This keeps the pipeline fast while allowing the model to capture relationships across modalities that plain concatenation misses.

<strong><u>Proposed Solution 2:</u> To design comprehensive class-imbalance mitigation strategy </strong> 

To comprehensively address class imbalance sensitivity, we implement a multi-faceted mitigation strategy that employs Dynamic Focal Loss to automatically adjust class weighting based on real-time training statistics. It is followed up by Uncertainty-Aware Sampling that strategically prioritizes challenging cases and minority classes during batch selection. This approach is further strengthened by a Gradient Harmonizing Mechanism that balances gradient contributions across classes to prevent majority class dominance.

<strong><u>Proposed Solution 3:</u> To design a cohesive end-to-end training ecosystem </strong> 

To create a smooth all-in-one training process, we design a unified system that combines the initial learning (pretraining), refinement (fine-tuning), and final task (classification) into a single, connected workflow. This allows the entire model to learn together efficiently. We use techniques like gradient accumulation to work with manageable chunks of data without losing this connectedness, and we carefully coordinate the learning speed across all parts of the model. Most importantly, this setup can ensure that learning signals flow back seamlessly through every component—from the final decision all the way back to each individual medical scan and data point—eliminating stops and gaps that would otherwise slow down and complicate the training.

<strong><u>Proposed Solution 4:</u> To adopt consistency regularised Modality-Aware Learning Loss</strong> 

To address the limitations of static fusion and inconsistent modality contributions, we propose a novel loss formulation that jointly optimizes for classification accuracy, modality reliability, and cross-modal consensus. Our approach introduces a Consistency-Regularized Modality-Aware Loss composed of two key components: (1) an uncertainty-weighted multi-task loss that dynamically adjusts the influence of each modality based on its predictive confidence, and (2) a cross-modal consistency regularization term that penalizes disagreements between modality-specific predictions. The formulation will not only enhance robustness to noisy or missing data but also encourage the model to prioritize modalities with higher clinical reliability for each patient.


</div>

# THE FOLLOWING IS SUPPOSED TO BE DONE LATER

### Project Documents
- **Presentation:** [Project Presentation](https://docs.google.com/presentation/d/1j4fo9gVoOepcwr647Mpc6un3EALFBDTa/edit?usp=sharing&ouid=114530776915226606795&rtpof=true&sd=true)
- **Report:** [Project Report](https://drive.google.com/file/d/1nnPSIFQI072GNP2kQ2Ocp8RBqDnkqEpW/view?usp=sharing)

### Reference Paper
- [Multistage Alignment and Fusion for Multimodal Multiclass Alzheimer’s Disease Diagnosis](https://link.springer.com/chapter/10.1007/978-3-032-05182-0_37)

### Reference Dataset
- [ADNI Dataset](https://ida.loni.usc.edu/)


## Project Technicalities

### Terminologies
- **Hierarchical Cross-Modal Attention:** A fusion mechanism that dynamically models fine-grained interactions between different data modalities through bidirectional attention. 
- **Multimodal Fusion:** A paradigm that integrates information from multiple heterogeneous data sources—such as MRI, PET, and tabular clinical features—to improve diagnostic prediction.
- **LoRA (Low-Rank Adaptation):** A parameter-efficient fine-tuning technique that reduces trainable parameters by injecting trainable rank-decomposition matrices into pre-trained models.
- **TabPFN (Tabular Prior-Data Fitted Network):** A transformer-based foundation model for tabular data that performs in-context learning using pre-computed priors.
- **Focal Loss:** A loss function that addresses class imbalance by focusing training on hard-to-classify examples.
- **SMOTE (Synthetic Minority Over-sampling Technique):**  A data augmentation strategy that generates synthetic samples for minority classes to mitigate class imbalance.
- **AdaIN (Adaptive Instance Normalization):** A feature normalization technique that aligns statistical properties of feature maps between different modalities.
- **Monte Carlo Dropout:** An inference technique that approximates model uncertainty by performing multiple forward passes with dropout enabled.
- **Feature Alignment:** The process of projecting feature representations from different modalities into a shared latent space for semantic comparability.
- **Ensemble Classifier:** A machine learning approach that combines predictions from multiple base models through weighted voting for robust predictions.
- **SHAP (SHapley Additive exPlanations):** A game-theoretic approach to explain model outputs by providing global and local feature importance scores.

### Problem Statements
- **Problem 1:** Existing fusion strategies rely on simple concatenation which limits the model’s ability to capture complex cross-modal interactions     between imaging, clinical and cognitive features.
- **Problem 2:** Current frameworks suffer from bias or robustness due to the sensitivity in class imbalance.
- **Problem 3:** Most of the works deploy fragmented training pipelines for optimization which prevents true end-to-end learning.

### Loopholes or Research Areas
- **Concatenation:** Simple concatenation-based fusion fails to capture complex cross-modal interactions.
- **Imbalance in Dataset:** Class imbalance leads to biased performance across patient groups.
- **Computational Resources:** Memory-intensive architectures hinder deployment in resource-limited settings.

### Problem vs. Ideation: Proposed 3 Ideas to Solve the Problems
1. **Advanced Cross-Modal Integration:** Implements bidirectional cross-attention, AdaIN-style normalization, and self-attention refinement for MRI-PET feature fusion.
2. **Advanced Loss Functions:** Integrate loss functions (e.g., focal loss, consistency loss, contrastive loss, uncertainity) to better capture artistic nuances and structural details with parameter-efficient fine-tuning (LoRA) can stabilize training and improve generalization.
3. **Adaptive Ensemble Classification:** Overcomes class imbalance and optimization fragmentation by using an enhanced TabPFN ensemble with SMOTE balancing and dynamic modality weighting for the final diagnosis.

### Proposed Solution: Code-Based Implementation
This repository provides an implementation of the enhanced stable diffusion model using PyTorch. The solution includes:

- **Hierarchical Cross-Modal Fusion Architecture:** Implements a specialized block featuring AdaIN style transfer, bidirectional MRI-PET cross-attention, and context gating to dynamically align heterogeneous features.
- **Composite Loss Objective:** Combines focal loss, uncertainty regularization, contrastive loss, and consistency terms to robustly handle class imbalance and noisy data.
- **Optimized by Parameter-Efficient Tuning:** Utilizes Vision Transformer (ViT) encoders enhanced with Low-Rank Adaptation (LoRA) and FiLM conditioning to reduce computational cost while maintaining model capacity.
- **Enhanced TabPFN Ensemble:** Uses SMOTE-balanced data, MI+RF-based feature selection, and a dual-expert TabPFN ensemble to fuse imaging and clinical predictors for final CN/MCI/AD classification.

### Key Components
- **`imp_model_pretrain3D.py`**: Contains the modified ImprovedMultiModal3DClassifier and HierarchicalCrossModalFusion modules, which are the core architectural contributions replacing the standard ALBEF fusion.
- **`prepare_json_adni.py & prepare_tr_val_te_adni.py:`**: Scripts responsible for data preprocessing, matching MRI/PET timelines, and generating the 60/20/20 train-val-test splits with augmentation.
- **`imp_adaptive_lora.py:`**: Contains the implementation of advanced techniques including LoRA (Low-Rank Adaptation), AdaIN, FiLM, and Context Gating used to enhance the model's efficiency and expressiveness.
- **`imp_optimizers.py:`**: Contains the advanced optimization strategies, including layer-wise learning rate decay and custom schedulers to ensure stable convergence.
- **`ultimate_train_ADNI.py`**: Script to handle the primary execution for stage 1, handling the end-to-end training loop, custom loss integration with objective loss function and feature extraction.
- **`ultimate_train_ADNI.yaml:`**: The central configuration file driving the enhanced training pipeline, defining critical hyperparameters for LoRA, Focal Loss weights, layer-wise learning rates, and gradient accumulation settings.
- **`utils.py`**: Utility functions for metrics, logging, distributed training setup, and tracking smoothed training statistics (e.g., MetricLogger, SmoothedValue).
- **`imp_tabpfn_for_multiclass_classification_contrast_FOD_moca.py`**: Script for implementing stage 2 of the pipeline, utilizing the extracted features to train the TabPFN ensemble with SMOTE balancing, feature selection, and SHAP analysis.

## Model Workflow
The workflow of the Hierarchical Cross-Modal Attention framework for Alzheimer's Disease diagnosis integrates multimodal data through a structured pipeline:

1. **Input:**
   - **Neuroimaging Data:** T1w MRI and Tau PET scans are encoded using Vision Transformer (ViT) backbones enhanced with LoRA for parameter-efficient feature extraction.
   - **Tabular Data** Age, sex, and MoCA scores and other tabular data are processed through FiLM conditioning to modulate subsequent feature representations.
   
2. **Adaptive Hierarchical Fusion:**
   - **Feature Alignment:** Modality features undergo Adaptive Instance Normalization (AdaIN) for style transfer and distribution alignment.
   - **Cross-Modal Attention:** A hierarchical cross-attention mechanism enables bidirectional interaction between MRI and PET features.
   - **Feature Refinement:** Self-attention layers further refine the fused representations, with context gating dynamically weighting important features.

3. **Multi-Objective Optimization:**
      - **Loss Optimization:**  The combined loss function with four ways ensures balanced training across classification, uncertainty, and consistency objectives.

4. **Output Prediction:**
   - **Feature Integration:** Fused imaging features are combined with tabular data with better representations.
   - **Enhanced TabPFN:** The integrated features are processed through an enhanced TabPFN ensemble classifier having both imaging TabPFN and tabular TabPFN.
   - **Weighted Voting Ensemble:** Specialized classifiers for imaging and clinical data contribute through weighted voting.
   - **Final Output:** The system generates diagnostic classification for CN/MCI/AD cases with classification metrices.

  

## How to Run the Code

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/BRAIN-Lab-AI/Knowledge-Aware-Cross-Attention-for-Multimodal-Alzheimer-s-Diagnosis-from-Brain-MRI-.git
    cd multimodalAD/ALBEF
    ```

2. **Set Up the Environment:**
    Create a virtual environment and install the required dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    pip install -r requirements.txt
    ```
3. **Accessing Token:**
     Login Hugging Face and access the hugging face token to get access of TabPFN model.
    ```bash
    huggingface-cli login
    # Enter your Hugging Face access token when prompted

4. **Data Preparation:**
      Prepare image data files into JSON file types.
    ```bash
    # Navigate to data directory
      cd data

    # Prepare JSON files for ADNI dataset
      python prepare_json_adni.py

    # Split data into train/val/test sets
      python prepare_tr_val_te_adni.py


3. **Train the Improved Model:**
    Configure the training parameters in the provided configuration file and run:
    ```bash
    cd multimodalAD/ALBEF 
    
    # Test model initialization
    python imp_model_pretrain3D.py \
    --config ./configs/Pretrain3D.yaml \
    --output_dir output/imp_Pretrain3D \
    --device cuda
    
    # Run ultimate training pipeline
    python ultimate_train_ADNI.py \
    --config configs/ultimate_train_ADNI.yaml \
    --output_dir output/ultimate_train_ADNI \
    --device cuda \
    --seed 42
    ```

4. **TabPFN Classification:**
    Once training is complete, use the generated checkpoints to run the TabPFN pipeline for multiclass classification and feature importance analysis (SHAP).
    ```bash
    python imp_tabpfn_for_multiclass_classification_contrast_FOD_moca.py \
    --checkpoint output/ultimate_train_ADNI/checkpoint_best.pth \
    --device cuda \
    --seed 42
    ```

## Acknowledgments
- **Open-Source Communities:** Thanks to the contributors of PyTorch, Hugging Face, and other libraries for their amazing work.
- **Foundation Work:** This research builds upon the foundational work "MultimodalAD: Multistage Alignment and Fusion for Multimodal Multiclass Alzheimer's Disease Diagnosis" by Huang et al. We thank the authors for their valuable contributions and for making their code publicly available at: https://github.com/huangshuo343/multimodalAD.
- **Data Providers:** We gratefully acknowledge the Alzheimer's Disease Neuroimaging Initiative (ADNI) for providing the dataset used in this study. T
- **Resource Providers:** Gratitude to ABC-organization for providing the computational resources necessary for this project.
