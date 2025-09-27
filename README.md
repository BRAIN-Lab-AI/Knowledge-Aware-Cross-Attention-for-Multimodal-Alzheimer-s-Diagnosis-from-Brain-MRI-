# Knowledge-Aware-Cross-Attention-for-Multimodal-Alzheimer-s-Diagnosis-from-Brain-MRI

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
<li><strong>Problem 1:</strong> Existing fusion strategies rely on simple concatenation or fixed tabular models, limiting the model’s ability to capture complex cross-modal interactions between imaging, clinical, cognitive, and genetic features.</li>
<li><strong>Problem 2:</strong> There is a limited capability of multimodal AD models to distinguish MCI from CN/AD, partly because diffusion MRI (FOD) features are underutilised.</li>
<li><strong>Problem 3:</strong> Global feature importance interpretability methods fail to provide fine-grained, region-level, interaction-level, or case-level explanations that are necessary for clinical trust and understanding in individual diagnoses.</li>
<li><strong>Problem 4:</strong> Current frameworks often suffer from bias or robustness due to missing or noisy modalities (i.e. absent or incomplete data).</li>
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
To capture complex cross-modal interactions, I will extend the fusion module by inserting a lightweight attention gate or TabTransformer block before feeding features into TabPFN. This will allow the model to dynamically weight MRI, PET, cognitive scores, and genetic features per patient and learns how much to trust each modality for each patient. Practically, this can be a simple attention gate (learned weights) or a tiny transformer over “tokens” for MRI–PET, FOD, cognitive scores, demographics, and genetics. This keeps the pipeline fast while allowing the model to capture relationships across modalities that plain concatenation misses.

<strong><u>Proposed Solution 2:</u> To improve the utility of underperforming modalities (e.g., diffusion MRI) through auxiliary learning tasks</strong> 

To improve the discriminative power of underutilized modalities like diffusion MRI (FOD), I will introduce an auxiliary learning task with a multi-task objective that forces the Swin-FOD encoder to predict tract-level microstructural biomarkers (e.g., FA, MD values) alongside the main diagnosis. This setup should enrich the learned representations and increase FOD’s contribution, especially for challenging MCI cases. The goal is to boost the usefulness of diffusion information and improve separation of MCI from CN/AD without heavy computation.

<strong><u>Proposed Solution 3:</u> To establish interaction-aware explainability </strong> 

We will establish a multi-level interpretability protocol: 
(a) ROI-level SHAP by pooling imaging encoder activations and reporting region-level attributions (e.g., hippocampus, PCC) 
(b) SHAP interaction values, to quantify pairwise synergies (e.g., APOE4 × hippocampal atrophy, MoCA × PET-temporal signal); and 
(c) Presenting individual patient-level explanations that combine SHAP decision plots with MRI/PET heatmaps (Grad-CAM). 
These additions explain which regions and features matter, how they work together, and why a borderline MCI case was classified a certain way.

<strong><u>Proposed Solution 4:</u> To handle missing and noisy modalities by Modality Dropout and Adaptive Gating</strong> 

During training, we will randomly hide or down-weight modalities (modality “dropout”) so the model learns to cope with incomplete data. At inference, a simple gating mechanism reduces the influence of absent or low-quality inputs and relies more on the remaining signals. During optimization, random modality masks are applied to feature tokens; the pre-fusion module learns to reweight available information. In case of inference, missing modalities are zero-masked and the gate down-weights their influence. This makes the system practical for real settings where PET or some clinical fields may be unavailable.

<strong><u>Proposed Solution 5:</u> To adopt consistency regularised Modality-Aware Learning Loss</strong> 

To address the limitations of static fusion and inconsistent modality contributions, we propose a novel loss formulation that jointly optimizes for classification accuracy, modality reliability, and cross-modal consensus. Our approach introduces a Consistency-Regularized Modality-Aware Loss composed of two key components: (1) an uncertainty-weighted multi-task loss that dynamically adjusts the influence of each modality based on its predictive confidence, and (2) a cross-modal consistency regularization term that penalizes disagreements between modality-specific predictions. The formulation will not only enhance robustness to noisy or missing data but also encourage the model to prioritize modalities with higher clinical reliability for each patient.


</div>

# THE FOLLOWING IS SUPPOSED TO BE DONE LATER

### Project Documents
- **Presentation:** [Project Presentation](/presentation.pptx)
- **Report:** [Project Report](/report.pdf)

### Reference Paper
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

### Reference Dataset
- [LAION-5B Dataset](https://laion.ai/blog/laion-5b/)


## Project Technicalities

### Terminologies
- **Diffusion Model:** A generative model that progressively transforms random noise into coherent data.
- **Latent Space:** A compressed, abstract representation of data where complex features are captured.
- **UNet Architecture:** A neural network with an encoder-decoder structure featuring skip connections for better feature preservation.
- **Text Encoder:** A model that converts text into numerical embeddings for downstream tasks.
- **Perceptual Loss:** A loss function that measures high-level differences between images, emphasizing perceptual similarity.
- **Tokenization:** The process of breaking down text into smaller units (tokens) for processing.
- **Noise Vector:** A randomly generated vector used to initialize the diffusion process in generative models.
- **Decoder:** A network component that transforms latent representations back into image space.
- **Iterative Refinement:** The process of gradually improving the quality of generated data through multiple steps.
- **Conditional Generation:** The process where outputs are generated based on auxiliary inputs, such as textual descriptions.

### Problem Statements
- **Problem 1:** Achieving high-resolution and detailed images using conventional diffusion models remains challenging.
- **Problem 2:** Existing models suffer from slow inference times during the image generation process.
- **Problem 3:** There is limited capability in performing style transfer and generating diverse artistic variations.

### Loopholes or Research Areas
- **Evaluation Metrics:** Lack of robust metrics to effectively assess the quality of generated images.
- **Output Consistency:** Inconsistencies in output quality when scaling the model to higher resolutions.
- **Computational Resources:** Training requires significant GPU compute resources, which may not be readily accessible.

### Problem vs. Ideation: Proposed 3 Ideas to Solve the Problems
1. **Optimized Architecture:** Redesign the model architecture to improve efficiency and balance image quality with faster inference.
2. **Advanced Loss Functions:** Integrate novel loss functions (e.g., perceptual loss) to better capture artistic nuances and structural details.
3. **Enhanced Data Augmentation:** Implement sophisticated data augmentation strategies to improve the model’s robustness and reduce overfitting.

### Proposed Solution: Code-Based Implementation
This repository provides an implementation of the enhanced stable diffusion model using PyTorch. The solution includes:

- **Modified UNet Architecture:** Incorporates residual connections and efficient convolutional blocks.
- **Novel Loss Functions:** Combines Mean Squared Error (MSE) with perceptual loss to enhance feature learning.
- **Optimized Training Loop:** Reduces computational overhead while maintaining performance.

### Key Components
- **`model.py`**: Contains the modified UNet architecture and other model components.
- **`train.py`**: Script to handle the training process with configurable parameters.
- **`utils.py`**: Utility functions for data processing, augmentation, and metric evaluations.
- **`inference.py`**: Script for generating images using the trained model.

## Model Workflow
The workflow of the Enhanced Stable Diffusion model is designed to translate textual descriptions into high-quality artistic images through a multi-step diffusion process:

1. **Input:**
   - **Text Prompt:** The model takes a text prompt (e.g., "A surreal landscape with mountains and rivers") as the primary input.
   - **Tokenization:** The text prompt is tokenized and processed through a text encoder (such as a CLIP model) to obtain meaningful embeddings.
   - **Latent Noise:** A random latent noise vector is generated to initialize the diffusion process, which is then conditioned on the text embeddings.

2. **Diffusion Process:**
   - **Iterative Refinement:** The conditioned latent vector is fed into a modified UNet architecture. The model iteratively refines this vector by reversing a diffusion process, gradually reducing noise while preserving the text-conditioned features.
   - **Intermediate States:** At each step, intermediate latent representations are produced that increasingly capture the structure and details dictated by the text prompt.

3. **Output:**
   - **Decoding:** The final refined latent representation is passed through a decoder (often part of a Variational Autoencoder setup) to generate the final image.
   - **Generated Image:** The output is a synthesized image that visually represents the input text prompt, complete with artistic style and detail.

## How to Run the Code

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/enhanced-stable-diffusion.git
    cd enhanced-stable-diffusion
    ```

2. **Set Up the Environment:**
    Create a virtual environment and install the required dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Train the Model:**
    Configure the training parameters in the provided configuration file and run:
    ```bash
    python train.py --config configs/train_config.yaml
    ```

4. **Generate Images:**
    Once training is complete, use the inference script to generate images.
    ```bash
    python inference.py --checkpoint path/to/checkpoint.pt --input "A surreal landscape with mountains and rivers"
    ```

## Acknowledgments
- **Open-Source Communities:** Thanks to the contributors of PyTorch, Hugging Face, and other libraries for their amazing work.
- **Individuals:** Special thanks to bla, bla, bla for the amazing team effort, invaluable guidance and support throughout this project.
- **Resource Providers:** Gratitude to ABC-organization for providing the computational resources necessary for this project.
