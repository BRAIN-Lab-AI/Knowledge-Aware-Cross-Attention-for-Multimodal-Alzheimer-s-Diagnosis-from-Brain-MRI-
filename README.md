# Knowledge-Aware-Cross-Attention-for-Multimodal-Alzheimer-s-Diagnosis-from-Brain-MRI

## Project Metadata
### Authors
- **Team:** Sabrina Jahan Maisha
- **Supervisor Name:** Dr. Muzammil Behzad
- **Affiliations:** King Fahd University of Petroleum & Minerals (KFUPM)

## Introduction
<div align="justify">
**Alzheimer's Disease (AD)** is a progressive neurodegenerative disorder and the most common cause of dementia, representing a critical global health challenge. According to **WHO**, **57 million people** lived with dementia worldwide in 2021, with Alzheimer's disease being the most common form. Its progression from a preclinical stage to Cognitively Normal (CN), then to Mild Cognitive Impairment (MCI), and finally to Alzheimer’s disease (AD) underscores the paramount significance of early and accurate diagnosis. Early detection of MCI and accurate differentiation among these groups is crucial for timely intervention. However, this early detection is difficult due to the subtle and often overlapping brain changes between healthy aging, MCI, and early AD, making robust automated diagnosis a key research frontier. Traditional diagnostic methods rely on a mix of MRI, PET scans, neuropsychological tests, and clinical assessments, but no single modality is sufficient due to overlapping biomarkers and heterogeneous disease presentations. Recent advances in deep learning have shown promise in integrating multimodal data to improve diagnostic accuracy, interpretability, and clinical utility. While these approaches show improved performance, they introduce new challenges, particularly in effectively aligning and fusing data from these diverse and complex modalities to create a coherent and discriminative feature set for classification. 
</div>
<div align="justify">
This project builds on recent multimodal deep learning frameworks that combine neuroimaging, cognitive scores, and clinical data for automated AD classification. By leveraging feature extraction, contrastive alignment between modalities, and modern fusion strategies, the goal is to advance multi-class Alzheimer’s detection using different modalities such as Brain MRI images and structured clinical cognitive assessments to enhance model performance, robustness and interpretability.
</div>

## Problem Statement
<div align="justify">
Current state-of-the-art approaches achieve moderate performance in three-class AD diagnosis, indicating the need for more sophisticated multimodal integration strategies. The technical challenges include:

<strong>Problem 1:</strong> Existing fusion strategies rely on simple concatenation or fixed tabular models, limiting the model’s ability to capture complex cross-modal interactions between imaging, clinical, cognitive, and genetic features.
<strong>Problem 2:</strong> There is a limited capability of multimodal AD models to distinguish MCI from CN/AD, partly because diffusion MRI (FOD) features are underutilised.
<strong>Problem 3:</strong> Global feature importance interpretability methods fail to provide fine-grained, region-level, interaction-level, or case-level explanations that are necessary for clinical trust and understanding in individual diagnoses.
<strong>Problem 4:</strong> Current frameworks often suffer from bias or robustness due to missing or noisy modalities (i.e. absent or incomplete data).
</div>

## Application Area and Project Domain
Write 1-2 technical paragraphs (feel free to add images if you would like).

## What is the paper trying to do, and what are you planning to do?
Write 1-2 technical paragraphs (feel free to add images if you would like).


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
