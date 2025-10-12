Abstract

Brain tumor segmentation plays a crucial role in medical image analysis, forming the foundation for tumor diagnosis, progression monitoring, surgical planning, and radiotherapy assessment. Manual segmentation of brain MRI scans is an extremely labor-intensive process and is subject to inter-observer variability. To address these challenges, this project proposes an automated deep learning–based segmentation framework that integrates Swin V2 Transformers, Convolutional Neural Networks (CNNs), and Convolutional Block Attention Modules (CBAM) for improved spatial and contextual understanding. The model is fine-tuned on the BraTS 2020 dataset, which contains multimodal MRI scans annotated for tumor subregions including the enhancing tumor (ET), tumor core (TC), and whole tumor (WT). The architecture combines the local feature extraction power of CNNs with the global attention capabilities of transformers, while CBAM enhances feature refinement by emphasizing salient regions. Data augmentation techniques were applied to improve generalization, and fine-tuning strategies were employed to adapt pretrained weights to medical imaging data. The proposed system demonstrates competitive accuracy and Dice similarity coefficients, suggesting its potential for real-world clinical applications in tumor segmentation and analysis.

1. Introduction

Brain tumor segmentation is a critical process in neuroimaging, as it enables precise delineation of tumor boundaries from magnetic resonance imaging (MRI) scans. The segmentation process aids clinicians in assessing tumor size, volume, and growth patterns, which are essential for diagnosis and treatment planning. Traditional manual delineation, although accurate, is time-consuming and impractical in large-scale clinical settings. Thus, developing robust automated segmentation techniques is of paramount importance.

Convolutional Neural Networks (CNNs) have historically shown strong performance in segmentation tasks, largely due to their ability to capture hierarchical and spatially local features. However, CNNs have inherent limitations in modeling long-range dependencies, which are essential for understanding complex anatomical structures like brain tumors. Vision Transformers (ViTs), particularly Swin Transformers, have emerged as a powerful alternative, capable of modeling global contextual relationships through self-attention mechanisms. Unlike standard ViTs, Swin V2 introduces a hierarchical design and shifted window attention mechanism, allowing scalable and computationally efficient feature extraction at multiple resolutions.

This project integrates CNNs and Swin V2 Transformers in a unified architecture for brain tumor segmentation. Furthermore, CBAM layers are incorporated to improve focus on informative spatial regions, enhancing the network’s ability to differentiate tumor tissues from surrounding healthy structures. The model is fine-tuned using the BraTS 2020 dataset, which provides multimodal MRI scans with expert-annotated ground truth masks.

2. Problem Definition

Brain tumor segmentation involves classifying each voxel of an MRI volume as belonging to one of several classes: background, enhancing tumor, tumor core, and whole tumor. The goal is to automate this voxel-wise classification with high precision and reliability.

The segmentation process must address several challenges:

Variability of tumor shapes and intensities: Brain tumors exhibit significant variability in location, size, and intensity patterns across patients.

Multimodal complexity: MRI data includes multiple sequences (T1, T1ce, T2, FLAIR), each contributing unique information.

Limited annotated data: High-quality labeled medical data is scarce, and deep learning models often risk overfitting.

Need for global and local understanding: Capturing both fine details (edges, textures) and global dependencies (tumor context within the brain) is essential for accurate segmentation.

The objective of this project is to develop an end-to-end trainable segmentation system that:

Combines CNN and transformer modules to capture both local and global features.

Employs attention mechanisms (CBAM) for enhanced spatial and channel-level focus.

Achieves high Dice similarity scores and intersection-over-union (IoU) across tumor subregions.

3. Dataset Description

The BraTS 2020 dataset (Brain Tumor Segmentation Challenge) is one of the most comprehensive publicly available datasets for brain tumor analysis. It consists of multimodal MRI scans from patients diagnosed with gliomas, including both high-grade glioma (HGG) and low-grade glioma (LGG).

Each patient case contains four MRI modalities:

T1-weighted (T1): captures anatomical details.

T1-contrast-enhanced (T1ce): highlights active tumor regions.

T2-weighted (T2): emphasizes fluid-filled regions.

FLAIR: detects edema and abnormal tissue.

Every MRI volume has a spatial dimension of 240×240×155 voxels and is accompanied by a segmentation mask that labels voxels as:

0: Background

1: Necrotic and non-enhancing tumor core (NCR/NET)

2: Peritumoral edema (ED)

4: Enhancing tumor (ET)

For model training, the data were preprocessed using NIfTI file handling tools, with normalization applied per modality. The segmentation labels were combined to represent three primary regions:

Whole Tumor (WT): labels 1, 2, 4 combined

Tumor Core (TC): labels 1 and 4

Enhancing Tumor (ET): label 4 only

A total of 369 patient volumes were used, split into training, validation, and test sets (80%, 10%, 10%).

4. Data Preprocessing

Data preprocessing plays a vital role in the performance of the segmentation model. The following steps were performed:

Intensity Normalization: Each modality was normalized using z-score normalization to reduce intensity variance across scans.

Resampling and Cropping: MRI volumes were resampled to a uniform voxel spacing of 1mm³ and cropped around the brain region to remove background noise.

Patch Extraction: 3D patches of size 128×128×128 were extracted to reduce GPU memory usage while maintaining spatial context.

Data Augmentation: To increase data diversity and reduce overfitting, multiple augmentation techniques were applied, including:

Random rotations (±15°)

Horizontal and vertical flips

Random elastic deformations

Gaussian noise addition

Intensity shifts and scaling

Random cropping and scaling

The augmentation was performed in real time using TensorFlow and MONAI pipelines.

5. Model Architecture

The proposed architecture is a hybrid Swin V2 Transformer + CNN segmentation network, inspired by encoder-decoder designs like U-Net. The CNN serves as a local feature extractor in the encoder, while the Swin V2 Transformer captures global dependencies. The CBAM layers are embedded within both the encoder and decoder to enhance attention on salient features.

5.1 Encoder

The encoder consists of several convolutional blocks, each followed by batch normalization and ReLU activation. These blocks capture low-level features such as edges and intensity transitions. The encoded feature maps are then passed to Swin Transformer blocks, where multi-head self-attention operates within non-overlapping shifted windows, ensuring global spatial understanding without excessive computational cost.

5.2 Swin V2 Transformer Integration

The Swin V2 Transformer replaces standard CNN down-sampling layers. Its hierarchical design divides the image into fixed-size patches (e.g., 4×4 or 8×8) and processes them using shifted window attention. Each layer consists of:

Layer normalization

Multi-head self-attention (MSA)

Multi-layer perceptron (MLP)

Residual connections for stability

This component models long-range dependencies crucial for understanding the spatial relationships between tumor regions.

5.3 CBAM Layer Integration

The Convolutional Block Attention Module (CBAM) enhances feature maps using two attention mechanisms:

Channel Attention: Focuses on important feature channels via global pooling operations.

Spatial Attention: Applies a convolutional filter to emphasize spatially important regions.

By integrating CBAM, the model dynamically emphasizes tumor-relevant features and suppresses irrelevant background responses.

5.4 Decoder

The decoder mirrors the encoder, consisting of up-sampling and convolutional blocks. Skip connections between encoder and decoder layers preserve spatial details. The final layer uses a 1×1 convolution followed by a softmax activation to generate voxel-wise probability maps for each tumor region.

6. Training Procedure

The model was fine-tuned on the BraTS 2020 dataset using pretrained weights from ImageNet (for CNN) and the Swin V2 base model (pretrained on ImageNet-22K). The following configurations were used:

Optimizer: AdamW



9. Discussion

The experimental results indicate that combining CNN and Swin V2 Transformer architectures yields superior performance in medical image segmentation. CNN layers effectively learn low-level spatial details, while transformer modules capture global relationships, overcoming the locality limitation of convolutional kernels. CBAM further enhances interpretability by directing the network’s focus toward meaningful areas.

The Swin Transformer’s hierarchical design allows efficient computation even on high-resolution images. Its shifted window mechanism ensures continuity across patch boundaries, making it suitable for medical images where anatomical consistency is crucial. The fine-tuning of pretrained weights on BraTS data reduced the need for massive medical datasets while still achieving robust feature learning.

10. Limitations

Despite its success, the proposed approach has limitations. The primary constraint is the computational cost associated with transformer blocks, which increases memory consumption and training time. Processing full 3D MRI volumes remains challenging on consumer GPUs, necessitating patch-based training that may lose some contextual information.

Additionally, while the BraTS dataset provides high-quality annotations, it represents a limited set of tumor types. The model’s generalizability to other pathologies or MRI scanners may require additional fine-tuning and domain adaptation.

11. Future Work

Future research will focus on several enhancements:

3D Transformer Extension: Expanding the model to fully 3D Swin Transformers to capture volumetric context without patch flattening.

Multi-modal Fusion Networks: Developing fusion mechanisms to jointly process all MRI modalities, preserving inter-sequence relationships.

Semi-supervised Learning: Leveraging unlabeled medical data using pseudo-labeling or contrastive pretraining.

Clinical Deployment: Integrating the trained model into medical imaging systems for real-time tumor segmentation and visualization.

12. Conclusion

This project presents a robust, hybrid deep learning framework that integrates Swin V2 Transformers, CNNs, and CBAM layers for brain tumor segmentation. By leveraging both local and global contextual features, the system achieves superior accuracy and generalization on the BraTS 2020 dataset. The combination of transformer-based attention and convolutional inductive biases proves effective for complex medical imaging tasks. The model’s high Dice similarity scores, accurate boundary predictions, and consistent performance across tumor regions make it a promising candidate for aiding clinicians in brain tumor diagnosis and treatment planning.

13. References

Menze, B. H. et al., “The Multimodal Brain Tumor Image Segmentation Benchmark (BraTS),” IEEE Transactions on Medical Imaging, 2015.

Liu, Z. et al., “Swin Transformer V2: Scaling Up Capacity and Resolution,” CVPR, 2022.

Woo, S. et al., “CBAM: Convolutional Block Attention Module,” ECCV, 2018.

Ronneberger, O. et al., “U-Net: Convolutional Networks for Biomedical Image Segmentation,” MICCAI, 2015.

Hatamizadeh, A. et al., “Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors,” CVPR Workshops, 2022.

Kamnitsas, K. et al., “Efficient Multi-Scale 3D CNN with Fully Connected CRF for Accurate Brain Lesion Segmentation,” Medical Image Analysis, 2017.
