# Brain-Tumor-Segmentation-Using-Swin-V2-Transformers-and-CNNs
This repository implements a hybrid brain tumor segmentation model that combines Swin V2 transformer blocks with convolutional encoder–decoder components and Convolutional Block Attention Modules (CBAM). The network is trained and fine-tuned on the BraTS 2020 dataset to segment 
Data: BraTS 2020

This project uses the BraTS 2020 dataset. The dataset is not included in the repository. Follow official BraTS instructions to download and prepare the data.
Expected input: co-registered, skull-stripped 4-modal MRI volumes per patient (T1, T1Gd, T2, FLAIR).
The dataset loader implements per-case normalization and consistent patch/volume cropping for training.
Preprocessing
Intensity normalization per modality (z-score or percentile clipping + min-max).
Resampling to consistent voxel spacing (if required).
Optional brain-masking to focus training on intracranial region.
Patch extraction or whole-volume training depending on GPU memory.
Data augmentation
Augmentations applied during training (configurable):
Random flips and rotations (90-degree multiples and small-angle random rotations).
Random contrast/brightness adjustments and Gaussian noise.
Elastic deformation and random scaling.
Random crop/patch sampling with balanced foreground (ensure tumor presence probability).
All augmentations are implemented as on-the-fly transforms in src/data/augmentations.py.
Model architecture

Encoder path: convolutional blocks to extract multi-scale local features. CBAM modules are applied after selected convolutional blocks to reweight channel and spatial features.

Transformer path: Swin V2 blocks operate on patch embeddings at multiple scales to capture global and long-range contextual relationships. Where relevant, swin features are fused with convolutional features via skip connections and attention-guided fusion layers.

Decoder path: upsampling and convolutional refinement to reconstruct segmentation maps, using fused features from both transformer and CNN streams.

Output: multi-class segmentation logits for {background, ET, TC, WT} or a commonly used mapping depending on loss/metric pipeline.

Files of interest:

src/models/swinv2_unet.py — full hybrid implementation and fusion blocks.

src/models/cbam.py — CBAM (channel + spatial attention) implementation.

Loss functions and optimization

Typical losses used: combination of Dice loss and cross-entropy (e.g., Loss = DiceLoss + λ * CrossEntropy). For class imbalance, consider using weighted CE or focal variants.

Optimizer: AdamW or Adam with weight decay.

Scheduler: cosine annealing or step LR scheduler.

Example hyperparameters (replace/adjust as necessary):

Learning rate: 1e-4 (base), with warmup for transformer weights.

Batch size: depends on GPU memory (common values: 1–4 for full-volume, 8–16 for patches).

Epochs: 200–500 depending on early stopping and compute budget.

Weight decay: 1e-5.
