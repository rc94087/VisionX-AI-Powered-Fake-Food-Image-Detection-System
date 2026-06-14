
# Phase 2: AI vs Real Image Classifier

## Objective
Train a binary classifier to detect AI-generated vs real food damage images.

## Folder Structure
dataset/
├── train/real, train/ai
├── val/real, val/ai
└── test/real, test/ai

## Steps
1. Install dependencies:
   pip install -r requirements.txt
2. Add images to dataset folders
3. Train model:
   python train_model.py
4. Model saved as ai_vs_real_food_detector.h5

## Highlights
- EfficientNet-B0 transfer learning
- Data augmentation for adversarial robustness
- Metadata forensics layer
