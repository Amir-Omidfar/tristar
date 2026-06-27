# TriStart Computer Vision Take Home Project 
## Objective
Evaluate the given dataset to determine if parts are good or bad. You have 4 days to complete the project.
This can be done with supervised or unsupervised methods. NOTE: if choosing supervised
methods then use the test set to get good/bad images and redo split.
### Dataset
- [dataset overview](https://github.com/stepanje/MPDD?tab=readme-ov-file)
- [dataset download link](https://drive.google.com/file/d/1b3dcRqTXR7LZkOEkVQ9qO_EcKzzC2EEI/view)

### Parts: 
- Bracket (brown, white, black) and 
- Metal plate
You may choose one part or analyze multiple parts for additional exploration.
    - Do not choose connector or tubes
### Data structure:
- Ground truth: Labeled data for test folder
- Training split: Images for model training are only good parts
- Test split: Images for evaluating model performance
- Subfolders distinguish between good and bad parts
- Some training folders only contain good parts, but you can see more bad parts in
the ground truth folder.
- NOTE: You do not need to do an anomaly detection model, you are free to split
up test set to use a supervised learning approach

## Requirements
1. Code Repository:
    - Create a github repo and push all code to it
    - Add ReadME.md to show how to run repo
2. Preprocessing
    - Implement any preprocessing techniques to prepare data for model training
3. Model Creation
    - Build and train models
4. Post Processing
    - Based on model results look at any post processing techniques to improve
performance.

## Design and Submission
- You have the freedom to choose the model and pre/post processing techniques
    - Example techniques: Object detection, segmentation, classification, and/or
anomaly detection
- Either supervised or unsupervised can done
    - Feel free to use multiple models to compare performance
    - A link to the github repository and any accompany results/plots should be submitted
    - Be prepared to discuss your take-home project in a follow-up meeting
## Extra Note
- Please inform us if you have any questions
- We will assess code structure, model pipeline performance, and chosen techniques
- Even if project isn’t finished we can still review and go over repo with you
## Steps:
1. Virtual Environment:
    ```
    conda create -n tristar_test python=3.12
    conda activate tristar_test
    ```
2. Download the dataset from the link above and place it inside the repository at the root level.
## Supervised Learning Method:
1. First classify the item type from the given 4 classes: 1. Brackets (white, brown, blakc) or Metal plate.
2. Then detect whether it's a defect or not. 
***The colors and shapes are fundamentally different. A lightweight backbone like ResNet18 or MobileNetV3 will easily hit near-100% accuracy on this task with very little training.***
               ┌──────────────┐
               │  Input Image │
               └──────┬───────┘
                      │
                      ▼
             ┌──────────────────┐
             │     Stage 1:     │
             │ Part Classifier  │
             └────────┬─────────┘
                      │
      ┌───────────────┼───────────────┬───────────────┐
      ▼               ▼               ▼               ▼
 ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
 │  Metal  │     │  Black  │     │  Brown  │     │  White  │
 │  Plate  │     │ Bracket │     │ Bracket │     │ Bracket │
 └────┬────┘     └────┬────┘     └────┬────┘     └────┬────┘
      │               │               │               │
      ▼               ▼               ▼               ▼
┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐
│ Stage 2A  │   │ Stage 2B  │   │ Stage 2C  │   │ Stage 2D  │
│Defect Det.│   │Defect Det.│   │Defect Det.│   │Defect Det.│
└───────────┘   └───────────┘   └───────────┘   └───────────┘

## Resnet18 is used for the first part and here's the performance:
Class mapping found: {'bracket_black': 0, 'bracket_brown': 1, 'bracket_white': 2, 'metal_plate': 3}
Loaded 665 training images, 190 validation images, and 96 test images.

============================================================
Evaluating model on the completely unseen test set...
============================================================

CLASSIFICATION REPORT:
               precision    recall  f1-score   support

bracket_black       1.00      1.00      1.00        44
bracket_brown       1.00      1.00      1.00        23
bracket_white       1.00      1.00      1.00        16
  metal_plate       1.00      1.00      1.00        13

     accuracy                           1.00        96
    macro avg       1.00      1.00      1.00        96
 weighted avg       1.00      1.00      1.00        96

============================================================
CONFUSION MATRIX:
============================================================
True \ Pred     bracket_black  bracket_brown  bracket_white  metal_plate    
----------------------------------------------------------------------------
bracket_black   44             0              0              0              
bracket_brown   0              23             0              0              
bracket_white   0              0              16             0              
metal_plate     0              0              0              13             
============================================================