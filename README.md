# Document Verification System

## Overview
This project focuses on document verification by integrating multiple machine learning models for document classification, OCR-based text extraction, text similarity checking, face comparison and face recognition. The system verifies a user's identity by analyzing submitted documents using deep learning techniques.

**About the Contributor**

**Ganesh Chaudhary — Front-end Developer**
Hello world!. In this project, my responsibilites to build the front-end part and I fulfilled through this:
1. Built a user-friendly interface using HTML, CSS, JavaScript, and Bootstrap
2. Prioritized mobile-responsive design
3. Optimized front-end code for performance
4. Integrated front-end, back-end, and YOLO model into a seamless workflow

## Training and Evaluation

### 1. TrOCR Model
#### 1.1. Synthetic Data Preparation
- The dataset was generated using a Nepali corpus from Hugging Face: [Nepali-English Translation Dataset](https://huggingface.co/datasets/ashokpoudel/nepali-english-translation-dataset)
- Steps involved:
  - Extracting text from the corpus.
  - Generating synthetic text samples with 1-3 words.
  - Creating numeric text data similar to Nepali citizenship documents.
  - Merging text datasets.
  - Generating synthetic images with various fonts, backgrounds, and noise.
- [Data Preparation Code Directory](training_evaluation/ocr/ocrdataset/)

#### 1.2. Training TrOCR Model
- The TrOCR model was fine-tuned using Microsoft’s [`trocr-small-printed`](https://huggingface.co/microsoft/trocr-small-printed) model.
- Training configurations:
  - **Batch size:** 4
  - **Epochs:** 9 (first dataset), 7 (second dataset)
  - **Tokenizer:** `microsoft/trocr-small-printed`
  - **Optimizer:** Adam
  - **Learning rate:** Adjusted with decay
- [Training Notebook Directory](training_evaluation/ocr/kaggle_nepali_train/)

#### 1.3. CER Calculation and Evaluation
- The model is evaluated using **Character Error Rate (CER)**.
- Evaluation script: [CER Calculation](training_evaluation/ocr/cer.ipynb)

### 2. ResNet50 Classification Model
#### 2.1. Training ResNet50 Model
- The ResNet50 model is fine-tuned for document classification.
- Categories: Citizenship Front, Citizenship Back, ID Card, Random.
- Training configurations:
  - **Dataset Split:** Train/Validation/Test
  - **Augmentation:** Exposure adjustments, noise addition, rotation
  - **Layer Freezing:** 70% of layers frozen
  - **Optimizer:** Adam
  - **Learning Rate:** Exponential decay
- [Training Notebook](training_evaluation/document_classifier/training.ipynb)

## Inference

- [Inference Code Directory](training_evaluation/inference/)
  ```sh
  cd training_evaluation/inference
  ```
- **Install `dlib` for a specific Python version and the requirements**:
  ```sh
  pip install dlib/<dlib for your python version>
  ```
  ```
  pip install -r requirements.txt
  ```

- **Download Pre-trained Models**:
  - [Nepali TrOCR Model](https://drive.google.com/drive/folders/1rjHBUjDJAwNAuaevrZa7RMujwjWXZUOJ?usp=drive_link)
  -- place TrOCR model in path `training_evaluation/inference/models/trocr/part_3/` 
  - [ResNet50 Classifier Model](https://drive.google.com/drive/folders/1EAE6bRMMTRR-ORY1WIg5NnTvaopeo0mJ?usp=drive_link)
  -- place ResNet50 model in path `training_evaluation/inference/models/document_classifier/`
- **Set the path of documents and Run inference script**:
  ```sh
  python inference.py
  ```

## Web Application

- **Navigate to the WebApp directory**:
  ```sh
  cd frontend_backend/
  ```
- **Install dependencies**:
  ```sh
  pip install -r requirements.txt
  ```
- **Run the web application**:
  ```sh
  python manage.py runserver
  ```

The Complete Project Report of our Document Verification System is Available [HERE](Major_Project.pdf)
