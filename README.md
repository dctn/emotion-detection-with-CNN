# Emotion Detection with CNN

A deep-learning based **facial emotion recognition** system built using
a **Convolutional Neural Network (CNN)**. This project classifies human
facial expressions into categories such as **Happy, Sad, Angry, Neutral,
Surprise, Fear, Disgust**.

The trained model used in this project is uploaded on Kaggle:\
https://www.kaggle.com/models/strangerias/emotion-detection

## 🚀 Features

-   Detects emotions from facial images using CNN
-   Pretrained model available for direct usage
-   Clean modular code (training + prediction)
-   Works on CPU or GPU
-   Can be integrated into SaaS apps, websites, and mobile apps

## 🧠 Model Information

-   Format: `.pt`
-   Framework: Pytorch
-   Trained on: FER-2013 + custom dataset
-   Accuracy: \~60%
-   Supported emotions: Angry, Disgust, Fear, Happy, Sad, Surprise,
    Neutral

## 📂 Project Structure

    emotion-detectiion-with-CNN/
    │── model/
    │    └── emotion_model.pt
    │── dataset/
    │── notebooks/
    │    └── training.ipynb
    │── src/
    │    ├── train.py
    │    ├── predict.py
    │    ├── utils.py
    │── requirements.txt
    │── README.md

## 🛠 Installation

### Clone the repo

``` bash
git clone https://github.com/dctn/emotion-detectiion-with-CNN.git
cd emotion-detectiion-with-CNN
```

### Install dependencies

``` bash
pip install -r requirements.txt
```

### Download the trained model

Download from Kaggle and place inside `model/alex_model_v6_data_arugmention.pt`.

## 📝 License

MIT License.
