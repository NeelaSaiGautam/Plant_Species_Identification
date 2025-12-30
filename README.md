# Plant_Species_Identification
🌱 Plant Species Identification using CNN & VGG16

📌 Project Overview

Plant Species Identification is a machine learning–based application that identifies plant species from an image and provides detailed information about the plant. The system uses Convolutional Neural Networks (CNN) with a VGG16 pre-trained model to classify plant images and display useful details such as common name, scientific name, water requirements, environmental conditions, and uses.

This application is helpful for gardening enthusiasts, botany students, and medical students who want quick and reliable plant information.

🎯 Problem Statement

Many people are interested in growing plants but lack knowledge about:

Medicinal vs harmful plants

Proper care requirements

Uses and environmental conditions

Existing applications mainly identify plant names but do not provide complete information about the plant.

💡 Proposed Solution

User uploads a plant image

Image is processed using CNN + VGG16

The model predicts the plant species

Related plant information is fetched from a CSV dataset

Results are displayed through a Streamlit web interface

🛠️ Tech Stack

Programming Language: Python 3.10

Machine Learning: CNN, VGG16

Libraries: TensorFlow, Keras, NumPy, Pandas, Matplotlib

Frontend: Streamlit

IDE: PyCharm

Dataset Source: Kaggle (99 plant species)

🧠 System Architecture

User uploads plant image

Image preprocessing (resize, normalization)

Feature extraction using VGG16

Classification using trained CNN model

Plant details fetched from CSV file

Results displayed on web UI

✨ Features

Image-based plant identification

Uses pre-trained VGG16 for better accuracy

Displays:

Common Name

Scientific Name

Water Consumption

Environmental Conditions

Plant Uses

User-friendly web interface

Supports 99 plant species

⚙️ Software Requirements

Python 3.10

PyCharm Community Edition

Streamlit

Anaconda Navigator

Required Python packages:

numpy
pandas
matplotlib
tensorflow
keras
streamlit
pillow

💻 Hardware Requirements

Processor: 64-bit, Quad Core (2.5 GHz minimum)

RAM: 6 GB

Storage: 64 GB

OS: Windows 10

🚀 How to Run the Project

Clone the repository

git clone https://github.com/your-username/plant-species-identification.git


Install required packages

pip install -r requirements.txt


Run the Streamlit app

streamlit run app.py


Upload a plant image and click Predict

📊 Results

Successfully identifies plant species from images

Displays accurate plant information

Tested with multiple plant images

Achieved good accuracy using CNN + VGG16

🧪 Testing

Multiple test cases executed with different plant images

All test cases passed successfully

Verified both prediction accuracy and UI output

🔮 Future Enhancements

Increase dataset size to improve accuracy

Support more plant species

Deploy as a mobile application

Improve UI/UX

Add multilingual support
