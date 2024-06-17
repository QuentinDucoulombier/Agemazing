# AgeMazing

## Introduction

The AgeMazing project aims to develop a face detection system capable of accurately estimating the age of individuals from both images and real-time video streams. This system involves designing and implementing an algorithm that can effectively identify and localize faces within static images and dynamic video frames, followed by an age estimation model that analyzes facial features to predict the age of the person.

## Project Structure

- **`agemazing_detector.py`**: Script for real-time face detection and age prediction.
- **`agemazing_model.ipynb`**: Notebook containing the model implementation and training.
- **`model-dataset`**: Directory containing the pre-trained h5 models.
- **`face-detection-dataset`**: Directory containing the face detection dataset.
- **`AgeMazing-FinalReport.pdf`**: Detailed report of the project.

## Installation

1. Clone the repository.

   ```bash
   git clone https://github.com/QuentinDucoulombier/Agemazing.git
   ```

2. Navigate to the project directory.

   ```bash
   cd AgeMazing
   ```

3. **Make sure you have Python 3.7.10 installed.**

4. Install the required libraries.

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Download the pre-trained models (H5 files) from Google Drive.
   [H5 Files on Google Drive](https://drive.google.com/drive/folders/1ueZw08YBmm0tp6H1Qd5jPoHWJuRImIo3?usp=sharing)
2. Place the downloaded H5 files into the `model-dataset` directory.
3. Run the real-time face detection and age prediction script.

   ```bash
   python agemazing_detector.py
   ```

4. Follow the prompts to select the pre-trained model you want to use for age prediction.

## Model on Kaggle

You can find the notebook and model implementation on Kaggle:
[AgeMazing on Kaggle](https://www.kaggle.com/code/quentinducoulombier/agemazing-model)

## Project Report

For detailed information about the project, refer to the [AgeMazing Final Report](./assets/AgeMazing-FinalReport.pdf).

## Support

If you encounter any issues, please open an issue on GitHub: [AgeMazing Issues](https://github.com/QuentinDucoulombier/Agemazing/issues) or send an email to ducoulombi@cy-tech.fr.
