# Spiking Neural Network (SNN) for MNIST Digit Classification

This project implements a Spiking Neural Network (SNN) using Leaky Integrate-and-Fire (LIF) neurons to classify handwritten digits from the MNIST dataset. It uses PyTorch for model building and training.

## 📁 Project Structure

IMAGE/
├── DATASET/ # (Optional) Directory for dataset or processed images
├── about.txt # Miscellaneous description (not required for training)
├── ds_visual.py # Script to visualize dataset samples
├── parameters.pt # Trained model weights
├── run_digit.py # Inference script
├── train_digit.py # Script to train the model

markdown

## 🚀 Features

- Fully functional SNN architecture built with PyTorch
- Basic dataset visualization utility
- Save/load trained weights
- Designed to work with MNIST-style grayscale digit images (28x28)

## 🔧 Requirements

Install all required packages using:

```bash
pip install -r requirements.txt
```
🧠 Training the Model
To train the model from scratch:

```bash
python train_digit.py
```
This will save the trained weights to parameters.pt.

🔍 Running Inference
To run inference using a pre-trained model:

```bash
python run_digit.py
```
Make sure parameters.pt exists or update the path to your weights.

📊 Visualizing the Data
You can visualize samples using:

```bash

python ds_visual.py
```
📌 Notes
Make sure the dataset is in the right format (e.g., MNIST-like 28x28 grayscale images).

Update paths and file handling inside scripts based on your local file structure if necessary.

Made with ❤️ by Adhish J S

yaml

---

### 📦 `requirements.txt`

```text
torch
torchvision
matplotlib
numpy
You can generate this yourself by running:
```
```bash

pip freeze > requirements.txt
```
But for now, the above is sufficient unless you use additional libraries.
