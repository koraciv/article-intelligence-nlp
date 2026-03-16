# Article Intelligence: NLP Product Categorization

## 📌 Project Overview
E-commerce platforms receive thousands of new products daily, often with messy, unstructured text descriptions. Manually categorizing these articles is highly inefficient and prone to human error.

This project demonstrates a scalable **Natural Language Processing (NLP)** pipeline. It utilizes **PyTorch** to build a Deep Learning neural network that automatically reads unstructured product descriptions and classifies them into structured retail categories. 

**Business Value:** Automates master data management, enriches article data, and accelerates the time-to-market for new retail inventory.

## 🛠️ Tech Stack
* **Deep Learning Framework:** PyTorch (`torch.nn`, `DataLoader`)
* **Natural Language Processing:** scikit-learn (TF-IDF Vectorization)
* **Data Engineering:** Pandas, NumPy
* **Language:** Python 3.x

## 🏗️ Architecture & Workflow
1. **Text Preprocessing:** Ingests raw article descriptions and utilizes `TfidfVectorizer` to convert unstructured text into sparse numerical matrices.
2. **Custom Data Loaders:** Wraps the vectorized data in custom PyTorch `Dataset` and `DataLoader` classes for memory-efficient batch processing.
3. **Neural Architecture:** Implements a Multi-Layer Perceptron (Feedforward Neural Network) with ReLU activations and Dropout layers to prevent overfitting.
4. **Optimization:** Utilizes the Adam optimizer and Cross-Entropy Loss to iteratively train the model over multiple epochs.

## 📂 Project Structure
```text
├── data/                   # Data directory (Dataset excluded via .gitignore)
├── src/                    # Production Python scripts
│   ├── dataset.py          # Data ingestion, TF-IDF, and PyTorch Dataset class
│   ├── model.py            # PyTorch Neural Network architecture
│   ├── train.py            # Backpropagation and training loop
│   └── main.py             # Orchestration script
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignored files and directories
└── README.md               # Project documentation
