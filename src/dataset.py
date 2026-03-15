import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

class ArticleDataset(Dataset):
    """Custom PyTorch Dataset for E-Commerce Article Data"""
    def __init__(self, texts, labels):
        self.texts = torch.tensor(texts, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def prepare_data(file_path):
    print("Loading article descriptions...")
    df = pd.read_csv(file_path)
    
    # Assuming columns 'description' and 'category'
    texts = df['description'].fillna("")
    labels = df['category']

    print("Vectorizing text using TF-IDF...")
    # Cap features to 5000 to manage memory efficiently
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(texts).toarray() 

    print("Encoding categorical labels...")
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    return X, y, vectorizer, encoder
