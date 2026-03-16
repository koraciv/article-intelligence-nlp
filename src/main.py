import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dataset import prepare_data, ArticleDataset
from model import ArticleClassifier
from train import train_model

def run_pipeline():
    data_path = "../data/ecommerce_data.csv"
    
    if not os.path.exists(data_path):
        print(f"Error: Place 'ecommerce_data.csv' in the data/ directory.")
        return

    # 1. Prepare Data
    X, y, vectorizer, encoder = prepare_data(data_path)
    input_dim = X.shape[1]
    num_classes = len(encoder.classes_)

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Create PyTorch Datasets and DataLoaders
    train_dataset = ArticleDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 4. Initialize and Train Model
    print(f"Initializing Neural Network (Input: {input_dim}, Classes: {num_classes})")
    model = ArticleClassifier(input_dim, num_classes)
    
    trained_model = train_model(model, train_loader, epochs=5)

if __name__ == "__main__":
    run_pipeline()
