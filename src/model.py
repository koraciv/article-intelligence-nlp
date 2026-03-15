import torch.nn as nn

class ArticleClassifier(nn.Module):
    """Feedforward Neural Network for Text Classification"""
    def __init__(self, input_dim, num_classes):
        super(ArticleClassifier, self).__init__()
        # Hidden layer with 512 neurons
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        # Dropout prevents the model from memorizing the data (overfitting)
        self.dropout = nn.Dropout(0.3)
        # Output layer mapping to the number of categories
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
