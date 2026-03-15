import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, dataloader, epochs=5):
    print("Initializing training loop...")
    # CrossEntropyLoss is standard for multi-category classification
    criterion = nn.CrossEntropyLoss()
    # Adam is a highly efficient optimization algorithm
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for texts, labels in dataloader:
            # 1. Clear gradients
            optimizer.zero_grad()
            # 2. Forward pass (predict)
            outputs = model(texts)
            # 3. Calculate error
            loss = criterion(outputs, labels)
            # 4. Backward pass (learn)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(dataloader):.4f}")
    
    print("Training complete.")
    return model
