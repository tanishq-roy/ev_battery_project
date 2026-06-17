import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
from data import load_and_preprocess_data
from train_sklearn import build_pipeline

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.network(x)

def train_pytorch():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data()
    
    pipeline = build_pipeline(X_train)
    preprocessor = pipeline.named_steps['preprocessor']
    
    X_train_processed = preprocessor.fit_transform(X_train)
    if hasattr(X_train_processed, 'toarray'):
        X_train_processed = X_train_processed.toarray()
    X_train_processed = X_train_processed.astype('float32')
    
    X_test_processed = preprocessor.transform(X_test)
    if hasattr(X_test_processed, 'toarray'):
        X_test_processed = X_test_processed.toarray()
    X_test_processed = X_test_processed.astype('float32')
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(preprocessor, 'models/torch_preprocessor.pkl')
    
    train_dataset = TensorDataset(torch.tensor(X_train_processed), torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = MLP(input_dim=X_train_processed.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Training PyTorch MLP on {device}...")
    for epoch in range(10): 
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/10, Loss: {epoch_loss/len(train_loader):.4f}")
        
    torch.save(model.state_dict(), 'models/mlp_weights.pth')
    print("Model saved to models/mlp_weights.pth")

if __name__ == "__main__":
    train_pytorch()
