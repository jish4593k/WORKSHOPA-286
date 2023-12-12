# train_model.py
import torch.optim as optim
from torch.utils.data import DataLoader
from .ml_model import MovieRatingPredictor
from .datasets import MovieDataset  # Implement a custom dataset

def train_model():
    model = MovieRatingPredictor(input_size=1, output_size=1)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Load your custom dataset using DataLoader

    for epoch in range(100):
        for data in DataLoader(your_movie_dataset, batch_size=1, shuffle=True):
            inputs, targets = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # Save the trained model
    torch.save(model.state_dict(), 'path/to/saved_model.pth')
