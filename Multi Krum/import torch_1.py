import torch
from torch.utils.data import Dataset, DataLoader
import random
from sklearn.feature_extraction.text import CountVectorizer

# Synthetic dataset
class AlphanumericDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.categories = ["A1", "B2", "C3", "D4"]  # 4 classes
        self.samples = []
        self.labels = []
        
        # Generate random alphanumeric sequences
        for _ in range(num_samples):
            category = random.choice(self.categories)
            text = f"{category}-{random.randint(100,999)}"  # e.g., "A1-123"
            self.samples.append(text)
            self.labels.append(self.categories.index(category))
        
        # Convert text to bag-of-words
        self.vectorizer = CountVectorizer(token_pattern=r"\b\w+\b")
        self.X = self.vectorizer.fit_transform(self.samples).toarray()
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.LongTensor([self.labels[idx]])

# Initialize dataset
dataset = AlphanumericDataset()
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)