"""
Task II - Sentiment Analysis using BERT
This script implements sentiment analysis on Amazon reviews using BERT
Author: William Brandon
Date: 15 April 2025
"""

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and prepare data
print("Loading data...")
df = pd.read_excel("amazon_reviews-1.xlsx")
df = df.dropna(subset=['reviewText', 'overall'])
df['reviewText'] = df['reviewText'].astype(str)

# Convert ratings to labels (0-4)
df['label'] = df['overall'] - 1

# Split data
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['reviewText'].values,
    df['label'].values,
    test_size=0.2,
    random_state=42
)

print(f"Training samples: {len(train_texts)}, Test samples: {len(test_texts)}")

# Create dataset class
class AmazonDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Initialize tokenizer and model
print("Loading BERT model and tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=5
)
model.to(device)

# Create datasets and dataloaders
train_dataset = AmazonDataset(train_texts, train_labels, tokenizer)
test_dataset = AmazonDataset(test_texts, test_labels, tokenizer)

batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# Training parameters
epochs = 3
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Training loop
print("Starting training...")
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{epochs} - Average training loss: {avg_train_loss:.4f}")

    # Evaluation
    model.eval()
    predictions = []
    actual_labels = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs.logits, dim=1)

            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())

    accuracy = accuracy_score(actual_labels, predictions)
    print(f"Epoch {epoch+1}/{epochs} - Test accuracy: {accuracy:.4f}")

# Final evaluation
print("\nFinal Evaluation:")
print(classification_report(actual_labels, predictions, target_names=['1 star', '2 stars', '3 stars', '4 stars', '5 stars']))

# Save the model
print("Saving model...")
model.save_pretrained('./bert_sentiment_model')
tokenizer.save_pretrained('./bert_sentiment_model')
print("Model saved successfully!")