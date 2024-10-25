import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from dataset import PairDataset, create_train_val_split, create_pairs_from_csv
from utils import save_checkpoint
import logging

# Logging configuration
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')


class LaBSEContrastive(nn.Module):
    def __init__(self):
        super(LaBSEContrastive, self).__init__()
        # Load pre-trained LaBSE model and tokenizer from HuggingFace
        self.labse = AutoModel.from_pretrained('sentence-transformers/LaBSE')
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/LaBSE')

    def forward(self, texts):
        # Tokenize input text data
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
        inputs = {key: value.to(next(self.labse.parameters()).device) for key, value in inputs.items()}
        # Pass the tokenized input through the LaBSE model
        outputs = self.labse(**inputs)
        return outputs.pooler_output


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embedding1, embedding2, label):
        # Calculate pairwise distance between embeddings
        distance = torch.nn.functional.pairwise_distance(embedding1, embedding2)
        # Compute contrastive loss based on the distance and label (positive/negative pairs)
        loss = 0.5 * (label * torch.pow(distance, 2) +
                      (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return loss.mean()


def train_contrastive_model(file_path, num_epochs=5, batch_size=16, lr=2e-7, checkpoint_dir="checkpoints"):
    # Create train and validation splits
    train_df, val_df = create_train_val_split(file_path)
    train_pairs = create_pairs_from_csv(train_df)
    val_pairs = create_pairs_from_csv(val_df)

    # Create dataset and dataloader
    train_dataset = PairDataset(train_pairs)
    val_dataset = PairDataset(val_pairs)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = LaBSEContrastive()
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in progress_bar:
            text1, text2, labels = batch
            labels = labels.to(device)

            # Forward pass to get embeddings for both texts
            embeddings1 = model(list(text1))
            embeddings2 = model(list(text2))

            # Zero gradients, perform backpropagation, and update weights
            optimizer.zero_grad()
            loss = criterion(embeddings1, embeddings2, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Update progress bar and log loss
            if batch_idx % 10 == 0:
                progress_bar.set_postfix(loss=loss.item())
                logging.info(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} completed. Average Training Loss: {avg_train_loss:.4f}")
        logging.info(f"Epoch {epoch+1}/{num_epochs} completed. Average Training Loss: {avg_train_loss:.4f}")

        # Validation step
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                text1, text2, labels = batch
                labels = labels.to(device)

                embeddings1 = model(list(text1))
                embeddings2 = model(list(text2))

                val_loss = criterion(embeddings1, embeddings2, labels)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} completed. Average Validation Loss: {avg_val_loss:.4f}")
        logging.info(f"Epoch {epoch+1}/{num_epochs} completed. Average Validation Loss: {avg_val_loss:.4f}")

        # Save checkpoint
        save_checkpoint(model, optimizer, epoch+1, avg_val_loss, checkpoint_dir)

    # Save the final model
    final_model_path = os.path.join(checkpoint_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved: {final_model_path}")
    logging.info(f"Final model saved: {final_model_path}")


if __name__ == "__main__":
    # Example usage
    train_contrastive_model('data/contrastiveData.csv', batch_size=128)
