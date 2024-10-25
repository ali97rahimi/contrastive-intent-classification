import pandas as pd
import random
from itertools import combinations
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

def create_train_val_split(file_path, test_size=0.2, random_seed=42):
    """
    Splits the dataset into training and validation sets.
    
    Args:
        file_path (str): Path to the CSV file containing the data.
        test_size (float): Proportion of the data to be used for validation.
        random_seed (int): Random seed for reproducibility.
    
    Returns:
        train_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.
    """
    df = pd.read_csv(file_path)
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_seed)
    return train_df, val_df


def create_pairs_from_csv(df, random_seed=42):
    """
    Creates positive and negative pairs from a dataframe.
    
    Args:
        df (pd.DataFrame): Dataframe containing 'Text' and 'Intent' columns.
        random_seed (int): Random seed for reproducibility.
    
    Returns:
        pairs (list): List of text pairs with labels (1 for positive pairs, 0 for negative pairs).
    """
    intent_groups = df.groupby('Intent')['Text'].apply(list).to_dict()

    positive_pairs = []
    negative_pairs = []

    # Create positive pairs (same intent)
    for intent, texts in intent_groups.items():
        if len(texts) > 1:
            for pair in combinations(texts, 2):
                positive_pairs.append((pair[0], pair[1], 1))

    # Create negative pairs (different intents)
    intents = list(intent_groups.keys())
    for i in range(len(intents)):
        for j in range(i + 1, len(intents)):
            texts1 = intent_groups[intents[i]]
            texts2 = intent_groups[intents[j]]
            for text1 in texts1:
                for text2 in texts2:
                    negative_pairs.append((text1, text2, 0))

    # Balance positive and negative pairs
    num_positive = len(positive_pairs)
    num_negative = len(negative_pairs)

    if num_negative > num_positive:
        random.seed(random_seed)
        negative_pairs = random.sample(negative_pairs, num_positive)
    elif num_positive > num_negative:
        random.seed(random_seed)
        positive_pairs = random.sample(positive_pairs, num_negative)

    pairs = positive_pairs + negative_pairs
    random.shuffle(pairs)

    return pairs


class PairDataset(Dataset):
    """
    Dataset class for handling text pairs and labels.
    
    Args:
        pairs (list): List of text pairs and labels.
    
    Returns:
        Dataset object for PyTorch.
    """
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        text1, text2, label = self.pairs[idx]
        return text1, text2, torch.tensor(label, dtype=torch.float32)
