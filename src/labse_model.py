import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class LaBSEContrastive(nn.Module):
    """
    LaBSE model class for contrastive learning. This model uses LaBSE embeddings
    to calculate the similarity between pairs of sentences.
    
    Args:
        nn.Module: Base class for all neural network modules.
    """
    def __init__(self):
        super(LaBSEContrastive, self).__init__()
        # Load pre-trained LaBSE model and tokenizer from HuggingFace
        self.labse = AutoModel.from_pretrained('sentence-transformers/LaBSE')
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/LaBSE')

    def forward(self, texts):
        """
        Forward pass for the LaBSE model. Tokenizes and passes the input text to the model
        and returns the output embeddings (pooled output).
        
        Args:
            texts (list of str): List of input sentences.
        
        Returns:
            torch.Tensor: Pooled embeddings for the input sentences.
        """
        # Tokenize the input texts
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
        # Move the tokenized inputs to the same device as the model parameters
        inputs = {key: value.to(next(self.labse.parameters()).device) for key, value in inputs.items()}
        # Get the embeddings from LaBSE model
        outputs = self.labse(**inputs)
        # Return the pooled output (embedding representation for each sentence)
        return outputs.pooler_output
