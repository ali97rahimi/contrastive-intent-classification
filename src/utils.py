import os
import torch
import logging

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir="checkpoints"):
    """
    Saves a checkpoint of the model during training.

    Args:
        model (nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): The current epoch.
        loss (float): The current loss.
        checkpoint_dir (str): Directory where to save the checkpoint.
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    logging.info(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Loads a saved checkpoint for the model and optimizer.

    Args:
        model (nn.Module): The model to load the weights into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.
        checkpoint_path (str): Path to the saved checkpoint.

    Returns:
        int: The epoch to resume training from.
        float: The last recorded loss.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded from {checkpoint_path}, epoch {epoch}, loss {loss}")
    logging.info(f"Checkpoint loaded from {checkpoint_path}, epoch {epoch}, loss {loss}")
    
    return epoch, loss


def set_logger(log_file="training.log"):
    """
    Sets up logging to both a file and the console.

    Args:
        log_file (str): File to log the training process.
    """
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s',
        filemode='w'
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)
    logging.info("Logging is set up.")


def count_parameters(model):
    """
    Counts the total number of trainable parameters in a model.

    Args:
        model (nn.Module): The model whose parameters need to be counted.

    Returns:
        int: The total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_training_progress(epoch, batch_idx, total_batches, loss):
    """
    Logs the progress of training after every batch.

    Args:
        epoch (int): The current epoch.
        batch_idx (int): The current batch index.
        total_batches (int): The total number of batches.
        loss (float): The current loss.
    """
    progress_message = f"Epoch [{epoch}], Batch [{batch_idx}/{total_batches}], Loss: {loss:.4f}"
    print(progress_message)
    logging.info(progress_message)

