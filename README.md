
# LaBSE Contrastive Learning for Intent Classification

This repository implements a contrastive learning model using the LaBSE (Language-Agnostic BERT Sentence Embedding) model for intent classification. The model is trained to distinguish between pairs of sentences that have the same intent (positive pairs) and those that have different intents (negative pairs).

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Dataset Preparation](#dataset-preparation)
- [Model Checkpoints](#model-checkpoints)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

```
├── checkpoints/          # Directory to store model checkpoints
├── data/                 # Folder containing your dataset
├── labse_model.py        # Contains the LaBSE model for contrastive learning
├── dataset.py            # Functions for preparing datasets and creating pairs
├── contrastive_model.py  # Main script for training the contrastive learning model
├── utils.py              # Helper functions for logging, saving/loading checkpoints
├── requirements.txt      # Required dependencies
└── README.md             # Project documentation
```

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/contrastive-labse-intent-classification.git
    cd contrastive-labse-intent-classification
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training the Model

To train the contrastive learning model with your dataset, ensure your data is in a CSV format containing two columns: `Text` (the sentences) and `Intent` (the corresponding intent labels).

Run the training script:

```bash
python contrastive_model.py --data_path data/your_dataset.csv --epochs 10 --batch_size 64 --learning_rate 2e-5
```

Arguments:
- `--data_path`: Path to the CSV file containing the dataset.
- `--epochs`: Number of training epochs.
- `--batch_size`: Size of the batches for training.
- `--learning_rate`: Learning rate for the optimizer.

### Dataset Preparation

The dataset must be in a CSV format with two columns:
- **Text**: The input sentence.
- **Intent**: The intent label of the sentence.

You can modify the dataset preparation logic in `dataset.py` as needed.

## Model Checkpoints

During training, model checkpoints will be saved in the `checkpoints/` directory after each epoch. If training is interrupted or you want to resume from a specific point, you can use the `load_checkpoint` function in `utils.py` to restore the model and optimizer states.

```python
from utils import load_checkpoint

# Example usage:
epoch, loss = load_checkpoint(model, optimizer, 'checkpoints/model_epoch_5.pth')
```

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests for improvements or fixes.

To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add a feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
