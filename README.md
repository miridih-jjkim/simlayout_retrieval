# SimLayout Retrieval

This project provides tools for extracting embeddings and calculating similarity scores between images using a pre-trained model.

## Requirements

- Python 3.x
- PyTorch
- Other dependencies as specified in `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/miridih-jjkim/simlayout_retrieval.git
   cd simlayout_retrieval   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt   ```

## Usage

### Embedding Extraction

To extract embeddings from images, use the following command:

- **`/data/decoreted/test`**: Path to the directory containing the images for which embeddings need to be extracted.
- **`/data/ckpt_codetr/epoch_15.pth`**: Path to the pre-trained model checkpoint.
- **`--out-dir /data/sample_results`**: Directory where the extracted embeddings will be saved.

### Similarity Calculation

To calculate similarity scores between images, run:


This script will compute similarity scores for a set of target images against all other images in the specified directory and save the results in JSON format.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [DreamSim](https://github.com/dreamsim) for the pre-trained model and embedding extraction framework.