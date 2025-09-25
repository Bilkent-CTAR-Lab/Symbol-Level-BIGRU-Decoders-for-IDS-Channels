# Symbol-Level-BIGRU-Decoders-for-IDS-Channels

Symbol-Level Training for Insertion/Deletion Channels

This repository provides TensorFlow/Keras implementations for training and evaluating Bi-directional RNN-based models (LSTM/GRU) on insertion/deletion communication channels.  
The framework simulates noisy channels with insertions, deletions, and substitutions, and trains end-to-end neural architectures for error correction.  

---

## 📂 Project Structure

.
├── README.md # Project description and usage
├── requirements.txt # Python dependencies
├── LICENSE # License file
├── .gitignore # Files to ignore in git

├── src/ # Core source code
│ ├── main.py # Main training script
│ ├── test.py # Evaluation script
│ ├── functions.py # Utilities (train_step, saving, etc.)
│ ├── models.py # Model architectures (Bi-LSTM, GRU, etc.)
│ ├── datasets.py # Dataset creation functions
│ ├── test_functions.py # Testing utilities
│ ├── channel_models.py # Channel simulation
│ ├── decoders.py # Decoding functions
│ ├── demappers.py # Demapping functions
│ ├── decoder_demapper.py # Decoder + demapper utilities
│ └── marker_related.py # Marker code utilities

├── data/
│ └── Matrices/ # Parity-check matrices (.mat files)

├── results/ # Auto-generated results (ignored in git)
│ └── training.log

├── figures/ # Plots, diagrams, visualizations

└── notebooks/
└── result_print.ipynb # Example analysis notebook


---

## ⚙️ Installation

Clone the repository:

git clone https://github.com/your-username/your-repo-name.git

cd your-repo-name


Install dependencies:
pip install -r requirements.txt


---

## 🚀 Usage

### 🏋️ Training

Run the training script:


python src/main.py

Example with custom parameters:
python src/main.py --epochs 100 --batch-size 32 --lr 0.001 --H 2


### 📝 Arguments

Some key options:

| Argument       | Default | Description                                  |
|----------------|---------|----------------------------------------------|
| `--symbol-bit` | 6       | Number of bits per symbol                    |
| `--Nc`         | 30      | Number of bits to insert markers             |
| `--epochs`     | 300     | Number of training epochs                    |
| `--batch-size` | 16      | Training batch size                          |
| `--lr`         | 9e-4    | Learning rate                                |
| `--H`          | 3       | Parity-check matrix size (1=Small, 4=XLarge) |

---

## 📊 Results

- Training results are stored in:

results/<experiment_name>/train_results.npy

- Test results are saved under:

results/<experiment_name>/test_results/

- Model weights are stored in `.keras` or `.h5` format for reuse.

---

## 🧪 Example

Train with symbol size = 6, medium matrix, 50 epochs:

python src/main.py --symbol-bit 6 --H 2 --epochs 50


---

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- TensorFlow/Keras for the deep learning framework  
- (Optional) Add citation if this work is part of your thesis or publications  
