**TRANSFORMER-BASED SENTIMENT ANALYSIS PIPELINE**  
This repository contains an end-to-end text classification pipeline that fine-tunes a pre-trained BERT model using Hugging Face Transformers and PyTorch on the IMDb dataset (50,000+ reviews) for sentiment analysis. The pipeline achieved approximately 88% test accuracy.

**FEATURES**  
**Modular Data Processing:** Efficient tokenization and preprocessing using Hugging Face Datasets.  
**Robust Training:** Implements checkpointing and resume functionality for long-running training sessions.  
**Evaluation Framework:** Includes scripts for model evaluation and performance tracking.

**TECHNOLOGIES USED**  
**Programming Language:** Python  
**Deep Learning Framework:** PyTorch  
**NLP Library:** Hugging Face Transformers  
**Data Handling:** Hugging Face Datasets, scikit-learn  
**Other Tools:** Git, GitHub

**SETUP AND INSTALLATION**  
**Clone the Repository:**  
git clone https://github.com/your-username/text-classification-project.git  
cd text-classification-project  

**Create and Activate a Virtual Environment:**  
python -m venv env  
(Activate the virtual environment on Windows with: .\env\Scripts\Activate.ps1)

**Install Dependencies:**  
pip install torch torchvision torchaudio  
pip install transformers datasets scikit-learn accelerate evaluate

**USAGE**  
**Train the Model:**  
python src/train.py  
This script fine-tunes the BERT model on the IMDb dataset. Checkpoints are saved during training.

**Evaluate the Model:**  
python src/evaluate.py  
This script loads the trained model and evaluates its performance on the test set.

**PROJECT STRUCTURE**  
text-classification-project/  
├── env/                   (Virtual environment; not tracked)  
├── results/               (Model checkpoints; ignored)  
├── saved_model/           (Final saved model; ignored)  
├── src/                   (Source code: data processing, training, evaluation)  
│   ├── data_processing.py  
│   ├── evaluate.py  
│   ├── model.py  
│   └── train.py  
├── .gitignore             (Specifies files/directories to ignore)  
├── LICENSE  
└── README.md              (This file)

**LICENSE**  
This project is licensed under the MIT License. See the LICENSE file for details.
