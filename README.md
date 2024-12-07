<div align="center">

# Linformer: Transformer with Linear Complexity

<div style="display: flex; justify-content: center; gap: 10px;">

[![Medium](https://img.shields.io/badge/Medium-%23000000.svg?logo=medium&logoColor=white)](#)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](#)
[![X](https://img.shields.io/badge/X-%23000000.svg?logo=X&logoColor=white)](#)
[![Kaggle](https://img.shields.io/badge/Kaggle-white?logo=kaggle)](#)

</div>

</div>

## 👋 Inroduction
So, Our Tradinal Transformer that Prposed in Attention is all you need paper it's solve the biggest problem of RNN (Recurrent Neural Network) is to gradient vanishing, but the Transformer requires too much amount of time and computation to train and evaluate. 

## 📖 About the Linformer 
The Linformer is proposed first time for the linear complexity attention mechanism. Paper: [Linformer: Self-Attention with Linear Complexity](https://arxiv.org/pdf/2006.04768)
### Why Linformer? 
Traditional transformers have quadratic time complexity which depends on the D_Model but the Linformer depends on the Sequence Length.
Also, Linformer allows you to project Key-value pair sharing and Headwise sharing which reduces the computation of time and memory.

# ⚒️ Basic requirements

[![Made with Python](https://img.shields.io/badge/Python->=3.10-orange?logo=python&logoColor=lightgray)](https://python.org "Go to Python homepage")
[![PyTorch](https://img.shields.io/badge/PyTorch->=2.4.1-red?logo=pytorch&logoColor=white)](https://pytorch.org "Go to PyTorch homepage")


## 🗂️ Repository Structure
```bash
linformer/
├── model/                    # Linformer implementation
    ├── __init__.py
│   ├── Linformer.py          # Core Linformer model
│   ├── embeddings.py         # Embeddings implementation
│   └── attention.py          # Attention implementation
├── requirements.txt          # Python dependencies
├── train.py                  # Training script
├── config.py                 # Set configurations for model training
├── dataset.py                # DataLoader stuff like that 
├── utils.py                  # some use full functions
└── README.md                 # Project documentation
```

## 🏋️‍♀️ Training The Model
Too run the script:
```bash
python train.py --epoch 10
```
> **Note:** 
> Arguments <br>
> --epoch 10 set the number of epoch to 10.<br>
> --workers 2 set the number of workers in dataloader to 2.<br>
> --datalen 500 set the dataset size to 500.<br>
> --srclang "en" set the src lang to "en".<br>
> --tgtlang "it" set the tgt lang to "it".<br>

## 🤝 Acknowledgements
- Liformer Paper: *[Linformer: Self-Attention with Linear Complexity](https://arxiv.org/pdf/2006.04768)*
- Special thanks to @hkproj
