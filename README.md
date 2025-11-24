# LLM Building Pipeline

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)  ![License](https://img.shields.io/badge/license-MIT-green)

**End-to-end LLM pipeline** from tokenization and data prep to Transformer pretraining, SFT, reward modeling, RLHF, and PPO. Includes architecture improvements from base Transformers to MoE and other modern designs. Fully modular, educational, and customizable for building advanced LLMs.

![Transformer Architecture](https://miro.medium.com/v2/resize:fit:1400/1*y5k-nW2i-PzNeEO7FNHl0A.png)

## 📋 Table of Contents

- [Features](#-features)
- [Directory Structure](#-directory-structure)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Training](#training)
  - [Generation](#generation)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

- **🧩 Modular Architecture**: Includes implementations for different GPT versions (`gptv1`, `gptv2`) allowing for easy experimentation and upgrades.
- **⚙️ Configurable Training**: All training parameters are easily adjustable via `config.yaml`.
- **🎓 Educational Focus**: Designed to be a learning resource for understanding the complete lifecycle of LLM development.
- **🚀 Modern Techniques**: Incorporates best practices in transformer architecture and training loops.
- **📈 Scalable Design**: Ready for expansion into SFT, RLHF, and PPO.

## 📂 Directory Structure

```
.
├── Pretraining-Phase/      # Core code for pretraining
│   ├── gptv1/              # GPT v1 implementation
│   ├── gptv2/              # GPT v2 implementation
│   ├── trainer.py          # Main training script
│   ├── generation.py       # Text generation script
│   └── config.yaml         # Configuration file
├── Data/                   # Dataset storage
└── README.md               # Project documentation
```

## 🛠 Installation

This project uses `uv` for fast dependency management, but supports standard `pip` as well.

### Prerequisites

- Python >= 3.11
- CUDA-capable GPU (recommended for training)

### Steps

1.  **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/llm-building-pipeline.git
    cd llm-building-pipeline
    ```

2.  **Install dependencies**

    Using `uv` (Recommended):
    ```bash
    uv sync
    ```

    Using `pip`:
    ```bash
    pip install .
    ```

## 🚀 Usage

### Training

To start pretraining the model, run the `trainer.py` script. You can adjust hyperparameters in `Pretraining-Phase/config.yaml`.

```bash
python Pretraining-Phase/trainer.py
```

**Configuration (`config.yaml`):**
```yaml
training:
  max_iters: 50000
  batch_size: 64
  learning_rate: 6e-4
```

### Generation

To generate text using a trained model (or checkpoint):

```bash
python Pretraining-Phase/generation.py
```

Ensure you have a trained model checkpoint (e.g., `runs/best_model.pt`) available.

## 🗺 Roadmap

- [x] **Pretraining Phase**: Basic Transformer implementation and training loop.
- [ ] **SFT (Supervised Fine-Tuning)**: Instruction tuning on datasets.
- [ ] **Reward Modeling**: Training a reward model for RLHF.
- [ ] **RLHF (Reinforcement Learning from Human Feedback)**: PPO implementation.
- [ ] **MoE (Mixture of Experts)**: Advanced architecture support.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/Feature`)
3. Commit your Changes (`git commit -m 'Add some Feature'`)
4. Push to the Branch (`git push origin feature/Feature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
