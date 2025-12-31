# Neural CDE for Multimodal Emotion Recognition

This repository contains the official implementation of the thesis "Neural Controlled Differential Equations for Multimodal Emotion Recognition".

## Project Structure

The project is organized as follows:

```
Major_Project/
├── src/                      # Core implementation modules
│   ├── classifier/           # Classification layers
│   ├── continuous_path/      # Spline interpolation logic
│   ├── evaluation/           # Metrics and evaluation tools
│   ├── fusion_transformer/   # Multimodal transformer architecture
│   ├── latent_discretization/# Neural CDE solver integration
│   ├── neural_cde/           # Neural CDE layer definitions
│   ├── normalization/        # Data normalization utilities
│   ├── preprocessing/        # Signal filtering and processing
│   ├── training/             # Training loops and dataset loaders
│   └── utils/                # Configuration and helper functions
├── scripts/                  # Standalone scripts
│   ├── cross_eval/           # Cross-dataset evaluation scripts & models
│   ├── verify_steps/         # Verification and visualization tools
│   └── ...                   # Utility scripts
├── docs/                     # Documentation
│   ├── latex/                # Thesis LaTeX source code
│   └── references/           # Research papers and reviews
├── data/                     # Dataset storage (excluded from git)
├── results/                  # Generated outputs and plots
├── demo_app/                 # Interactive Web Application
└── main.py                   # Main entry point
```

## How to Run

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Train the Model**:
    ```bash
    python main.py --mode train
    ```

3.  **Run the Demo App**:
    ```bash
    python demo_app/backend.py
    ```
    Then open `http://127.0.0.1:5000` in your browser.

## Thesis

The full thesis PDF is available in `docs/latex/neural_cde_thesis.pdf`.
