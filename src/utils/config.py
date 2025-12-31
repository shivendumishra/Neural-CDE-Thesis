import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

# Dataset Config
FS_ECG = 700
FS_EDA = 4
FS_ACC = 32

# Preprocessing Constants
ECG_LOW_CUT = 5.0
ECG_HIGH_CUT = 15.0
EDA_LOW_PASS_CUT = 1.0

# Input Channels (Raw Feature Dims)
INPUT_CHANNELS_ECG = 1  # RR Interval only
INPUT_CHANNELS_EDA = 2  # Phasic, Tonic
INPUT_CHANNELS_ACC = 4  # Mean, Var, Energy, Entropy

# Model Config
LATENT_DIM = 16
HIDDEN_DIM = 32
LATENT_SAMPLING_RATE = 2.0  # Hz (sampling z(t))

# Transformer Config
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 2
TRANSFORMER_DIM = 32  # Must match combined latent dims or be projected

# Training Config
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 50
SEED = 42

# Evaluation
TARGET_CLASSES = [1, 2, 3] # 1: Baseline, 2: Stress, 3: Amusement
CLASS_NAMES = ['Baseline', 'Stress', 'Amusement']
