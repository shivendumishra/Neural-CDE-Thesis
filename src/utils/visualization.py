import matplotlib.pyplot as plt
import numpy as np
import os

def plot_signal(time, signal, label, title, save_path=None):
    """
    Plots a physiological signal.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(time, signal, label=label)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_training_history(history, save_dir):
    """
    Plots training and validation loss/accuracy.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Loss
    plt.figure()
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_history.png'))
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'accuracy_history.png'))
    plt.close()
