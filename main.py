import argparse
import sys
import os

# Ensure project root is in path
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
sys.path.append(os.path.join(root, 'src'))

from training.train_loso import train_loso_pipeline
from utils.config import PROJECT_ROOT

def main():
    parser = argparse.ArgumentParser(description="Multimodal Emotion Recognition using Neural CDEs")
    
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'preprocess', 'evaluate'],
                        help='Execution mode: train (LOSO), preprocess (run feature extraction), or evaluate.')
    parser.add_argument('--data_dir', type=str, default=os.path.join(PROJECT_ROOT, 'data', 'raw', 'WESAD'),
                        help='Path to data directory (specifically the raw folder containing S2, S3...)')
    
    args = parser.parse_args()
    
    print(f"--- Starting Agent: Multimodal Emotion Recognition ---")
    print(f"Mode: {args.mode}")
    print(f"Project Root: {PROJECT_ROOT}")
    
    if args.mode == 'train':
        print("Initializing Leave-One-Subject-Out (LOSO) Training...")
        try:
            train_loso_pipeline(data_root=args.data_dir)
        except Exception as e:
            print(f"Training Failed: {e}")
            import traceback
            traceback.print_exc()
            
    elif args.mode == 'preprocess':
        print("Preprocessing raw WESAD data...")
        # Placeholder for batch preprocessing script calls
        # process_all_subjects(args.data_dir)
        print("Preprocessing module not fully linked in main.py demo. Run preprocessing scripts individually.")
        
    elif args.mode == 'evaluate':
        print("Evaluation mode selected.")
        # load_and_evaluate(...)
        pass
        
if __name__ == '__main__':
    main()
