import json
import matplotlib.pyplot as plt
import os
import glob

def parse_log_file(log_path):
    """Reads a JSON-formatted log file and extracts metrics."""
    epochs_pre = []
    losses_ita = []
    losses_itm = []
    
    epochs_fine = []
    losses_cls = []
    accuracies = []
    
    print(f"Parsing log file: {log_path}")
    
    with open(log_path, 'r') as f:
        for line in f:
            try:
                log = json.loads(line)
                
                # Pre-training Data (Look for 'train_loss_ita')
                if 'train_loss_ita' in log: 
                    epochs_pre.append(log['epoch'])
                    losses_ita.append(log['train_loss_ita'])
                    losses_itm.append(log['train_loss_itm'])
                
                # Fine-tuning Data Train (Look for 'train_loss_cls')
                elif 'train_loss_cls' in log: 
                    epochs_fine.append(log['epoch'])
                    losses_cls.append(log['train_loss_cls'])
                
                # Fine-tuning Data Test/Val (Look for 'test_acc')
                elif 'test_acc' in log: 
                     accuracies.append(log['test_acc'])
                     
            except json.JSONDecodeError:
                continue
    
    # Organize data into a dictionary
    data = {}
    if epochs_pre:
        data['pre'] = {'epochs': epochs_pre, 'ita': losses_ita, 'itm': losses_itm}
        print(f"  -> Found {len(epochs_pre)} pre-training epochs.")
    if epochs_fine:
        # Align accuracies length with epochs if needed
        if len(accuracies) > len(epochs_fine):
            accuracies = accuracies[:len(epochs_fine)]
        
        data['fine'] = {'epochs': epochs_fine, 'cls': losses_cls, 'acc': accuracies}
        print(f"  -> Found {len(epochs_fine)} fine-tuning epochs.")
        
    return data

def plot_pretrain_losses(data, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(data['epochs'], data['ita'], label='ITA Loss (Alignment)', color='blue')
    plt.plot(data['epochs'], data['itm'], label='ITM Loss (Matching)', color='orange')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Pre-training Losses (Task 1: Learn 3D Features)')
    plt.legend()
    plt.grid(True)
    
    out_path = os.path.join(save_dir, 'pretrain_losses.png')
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")
    plt.close()

def plot_finetune_losses(data, save_dir):
    plt.figure(figsize=(10, 5))
    
    # Plot Loss on left axis
    fig, ax1 = plt.subplots(figsize=(10,6))
    
    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Classification Loss', color=color)
    ax1.plot(data['epochs'], data['cls'], color=color, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Plot Accuracy on right axis (if available)
    if data['acc']:
        ax2 = ax1.twinx()  
        color = 'tab:green'
        ax2.set_ylabel('Test Accuracy', color=color)
        # Generate x-axis for accuracy (assuming 1 val per epoch)
        val_epochs = data['epochs'][:len(data['acc'])]
        ax2.plot(val_epochs, data['acc'], color=color, linestyle='--', label='Test Accuracy')
        ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Fine-tuning Performance (Task 2: ADNI Classification)')
    fig.tight_layout()
    
    out_path = os.path.join(save_dir, 'finetune_performance.png')
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")
    plt.close()

if __name__ == "__main__":
    # Find all log.txt files recursively in the current directory
    # This ensures we find the log whether it's in output/Pretrain3D or output/train_ADNI
    log_files = glob.glob('**/log.txt', recursive=True)
    
    if not log_files:
        print("No log.txt files found! Make sure you have run the training scripts.")
    
    for log_path in log_files:
        # Determine where to save the plot (same dir as the log file)
        save_dir = os.path.dirname(log_path)
        
        results = parse_log_file(log_path)
        
        if 'pre' in results:
            plot_pretrain_losses(results['pre'], save_dir)
        
        if 'fine' in results:
            plot_finetune_losses(results['fine'], save_dir)
            
    print("\nProcessing Complete.")