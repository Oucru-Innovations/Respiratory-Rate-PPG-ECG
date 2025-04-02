import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

def sanity_check_rr_distribution():
    try:
        targets = np.loadtxt("dataset/train_target.csv", delimiter=',')
        data = np.loadtxt("dataset/train_data.csv", delimiter=',')

        print("\n🧪 Sanity Check: RR Distribution Stats")
        print("--------------------------------------------------")
        print(f"Total samples        : {len(targets)}")
        print(f"Mean RR              : {np.mean(targets):.2f}")
        print(f"Std Dev              : {np.std(targets):.2f}")
        print(f"Min / Max            : {np.min(targets):.2f} / {np.max(targets):.2f}")
        print(f"Unique RR values     : {len(np.unique(targets))}")
        
        plt.figure(figsize=(8, 4))
        sns.histplot(targets, bins=30, kde=True)
        plt.title("Respiratory Rate Distribution")
        plt.xlabel("Respiratory Rate (bpm)")
        plt.ylabel("Count")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # 🧠 Try overfitting on 1 batch
        print("\n🎯 Overfit Check: Sampling 32 examples...")
        x_train, x_val, y_train, y_val = train_test_split(
            data[:32], targets[:32], test_size=0.5, random_state=42
        )

        print(f"Train sample shape: {x_train.shape}")
        print(f"Val   sample shape: {x_val.shape}")
        print(f"Sample targets train: {y_train.tolist()}")
        print(f"Sample targets val  : {y_val.tolist()}")
    except Exception as e:
        print(f"[ERROR] Sanity check failed: {e}")

# Call it
sanity_check_rr_distribution()
