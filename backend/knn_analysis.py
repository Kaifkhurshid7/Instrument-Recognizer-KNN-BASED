# knn_analysis.py
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import itertools

# --- Configuration ---
DATABASE_FILE = 'reference_database.pkl'
BEST_K_VALUE = 7 # 
FEATURE_LENGTH = 10

def load_data():
    """Loads features (X) and labels (y) from the database file."""
    try:
        with open(DATABASE_FILE, 'rb') as f:
            database = pickle.load(f)
    except FileNotFoundError:
        print(f"ERROR: {DATABASE_FILE} not found. Run build_database.py first!")
        return None, None, None, None

    X, y, class_names = [], [], []
    
    
    class_names = sorted(list(database.keys()))
    class_map = {name: i for i, name in enumerate(class_names)}

    for instrument_name, data in database.items():
        class_id = class_map[instrument_name]
        for fingerprint in data['fingerprints']:
            
            if len(fingerprint) == FEATURE_LENGTH:
                X.append(fingerprint)
                y.append(class_id)
            
    return np.array(X), np.array(y), class_names, class_map

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """Prints and plots the confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False, 
                xticklabels=classes, yticklabels=classes, linewidths=.5, linecolor='gray')
    
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Graph saved: confusion_matrix.png")

    
def plot_k_vs_accuracy(X_train_scaled, y_train, X_test_scaled, y_test):
    """Plots accuracy vs. K value to justify the chosen K."""
    neighbors = np.arange(1, 20, 2) # Test odd K values from 1 to 19
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))

    for i, k in enumerate(neighbors):
        knn = KNeighborsClassifier(n_neighbors=k, metric='cosine', weights='distance')
        knn.fit(X_train_scaled, y_train)
        
        train_accuracy[i] = knn.score(X_train_scaled, y_train)
        test_accuracy[i] = knn.score(X_test_scaled, y_test)

    plt.figure(figsize=(10, 6))
    plt.title('k-NN: Accuracy vs. Number of Neighbors (K)')
    plt.plot(neighbors, test_accuracy, label='Testing Accuracy', marker='o', color='#61dafb')
    plt.plot(neighbors, train_accuracy, label='Training Accuracy', marker='o', color='#90caf9')
    plt.legend()
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('k_vs_accuracy.png')
    print("Graph saved: k_vs_accuracy.png")
    # Trigger image generation for the poster
    
def run_analysis():
    X, y, class_names, class_map = load_data()

    if X is None:
        return

    # 1. Split Data (70% Train, 30% Test for evaluation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # 2. Scale Data (CRITICAL for Distance-based models like k-NN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n--- Model Evaluation: k-NN (K=7, Cosine Distance) ---")

    # 3. Plot K vs. Accuracy (Justification for best K)
    plot_k_vs_accuracy(X_train_scaled, y_train, X_test_scaled, y_test)

    # 4. Train the Final Model for Evaluation (K=7)
    knn_final = KNeighborsClassifier(n_neighbors=BEST_K_VALUE, metric='cosine', weights='distance')
    knn_final.fit(X_train_scaled, y_train)

    # 5. Predict and Evaluate
    y_pred = knn_final.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nFinal k-NN Test Accuracy (K={BEST_K_VALUE}): {accuracy * 100:.2f}%")
    
    # 6. Generate and Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, class_names, title=f'k-NN Confusion Matrix (K={BEST_K_VALUE}, Accuracy={accuracy*100:.1f}%)')
    
if __name__ == '__main__':
    run_analysis()