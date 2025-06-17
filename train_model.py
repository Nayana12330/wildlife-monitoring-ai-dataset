import os
from PIL import Image
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from feature_extraction import extract_features

def load_dataset(dataset_path='dataset'):
    features = []
    labels = []
    for class_name in sorted(os.listdir(dataset_path)):
        class_folder = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_folder):
            continue
        for img_file in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_file)
            try:
                img = Image.open(img_path).convert('RGB')
                feat = extract_features(img)
                features.append(feat)
                labels.append(class_name)
            except Exception as e:
                print(f"[ERROR] Failed to load image {img_path}: {e}")
    return np.array(features), np.array(labels)

def main():
    print("[INFO] Loading dataset...")
    X, y = load_dataset()
    
    if len(X) == 0:
        print("[ERROR] No images loaded. Please check your dataset directory structure.")
        return

    print(f"[INFO] Total samples: {len(y)}")
    print(f"[INFO] Classes found: {np.unique(y)}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    if len(np.unique(y_enc)) < 2:
        print("[WARNING] Only one class found. Training model anyway.")
        model = RandomForestClassifier()
        model.fit(X, y_enc)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        print(f"[INFO] Training completed. Accuracy on training data: {model.score(X_train, y_train):.2f}")

    # Save the model and label encoder
    joblib.dump(model, 'model.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    print("[INFO] Model and label encoder saved as 'model.pkl' and 'label_encoder.pkl'.")

if __name__ == "__main__":
    main()
