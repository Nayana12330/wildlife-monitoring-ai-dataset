import numpy as np

def generate_synthetic_dataset():
    np.random.seed(42)
    classes = ['deer', 'tiger', 'elephant', 'monkey', 'bird']
    num_samples_per_class = 100
    feature_length = 1000  # length of feature vector

    features = []
    labels = []

    for cls in classes:
        for _ in range(num_samples_per_class):
            feat = np.random.rand(feature_length)
            features.append(feat)
            labels.append(cls)

    features = np.array(features)
    labels = np.array(labels)

    np.save('features.npy', features)
    np.save('labels.npy', labels)

    print(f"Synthetic dataset created with {len(labels)} samples.")

if __name__ == "__main__":
    generate_synthetic_dataset()
