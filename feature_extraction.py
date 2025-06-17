from skimage.feature import hog
from skimage.color import rgb2gray
import numpy as np
from PIL import Image

def extract_features(image):
    """
    Extract HOG features from a PIL image after resizing it to 128x128.
    """
    # Resize image for consistent feature size
    image = image.resize((128, 128))
    
    # Convert to numpy array
    image_np = np.array(image)
    
    # Convert RGB to grayscale if needed
    if len(image_np.shape) == 3:
        gray_image = rgb2gray(image_np)
    else:
        gray_image = image_np

    # Extract HOG features
    features, hog_image = hog(
        gray_image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=True,
        feature_vector=True
    )
    return features
