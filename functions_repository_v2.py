# Imports

import os
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import sklearn.model_selection
import random
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split


# --------------------------------------------------------------------
# Data Preprocessing
# Set constants for data loading and preprocessing

DATA_PATH = "/content/rare_species"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE
CLASS_THRESHOLD = 10
MIN_RESOLUTION = (100, 100)

# Define a seed for replications
SEED = 22
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Define the mean and the standard deviation expeted to our models (RGB patterns)
IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
IMAGENET_STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

def image_validator(path, min_resolution = MIN_RESOLUTION):
    """
    Checks whether an image exists and meets a minimum resolution.

    Requires:
        - path (str): Full path to the image file.
        - min_resolution (tuple): Minimum resolution as (width, height).

    Ensures:
        - Returns True if the image exists, is readable, and meets the resolution requirement.
        - Returns False otherwise (missing, unreadable, or low-resolution image).
    """
    try:
        with Image.open(path) as img:
            return img.size[0] >= min_resolution[0] and img.size[1] >= min_resolution[1]
    except:
        return False


def load_and_validate_data(data_path):
    """
    Loads metadata and filters out invalid or low-resolution image entries.

    Requires:
        - data_path (str): Path to the folder containing 'metadata.csv' and image files.

    Ensures:
        - Returns a cleaned DataFrame with only valid image entries.
        - Adds a 'full_path' column with the absolute path to each image.
    """
    metadata = pd.read_csv(os.path.join(data_path, 'metadata.csv'))
    metadata['full_path'] = metadata['file_path'].apply(lambda x: os.path.join(DATA_PATH, x))
    metadata['valid'] = metadata['full_path'].apply(lambda x: os.path.exists(x) and image_validator(x))
    return metadata[metadata['valid']].drop(columns=['valid'])


def analyze_class_distribution(metadata):
    """
    Plots the distribution of image counts per family and prints rare classes.

    Requires:
        - metadata (pd.DataFrame): Must include a 'family' column with image labels.
        - CLASS_THRESHOLD (int): Global variable defining what counts as a 'rare' class.

    Ensures:
        - Displays a histogram showing how many images each family has.
        - Prints the number of families with fewer than CLASS_THRESHOLD images.
    """
    family_counts = metadata['family'].value_counts()

    plt.figure(figsize=(12, 6))
    sns.histplot(family_counts, bins=50)
    plt.title("Images per Family Distribution")
    plt.xlabel("Number of Images")
    plt.ylabel("Number of Families")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    rare_classes = family_counts[family_counts < CLASS_THRESHOLD]
    print(f"Rare families (<{CLASS_THRESHOLD} images): {len(rare_classes)}")


def create_datasets(metadata):
    """
    Prepares train and validation datasets from metadata, applying preprocessing and batching.

    Requires:
        - metadata (DataFrame): Cleaned metadata including 'full_path', 'family', and 'phylum' columns.

    Ensures:
        - Stratified train/val split on 'family'.
        - Normalized and resized image tensors.
        - Encoded phylum and family labels using StringLookup.
        - Returns train and validation datasets ready for training.

    Returns:
        - train_ds (tf.data.Dataset): Preprocessed and batched training set.
        - val_ds (tf.data.Dataset): Preprocessed and batched validation set.
        - family_lookup (StringLookup): Mapping for family class labels.
    """

    file_paths = metadata['full_path'].values
    family_labels = metadata['family'].values
    phylum_labels = metadata['phylum'].values

    # Stratified split on family labels
    train_df, val_df = train_test_split(
        metadata,
        test_size=0.2,
        stratify=family_labels,
        random_state=SEED
    )

    # Lookup tables for string labels
    family_lookup = tf.keras.layers.StringLookup()
    family_lookup.adapt(family_labels)

    phylum_lookup = tf.keras.layers.StringLookup()
    phylum_lookup.adapt(phylum_labels)

    # Image preprocessing function
    def preprocess(file_path, family, phylum):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = img / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = tf.ensure_shape(img, [224, 224, 3])
        phylum_idx = phylum_lookup(phylum)
        family_idx = family_lookup(family)
        return (img, phylum_idx), family_idx

    # Optional data augmentation
    def augment(image_phylum, label):
        image, phylum = image_phylum
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.3)
        image = tf.image.rot90(image, tf.random.uniform([], 0, 4, dtype=tf.int32))
        return (image, phylum), label

    # Final dataset creator
    def make_dataset(df):
        return tf.data.Dataset.from_tensor_slices(dict(df)) \
            .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE) \
            .batch(BATCH_SIZE) \
            .prefetch(tf.data.AUTOTUNE)

    return make_dataset(train_df), make_dataset(val_df), family_lookup
 
 # Building Models

def build_custom_cnn(num_phyla, num_families):
    """
    Builds a simple custom CNN model for image classification with auxiliary phylum input.

    Requires:
        - num_phyla (int): Number of unique phylum classes (for embedding layer).
        - num_families (int): Number of output classes (families) for softmax classification.

    Ensures:
        - Image input passes through 3 convolutional + pooling blocks.
        - Phylum input is embedded and concatenated with visual features.
        - Fully connected layers apply dropout regularization to prevent overfitting.
        - Output layer predicts one of the family classes via softmax.

    Returns:
        - model (tf.keras.Model): Compiled Keras model ready for training.
    """

    img_input = tf.keras.Input(shape=(224, 224, 3), name="image")
    phylum_input = tf.keras.Input(shape=(), dtype=tf.int32, name="phylum")

    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(img_input)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.Flatten()(x)

    phylum_embed = tf.keras.layers.Embedding(input_dim=num_phyla, output_dim=16)(phylum_input)
    phylum_embed = tf.keras.layers.Flatten()(phylum_embed)

    concat = tf.keras.layers.Concatenate()([x, phylum_embed])
    x = tf.keras.layers.Dense(256, activation='relu')(concat)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    output = tf.keras.layers.Dense(num_families, activation='softmax')(x)

    return tf.keras.Model(inputs=[img_input, phylum_input], outputs=output, name="CustomCNN")


def build_pretrained_model(base_model, name, num_phyla, num_families):
    """
    Builds a transfer learning model using a frozen pre-trained base and phylum metadata.

    Requires:
        - base_model (tf.keras.Model): A pre-trained convolutional base (e.g., VGG16, ResNet).
        - name (str): Name for the returned model.
        - num_phyla (int): Number of unique phylum classes (used for embedding).
        - num_families (int): Number of output classes (families) for classification.

    Ensures:
        - The base model is frozen (non-trainable).
        - Visual features from images are extracted using the pre-trained base.
        - Phylum information is embedded and concatenated with image features.
        - Fully connected layers combine both modalities and apply dropout for regularization.
        - Final layer outputs family class probabilities using softmax.

    Returns:
        - model (tf.keras.Model): A compiled model ready for training on image + metadata input.
    """

    base_model.trainable = False
    image_input = tf.keras.Input(shape=(224, 224, 3), name="image")
    phylum_input = tf.keras.Input(shape=(), dtype=tf.int32, name="phylum")

    x = base_model(image_input, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    phylum_embed = tf.keras.layers.Embedding(input_dim=num_phyla, output_dim=16)(phylum_input)
    phylum_embed = tf.keras.layers.Flatten()(phylum_embed)
    phylum_embed = tf.keras.layers.Dense(64, activation='relu')(phylum_embed)

    x = tf.keras.layers.Concatenate()([x, phylum_embed])
    x = tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    output = tf.keras.layers.Dense(num_families, activation='softmax')(x)

    return tf.keras.Model(inputs=[image_input, phylum_input], outputs=output, name=name)