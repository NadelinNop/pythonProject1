import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

from PIL import Image
# GPU Configuration
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
if tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None):
    print("CUDA is available")
else:
    print("CUDA is not available")
# Define image data generators
# Enhanced Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2
)
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    'classification_data/train_data',
    target_size=(128, 128),
    batch_size=32,
    class_mode='sparse'
)

val_generator = val_test_datagen.flow_from_directory(
    'classification_data/val_data',
    target_size=(128, 128),
    batch_size=32,
    class_mode='sparse'
)

test_generator = val_test_datagen.flow_from_directory(
    'classification_data/test_data',
    target_size=(128, 128),
    batch_size=32,
    class_mode='sparse'
)


def build_resnet18(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)

    def residual_block(x, filters, downsample=False):
        strides = 2 if downsample else 1
        shortcut = x
        x = layers.Conv2D(filters, (3, 3), padding='same', strides=strides, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, (3, 3), padding='same', activation=None)(x)
        x = layers.BatchNormalization()(x)
        if downsample or x.shape[-1] != shortcut.shape[-1]:
            shortcut = layers.Conv2D(filters, (1, 1), strides=strides, activation=None)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        x = layers.add([x, shortcut])
        x = layers.ReLU()(x)
        return x

    x = residual_block(x, 64)
    x = residual_block(x, 128, downsample=True)
    x = residual_block(x, 256, downsample=True)
    x = residual_block(x, 512, downsample=True)
    x = layers.GlobalAveragePooling2D()(x)
    embedding = layers.Flatten()(x)

    model = models.Model(inputs, embedding)
    return model

input_shape = (128, 128, 3)
embedding_model = build_resnet18(input_shape)

num_classes=4000
# Define a classifier using the embedding model
inputs = layers.Input(shape=input_shape)
x = embedding_model(inputs)
outputs = layers.Dense(num_classes, activation='softmax')(x)
classifier_model = models.Model(inputs, outputs)

classifier_model.compile(optimizer='adam', loss='Dsparse_categorical_crossentropy', metrics=['accuracy'])
history = classifier_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)
#classifier_model.save('Classifier_Model.h5')
import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    ap_distance = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    an_distance = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    return tf.maximum(ap_distance - an_distance + margin, 0.0)


def generate_triplets(generator):
    class_indices = generator.class_indices
    classes = list(class_indices.keys())
    class_to_indices = {class_name: np.where(generator.classes == class_idx)[0] for class_name, class_idx in class_indices.items()}
    sample_size = 100000
    triplets = []
    for class_name, indices in class_to_indices.items():
        if len(indices) < 2:
            continue
        negative_classes = [other_class for other_class in classes if other_class != class_name]
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                anchor_idx = indices[i]
                positive_idx = indices[j]
                negative_class = np.random.choice(negative_classes)
                negative_idx = np.random.choice(class_to_indices[negative_class])
                triplets.append((anchor_idx, positive_idx, negative_idx))
                print("ss")
                if len(triplets) >= sample_size:
                    break
            if len(triplets) >= sample_size:
                break
        if len(triplets) >= sample_size:
            break

    with open('triplets.pkl', 'wb') as f:
        pickle.dump(triplets, f)
    print(f"Triplets saved to ")

    return triplets


def triplet_loss_training(embedding_model, triplets, generator, epochs=10, batch_size=16):
    optimizer = tf.keras.optimizers.Adam()


    for epoch in range(epochs):
        epoch_loss = 0.0
        np.random.shuffle(triplets)  # Shuffle triplets each epoch
        print(f"Epoch {epoch + 1}/{epochs} started.")

        # Process triplets in batches
        num_batches = len(triplets) // batch_size
        print(f"Number of batches: {num_batches}")

        for batch_num in range(num_batches):
            start_idx = batch_num * batch_size
            end_idx = start_idx + batch_size
            batch_triplets = triplets[start_idx:end_idx]

            # Prepare batch data
            anchor_imgs = []
            positive_imgs = []
            negative_imgs = []
            print(f"Processing batch {batch_num + 1}/{num_batches}.")

            for anchor_idx, positive_idx, negative_idx in batch_triplets:
                # Corrected indexing to handle batch size properly
                anchor_batch = anchor_idx // generator.batch_size
                positive_batch = positive_idx // generator.batch_size
                negative_batch = negative_idx // generator.batch_size

                anchor_img_idx = anchor_idx % generator.batch_size
                positive_img_idx = positive_idx % generator.batch_size
                negative_img_idx = negative_idx % generator.batch_size

                anchor_img = generator[anchor_batch][0][anchor_img_idx]
                positive_img = generator[positive_batch][0][positive_img_idx]
                negative_img = generator[negative_batch][0][negative_img_idx]

                anchor_imgs.append(anchor_img)
                positive_imgs.append(positive_img)
                negative_imgs.append(negative_img)

            anchor_imgs = np.array(anchor_imgs)
            positive_imgs = np.array(positive_imgs)
            negative_imgs = np.array(negative_imgs)

            # Perform forward and backward pass
            with tf.GradientTape() as tape:
                anchor_embs = embedding_model(anchor_imgs)
                positive_embs = embedding_model(positive_imgs)
                negative_embs = embedding_model(negative_imgs)

                loss = triplet_loss(anchor_embs, positive_embs, negative_embs)
            grads = tape.gradient(loss, embedding_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, embedding_model.trainable_variables))

            epoch_loss += np.sum(loss.numpy())

        avg_epoch_loss = epoch_loss / num_batches
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_epoch_loss:.4f}')
        embedding_model.save(f'embedding_model_epoch_{epoch + 1:02d}.h5')
        print(f"Model saved for epoch {epoch + 1}")



def load_triplets(filename='triplets.pkl'):
    with open(filename, 'rb') as f:
        triplets = pickle.load(f)
    print(f"Triplets loaded from {filename}")
    return triplets

# Load triplets for training
#triplets = load_triplets('triplets.pkl')

# Train the model with the loaded triplets
#triplet_loss_training(embedding_model, triplets, train_generator)
triplets = generate_hard_triplets(train_generator, embedding_model)
triplet_loss_training(embedding_model, triplets, train_generator)

embedding_model.save('EmbeddingModel.h5')

#embedding_model.save('Embedding_Model.h5')
# Load the models for evaluation
classifier_model = tf.keras.models.load_model('ClassifierModel.h5')
embedding_model = tf.keras.models.load_model('Embedding_Model.h5')


import os
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score
from PIL import Image


def preprocess_image(img_path):
    img_full_path = os.path.join(img_path)
    print(f"Loading image from path: {img_full_path}")  # Debug statement
    if not os.path.exists(img_full_path):
        raise FileNotFoundError(f"File {img_full_path} does not  exist")
    img = Image.open(img_full_path).convert('RGB')
    img = img.resize((128, 128))
    img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    return np.expand_dims(img, axis=0)

def evaluate_verification(embedding_model, anti_spoofing_model, verification_pairs_file, verification_data_folder):
    pairs = pd.read_csv(verification_pairs_file, sep=' ', header=None, names=['img1', 'img2', 'label'])
    img1_paths = pairs['img1'].values
    img2_paths = pairs['img2'].values
    labels = pairs['label'].values
    print(labels)
    embeddings1 = []
    embeddings2 = []
    valid_labels = []

    for img1_path, img2_path, label in zip(img1_paths, img2_paths, labels):
        try:
            img1 = preprocess_image(os.path.join(img1_path))
            img2 = preprocess_image(os.path.join( img2_path))
        except FileNotFoundError as e:
            print(e)
            continue


        emb1 = embedding_model.predict(img1)
        emb2 = embedding_model.predict(img2)

        embeddings1.append(emb1)
        embeddings2.append(emb2)
        valid_labels.append(label)

        if len(valid_labels) % 100 == 0:
            print(f"Processed {len(valid_labels)} valid pairs so far")

    # Ensure we have consistent lengths for labels and similarities
    if len(embeddings1) != len(embeddings2):
        raise ValueError(
            f"Mismatch in lengths: embeddings1 ({len(embeddings1)}) vs embeddings2 ({len(embeddings2)})")

    similarities_cosine = [cosine_similarity(emb1, emb2).item() for emb1, emb2 in zip(embeddings1, embeddings2)]
    similarities_euclidean = [-euclidean_distances(emb1, emb2).item() for emb1, emb2 in zip(embeddings1,
                                                                                            embeddings2)]  # Use negative because lower distance means higher similarity

    if len(valid_labels) != len(similarities_cosine):
        raise ValueError(
            f"Mismatch in lengths: labels ({len(valid_labels)}) vs similarities ({len(similarities_cosine)})")

    if len(valid_labels) == 0:
        raise ValueError("No valid pairs processed. Check your verification pairs and data paths.")

    print(f"Labels distribution: {pd.Series(valid_labels).value_counts().to_dict()}")  # Debug statement

    if len(set(valid_labels)) == 1:
        print("Only one class present in y_true. Valid labels:", valid_labels)
        raise ValueError("Only one class present in y_true. ROC AUC score is not defined in that case.")

    auc_score_cosine = roc_auc_score(valid_labels,
                                     similarities_cosine)  # Use valid_labels which match similarities length
    auc_score_euclidean = roc_auc_score(valid_labels,
                                        similarities_euclidean)  # Use valid_labels which match similarities length

    print(f'Verification AUC Score (Cosine Similarity): {auc_score_cosine:.4f}')
    print(f'Verification AUC Score (Euclidean Distance): {auc_score_euclidean:.4f}')

    # Plot ROC curve
    fpr_cosine, tpr_cosine, _ = roc_curve(valid_labels, similarities_cosine)
    fpr_euclidean, tpr_euclidean, _ = roc_curve(valid_labels, similarities_euclidean)

    plt.figure()
    plt.plot(fpr_cosine, tpr_cosine, color='blue', lw=2,
             label=f'ROC curve (area = {auc_score_cosine:.4f}) (Cosine Similarity)')
    plt.plot(fpr_euclidean, tpr_euclidean, color='red', lw=2,
             label=f'ROC curve (area = {auc_score_euclidean:.4f}) (Euclidean Distance)')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()


# Evaluate the verification performance
anti_spoofing_model = tf.keras.models.load_model('Anti_Spoofing_Model.h5')
# Evaluate the verification performance
evaluate_verification(embedding_model, anti_spoofing_model, 'verification_pairs_val.txt', 'verification_data')