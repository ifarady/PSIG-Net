from tensorflow.keras.layers import Input, Flatten, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import os


def load_images_from_folder(folder_path, target_size=(224, 224)):
    return [image.img_to_array(image.load_img(os.path.join(folder_path, fname), target_size=target_size))
            for fname in os.listdir(folder_path) if fname.endswith(('.jpg', '.png', '.jpeg'))]

# ori = load_images_from_folder('ADD_PATH_HERE')
# gen = load_images_from_folder('ADD_PATH_HERE')

# Create Positive and Negative Pairs
def create_pairs(ori, gen):
    pairs = []
    labels = []

    # Positive pairs
    for i in range(len(ori)):
        for j in range(i+1, len(ori)):
            pairs.append([ori[i], ori[j]])
            labels.append(1)

    for i in range(len(gen)):
        for j in range(i+1, len(gen)):
            pairs.append([gen[i], gen[j]])
            labels.append(1)
    
    # Negative pairs
    for i in range(len(ori)):
        for j in range(len(gen)):
            pairs.append([ori[i], gen[j]])
            labels.append(0)

    for i in range(len(gen)):
        for j in range(len(ori)):
            pairs.append([gen[i], ori[j]])
            labels.append(0)

    return np.array(pairs), np.array(labels)

pairs, labels = create_pairs(ori, gen)

# Shuffle Pairs
perm = np.random.permutation(len(pairs))
pairs, labels = pairs[perm], labels[perm]

# Process Pairs for Model Training
def process_pairs(pairs):
    pair1 = []
    pair2 = []

    for p in pairs:
        pair1.append(image.img_to_array(p[0]))
        pair2.append(image.img_to_array(p[1]))

    return [np.array(pair1), np.array(pair2)]


pair_images = process_pairs(pairs)
print("preprocessing done")

def create_base_model(input_shape):
    '''
    simple EfficientNetB7 model

    base_model = EfficientNetB7(input_shape=input_shape, weights='imagenet', include_top=False)
    x = base_model.output
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    '''
    return Model(inputs=base_model.input, outputs=x)

# Contrastive Loss
def contrastive_loss(y_true, y_pred):
    # https://pyimagesearch.com/2021/01/18/contrastive-loss-for-siamese-networks-with-keras-and-tensorflow/
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    margin = 1
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

# Create the Siamese Network
def create_siamese_network(input_shape):
    base_network = create_base_model(input_shape)
    
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    vector_a = base_network(input_a)
    vector_b = base_network(input_b)
    
    # Compute L2 distance between vectors
    l2_distance_layer = Lambda(lambda tensors: tf.norm(tensors[0] - tensors[1], axis=-1, ord=2))
    l2_distance = l2_distance_layer([vector_a, vector_b])
    
    model = Model(inputs=[input_a, input_b], outputs=l2_distance)
    return model

model = create_siamese_network((224, 224, 3))
model.compile(optimizer='adam', loss=contrastive_loss)


pair1_images, pair2_images = pair_images
model.fit([pair1_images, pair2_images], labels, batch_size=32, epochs=1000)