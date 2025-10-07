import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def load_and_preprocess_data():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    train_images = train_images / 127.5 - 1
    test_images = test_images / 127.5 - 1
    return (train_images, train_labels), (test_images, test_labels)

def get_class_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def visualize_prediction(image, probabilities):
    classes = get_class_names()
    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    denormalized_image = (image + 1) / 2
    ax1.imshow(denormalized_image)
    ax1.axis('off')
    ax2.barh(np.arange(10), probabilities)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(classes)
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()
