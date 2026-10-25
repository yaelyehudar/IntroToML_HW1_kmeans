import sklearn 
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', as_frame=False)
data = mnist['data']
labels = mnist['target']

import numpy.random
idx = numpy.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]

def euclidean_distance(a, b):
    # a: Matrix of training images, shape (n, 784).
    # b: Single query image, shape (784,).
    return np.sqrt(np.sum((a - b) ** 2, axis=1))

def kNN(train_images, labels_vector, query_image, k):
    # Calculate the Euclidean distance between the query image and every image in the training set.
    distances = euclidean_distance(train_images, query_image)

    # returns the indices that would sort distances (from smallest to largest).
    # Select the indices of the k nearest training images.
    nearest_indices = np.argsort(distances)[:k]

    # Retrieve the true labels of the k nearest neighbors.
    nearest_labels = labels_vector[nearest_indices]

    # Find the unique labels among the nearest neighbors and count how many times each label appears.
    unique_labels, counts = np.unique(nearest_labels, return_counts=True)

    # Find the index of the label with the highest count, and return the corresponding label as the predicted label for the query image.
    return unique_labels[np.argmax(counts)]

# part b 
n = 1000
k = 10

# Use only the first n training images and their labels.
train_subset = train[:n]
labels_subset = train_labels[:n]

# Predict the label of each test image.
predictions = []

for query_image in test:
    predicted_label = kNN(train_subset, labels_subset, query_image, k)
    predictions.append(predicted_label)

predictions = np.array(predictions)

# Compare predictions with the true labels.
correct_predictions = predictions == test_labels

# The mean of a Boolean array gives the fraction of True values.
accuracy = np.mean(correct_predictions)

print(f"Accuracy (n={n}, k={k}): {accuracy:.2%}")

#part c 
n = 1000
k_values = range(1, 101)
accuracies = []
for k in k_values:
    predictons = [kNN(train[:n], train_labels[:n], query_image, k) for query_image in test]
    accuracy = np.mean(np.array(predictons) == test_labels)
    accuracies.append(accuracy)

best_k = np.argmax(accuracies) + 1  # +1 because k_values starts from 1
print(f"Best k: {best_k} with accuracy: {accuracies[best_k - 1]:.2%}")

plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o')
plt.title('k-NN Accuracy for Different k Values')
plt.xlabel('k (Number of Nearest Neighbors)')
plt.ylabel('Accuracy')
plt.xticks(np.arange(0, 101, 10))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#part d
k = 1
n_values = range(100, 5001, 100)
accuracies_n = []

for n in n_values:
    predictions = [kNN(train[:n], train_labels[:n], query_image, k) for query_image in test]
    accuracy = np.mean(np.array(predictions) == test_labels)
    accuracies_n.append(accuracy)

plt.figure(figsize=(10, 6))
plt.plot(n_values, accuracies_n, marker='o')
plt.title('k-NN Accuracy for Different Training Set Sizes (k=1)')
plt.xlabel('Number of Training Samples (n)')
plt.ylabel('Accuracy')
plt.xticks(np.arange(0, 5001, 500))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Part C, k=1:", accuracies[0])
print("Part D, n=1000:", accuracies_n[9])