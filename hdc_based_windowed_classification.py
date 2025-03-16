import numpy as np
import pickle
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from matplotlib import rcParams

def aggregate_embeddings(data, labels, hypervectors, D=1024, window_size=3):
    """
    Aggregate embeddings with window-based approach and positional encoding.
    Each opcode in a window is shifted based on its position:
    - First opcode is shifted window_size-1 times,
    - Second opcode is shifted window_size-2 times,
    - ..., until the last opcode which is not shifted.
    """
    # Initialize aggregated embeddings
    malware_aggregated_embd = np.zeros(D)
    benign_aggregated_embd = np.zeros(D)
    # print(len(data))
    # Iterate through each opcode list and label
    for opcodes, label in zip(data, labels):
        # Handle each window of opcodes
        # print(len(opcodes))
        for i in range(0, len(opcodes) - window_size + 1, window_size):
            window = opcodes[i:i + window_size]
            # Process each opcode in the window
            for j, opcode in enumerate(window):
                shift = window_size - 1 - j  # Calculate shift amount
                hypervector = hypervectors.get(opcode, np.zeros(D))  # Default to zero vector if not found
                # Shift the hypervector by repeating its last value 'shift' times at the end
                if shift > 0:
                    hypervector = np.roll(hypervector, shift)
                    hypervector[:shift] = 0  # Clear the first 'shift' elements
                # Aggregate the shifted hypervector
                if label == 1:  # Malware
                    malware_aggregated_embd += hypervector
                else:  # Benign
                    benign_aggregated_embd += hypervector

    binarized_malware = np.where(malware_aggregated_embd >= 1, 1, -1)
    binarized_benign = np.where(benign_aggregated_embd >= 1, 1, -1)
    return binarized_malware, binarized_benign
    # return malware_aggregated_embd, benign_aggregated_embd

def predict_sentiment_optimized(test_data, malware_vector, benign_vector, hypervectors, D=1024, window_size=3):
    """
    Predict sentiment using optimized hypervectors and window-based approach with positional encoding.
    Each opcode in a window is shifted based on its position.
    """
    pred_list = []
    for opcodes in test_data:
        pos_hypervector = np.zeros(D)
        # Handle each window of opcodes
        for i in range(0, len(opcodes) - window_size + 1, window_size):
            window = opcodes[i:i + window_size]
            # Process each opcode in the window
            for j, opcode in enumerate(window):
                shift = window_size - 1 - j  # Calculate shift amount
                hypervector = hypervectors.get(opcode, np.zeros(D))  # Default to zero vector if not found
                # Shift the hypervector by rolling and zeroing out wrapped elements
                if shift > 0:
                    hypervector = np.roll(hypervector, shift)
                    hypervector[:shift] = 0  # Clear the first 'shift' elements
                pos_hypervector += hypervector

        # Calculate similarity scores with the malware and benign class hypervectors
        malware_sim = cosine_similarity_1d(pos_hypervector, malware_vector)
        benign_sim = cosine_similarity_1d(pos_hypervector, benign_vector)
        
        # Determine the class based on higher similarity
        pred_list.append(1 if malware_sim > benign_sim else 0)  # 1 for malware, 0 for benign

    return pred_list

def cosine_similarity_1d(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    magnitude = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if magnitude == 0:
        return 0
    return dot_product / magnitude

def load_data_and_dict(train_path, test_path):
    with open(train_path, 'rb') as f:
        X_train, y_train = pickle.load(f)
    with open(test_path, 'rb') as f:
        X_test, y_test = pickle.load(f)
    return X_train, y_train, X_test, y_test

# Load data and hypervector dictionary
# dict_path = "sobol_opcode_hypervectors.pkl"
# dict_path = "opcode_hypervectors.pkl"
training_embeddings_file = "train_data.pkl"
testing_embeddings_file = "test_data.pkl"
# dict_path = "dicts//numpy_random_sequences_1024D_opcode.pkl"
dict_path = "dicts/numpy_random_sequences_1024D_opcode.pkl"
d_size = 1024
w_size = 6

with open(dict_path, 'rb') as file:
    hypervector_dict = pickle.load(file)
    hypervector_dict = {key: np.array(value) for key, value in hypervector_dict.items()}

X_train, y_train, X_test, y_test = load_data_and_dict(training_embeddings_file, testing_embeddings_file)


# Aggregation of hypervectors based on labels
time_start = time.time()
malware_vector, benign_vector = aggregate_embeddings(X_train, y_train, hypervector_dict, D = d_size, window_size= w_size)


# Prediction
predictions = predict_sentiment_optimized(X_test, malware_vector, benign_vector, hypervector_dict, D = d_size, window_size= w_size)
correct_predictions = sum(pred == actual for pred, actual in zip(predictions, y_test))
accuracy = correct_predictions / len(X_test) * 100

# model_data = {
#     'malware_vector': malware_vector,
#     'benign_vector': benign_vector
# }
# with open('hdc_model.pkl', 'wb') as file:
#     pickle.dump(model_data, file)

# print("Model saved successfully.")


# Confusion Matrix
cm = confusion_matrix(y_test, predictions)
tn, fp, fn, tp = cm.ravel()

# F1 Score, Precision, and Recall
f1 = f1_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)

time_end = time.time()

print(f"Total time for training and testing: {time_end - time_start:.2f} seconds")
print(f"Accuracy result with {dict_path} : {accuracy:.5f}%")
print(f"Precision: {precision:.5f}, Recall: {recall:.5f}, F1 Score: {f1:.5f}")
print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")

# # Set the font globally for all plots
# rcParams['font.family'] = 'Consolas'

# # Plotting the heatmap for the confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
# plt.title('Numpy Base Conf Matrix', fontsize = 16)
# plt.ylabel('True label', fontsize = 14)
# plt.xlabel('Predicted label', fontsize = 14)
# plt.show()
