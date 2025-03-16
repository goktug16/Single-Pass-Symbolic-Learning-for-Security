import os
import pickle
from sklearn.model_selection import train_test_split

def read_opcodes_from_folder(folder_path):
    """
    Read all txt files in the folder and compile a list of opcode sequences.
    """
    opcode_sequences = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r') as file:
                opcodes = file.read().split()  # Assuming opcodes are separated by whitespace
                opcode_sequences.append(opcodes[1:])
    return opcode_sequences

# Paths to the folders
goodware_folder = 'goodware'
malware_folder = 'malware'

# Read opcodes from both folders
goodware_opcodes = read_opcodes_from_folder(goodware_folder)
malware_opcodes = read_opcodes_from_folder(malware_folder)

# Combine the goodware and malware lists, and create a label list
combined_opcodes = goodware_opcodes + malware_opcodes
labels = [0] * len(goodware_opcodes) + [1] * len(malware_opcodes)  # 0 for goodware, 1 for malware

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(combined_opcodes, labels, test_size=0.20, random_state=42)



# Save the training and testing data to pickle files
with open('train_data.pkl', 'wb') as f:
    pickle.dump((X_train, y_train), f)
with open('test_data.pkl', 'wb') as f:
    pickle.dump((X_test, y_test), f)

print("Training and testing data have been saved to 'train_data.pkl' and 'test_data.pkl'.")
