import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch

def normalize_data(data, attribute_columns, target_column):
    # MinMax Normalization
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Normalize the attribute columns using MinMaxScaler and convert the result back to a DataFrame
    x = scaler.fit_transform(data[attribute_columns])
    x = pd.DataFrame(x, columns=attribute_columns)

    # Extract the target variable values and reshape them for normalization
    y = data[target_column].values.reshape(-1, 1)
    # Normalize the target variable using MinMaxScaler and convert the result back to a DataFrame
    y = scaler.fit_transform(y)
    y = pd.DataFrame(y, columns=[target_column])
    return x,y

# Function to split the data path wise as path changes every 3000 samples for given data.
def custom_train_test_split(data, labels, samples_per_block=3000, train_samples_per_block=2500, test_samples_per_block=500):
    # Initialize lists to hold the training and testing data and labels
    train_data = []
    test_data = []
    train_labels = []
    test_labels = []

    # Determine the total number of samples in the data
    total_samples = len(data)

    # Loop over the data in blocks of the specified size (samples_per_block)
    for start in range(0, total_samples, samples_per_block):
        # Define the end of the current block
        end = start + samples_per_block
        # Define the end of the training samples within the current block
        train_end = start + train_samples_per_block

        # Adjust train_end and end if they exceed the total number of samples
        if train_end > total_samples:
            train_end = total_samples
        if end > total_samples:
            end = total_samples

        # Append the training and testing data and labels for the current block to the respective lists
        train_data.append(data[start:train_end])
        test_data.append(data[train_end:end])
        train_labels.append(labels[start:train_end])
        test_labels.append(labels[train_end:end])

    # Concatenate the lists into DataFrames and reset the index
    train_data = pd.concat(train_data).reset_index(drop=True)
    test_data = pd.concat(test_data).reset_index(drop=True)
    train_labels = pd.concat(train_labels).reset_index(drop=True)
    test_labels = pd.concat(test_labels).reset_index(drop=True)

    # Return the training and testing data and labels
    return train_data, test_data, train_labels, test_labels


def get_data(path, label):
    # Use pandas to read the Excel file located at the specified file path
    data = pd.read_excel(path)

    # Define the attribute columns and the target variable without frequency
    attribute_columns = [f'Power_{i}' for i in range(1, 77)] + [f'ASE_{i}' for i in range(1, 77)] + [f'NLI_{i}' for i in range(1, 77)]+ ['No. Spans'] + ['Total Distance(m)']
    # Label
    target_column = label

    # Normalize the data
    x,y = normalize_data(data, attribute_columns, target_column)

    # Perform the custom train-test split on the normalized data
    X_train, X_test, Y_train, Y_test = custom_train_test_split(x[attribute_columns], y[target_column])
    return X_train, X_test, Y_train, Y_test

def prepare_fldata(num_partitions: int, batch_size: int, validation_split: float = 0.2):
    X_train, X_test, Y_train, Y_test = get_data('DataSet_EU_3k_5k.xlsx', 'GSNR_1')

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test.values, dtype=torch.long)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
    
    # Split training set into partitions
    partition_size = len(train_dataset) // num_partitions
    lengths = [partition_size] * num_partitions
    datasets = random_split(train_dataset, lengths, torch.Generator().manual_seed(42))
    
    # Split each partition into train/val and create DataLoaders
    trainloaders = []
    valloader = []
    for ds in datasets:
        len_val = int(len(ds) * validation_split)
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
        valloader.append(DataLoader(ds_val, batch_size=batch_size))
    
    testloader = DataLoader(test_dataset, batch_size=batch_size)
    
    return trainloaders, valloader, testloader


