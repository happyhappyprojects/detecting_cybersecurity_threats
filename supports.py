import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np


def secs_to_mins_secs(secs, round_secs=2):
    return secs//60, round(secs%60, round_secs)

class CyberDataset(Dataset):
    def __init__(self, cvs_path, num_labels):
        df = pd.read_csv(cvs_path)
        df = self.preprocess_df(df)
        self.categorical_col_indexes = self.get_categorical_column_indices(df.iloc[:,:-1])
        # Assueme label column is the last column
        self.data = torch.tensor(df.iloc[:,:-1].to_numpy()).float()
        self.labels = torch.tensor(df.iloc[:,-1].to_numpy()).float()
        self.num_classes = [len(torch.unique(self.data[:, i])) for i in self.categorical_col_indexes]
        self.num_labels= num_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        numerical_features = np.delete(self.data[idx,:], self.categorical_col_indexes, axis=0)
        processed_data = numerical_features
        for i, col_i in enumerate(self.categorical_col_indexes):
            one_hot_encoded = F.one_hot(self.data[idx,col_i].long(), num_classes = self.num_classes[i])
            processed_data = torch.cat((processed_data, one_hot_encoded), dim=0)
        # processed_labels = F.one_hot(self.labels[idx].long(), num_classes = self.num_labels)
        return processed_data, self.labels[idx].unsqueeze(-1)

    @staticmethod
    def preprocess_df(df):
        """Transforms data according to recommendations by the paper"""
        df['processId'] = [1 if x in [0,1,2] else 0 for x in df['processId']]
        df['parentProcessId'] = [1 if x in [0,1,2] else 0 for x in df['parentProcessId']]
        df['userId'] = [1 if x > 1000 else 0 for x in df['userId']]
        df['mountNamespace'] = [1 if x == 4026531840 else 0 for x in df['mountNamespace']]
        # For returnValue mapping and meaniing
        # '<0' : 0 represents 'bad errors'; '=0' : 1 represents 'success'; 
        # '>0' : 2 represents 'success and signalling something to the parent process'
        df['returnValue'] = [0 if x < 0 else 1 if x == 0 else 2 for x in df['returnValue']]
        df['returnValue'] = df['returnValue'].astype(pd.CategoricalDtype(categories=[0,1,2]))
        # There's no transformation recommended by the paper; scale it for training
        df['threadId'] = (df['threadId'] - df['threadId'].mean()) / (df['threadId'].std())
        df['threadId'] = df['threadId'].astype('float32')
        return df
    
    @staticmethod
    def get_categorical_column_indices(df):
        """
        Returns a list of indices of categorical columns in a Pandas DataFrame.
        Args:
            df (pd.DataFrame): The input DataFrame.
        Returns:
            list: A list of indices of categorical columns.
        """
        categorical_indices = []
        for i, col_name in enumerate(df.columns):
            if pd.api.types.is_categorical_dtype(df[col_name]):
                categorical_indices.append(i)
        return categorical_indices
    

class CyberNN(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, 12)
        self.bn1 = nn.BatchNorm1d(12)
        self.fc2 = nn.Linear(12, 6)
        self.bn2 = nn.BatchNorm1d(6)
        self.fc3 = nn.Linear(6, dim_out)

        init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.fc3.weight, nonlinearity='sigmoid')

    def forward(self, x):
        x = F.elu(self.bn1(self.fc1(x)))
        x = F.elu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Return trainning loss of one epoch"""
    model.train() 
    epoch_loss = 0 
    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        batch_loss = criterion(outputs, labels)
        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss.item()
    return epoch_loss/len(dataloader)

def validate_epoch(model, dataloader, criterion, device, accuracy_metric):
    """Return trainning loss and accuracy of one epoch"""
    model.eval()
    epoch_loss = 0
    accuracy_metric.reset()
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            batch_loss = criterion(outputs, labels)
            epoch_loss += batch_loss.item()
            preds = outputs.argmax(dim=-1).unsqueeze(-1)
            accuracy_metric.update(preds, labels)
    return epoch_loss/len(dataloader), accuracy_metric.compute().item()


def test_model(model, dataloader, criterion, device, accuracy_metric):
    """Return test loss and accuracy
    It's the same process as validate epoch""" 
    return validate_epoch(model, dataloader, criterion, device, accuracy_metric)