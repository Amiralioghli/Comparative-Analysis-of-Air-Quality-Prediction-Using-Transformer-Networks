import pandas as pd
import numpy as np
import torch


device = (
    "mps"
    if getattr(torch, "has_mps", False)
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

def is_ne_in_df(df:pd.DataFrame):
    for col in df.columns:
        true_bool = (df[col] == "n/e")
        if any(true_bool):
            return True
    return False



def is_ne_in_df(df:pd.DataFrame):
    for col in df.columns:
        true_bool = (df[col] == "n/e")
        if any(true_bool):
            return True
    return False

def to_numeric_and_downcast_data(df: pd.DataFrame):
    fcols = df.select_dtypes('float').columns
    icols = df.select_dtypes('integer').columns
    df[fcols] = df[fcols].apply(pd.to_numeric, downcast='float')
    df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')
    return df

def read_train_test_val_data(File_Path,train_size = 70, val_size = 90 ,verbose=True):
    dataset = pd.read_csv(File_Path , sep=';', parse_dates={'date' : ['Date', 'Time']}, infer_datetime_format=True, 
                     low_memory=False, na_values=['nan','?'])
    dataset['date'] = pd.to_datetime(dataset['date'])
    dataset = dataset.sort_values(by = 'date')
    dataset = dataset.set_index("date")
    
    for j in range(0,7):        
            dataset.iloc[:,j]=dataset.iloc[:,j].fillna(dataset.iloc[:,j].mean())
    df = dataset.resample('17T').mean()
    
    if is_ne_in_df(df):
        raise ValueError("data frame contains 'n/e' values. These must be handled")
    df = to_numeric_and_downcast_data(df)
    
    #scaler = MinMaxScaler(feature_range = (0 , 1))
    #df = scaler.fit_transform(df)
    
    data_mean = df.mean(axis=0)
    data_std = df.std(axis=0)
    df = (df - data_mean) / data_std
    stats = (data_mean, data_std)
    
    train = df[:len(df) * train_size // 100]
    val = df[len(train) : len(train) + ((len(df) - len(train)) * val_size) // 100]
    test = df[len(val) + len(train) : ]
    return np.array(train) , np.array(val) , np.array(test)

# scaler = MinMaxScaler(feature_range=(0 , 1))
# train_set = scaler.fit_transform(train.values)
# test_set = scaler.fit_transform(test.values)
# val_set = scaler.fit_transform(val.values)

class CustomDataset(Dataset):
    def __init__(self, sequence, input_sequence_length, target_sequence_length, multivariate=True, target_feature=0):
        self.sequence = sequence
        self.input_sequence_length = input_sequence_length
        self.target_sequence_length = target_sequence_length
        self.window_size = input_sequence_length + target_sequence_length
        self.multivariate = multivariate
        self.target_feature = target_feature

    def __len__(self):
        return len(self.sequence) - self.window_size + 1

    def __getitem__(self, idx):
        src = self.sequence[idx:idx + self.input_sequence_length]
        trg = self.sequence[idx + self.input_sequence_length - 1:idx + self.window_size -1]
        
        if self.multivariate:
            trg_y = [obs[self.target_feature] for obs in self.sequence[idx + self.input_sequence_length:idx + self.input_sequence_length + self.target_sequence_length]]
            trg_y = torch.tensor(trg_y).unsqueeze(1).to(device)  # Adding a dimension for sequence length
        else:
            trg_y = self.sequence[idx + self.input_sequence_length:idx + self.input_sequence_length + self.target_sequence_length]
            trg_y = torch.tensor(trg_y).to(device)  # Adding a dimension for sequence length

        src = torch.tensor(src).to(device)  # Adding a dimension for features
        trg = torch.tensor(trg).to(device)  # Adding a dimension for features

        return src, trg, trg_y
