import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torch.utils.data import DataLoader
from Preprocessing_Windowing_Datasetcreation import CustomDataset, read_train_test_val_data


input_sequence_length = 10
target_sequence_length = 10

path = "Datasets//Dataset_name.csv"
train , test , val = read_train_test_val_data(path)

train_dataset = CustomDataset(train, input_sequence_length, target_sequence_length, multivariate=True, target_feature=0)
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=False , drop_last=True)
test_dataset = CustomDataset(test, input_sequence_length, target_sequence_length, multivariate=True, target_feature=0)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False , drop_last=True)
val_dataset = CustomDataset(val, input_sequence_length, target_sequence_length, multivariate=True, target_feature=0)
val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False , drop_last=True)

device = (
    "mps"
    if getattr(torch, "has_mps", False)
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)



# 1. Encoder-only Training process

from Encoder_only import Encoder

model = Encoder(num_layers=1, D=32, H=1, hidden_mlp_dim=100,
                                       inp_features=7, out_features=1, dropout_rate=0.1).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

epochs = 20
early_stop_count = 0
min_test_loss = float('inf')


train_losses = []
test_losses1 = []

for epoch in tqdm(range(epochs)):
    model.train()
    for batch in train_loader:
        x_batch, y_batch = batch
   
        optimizer.zero_grad()
        out, _ = model(x_batch)
        train_loss = criterion(y_batch , out)
        train_loss.backward()
        optimizer.step()
        
    model.eval()
    test_losses = []
    with torch.no_grad():
        for batch in test_loader:
            x_batch, y_batch = batch

            out, _ = model(x_batch)
            test_loss = criterion(y_batch , out)
            test_losses.append(test_loss.item())

    test_loss = np.mean(test_losses)
    scheduler.step(test_loss)

    if test_loss < min_test_loss:
        min_test_loss = test_loss
        early_stop_count = 0
    else:
        early_stop_count += 1

    if early_stop_count >= 5:
        print("Early stopping!")
        break
    train_losses.append(train_loss)
    test_losses1.append(test_loss)
    print(f"Epoch {epoch + 1}/{epochs}, train loss: {train_loss}, test Loss: {test_loss}")

# 2. Decoder-only Training process

from Decoder_only import Decoder

def create_look_ahead_mask(size1, size2, device=device):
    mask = torch.ones((size1, size2), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask  # (size, size)

model = Decoder(num_layers=1, D=32, H=1, hidden_mlp_dim=32,
                                       inp_features=7, out_features=1, dropout_rate=0.1).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

epochs = 20
early_stop_count = 0
min_test_loss = float('inf')

train_losses = []
test_losses1 = []

for epoch in tqdm(range(epochs)):
    model.train()
    for batch in train_loader:
        x_batch, y_batch = batch
   
        S1 = x_batch.shape[1]
        mask = create_look_ahead_mask(S1 , S1)
        
        optimizer.zero_grad()
        out, _ = model(x_batch, mask)
        train_loss = criterion(y_batch , out)
        train_loss.backward()
        optimizer.step()
        
    model.eval()
    test_losses = []
    with torch.no_grad():
        for batch in test_loader:
            x_batch, y_batch = batch
            
            S1 = x_batch.shape[1]
            mask = create_look_ahead_mask(S1 , S1)
            
            out, _ = model(x_batch , mask)
            test_loss = criterion(y_batch , out)
            test_losses.append(test_loss.item())

    test_loss = np.mean(test_losses)
    scheduler.step(test_loss)

    if test_loss < min_test_loss:
        min_test_loss = test_loss
        early_stop_count = 0
    else:
        early_stop_count += 1

    if early_stop_count >= 5:
        print("Early stopping!")
        break
    train_losses.append(train_loss)
    test_losses1.append(test_loss)
    print(f"Epoch {epoch + 1}/{epochs}, train loss: {train_loss}, test Loss: {test_loss}")
    

# 3. Enc and Dec Transformer Training process

from Enc_Dec_Transformer import Transformer

def generate_square_subsequent_mask(dim1, dim2):
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)

model = Transformer(num_layers = 1, d_model=32, num_heads=1, ffn_hidden=10, inp_features=7, out_features=1, drop_prob=0.1)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

epochs = 20
early_stop_count = 0
min_test_loss = float('inf')


train_losses = []
test_losses1 = []

for epoch in tqdm(range(epochs)):
    model.train()
    for batch in train_loader:
        x_batch, y_batch , trg = batch
   
        optimizer.zero_grad()
        target_sequence_length = y_batch.shape[1]
        input_sequence_length = x_batch.shape[1]
        tgt_mask = generate_square_subsequent_mask(
            dim1=target_sequence_length,
            dim2=target_sequence_length
           )
        src_mask = generate_square_subsequent_mask(
            dim1=target_sequence_length,
            dim2=input_sequence_length
           )
        
        out = model(x_batch , y_batch , src_mask = None, tgt_mask = tgt_mask)
        train_loss = criterion(trg , out)
        train_loss.backward()
        optimizer.step()
        
    model.eval()
    test_losses = []
    with torch.no_grad():
        for batch in test_loader:
            x_batch, y_batch , trg = batch
            
            tgt_mask = generate_square_subsequent_mask(
                dim1=target_sequence_length,
                dim2=target_sequence_length
               )
            src_mask = generate_square_subsequent_mask(
                dim1=target_sequence_length,
                dim2=input_sequence_length
               )

            out = model(x_batch , y_batch , src_mask = None, tgt_mask = tgt_mask)
            test_loss = criterion(trg , out)
            test_losses.append(test_loss.item())

    test_loss = np.mean(test_losses)
    scheduler.step(test_loss)

    if test_loss < min_test_loss:
        min_test_loss = test_loss
        early_stop_count = 0
    else:
        early_stop_count += 1

    if early_stop_count >= 5:
        print("Early stopping!")
        break
    train_losses.append(train_loss)
    test_losses1.append(test_loss)
    print(f"Epoch {epoch + 1}/{epochs}, train loss: {train_loss}, test Loss: {test_loss}")
    

