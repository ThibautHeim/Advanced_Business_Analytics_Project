import torch
import torch.nn as nn
from tqdm import tqdm
from argparse import ArgumentParser
import os, sys
import matplotlib.pyplot as plt


class BusDelayPredictor(nn.Module):
    def __init__(self, input_size, hidden_size = 32, num_layers = 1):
        """
        Args:
            input_size: Number of features per bus stop (delay + hour_minutes + weekday + month + weather + events + ...).
            hidden_size: Number of units in the LSTM hidden state (the "memory" capacity).
            num_layers: Number of stacked LSTM layers (usually 1 or 2).
        """
        super(BusDelayPredictor, self).__init__()
        
        # The core LSTM layer: Handles the recurrence and gate logic
        self.lstm = nn.LSTM(
            input_size=input_size,   
            hidden_size=hidden_size, 
            num_layers=num_layers,   
            batch_first=True         # Input format: (Batch, Sequence, Features)
        )
        
        # Output Layer: Maps the hidden state (h_t) to a single continuous value (predicted delay)
        self.fc = nn.Linear(hidden_size, 1)
    
    def init_hidden(self, device, h_O, c_O):
        """
        Initializes the hidden state (h_0) and cell state (c_0) for the LSTM.
        Args:
            batch_size: The number of sequences in a batch.
            device: The device (CPU or GPU) where the tensors should be allocated.
            h_O: Initial hidden state.
            c_O: Initial cell state.
        Returns:
            A tuple of (h_0, c_0) initialized to the provided initial states.
        """
        # change the hidden and cell states to the provided initial states
        h_0 = h_O.to(device)
        c_0 = c_O.to(device)
        return (h_0, c_0)   
    
    def init_hidden_zero(self, device, batch_size):
        """
        Initializes the hidden state (h_0) and cell state (c_0) for the LSTM to zeros.
        Args:
            batch_size: The number of sequences in a batch.
            device: The device (CPU or GPU) where the tensors should be allocated.
        Returns:
            A tuple of (h_0, c_0) initialized to zeros.
        """ 
        h_0 = torch.zeros(1, batch_size, self.lstm.hidden_size).to(device)  # (num_layers, batch_size, hidden_size)
        c_0 = torch.zeros(1, batch_size, self.lstm.hidden_size).to(device)  # (num_layers, batch_size, hidden_size)
        return (h_0, c_0)
    

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
        Returns:
            prediction: The predicted delay for the next target step.
        """
        
        # 'out' contains the hidden states (h_t) for EVERY time step in the sequence
        # '_' is a tuple containing the final (h_n, c_n); we don't use it here
        # out shape: (batch_size, seq_len, hidden_size)
        out, _ = self.lstm(x)
        
        # Extract only the hidden state of the LAST time step (the last bus stop)
        # We use [:, -1, :] to select all batches, the last index of the sequence, and all features
        last_stop_output = out[:, -1, :]
        
        # Pass the final hidden state through the Fully Connected (Dense) layer
        prediction = self.fc(last_stop_output)
        
        return prediction
    
# Structure of a batch : [batch_size x seq_len x num_features]
    
def train_one_epoch(model, train_loader, criterion, optimizer, batch_size, seq_len, device):
    
    total_loss = 0
    for batch in train_loader:
        batch_loss = 0
        # For sequence in batch
        for s in range(batch_size):
            h0, c0 = model.init_hidden_zero(device, batch_size)
            for t in range(1,seq_len): # if n stops including departure and terminus, we have seq_len = n+1
                input = batch[s,:t,:].unsqueeze(0)  # Shape: (1, t, num_features)
                target = batch[s,t:t+1,0].unsqueeze(0)  # Shape: (1, 1, 1) - Assuming the first feature is the delay
                if t == 1:
                    pred = model.lstm(input, (h0, c0))[0][:, -1, :]
                    pred = model.fc(pred)
                else :
                    pred = model.forward(input)
                loss = criterion(pred.squeeze(), target.squeeze())  # Assuming the first feature is the delay
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
        total_loss += batch_loss / batch_size
    return total_loss


def validate(model, test_loader, criterion, batch_size, seq_len, device,):
    total_loss = 0
    for batch in test_loader:
        batch_loss = 0
        for s in range(batch_size):
            h0, c0 = model.init_hidden_zero(device, batch_size)
            for t in range(1,seq_len): # if n stops including departure and terminus, we have seq_len = n+1
                input = batch[s,:t,:].unsqueeze(0)  # Shape: (1, t, num_features)
                target = batch[s,t:t+1,0].unsqueeze(0)  # Shape: (1, 1, 1) - Assuming the first feature is the delay
                if t == 1:
                    pred = model.lstm(input, (h0, c0))[0][:, -1, :]
                    pred = model.fc(pred)
                else :
                    pred = model.forward(input)
                loss = criterion(pred.squeeze(), target.squeeze())  # Assuming the first feature is the delay
                batch_loss += loss.item()
        total_loss += batch_loss / batch_size
    return total_loss


def main(
        num_features, 
        num_hidden,
        train_loader, 
        test_loader, 
        batch_size, 
        seq_len, 
        device, 
        num_epochs, 
        model_name):
    model = BusDelayPredictor(input_size=num_features, hidden_size=64)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_losses = []
    validation_losses = []
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        # training
        model.train()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, batch_size, seq_len, device)
        train_losses.append(train_loss)
        # validation
        model.eval()
        val_loss = validate(model, test_loader, criterion, batch_size, seq_len, device)
        validation_losses.append(val_loss)
        #update the progress bar with the current losses
        tqdm.write(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f}")
    # save the model in the output directory
    
    if not os.path.exists("output_LSTM"):
        os.makedirs("output_LSTM")
    torch.save(model.state_dict(), "output_LSTM/" + model_name + ".pth")
    return model, train_losses, validation_losses



if __name__ == "__main__":

    parser = ArgumentParser(description="Train an LSTM model for bus delay prediction.")
    parser.add_argument("--num_features", type=int, default=10, help="Number of features per bus stop.")
    parser.add_argument("--num_hidden", type=int, default=64, help="Number of hidden units in the LSTM.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--seq_len", type=int, default=10, help="Sequence length (number of bus stops in a sequence).")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--model_name", type=str, default="bus_delay_predictor", help="Name for saving the trained model.")
    args = parser.parse_args()

    train_loader, test_loader = None, None  # Replace with actual DataLoader instances
    model, train_losses, validation_losses = main(
        num_features=args.num_features,
        num_hidden=args.num_hidden,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_epochs=args.num_epochs,
        model_name=args.model_name
    )
    fig = plt.figure()
    plt.plot(range(len(train_losses)), train_losses, label="Training Loss")
    plt.plot(range(len(validation_losses)), validation_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig("output_LSTM/" + args.model_name + ".png")
