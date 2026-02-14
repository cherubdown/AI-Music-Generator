import torch
import torch.nn as nn

class BachLSTM(nn.Module):
    def __init__(self, input_size=128, hidden_size=512, num_layers=2, output_size=128, dropout=0.2):
        super(BachLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden=None):
        """
        x shape: (batch_size, seq_len, input_size)
        hidden: (h, c)
        """
        out, hidden = self.lstm(x, hidden)
        
        # Decode the hidden state of the last time step
        # out shape: (batch, seq, hidden)
        out = self.fc(out)
        
        return out, hidden

if __name__ == "__main__":
    # Test
    model = BachLSTM()
    x = torch.randn(2, 64, 128) # Batch size 2, seq len 64, input 128
    out, hidden = model(x)
    print(f"Output shape: {out.shape}") # Should be (2, 64, 128)
