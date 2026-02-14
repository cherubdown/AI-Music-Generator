import torch
import numpy as np
import muspy
from src.models.lstm import BachLSTM
import argparse
from pathlib import Path
import os
from src.data.dataset import BachDataset # Assuming same resolution logic or similar

def generate(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    # We need to know specific hyperparameters used during training.
    # Ideally saved in config, but for now hardcoded/arg-passed.
    input_size = 128
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    output_size = 128
    
    model = BachLSTM(input_size, hidden_size, num_layers, output_size).to(device)
    
    checkpoint_path = args.checkpoint_path
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading model from {checkpoint_path}...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # Include resolution logic from dataset
    resolution = 4 
    
    # Initial sequence
    # Use random noise or zeros? Or a real snippet?
    # Random noise is safer to start exploration.
    # Or start with a single chord if we had one.
    # Let's start with random noise for now, or zeros.
    # Since it's piano roll generation, maybe zeros is silence.
    # Maybe random active notes?
    
    # Shape: (1, seq_length, 128)
    # Start with a short seed or full sequence?
    # LSTM can take variable length if we manage hidden state.
    # Let's start with a seed of length 1 (random chord) or seq_length (random noise)
    
    # Initialize hidden state
    hidden = None
    
    # Initial input: (1, 1, 128)
    # Let's seed with a C major chord or something simple?
    # C major: C4 (60), E4 (64), G4 (67), C5 (72)
    # Indices: 60, 64, 67, 72.
    current_input = torch.zeros(1, 1, 128).to(device)
    current_input[0, 0, [60, 64, 67, 72]] = 1.0 # C Major
    
    generated_seq = [current_input.squeeze().cpu().numpy()]
    
    print(f"Generating {args.length} steps...")
    
    with torch.no_grad():
        for _ in range(args.length):
            # Forward pass
            out, hidden = model(current_input, hidden)
            
            # Out shape: (1, 1, 128)
            # Apply sigmoid and temperature
            probs = torch.sigmoid(out / args.temperature)
            
            # Sample: Bernoulli
            # Or threshold?
            # Bernoulli is standard for piano roll (multi-hot)
            current_input = torch.bernoulli(probs)
            
            # Append to result
            generated_seq.append(current_input.squeeze().cpu().numpy())
            
    # Convert to MusPy music object
    # Stack: (time, 128)
    pianoroll = np.stack(generated_seq)
    
    # Cast to bool for encode_velocity=False
    pianoroll = pianoroll.astype(bool)
    
    # MusPy expects pianoroll as (time, 128)
    # Create muspy object
    music = muspy.from_pianoroll_representation(
        pianoroll, 
        resolution=resolution, 
        encode_velocity=False
    )
    
    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_filename)
    muspy.write_midi(output_path, music)
    print(f"Saved generated music to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Bach Music')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='outputs/generated', help='Output directory')
    parser.add_argument('--output_filename', type=str, default='generated.mid', help='Output filename')
    parser.add_argument('--length', type=int, default=128, help='Length of generation in steps')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--hidden_size', type=int, default=512, help='Hidden size (must match training)')
    parser.add_argument('--num_layers', type=int, default=2, help='Num layers (must match training)')
    
    args = parser.parse_args()
    generate(args)
