import muspy
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from music21 import corpus, converter
from tqdm import tqdm

class BachDataset(Dataset):
    def __init__(self, root=None, download=False, resolution=4, seq_length=64, transpose=True):
        """
        Args:
            root (str): Not used for music21 corpus (dataset is built-in).
            download (bool): Not used for music21 corpus.
            resolution (int): Time steps per quarter note.
            seq_length (int): Sequence length for training.
            transpose (bool): Whether to augment data by transposing to all keys.
        """
        self.seq_length = seq_length
        self.resolution = resolution
        
        self.processed_data = []
        self._load_and_process_data(transpose)
        
    def _load_and_process_data(self, transpose):
        print("Loading JSB Chorales from music21...")
        # music21 has about 400 chorales
        chorales = corpus.chorales.Iterator(returnType='filename')
        
        # Limit for testing/debugging if needed, or load all
        # loading all might take a while, let's look at a few first if testing, but for real training we need all.
        # Let's try loading a subset first to ensure it works, then remove limit? 
        # No, let's load all.
        
        # We need to filter for 'bwv' to get Bach chorales generally
        bach_chorales = [c for c in chorales if 'bwv' in str(c)]
        print(f"Found {len(bach_chorales)} chorales.")

        for chorale_path in tqdm(bach_chorales, desc="Processing chorales"):
            try:
                # Parse with music21
                c_m21 = corpus.parse(chorale_path)
                
                # Convert to MusPy
                music = muspy.from_music21(c_m21)
                
                # Adjust resolution if needed
                if music.resolution != self.resolution:
                    music.adjust_resolution(self.resolution)
                
                # Transpose to all keys if requested
                keys = range(-6, 6) if transpose else [0]
                
                for semitones in keys:
                    transposed_music = music.deepcopy()
                    if semitones != 0:
                        transposed_music.transpose(semitones)
                    
                    # Convert to piano-roll
                    # Shape: (time, 128)
                    try:
                        pianoroll = muspy.to_pianoroll_representation(
                            transposed_music, 
                            encode_velocity=False
                        )
                    except TypeError:
                         # Fallback if older muspy version or different API
                         pianoroll = transposed_music.to_pianoroll_representation(encode_velocity=False)

                    
                    # Debug print for first item
                    if len(self.processed_data) == 0 and semitones == 0:
                         print(f"Debug: shape={pianoroll.shape}, non-zero={np.count_nonzero(pianoroll)}")

                    # Binarize
                    pianoroll = (pianoroll > 0).astype(np.float32)
                    
                    # Split into sequences
                    num_steps = pianoroll.shape[0]
                    # Filter out short sequences
                    if num_steps < self.seq_length:
                        # print(f"Skipping short sequence: {num_steps} < {self.seq_length}")
                        continue
                        
                    for i in range(0, num_steps - self.seq_length, self.seq_length): 
                        seq = pianoroll[i : i + self.seq_length]
                        if seq.shape[0] == self.seq_length:
                            self.processed_data.append(seq)
            except Exception as e:
                # Some files might fail validation or parsing
                print(f"Error processing {chorale_path}: {e}")
                continue
                        
        print(f"Processed {len(self.processed_data)} sequences.")

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        seq = self.processed_data[idx]
        seq_tensor = torch.from_numpy(seq)
        
        x = seq_tensor[:-1]
        y = seq_tensor[1:]
        
        return x, y

if __name__ == "__main__":
    # Test
    # Disable transpose for speed in test
    dataset = BachDataset(transpose=False)
    print(f"Dataset size: {len(dataset)}")
    if len(dataset) > 0:
        x, y = dataset[0]
        print(f"Sample shape: x={x.shape}, y={y.shape}")
