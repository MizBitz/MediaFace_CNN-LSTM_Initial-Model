import json
import random
import numpy as np

# ==========================================
#           CONFIGURATION
# ==========================================
OUTPUT_FILE = "morse_sequences.jsonl"
NUM_SAMPLES_PER_CHAR = 500  # 500 examples per character

# 1. ITU-R M.1677-1 OFFICIAL DICTIONARY
# This includes the standard letters, numbers, and punctuation.
MORSE_CODE_DICT = {
    # Letters
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 
    'F': '..-.', 'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 
    'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---', 
    'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-', 
    'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--', 
    'Z': '--..',
    
    # Numbers
    '0': '-----', '1': '.----', '2': '..---', '3': '...--', '4': '....-', 
    '5': '.....', '6': '-....', '7': '--...', '8': '---..', '9': '----.',
    
    # Punctuation (Standard ITU-R M.1677-1)
    '.': '.-.-.-',  # Period
    ',': '--..--',  # Comma
    '?': '..--..',  # Question Mark
    "'": '.----.',  # Apostrophe
    '/': '-..-.',   # Slash
    '(': '-.--.',   # Open Parenthesis
    ')': '-.--.-',  # Close Parenthesis
    ':': '---...',  # Colon
    '=': '-...-',   # Double Dash / Equals
    '+': '.-.-.',   # Plus
    '-': '-....-',  # Hyphen
    '"': '.-..-.',  # Quote
    '@': '.--.-.'   # At Sign
}

# 2. SIMPLE TIMING CONFIGURATION (Seconds)
# We ignore WPM formulas and just use "Short vs Long" logic.
DOT_MEAN = 0.15   # Average time for a Dot (0.25s)
DASH_MEAN = 0.65  # Average time for a Dash (0.75s)
JITTER = 0.1     # Standard Deviation (Randomness)

def generate_duration(symbol):
    """
    Generates a duration with simple random noise.
    No complex WPM math, just Dot vs Dash.
    """
    if symbol == '.':
        # Generate random time centered around 0.25s
        duration = np.random.normal(DOT_MEAN, JITTER)
        return max(0.05, min(duration, 0.5)) # Clamp between 0.05s and 0.5s
        
    elif symbol == '-':
        # Generate random time centered around 0.75s
        duration = np.random.normal(DASH_MEAN, JITTER)
        return max(0.5, min(duration, 1.5)) # Clamp between 0.5s and 1.5s
        
    return 0.0

def main():
    print(f"Generating data using official ITU-R M.1677-1 codes for {len(MORSE_CODE_DICT)} symbols...")
    
    samples = []
    
    for char, code in MORSE_CODE_DICT.items():
        for _ in range(NUM_SAMPLES_PER_CHAR):
            
            # Generate the float sequence based on the code
            raw_durations = [generate_duration(symbol) for symbol in code]
            
            sample = {
                "raw_durations": raw_durations,
                "morse_seq": code,
                "label": char
            }
            samples.append(sample)

    # Shuffle the dataset
    random.shuffle(samples)

    # Save to file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"Successfully generated {len(samples)} samples.")
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()