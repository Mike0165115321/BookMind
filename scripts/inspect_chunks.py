
import os
import sys
import pickle

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

def inspect():
    data_path = os.path.join(config.STORAGE_DIR, f"{config.INDEX_NAME}_data.pkl")
    if not os.path.exists(data_path):
        return

    with open(data_path, "rb") as f:
        chunks = pickle.load(f)
    
    print("🔍 First 5 chunks samples:")
    for i in range(min(5, len(chunks))):
        print(f"--- Chunk {i} ---")
        print(chunks[i][:200]) # Show first 200 chars

if __name__ == "__main__":
    inspect()
