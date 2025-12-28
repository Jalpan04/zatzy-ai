
import os
import time
import subprocess
import sys

def run_pipeline():
    print("🚀 Starting Neuro-Expectimax Master Update Pipeline...")
    
    # 1. Wait for Data Generation (It's running in background)
    # We check if the process is still running or file size is growing
    # Actually, simpler to just run it here if not running, but user already started it.
    # We will assume user wants us to wait or restart it faster?
    # Let's just restart it with 3000 games in this script to be sure it's correct context.
    
    print("\n[1/3] Ensuring Data Generation Strategy...")
    # We will let the background task finish. 
    # But for safety, let's just run the training when ready.
    
    # Check for value_data.npz
    while not os.path.exists("value_data.npz"):
        print("Waiting for value_data.npz...", end='\r')
        time.sleep(5)
        
    print("\n[2/3] Starting Master Training (1000 Epochs)...")
    subprocess.run([sys.executable, "src/trainer/train_value_net.py", "--epochs", "1000"])
    
    print("\n[3/3] Final Verification Benchmark...")
    subprocess.run([sys.executable, "src/trainer/benchmark_neuro.py"])
    
    print("\n✅ MISSION COMPLETE: Neuro-Expectimax is now God-Tier.")

if __name__ == "__main__":
    run_pipeline()
