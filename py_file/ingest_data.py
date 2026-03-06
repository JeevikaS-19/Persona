import os
import pandas as pd
import glob
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IngestionEngine")

def ingest_from_usb():
    """
    Step 1: GUI Inbound - Select USB/Source Folder
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    
    print("\n--- Persona PyTorch Ingestion Engine ---")
    source_folder = filedialog.askdirectory(title="Select Source Folder (USB Drive)")
    root.destroy()
    
    if not source_folder:
        print("[!] No folder selected. Operation cancelled.")
        return

    source_path = Path(source_folder)
    print(f"[*] Scanning: {source_path}")

    # Step 2: Deep Scanning for report_*.csv
    csv_files = list(source_path.rglob("report_*.csv"))
    if not csv_files:
        print("[!] No report files found in the source directory.")
        return

    all_data = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Record relative origin to help find files later
            # Step 2 requirement: add column tracking source info
            df['team_member_id'] = csv_file.parent.name
            
            # Step 2 requirement: combine USB folder path with filename
            # We store the absolute path on the CURRENT system to verify existence
            # and allow the PyTorch DataLoader to find it.
            df['absolute_path'] = df['filename'].apply(lambda x: str(csv_file.parent / x))
            
            all_data.append(df)
            logger.info(f"Ingested {len(df)} rows from {csv_file.name}")
        except Exception as e:
            logger.error(f"Failed to process {csv_file}: {e}")

    if not all_data:
        print("[!] No valid data collected.")
        return

    # Merge & Clean
    master_df = pd.concat(all_data, ignore_index=True)
    
    # Identify broken entries (image exists check)
    print("[*] Verifying image integrity on source drive...")
    idx_to_drop = []
    for idx, row in master_df.iterrows():
        if not os.path.exists(row['absolute_path']):
            idx_to_drop.append(idx)
    
    if idx_to_drop:
        print(f"[!] Warning: {len(idx_to_drop)} rows have missing images and will be dropped.")
        master_df = master_df.drop(idx_to_drop)

    # Save to local project root
    master_csv_path = "master_dataset.csv"
    master_df.to_csv(master_csv_path, index=False)
    print(f"\n[SUCCESS] Master dataset saved to: {os.path.abspath(master_csv_path)}")

    # Step 4: Summary Statistics
    # Assume 'label' column exists (0 for Real, 1 for Fake is standard, 
    # but we'll check for 'HUMAN'/'DEEPFAKE' strings too)
    if 'label' in master_df.columns:
        counts = master_df['label'].value_counts()
        real_count = counts.get('HUMAN', counts.get(0, 0))
        fake_count = counts.get('DEEPFAKE', counts.get(1, 0))
        
        balance_ratio = fake_count / real_count if real_count > 0 else 0
        
        print("\n" + "="*30)
        print("DATASET SUMMARY")
        print("-" * 30)
        print(f"Total Real Frames: {real_count}")
        print(f"Total Fake Frames: {fake_count}")
        print(f"Balance Ratio (Fake/Real): {balance_ratio:.2f}")
        print("="*30)
    else:
        print("[!] 'label' column not found in dataset. Statistics skipped.")

if __name__ == "__main__":
    ingest_from_usb()
