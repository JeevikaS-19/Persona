# Team Instruction Guide: Forensic Data Collection

Welcome to the Persona Data Collection team. This guide explains how to record, audit, and save forensic data for our master PyTorch training dataset.

## 1. Quick Setup
Ensure you have the required dependencies installed:
```bash
pip install opencv-python mediapipe pandas matplotlib moviepy torch
```

## 2. Running an Audit
To start the forensic collection engine, run:
```bash
python py_file/main.py
```

You will see two options:
*   **1. Upload**: Select a pre-recorded video file from your drive.
*   **2. Webcam**: Record a live 5-10 second clip (Press **'Q'** to finish recording and start analysis).

## 3. Saving to Your Pendrive (CRITICAL)
Once the forensic analysis is complete, the terminal will display the scores and then ask:
`Save Forensic Report? (y/n):`

1.  Type **`y`** and press Enter.
2.  A **Windows File Explorer** window will pop up.
3.  Navigate to your **Pendrive (USB)**.
4.  **Create a folder** named after yourself (e.g., `TEAM_MEMBER_01`).
5.  Save the report inside that folder.

## 4. Folder Organization
For our central system to find your work, please keep your pendrive organized like this:
```text
[USB Drive]
└── /Forensic_Work/
    └── /Your_Name/
        ├── report_20240306.csv
        ├── report_20240307.csv
        └── [Any face-crop images generated]
```

## 5. Tips for High-Quality Data
*   **Duration**: Aim for at least 5-10 seconds per clip.
*   **Lighting**: Ensure your face is well-lit for the Corneal Physics (Reflection) test.
*   **Motion**: Moderate head movement helps the Biometric Jitter test identify organic patterns.

---
*Thank you for contributing to the Persona Master Dataset!*
