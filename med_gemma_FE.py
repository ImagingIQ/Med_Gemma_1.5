import os
import pydicom
import torch
import numpy as np
from PIL import Image
from transformers import pipeline

# =========================
# Configuration
# =========================
PATIENT_ROOT = "/data/Ripan/VLM/PATIENT_632/1.2.840.113619.2.495.13667961.51077.31284.1736223200.191"
SLICES_PER_SERIES = 50
MODEL_ID = "google/medgemma-1.5-4b-it"

print("====================================")
print("MRI MULTI-SERIES VLM PIPELINE STARTED")
print("====================================")
print(f"Patient root directory: {PATIENT_ROOT}")
print(f"Slices per series: {SLICES_PER_SERIES}")
print(f"Model: {MODEL_ID}")
print("------------------------------------")

# =========================
# Helper: MRI Normalization
# =========================
def process_mri_slice(dcm):
    print("    ▶ Processing DICOM slice")
    arr = dcm.pixel_array.astype(np.float32)
    print(f"      - Raw pixel shape: {arr.shape}")
    arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)
    arr = (arr * 255).astype(np.uint8)
    print("      - Normalized to 8-bit RGB")
    return Image.fromarray(arr).convert("RGB")


# =========================
# Load Single Folder DICOM Series
# =========================
def load_single_folder_series(root_dir):
    print("\n[STEP 1] Scanning folder for DICOM files...")

    dicom_files = [
        f for f in os.listdir(root_dir)
        if f.lower().endswith((".dcm", ".dic"))
    ]

    print(f"  ✔ Found {len(dicom_files)} DICOM files")

    if not dicom_files:
        print("  ✖ No DICOM files found")
        return []

    slices = []
    for f in dicom_files:
        path = os.path.join(root_dir, f)
        try:
            dcm = pydicom.dcmread(path)
            slices.append(dcm)
        except Exception as e:
            print(f"    ✖ Failed to read {f}: {e}")

    print(f"  ✔ Loaded {len(slices)} slices")

    print("  ▶ Sorting slices by InstanceNumber")
    slices.sort(
        key=lambda x: int(getattr(x, "InstanceNumber", 0))
    )

    print("  ▶ Sampling representative slices")
    indices = np.linspace(
        0, len(slices) - 1,
        min(SLICES_PER_SERIES, len(slices)),
        dtype=int
    )

    images = []
    for idx in indices:
        print(f"    ▶ Converting slice index {idx}")
        images.append(process_mri_slice(slices[idx]))

    print(f"  ✔ Converted {len(images)} slices to images")

    # Return as one pseudo series
    return [("MRI_Study", images)]


# =========================
# Initialize Model Pipeline
# =========================
print("\n[STEP 4] Initializing MedGemma pipeline...")
pipe = pipeline(
    "image-text-to-text",
    model=MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
print("  ✔ Model pipeline initialized")

# =========================
# Build Multimodal Prompt
# =========================
print("\n[STEP 5] Building multimodal prompt...")
grouped_series = load_single_folder_series(PATIENT_ROOT)

content_payload = []

for series_name, images in grouped_series:
    print(f"\n[STEP 6] Adding series to prompt: {series_name}")
    content_payload.append({
        "type": "text",
        "text": f"[Series: {series_name}]"
    })
    print("  ✔ Added series separator text")

    for i, img in enumerate(images):
        content_payload.append({
            "type": "image",
            "image": img
        })
        print(f"  ✔ Added image {i + 1}/{len(images)}")

print("\n[STEP 7] Adding final instruction prompt")
content_payload.append({
    "type": "text",
    "text": (
        "You are a radiologist analyzing a brain MRI study. "
        "Each image group corresponds to the labeled series above it. "
        "Analyze systematically using this structured approach:\n\n"
        
        "**STEP 1: Chain of Thought (CoT) Analysis**\n"
        "Think step-by-step about the images provided. Analyze each sequence individually "
        "and look for any abnormalities, signal changes, or structural issues. "
        "Document your internal reasoning process here.\n\n"
        
        "**STEP 2: Sequence-Specific Signal Analysis**\n"
        "For each sequence (e.g., T1, T2, FLAIR, DWI), describe the tissue signal "
        "characteristics and any focal signal abnormalities detected.\n\n"
        
        "**STEP 3: Summarized Findings of Features**\n"
        "Summarize the key findings across all sequences. Focus on the most relevant "
        "anatomical and pathological features observed.\n\n"
        
        "**STEP 4: Health Status**\n"
        "Determine the overall health status based on this study. Return either "
        "[NORMAL] or [ABNORMAL] followed by a brief justification."
    )
})

print(f"  ✔ Total multimodal elements: {len(content_payload)}")


# =========================
# Run Inference
# =========================
print("\n[STEP 8] Running model inference...")
outputs = pipe(
    text=[{
        "role": "user",
        "content": content_payload
    }],
    do_sample=True,
    temperature=0.1,
    top_p=0.9,


)
print("  ✔ Inference completed")

# =========================
# Output
# =========================
print("\n====================================")
print("=== MULTI-SERIES RADIOLOGY REPORT ===")
print("====================================\n")
print(outputs[0]["generated_text"][-1]["content"])

print("\n====================================")
print("PIPELINE FINISHED SUCCESSFULLY")
print("====================================")
