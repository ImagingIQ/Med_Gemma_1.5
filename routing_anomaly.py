import os
import pydicom
import torch
import numpy as np
from PIL import Image
from transformers import pipeline

# =========================
# Configuration
# =========================
PATIENT_ROOT = "/data/Ripan/VLM/PATIENT_705"
SLICES_PER_SERIES = 100
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
    # arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)
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
    modalities = set()
    series_descriptions = set()

    for f in dicom_files:
        path = os.path.join(root_dir, f)
        try:
            dcm = pydicom.dcmread(path)
            slices.append(dcm)
            
            # Extract metadata
            modality = getattr(dcm, "Modality", "Unknown")
            print(f"      - Modality: {modality}")
            description = getattr(dcm, "SeriesDescription", "Unknown")
            print(f"      - Series Description: {description}")
            modalities.add(modality)
            series_descriptions.add(description)
            
        except Exception as e:
            print(f"    ✖ Failed to read {f}: {e}")

    print(f"  ✔ Identified Modalities: {', '.join(modalities)}")
    print(f"  ✔ Identified Series Descriptions: {', '.join(series_descriptions)}")
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
    series_name = list(series_descriptions)[0] if series_descriptions else "MRI_Study"
    return [(series_name, images)]


# =========================
# Initialize Model Pipeline
# =========================
print("\n[STEP 4] Initializing MedGemma pipeline...")
pipe = pipeline(
    "image-text-to-text",
    model=MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    temperature=0.2,
)
print("  ✔ Model pipeline initialized")

# =========================
# Main Execution Loop
# =========================
print("\n[STEP 5] Scanning for series folders...")

# Identify series folders
series_folders = []
if any(f.lower().endswith((".dcm", ".dic")) for f in os.listdir(PATIENT_ROOT)):
    series_folders.append(PATIENT_ROOT)
else:
    for d in os.listdir(PATIENT_ROOT):
        path = os.path.join(PATIENT_ROOT, d)
        if os.path.isdir(path) and any(f.lower().endswith((".dcm", ".dic")) for f in os.listdir(path)):
            series_folders.append(path)

print(f"  ✔ Found {len(series_folders)} series folders to process")

for folder in series_folders:
    print(f"\n\n{'='*60}")
    print(f"PROCESSING SERIES: {os.path.basename(folder)}")
    print(f"{'='*60}")

    grouped_series = load_single_folder_series(folder)
    if not grouped_series:
        continue

    content_payload = []
    for series_name, images in grouped_series:
        content_payload.append({
            "type": "text",
            "text": f"[Series: {series_name}]"
        })
        for img in images:
            content_payload.append({
                "type": "image",
                "image": img
            })

    content_payload.append({
        "type": "text",
        "text": (
            "You are a medical imaging router. Your task is to analyze the provided brain MRI images and categorize the study based on the presence of anomalies and tumors.\n\n"
            "Follow this structured analysis:\n\n"
            "**STEP 1: Chain of Thought (CoT) Analysis**\n"
            "Analyze the images step-by-step. Look for signal variations, mass effects, structural irregularities, or any clinical indicators of pathology. Document your findings clearly.\n\n"
            "**STEP 2: Final Routing Decision**\n"
            "Based on your analysis, provide a clear routing decision using the categories below.\n\n"
            "**Final Output Format:**\n"
            "THOUGHT: [Your step-by-step reasoning]\n"
            "ROUTING:\n"
            "- ANOMALY: [Present/Absent]\n"
            "- TUMOR: [Present/Absent]"
        )
    })

    print(f"\n[STEP 6] Running model inference for {os.path.basename(folder)}...")
    try:
        outputs = pipe(
            text=[{
                "role": "user",
                "content": content_payload
            }],
        )
        print("  ✔ Inference completed")

        print("\n====================================")
        print("=== MULTI-SERIES RADIOLOGY REPORT ===")
        print("====================================\n")
        report_text = outputs[0]["generated_text"][-1]["content"]
        print(report_text)

        # Append to a single patient report file
        output_file = "patient_radiology_report_650.txt"
        with open(output_file, "a") as f:
            f.write(f"\n\n{'='*60}\n")
            f.write(f"SERIES DESCRIPTION: {grouped_series[0][0]}\n")
            f.write(f"FOLDER: {os.path.basename(folder)}\n")
            f.write(f"{'='*60}\n")
            f.write(report_text)
            f.write("\n")
        print(f"\n  ✔ Findings appended to: {output_file}")

    except Exception as e:
        print(f"  ✖ Inference failed for {os.path.basename(folder)}: {e}")

print("\n====================================")
print("ALL SERIES PROCESSED SUCCESSFULLY")
print("====================================")
















