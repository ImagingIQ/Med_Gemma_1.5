
# import os
# import numpy as np
# import pydicom
# from PIL import Image
# import torch
# from transformers import pipeline
# from dotenv import load_dotenv

# # =========================
# # Environment Setup
# # =========================
# load_dotenv()

# # =========================
# # Robust MedGemma 1.5 Preprocessing
# # =========================
# def apply_medgemma_windowing(dcm):
#     """
#     Applies MedGemma 1.5's 3-channel windowing. 
#     Handles missing RescaleSlope/Intercept (common in MRI).
#     """
#     # 1. Safely convert to float and apply rescaling if tags exist
#     slope = float(getattr(dcm, 'RescaleSlope', 1.0))
#     intercept = float(getattr(dcm, 'RescaleIntercept', 0.0))
#     pixel_data = dcm.pixel_array.astype(np.float32) * slope + intercept
    
#     def normalize(data, wl, ww):
#         # wl: Window Level (Center), ww: Window Width
#         lower = wl - ww // 2
#         upper = wl + ww // 2
#         data = np.clip(data, lower, upper)
#         # Ensure we don't divide by zero if ww is 0
#         width = upper - lower
#         if width <= 0: return np.zeros_like(data).astype(np.uint8)
#         return ((data - lower) / width * 255).astype(np.uint8)

#     # 2. Determine Windowing Strategy
#     # If the DICOM has defined windows, use them; otherwise, use data statistics
#     default_wl = getattr(dcm, 'WindowCenter', np.median(pixel_data))
#     print(f"Window Center: {default_wl}")
#     default_ww = getattr(dcm, 'WindowWidth', np.std(pixel_data) * 4)
#     print(f"Window Width: {default_ww}")
    
#     # Handle cases where WL/WW might be a list (common in some DICOMs)
#     if isinstance(default_wl, pydicom.multival.MultiValue): default_wl = float(default_wl[0])
#     if isinstance(default_ww, pydicom.multival.MultiValue): default_ww = float(default_ww[0])

#     # MedGemma 1.5 Channels:
#     # Channel R: Wide (Global context)
#     # Channel G: Standard (Soft tissue/Signal focus)
#     # Channel B: Narrow (Fine contrast/Detail)
#     chan_r = normalize(pixel_data, default_wl, default_ww * 2) 
#     chan_g = normalize(pixel_data, default_wl, default_ww)
#     chan_b = normalize(pixel_data, default_wl, default_ww * 0.5)
    
#     rgb_image = np.stack([chan_r, chan_g, chan_b], axis=-1)
#     return Image.fromarray(rgb_image)

# def load_dicom_volume(dicom_dir, subsample_step=3):
#     """
#     Loads and windows a series, ensuring it handles MRI and CT correctly.
#     """
#     # Support both .dcm and .dic extensions
#     valid_ext = ('.dcm', '.dic')
#     files = [
#         pydicom.dcmread(os.path.join(dicom_dir, f)) 
#         for f in os.listdir(dicom_dir) if f.lower().endswith(valid_ext)
#     ]
    
#     if not files:
#         raise ValueError(f"No DICOM files found in {dicom_dir}")

#     # Sort by InstanceNumber to ensure the volume is in order (head-to-toe or vice versa)
#     files.sort(key=lambda x: int(getattr(x, "InstanceNumber", 0)))
    
#     # Subsample for VRAM efficiency
#     selected_files = files[::subsample_step]
#     print(f"Analyzing {len(selected_files)} slices (Subsampled from {len(files)} total).")
    
#     return [apply_medgemma_windowing(f) for f in selected_files]

# # =========================
# # Model Setup & Inference
# # =========================
# model_id = "google/medgemma-1.5-4b-it"

# pipe = pipeline(
#     "image-text-to-text",
#     model=model_id,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
#     token=os.environ.get("HF_TOKEN")

# )

# # Execution
# dicom_path = "/data/Ripan/VLM/PATIENT_680/1.2.840.113619.2.44.5554020.6880392.20466.1704591601.274"

# try:
#     vlm_images = load_dicom_volume(dicom_path, subsample_step=2) 
    
#     content_payload = [{"type": "image", "image": img} for img in vlm_images]
#     content_payload.append({
#         "type": "text",
#         "text": "Analyze this MRI series and provide a detailed radiology report including Findings and Impression."
#     })

#     outputs = pipe(text=[{"role": "user", "content": content_payload}],
#                     temperature=0.1,
#                     top_p=0.97,
#                     top_k=20
#                 )
#     print("\n--- FINAL REPORT ---\n")
#     print(outputs[0]["generated_text"][-1]["content"])

# except Exception as e:
#     print(f"Failed to process volume: {e}")








# import os
# import pydicom
# import torch
# import numpy as np
# from PIL import Image
# from transformers import pipeline

# # =========================
# # Configuration
# # =========================
# # Path to the patient's main folder
# PATIENT_ROOT = "/data/Ripan/VLM/PATIENT_553"
# # How many slices to take from EACH series 
# SLICES_PER_SERIES = 8

# # =========================
# # Helper: MRI Windowing
# # =========================
# def process_mri_slice(dcm):
#     """Simple robust normalization for MRI sequences."""
#     arr = dcm.pixel_array.astype(np.float32)
#     # Basic min-max scaling to 0-255
#     arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)
#     return Image.fromarray((arr * 255).astype(np.uint8)).convert("RGB")

# # =========================
# # Automated Loader
# # =========================
# def load_all_series(root_dir):
#     all_images = []
#     # Find all subdirectories (Series)
#     series_folders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
    
#     print(f"Found {len(series_folders)} series folders in {root_dir}")

#     for folder in series_folders:
#         dicom_files = [f for f in os.listdir(folder) if f.lower().endswith(('.dcm', '.dic'))]
#         if not dicom_files: continue
        
#         # Load and sort slices by Instance Number
#         slices = [pydicom.dcmread(os.path.join(folder, f)) for f in dicom_files]
#         slices.sort(key=lambda x: int(getattr(x, "InstanceNumber", 0)))
        
#         # Take a representative sample (Beginning, Middle, End of the series)
#         indices = np.linspace(0, len(slices)-1, SLICES_PER_SERIES, dtype=int)
#         for idx in indices:
#             all_images.append(process_mri_slice(slices[idx]))
            
#     return all_images

# # =========================
# # Execution
# # =========================
# model_id = "google/medgemma-1.5-4b-it"
# pipe = pipeline("image-text-to-text", 
#                 model=model_id, 
#                 torch_dtype=torch.bfloat16, 
#                 device_map="auto",
#                 temperature=0.1,
#                 top_p=1.0,
#                 top_k=50,
#                 num_tokens=1000
#                 )

# # 1. Gather slices from the entire patient directory
# print("Scanning patient directory and processing slices...")
# patient_case_images = load_all_series(PATIENT_ROOT)

# # 2. Build the multimodal prompt
# # We label them as a "Complete Study" so the model knows they belong together
# content_payload = [{"type": "image", "image": img} for img in patient_case_images]
# content_payload.append({
#     "type": "text",
#     "text": (
#         "You are looking at a complete MRI study for one patient, including multiple sequences. "
#         "Analyze all provided images. Identify any pathology, correlate findings across the different "
#         "sequences, and provide a clinical impression."
#     )
# })

# # 3. Generate Report
# print(f"Analyzing {len(patient_case_images)} slices across all series...")
# outputs = pipe(text=[{"role": "user", "content": content_payload}])

# print("\n=== MULTI-SERIES RADIOLOGY REPORT ===\n")
# print(outputs[0]["generated_text"][-1]["content"])
















import os
import pydicom
import torch
import numpy as np
from PIL import Image
from transformers import pipeline

# =========================
# Configuration
# =========================
PATIENT_ROOT = "/data/Ripan/VLM/PATIENT_821"
SLICES_PER_SERIES = 10
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
# Load & Group Series
# =========================
def load_all_series_grouped(root_dir):
    grouped_series = []

    print("\n[STEP 1] Scanning patient directory for series folders...")
    series_folders = [
        f.path for f in os.scandir(root_dir) if f.is_dir()
    ]
    print(f"  ✔ Found {len(series_folders)} series folders")

    for folder in sorted(series_folders):
        series_name = os.path.basename(folder)
        print(f"\n[STEP 2] Processing series folder: {series_name}")

        dicom_files = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".dcm", ".dic"))
        ]

        print(f"  ✔ Found {len(dicom_files)} DICOM files")

        if not dicom_files:
            print("  ✖ No DICOM files, skipping series")
            continue

        slices = []
        for f in dicom_files:
            path = os.path.join(folder, f)
            try:
                dcm = pydicom.dcmread(path)
                slices.append(dcm)
            except Exception as e:
                print(f"    ✖ Failed to read {f}: {e}")

        print(f"  ✔ Loaded {len(slices)} DICOM slices")

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
        print(f"    - Selected indices: {indices.tolist()}")

        images = []
        for idx in indices:
            print(f"    ▶ Converting slice index {idx}")
            images.append(process_mri_slice(slices[idx]))

        print(f"  ✔ Converted {len(images)} slices to images")

        grouped_series.append((series_name, images))

    print("\n[STEP 3] Completed loading all series")
    return grouped_series

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
# Build Multimodal Prompt
# =========================
print("\n[STEP 5] Building multimodal prompt...")
grouped_series = load_all_series_grouped(PATIENT_ROOT)

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
        
        "**STEP 1: Sequence-Specific Signal Analysis**\n"
        "For each provided sequence (T1, T2, FLAIR, DWI, ADC, T1+C, etc.):\n"
        "- Identify tissue signal characteristics (hyperintense, hypointense, isointense)\n"
        "- Note CSF, gray matter, white matter differentiation\n"
        "- Detect any focal signal abnormalities\n\n"
        
        "**STEP 2: Anatomic Localization & Morphology**\n"
        "For each identified abnormality, specify:\n"
        "- Location: Lobe (frontal, parietal, temporal, occipital), hemisphere (right/left), "
        "intra-axial vs extra-axial, supratentorial vs infratentorial\n"
        "- Morphology: Size (cm), shape (round, irregular), margins (well-defined, infiltrative)\n"
        "- Mass effect: Midline shift, sulcal effacement, ventricular compression, herniation signs\n"
        "- Surrounding features: Edema (vasogenic vs cytotoxic), hemorrhage, calcification\n\n"
        
        "**STEP 3: Multi-Sequence Correlation**\n"
        "Integrate findings across sequences:\n"
        "- T1 vs T2/FLAIR signal pattern\n"
        "- Enhancement characteristics (if contrast provided)\n"
        "- Diffusion restriction (DWI bright + ADC dark = acute infarct/abscess; DWI bright + ADC bright = T2 shine-through/viscous fluid)\n"
        "- Susceptibility artifacts (blood, calcification, air)\n\n"
        
        "**STEP 4: Differential Diagnosis**\n"
        "Rank likely pathologies based on imaging features:\n"
        "1. Most likely diagnosis with supporting features\n"
        "2. Alternative diagnoses to consider\n"
        "3. Key distinguishing features between possibilities\n\n"
        
        "**FINAL CLINICAL IMPRESSION**\n"
        "Provide a concise, radiology-report style summary including:\n"
        "- Primary finding with location and characteristics\n"
        "- Differential diagnosis (ranked)\n"
        "- Recommended follow-up or additional sequences if indicated"
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
