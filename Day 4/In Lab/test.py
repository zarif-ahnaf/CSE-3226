import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load images (grayscale)
src = cv2.imread("./a.jpg", 0)
target = cv2.imread("./ref_image.jpg", 0)


def manual_histogram_processing(image):
    #  Compute Histogram (n_k)
    hist = np.zeros(256)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            hist[image[i, j]] += 1

    #  Compute PDF (p_r = n_k / MN)
    num_pixels = image.shape[0] * image.shape[1]
    pdf = hist / num_pixels

    #  Compute CDF (Sum of PDF)
    cdf = np.cumsum(pdf)

    # Histogram Equalization Scaling (s = (L-1) * CDF)
    # We round to the nearest integer as per manual calculation rules in the PDF
    equalized_values = np.round(cdf * 255).astype(np.uint8)

    return hist, pdf, cdf, equalized_values


# --- Process Source Image ---
hist_s, pdf_s, cdf_s, s_k = manual_histogram_processing(src)

# --- Process Target Image ---
hist_t, pdf_t, cdf_t, v_q = manual_histogram_processing(target)

# --- Mapping (Inverse Transformation) ---
# We find z such that G(z) is closest to T(r)
mapping = np.zeros(256, dtype=np.uint8)

for i in range(256):
    # Calculate the absolute difference between source CDF value and all target CDF values
    diff = np.abs(s_k[i] - v_q)
    # Find the index (gray level) where the difference is minimal
    mapping[i] = np.argmin(diff)


# --- Apply Mapping to Create Matched Image ---
matched = mapping[src]

# Calculate final histogram for verification
hist_m, _, _, _ = manual_histogram_processing(matched)

# --- DISPLAY DATA TABLES (As requested for manual check) ---
pd.set_option("display.max_rows", 20)

table_source = pd.DataFrame(
    {
        "Gray_r": np.arange(256),
        "Count_nk": hist_s,
        "PDF_Pr": pdf_s,
        "CDF": cdf_s,
        "Eq_sk": s_k,
    }
)

table_target = pd.DataFrame(
    {
        "Gray_z": np.arange(256),
        "Count_nk": hist_t,
        "PDF_Pz": pdf_t,
        "CDF": cdf_t,
        "Eq_vq": v_q,
    }
)

mapping_summary = pd.DataFrame(
    {"Source_Gray": np.arange(256), "Mapped_to_Target": mapping}
)

print("\n===== SOURCE CALCULATION =====")
print(table_source.head(10))

print("\n===== TARGET CALCULATION =====")
print(table_target.head(10))

print("\n===== FINAL MAPPING =====")
print(mapping_summary.head(10))

# --- VISUALIZATION ---
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.title("Source Histogram")
plt.bar(range(256), hist_s)
plt.subplot(2, 3, 2)
plt.title("Target Histogram")
plt.bar(range(256), hist_t)
plt.subplot(2, 3, 3)
plt.title("Matched Histogram")
plt.bar(range(256), hist_m)

plt.subplot(2, 3, 4)
plt.title("Source Image")
plt.imshow(src, cmap="gray")
plt.subplot(2, 3, 5)
plt.title("Target Image")
plt.imshow(target, cmap="gray")
plt.subplot(2, 3, 6)
plt.title("Matched Image")
plt.imshow(matched, cmap="gray")

plt.tight_layout()
plt.show()
