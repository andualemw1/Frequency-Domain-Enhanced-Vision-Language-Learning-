
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 1. Load the image in grayscale
image_path = r'C:\Users\anduw\Desktop\VLM LLM\Chust Xray.jpeg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Could not load image at {image_path}. Check the file path.")

# 2. Compute the Forward Fourier Transform
f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)

# 3. Calculate the Magnitude Spectrum for visualization (log scale)
magnitude_spectrum = np.log(np.abs(fshift) + 1)

# 4. Create a High-Frequency Emphasis Filter Mask
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2

# Generate a base High-Pass Mask (0 at center, 1 everywhere else)
hpf_mask = np.ones((rows, cols), np.float32)
hpf_mask[crow-30:crow+30, ccol-30:ccol+30] = 0

# Apply High-Boost scaling: A + B * HPF
# A preserves the background structural layout; B emphasizes the edge contrast
A = 0.5 
B = 1.5
contrast_mask = A + B * hpf_mask

# 5. Apply the contrast enhancement mask to the frequency domain data
fshift_masked = fshift * contrast_mask

# 6. Compute the Inverse Fourier Transform to get the enhanced image back
f_ishift = np.fft.ifftshift(fshift_masked)
image_filtered = np.fft.ifft2(f_ishift)
image_filtered = np.abs(image_filtered)

# 7. Normalize images to 0-255 range so OpenCV can save them properly
image_processed_normalized = cv2.normalize(image_filtered, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
magnitude_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# --- VISUALIZATION (All 3 in one single window) ---
plt.figure(figsize=(15, 5))

# Subplot 1: Original Image
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Chest X-Ray")
plt.axis('off')

# Subplot 2: Frequency Spectrum
plt.subplot(1, 3, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title("Frequency Spectrum")
plt.axis('off')

# Subplot 3: Contrast Enhanced Result
plt.subplot(1, 3, 3)
plt.imshow(image_processed_normalized, cmap='gray')
plt.title("Contrast Enhanced X-Ray")
plt.axis('off')

plt.tight_layout()
plt.show()

# --- SAVING THE IMAGES TO DISK ---
save_path_spectrum = r'C:\Users\anduw\Desktop\VLM LLM\Chust Xray_spectrum.jpeg'
save_path_original_copy = r'C:\Users\anduw\Desktop\VLM LLM\Chust Xray_copy.jpeg'
save_path_enhanced = r'C:\Users\anduw\Desktop\VLM LLM\Chust Xray_enhanced.jpeg'

cv2.imwrite(save_path_spectrum, magnitude_normalized)
cv2.imwrite(save_path_original_copy, image)
cv2.imwrite(save_path_enhanced, image_processed_normalized)

print("All images (Original Copy, Spectrum, and Enhanced Contrast) successfully saved!")


