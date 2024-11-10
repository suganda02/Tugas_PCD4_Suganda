import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy.ndimage import histogram

def histogram_equalization(image):
    # Hitung histogram
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()  # Fungsi distribusi kumulatif
    cdf_normalized = cdf * hist.max() / cdf.max()  # Normalisasi CDF

    # Pemetaan intensitas
    cdf_mapped = np.floor(255 * cdf / cdf[-1]).astype(np.uint8)

    # Peta pixel ke intensitas baru
    image_equalized = cdf_mapped[image]
    return image_equalized

# Membaca citra
input_image = imageio.imread('IMG_20230908_200823 - Copy.jpg')  # Ganti dengan nama file citra Anda
if len(input_image.shape) == 3:  # Jika citra berwarna
    input_image = np.mean(input_image, axis=2).astype(np.uint8)  # Konversi ke grayscale

# Ekualisasi histogram
output_image = histogram_equalization(input_image)

# Menyimpan gambar hasil
imageio.imwrite('output_image.jpg', output_image)

# Menampilkan hasil
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Citra Awal')
plt.imshow(input_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Citra Setelah Ekualisasi')
plt.imshow(output_image, cmap='gray')
plt.axis('off')

plt.show()