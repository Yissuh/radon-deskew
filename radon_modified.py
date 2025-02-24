import time
from skimage.transform import radon
from skimage.filters import threshold_otsu
from skimage.morphology import opening, closing, square, disk
from skimage.measure import label, regionprops
from scipy.ndimage import median_filter
from PIL import Image, ImageOps
import numpy as np
from numpy.fft import rfft
import matplotlib.pyplot as plt

def extract_foreground(image_array):
    """Segment the foreground object from the background."""
    # Apply Otsu's thresholding
    thresh = threshold_otsu(image_array)
    binary = image_array > thresh  # True = object, False = background

    # Apply morphological closing to connect object parts
    binary = closing(binary, square(5))

    # Label connected regions
    labeled_img = label(binary)
    regions = regionprops(labeled_img)

    if not regions:
        return image_array  # Return original if no object detected

    # Find the largest connected component (assumed to be the object)
    largest_region = max(regions, key=lambda r: r.area)
    minr, minc, maxr, maxc = largest_region.bbox  # Bounding box

    # Crop to detected object
    cropped_object = image_array[minr:maxr, minc:maxc]
    return cropped_object

def binarize_image(image_path):
    """Segment object, convert to binary, and invert colors."""
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image_array = np.array(image)

    # Extract the foreground object
    object_array = extract_foreground(image_array)

    # Apply Otsu’s binarization
    thresh = threshold_otsu(object_array)
    binary = object_array > thresh

    # Apply median filtering to remove small noise
    binary = median_filter(binary, size=2)

    return Image.fromarray((binary * 255).astype(np.uint8))  # Convert back to image

def compute_radon_transform(image_array):
    """Compute the Radon transform of the given image array."""
    image_array = image_array - np.mean(image_array)  # Demean
    sinogram = radon(image_array)
    return sinogram

def detect_rotation(sinogram):
    """Detect the rotation angle of the text using the Radon transform."""
    rms_values = np.array([np.sqrt(np.mean(np.abs(line) ** 2)) for line in sinogram.T])
    rotation_angle = 90 - np.argmax(rms_values)
    return rotation_angle

def detect_text_orientation(image_array):
    """Improved detection of text orientation (upright vs upside-down)."""
    projection = np.sum(image_array, axis=1)  # Sum pixel values along rows

    # Smooth projection profile using a moving average filter (reduces noise)
    smooth_projection = np.convolve(projection, np.ones(5) / 5, mode="same")

    # Compare only the top and bottom 25% of the image
    quarter_len = len(smooth_projection) // 4
    top_section = np.sum(smooth_projection[:quarter_len])  # Top 25%
    bottom_section = np.sum(smooth_projection[-quarter_len:])  # Bottom 25%

    return "Upright" if top_section > bottom_section else "Upside-down"

def rotate_and_expand(image, angle):
    """Rotates an image while expanding the canvas size to fully fit the rotated content."""
    # Convert to RGBA (keeps transparency in rotation)
    image = image.convert("RGBA")
    
    # Rotate without cropping
    rotated_image = image.rotate(angle, expand=True, resample=Image.BICUBIC)
    
    # Convert back to RGB (remove alpha transparency)
    background = Image.new("RGB", rotated_image.size, (255, 255, 255))
    background.paste(rotated_image, (0, 0), rotated_image)
    
    return background

def deskew_and_correct_orientation(image_path, save_path="deskewed_expanded.jpg"):
    """Load, process, rotate the image while keeping full size, and detect text orientation."""
    
    # Start timing
    start_time = time.time()

    binary_image = binarize_image(image_path)
    image_array = np.array(binary_image, dtype=np.float64)
    
    sinogram = compute_radon_transform(image_array)
    rotation_angle = detect_rotation(sinogram)

    print(f"Rotation Detected: {rotation_angle:.2f} degrees")
    
    # Load original image
    image = Image.open(image_path)
    
    # Rotate with expanded canvas to prevent cropping
    corrected_image = rotate_and_expand(image, rotation_angle)

    # Detect text orientation (Upright vs Upside-down)
    rotated_array = np.array(corrected_image.convert("L"))
    text_orientation = detect_text_orientation(rotated_array)
    print(f"Detected Text Orientation: {text_orientation}")

    # Flip image only if it is upside-down
    if text_orientation == "Upside-down":
        corrected_image = rotate_and_expand(corrected_image, 180)

    # Save final image with expanded frame
    corrected_image.save(save_path)

    # Compute elapsed time
    elapsed_time = time.time() - start_time
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")

    # Plot images
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # axes[0].imshow(binary_image, cmap="gray")
    # axes[0].set_title("Binarized Image (Denoised)")
    
    # axes[1].imshow(sinogram.T, cmap="gray", aspect='auto')
    # axes[1].set_title("Radon Transform (Sinogram)")
    
    # axes[2].imshow(corrected_image, cmap="gray")
    # axes[2].set_title(f"Final Rotated Image ({text_orientation})")

    # plt.show()

    return corrected_image

# Example usage
deskew_and_correct_orientation("sample_images/sample1.jpg")
