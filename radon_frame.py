import time
from skimage.transform import radon
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square
from skimage.measure import label, regionprops
from scipy.ndimage import median_filter
from PIL import Image
import numpy as np

def extract_foreground(image_array):
    """Segment the foreground object from the background."""
    thresh = threshold_otsu(image_array)
    binary = image_array > thresh
    binary = closing(binary, square(5))
    labeled_img = label(binary)
    regions = regionprops(labeled_img)
    if not regions:
        return image_array
    largest_region = max(regions, key=lambda r: r.area)
    minr, minc, maxr, maxc = largest_region.bbox
    return image_array[minr:maxr, minc:maxc]

def binarize_frame(frame):
    """Segment object, convert to binary, and invert colors."""
    grayscale_array = np.array(frame.convert("L"))
    object_array = extract_foreground(grayscale_array)
    thresh = threshold_otsu(object_array)
    binary = object_array > thresh
    binary = median_filter(binary, size=2)
    return (binary * 255).astype(np.uint8)

def compute_radon_transform(image_array):
    """Compute the Radon transform of the given image array."""
    image_array = image_array - np.mean(image_array)
    return radon(image_array)

def detect_rotation(sinogram):
    """Detect the rotation angle of the text using the Radon transform."""
    rms_values = np.array([np.sqrt(np.mean(np.abs(line) ** 2)) for line in sinogram.T])
    return 90 - np.argmax(rms_values)

def detect_text_orientation(image_array):
    """Detect whether the text is upright or upside-down."""
    projection = np.sum(image_array, axis=1)
    smooth_projection = np.convolve(projection, np.ones(5) / 5, mode="same")
    quarter_len = len(smooth_projection) // 4
    return "Upright" if np.sum(smooth_projection[:quarter_len]) > np.sum(smooth_projection[-quarter_len:]) else "Upside-down"

def rotate_and_expand(image, angle):
    """Rotate an image while expanding the canvas to fit the rotated content."""
    image = image.convert("RGBA")
    rotated_image = image.rotate(angle, expand=True, resample=Image.BICUBIC)
    background = Image.new("RGB", rotated_image.size, (255, 255, 255))
    background.paste(rotated_image, (0, 0), rotated_image)
    return background

def deskew_and_correct_orientation(frame):
    """Process, rotate, and correct text orientation in an image frame."""
    start_time = time.time()
    binary_array = binarize_frame(frame)
    sinogram = compute_radon_transform(binary_array)
    rotation_angle = detect_rotation(sinogram)
    print(f"Rotation Detected: {rotation_angle:.2f} degrees")
    corrected_image = rotate_and_expand(frame, rotation_angle)
    rotated_array = np.array(corrected_image.convert("L"))
    text_orientation = detect_text_orientation(rotated_array)
    print(f"Detected Text Orientation: {text_orientation}")
    if text_orientation == "Upside-down":
        corrected_image = rotate_and_expand(corrected_image, 180)
    elapsed_time = time.time() - start_time
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
    return corrected_image


# import cv2
# from PIL import Image

# # # Load image using OpenCV
# # image_path = "sample_images/sample1.jpg"
# # cv_frame = cv2.imread(image_path)  # This loads as a NumPy array in BGR format

# # Convert OpenCV BGR to RGB (PIL uses RGB)
# cv_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# # Convert NumPy array to PIL Image
# frame = Image.fromarray(cv_frame_rgb)

# # Now you can use `frame` in deskew_and_correct_orientation()
# processed_image = deskew_and_correct_orientation(frame)

# # Show or save the processed image
# processed_image.show()
# # processed_image.save("output.jpg")
