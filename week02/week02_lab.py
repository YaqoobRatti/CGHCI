"""
Week 02 Lab: Image Processing Through a Four-Step Pipeline
Course: Perceptual Computing / Computer Graphics and Human-Computer Interaction
Student Name: Yaqoob Ratti
Roll Number: 2K24/CSEE/57
GitHub Username: YaqoobRatti
GitHub Repository: CGHCI
Folder: week02
"""

import numpy as np


# -------------------------------------------------------------
# Worked Sample 1: RGB to Grayscale
# -------------------------------------------------------------
def rgb_to_grayscale(r: int, g: int, b: int) -> int:
    """Convert one RGB pixel to a rounded grayscale luminosity value."""
    luminosity = 0.299 * r + 0.587 * g + 0.114 * b
    return int(round(luminosity))


# -------------------------------------------------------------
# Worked Sample 2: Brightness and Contrast Adjustment
# -------------------------------------------------------------
def adjust_brightness_contrast(image: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Apply O = alpha * I + beta and clip the result to [0, 255]."""
    image_float = image.astype(np.float32)
    adjusted = alpha * image_float + beta
    return np.clip(adjusted, 0, 255).astype(np.uint8)


# -------------------------------------------------------------
# Problem 1: Thresholding (1 mark)
# Step 1: Inputs and output: Grayscale image and threshold -> Binary image (0 or 255)
# Step 2: Rule: output 255 if pixel >= threshold else 0
# Step 3: Example: [[20, 128, 200], [100, 150, 250]] with threshold 128 -> [[0, 255, 255], [0, 255, 255]]
# -------------------------------------------------------------
def threshold_image(image: np.ndarray, threshold: int) -> np.ndarray:
    """Convert a grayscale image to black and white."""
    output = np.where(image >= threshold, 255, 0)
    return output.astype(np.uint8)


# -------------------------------------------------------------
# Problem 2: Image Memory (1 mark)
# Step 1: Inputs and output: width, height, bpp -> memory in bytes
# Step 2: Rule: bytes = width * height * bpp / 8
# Step 3: Example: 1920 x 1080 x 24 bpp / 8 = 6,220,800 bytes
# -------------------------------------------------------------
def display_memory_bytes(width: int, height: int, bpp: int) -> int:
    """Return image memory in bytes."""
    return (width * height * bpp) // 8


# -------------------------------------------------------------
# Problem 3: Average 9 Pixels (1 mark)
# Step 1: Inputs and output: 3x3 array -> single rounded integer average
# Step 2: Rule: average = sum of all 9 values / 9
# Step 3: Example: sum of 190 / 9 = 21 (rounded)
# -------------------------------------------------------------
def mean_filter_3x3(neighborhood: np.ndarray) -> int:
    """Return the rounded average of 9 pixels."""
    total = np.sum(neighborhood)
    return int(round(total / 9))


# -------------------------------------------------------------
# Problem 4: Change Image Contrast (1 mark)
# Step 1: Inputs and output: image array, in_min, in_max, out_min, out_max -> stretched array
# Step 2: Rule: new = (val - in_min) / (in_max - in_min) * (out_max - out_min) + out_min, clipped to [out_min, out_max]
# Step 3: Example: [50, 100, 150] from [50, 150] to [0, 255] -> [0.0, 127.5, 255.0]
# -------------------------------------------------------------
def contrast_stretch(
    image: np.ndarray,
    in_min: float,
    in_max: float,
    out_min: float = 0.0,
    out_max: float = 255.0,
) -> np.ndarray:
    """Change image values from one range to another."""
    stretched = (image.astype(np.float32) - in_min) / (in_max - in_min) * (out_max - out_min) + out_min
    return np.clip(stretched, out_min, out_max)


# -------------------------------------------------------------
# Problem 5: Find an Edge with Sobel (1 mark)
# Step 1: Inputs and output: 3x3 grayscale block -> gx, gy, and edge strength
# Step 2: Rule: gx = sum(block * Gx), gy = sum(block * Gy), strength = sqrt(gx^2 + gy^2)
# Step 3: Example: gx = 720, gy = 0, strength = 720.0
# -------------------------------------------------------------
def sobel_response(block: np.ndarray) -> tuple[float, float, float]:
    """Return gx, gy, and edge strength."""
    gx_kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    gy_kernel = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float32)

    gx = float(np.sum(block * gx_kernel))
    gy = float(np.sum(block * gy_kernel))
    strength = float(np.sqrt(gx**2 + gy**2))
    return gx, gy, strength


# -------------------------------------------------------------
# Verification Tests
# -------------------------------------------------------------
if __name__ == "__main__":
    print("Running Lab 02 tests...")

    # Sample 1
    sample_result = rgb_to_grayscale(180, 120, 60)
    assert sample_result == 131
    assert rgb_to_grayscale(80, 80, 80) == 80
    print("Sample 1 passed.")

    # Sample 2
    sample_image = np.array([[100, 150], [200, 50]], dtype=np.uint8)
    sample_output = adjust_brightness_contrast(sample_image, alpha=1.5, beta=-20)
    expected_output = np.array([[130, 205], [255, 55]], dtype=np.uint8)
    np.testing.assert_array_equal(sample_output, expected_output)
    extra = adjust_brightness_contrast(np.array([[240]], dtype=np.uint8), alpha=1.0, beta=30)
    assert extra[0, 0] == 255
    print("Sample 2 passed.")

    # Problem 1
    problem_1_input = np.array([[20, 128, 200], [100, 150, 250]], dtype=np.uint8)
    problem_1_expected = np.array([[0, 255, 255], [0, 255, 255]], dtype=np.uint8)
    np.testing.assert_array_equal(threshold_image(problem_1_input, 128), problem_1_expected)
    print("Problem 1 passed")

    # Problem 2
    expected_bytes = 1920 * 1080 * 24 // 8
    assert display_memory_bytes(1920, 1080, 24) == expected_bytes
    print("Problem 2 passed")

    # Problem 3
    problem_3_input = np.array([[10, 20, 10], [30, 50, 30], [10, 20, 10]], dtype=np.uint8)
    assert mean_filter_3x3(problem_3_input) == 21
    print("Problem 3 passed")

    # Problem 4
    problem_4_input = np.array([50, 100, 150], dtype=np.float32)
    expected = np.array([0.0, 127.5, 255.0], dtype=np.float32)
    np.testing.assert_allclose(contrast_stretch(problem_4_input, 50, 150), expected)
    print("Problem 4 passed")

    # Problem 5
    problem_5_input = np.array([[20, 20, 200], [20, 20, 200], [20, 20, 200]], dtype=np.float32)
    gx, gy, strength = sobel_response(problem_5_input)
    assert gx == 720.0 and gy == 0.0
    print("Problem 5 passed")

    print("\nAll tests completed successfully!")
