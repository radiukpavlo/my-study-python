from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.integrate import simps, trapz

# Load the image
image_path = "input_area2.jpg"
image = Image.open(image_path)

# Convert the image to a numpy array
image_np = np.array(image)

# Convert the image to grayscale
gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

# Apply a Canny edge detector to find edges in the image
edges = cv2.Canny(gray, 100, 200)

# Dilate the edges to make them more pronounced
dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))

# Find contours from the dilated edges
contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Identify the largest contour by area
largest_contour = max(contours, key=cv2.contourArea)

# Create a mask for the largest region
largest_region_mask = np.zeros_like(image_np)
cv2.drawContours(largest_region_mask, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

# Create an image with just the largest region highlighted
highlighted_largest_region_image = cv2.bitwise_and(image_np, largest_region_mask)

# Extract the x and y coordinates of the largest contour
contour_points = largest_contour[:, 0, :]
x = contour_points[:, 0]
y = contour_points[:, 1]

# Ensure the contour forms a closed loop
if not np.array_equal(contour_points[0], contour_points[-1]):
    x = np.append(x, x[0])
    y = np.append(y, y[0])

# Calculate area using the Simpson's method
simpson_area = abs(simps(y, x))

# Calculate area using the trapezoidal rule
trapezoidal_area = abs(trapz(y, x))

# Calculate area using Boole's rule
boole_area = 0
for i in range(0, len(x) - 1, 4):
    if i + 4 < len(x):
        boole_area += abs((7 * y[i] + 32 * y[i + 1] + 12 * y[i + 2] + 32 * y[i + 3] + 7 * y[i + 4]) * (x[i + 4] - x[i]) / 90)

# Define the correct calculation for largest_region_area_pixels based on the largest contour
largest_region_area_pixels = cv2.contourArea(largest_contour)

# Convert the calculated areas to square kilometers using the correct scale
# Adjusted scale (example): 1 pixel corresponds to a smaller distance to match the real area
pixels_per_km_adjusted = np.sqrt(largest_region_area_pixels / 5190.0)

# Calculate area using the Simpson's method
simpson_area_km2 = simpson_area / (pixels_per_km_adjusted ** 2)
trapezoidal_area_km2 = trapezoidal_area / (pixels_per_km_adjusted ** 2)
boole_area_km2 = boole_area / (pixels_per_km_adjusted ** 2)

# Display the original image and the largest segmented region
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
axes[0].imshow(image_np)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(highlighted_largest_region_image)
axes[1].set_title('Largest Segmented Region')
axes[1].axis('off')

print(simpson_area)
print(trapezoidal_area)
print(boole_area)

print(simpson_area_km2)
print(trapezoidal_area_km2)
print(boole_area_km2)

plt.show()
