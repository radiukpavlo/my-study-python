from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.integrate import simps, trapz


class AreaCalculator:
    """
    A class to calculate areas using different numerical integration methods.
    """

    def __init__(self, x, y):
        """
        Initialize the AreaCalculator with x and y coordinates of the contour.

        :param x: Array of x-coordinates of the contour.
        :param y: Array of y-coordinates of the contour.
        """
        self.x = x
        self.y = y

    def simpson_rule(self):
        """
        Calculate the area using Simpson's rule.

        :return: Area calculated using Simpson's rule.
        """
        return simps(self.y, self.x)

    def trapezoidal_rule(self):
        """
        Calculate the area using the Trapezoidal rule.

        :return: Area calculated using the Trapezoidal rule.
        """
        return trapz(self.y, self.x)

    def boole_rule(self):
        """
        Calculate the area using Boole's rule, which is a special case of
        higher-order Newton-Cotes formulas.

        :return: Area calculated using Boole's rule.
        """
        area = 0
        for i in range(0, len(self.x) - 1, 4):
            if i + 4 < len(self.x):
                # Boole's rule is applied to each group of 4 intervals
                area += (7 * self.y[i] + 32 * self.y[i + 1] + 12 * self.y[i + 2] +
                         32 * self.y[i + 3] + 7 * self.y[i + 4]) * (self.x[i + 4] - self.x[i]) / 90
        return area

class ImageProcessor:
    """
    A class to handle image processing tasks, such as edge detection,
    contour finding, and image masking.
    """

    def __init__(self, image_path):
        """
        Initialize the ImageProcessor with the path to an image.

        :param image_path: Path to the input image.
        """
        self.image_path = image_path
        self.image = Image.open(image_path)
        self.image_np = np.array(self.image)
        self.height, self.width = self.image_np.shape[:2]
        self.gray = cv2.cvtColor(self.image_np, cv2.COLOR_BGR2GRAY)

    def detect_edges(self):
        """
        Detect edges in the grayscale image using the Canny edge detector.

        :return: Dilated edges image, which emphasizes the edges.
        """
        edges = cv2.Canny(self.gray, 100, 200)
        dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
        return dilated_edges

    def find_largest_contour(self, edges):
        """
        Find the largest contour in the edge-detected image.

        :param edges: Image with edges detected.
        :return: The largest contour found in the image.
        """
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        return largest_contour

    def highlight_largest_region(self, largest_contour):
        """
        Highlight the largest segmented region by masking the rest of the image.

        :param largest_contour: The largest contour found in the image.
        :return: Image with only the largest region highlighted.
        """
        largest_region_mask = np.zeros_like(self.image_np)
        cv2.drawContours(largest_region_mask, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        highlighted_largest_region_image = cv2.bitwise_and(self.image_np, largest_region_mask)
        return highlighted_largest_region_image

    def extract_contour_points(self, largest_contour):
        """
        Extract x and y coordinates from the largest contour.

        :param largest_contour: The largest contour found in the image.
        :return: Tuple of x and y coordinates as numpy arrays.
        """
        contour_points = largest_contour[:, 0, :]
        x = contour_points[:, 0]
        y = contour_points[:, 1]

        # Ensure the contour forms a closed loop
        if not np.array_equal(contour_points[0], contour_points[-1]):
            x = np.append(x, x[0])
            y = np.append(y, y[0])

        return x, y

    def save_image_as_png(self, image, title, save_path):
        """
        Save an image as a PNG file.

        :param image: The image to be saved.
        :param title: The title of the image.
        :param save_path: The path to save the PNG file.
        """
        fig, ax = plt.subplots(figsize=(7.5, 7))
        ax.imshow(image)
        ax.set_title(title, fontweight='bold')
        ax.axis('off')
        fig.savefig(save_path, format="png", dpi=300, bbox_inches='tight', pad_inches=0.1)

def main():
    """
    Main function to execute the image processing and area calculation workflow.
    """
    image_path = "input_area2.jpg"
    pixels_per_km_adjusted = 24.25

    # Initialize the ImageProcessor with the image path
    processor = ImageProcessor(image_path)

    # Detect edges in the image
    edges = processor.detect_edges()

    # Find the largest contour based on the detected edges
    largest_contour = processor.find_largest_contour(edges)

    # Highlight the largest region in the image
    highlighted_image = processor.highlight_largest_region(largest_contour)

    # Extract x and y coordinates from the largest contour
    x, y = processor.extract_contour_points(largest_contour)

    # Initialize the AreaCalculator with the contour coordinates
    calculator = AreaCalculator(x, y)

    # Calculate area using different numerical methods
    simpson_area = abs(calculator.simpson_rule())
    trapezoidal_area = abs(calculator.trapezoidal_rule())
    boole_area = abs(calculator.boole_rule())

    # Convert pixel area to square kilometers using the adjusted pixels per km
    simpson_area_km2 = simpson_area / (pixels_per_km_adjusted ** 2)
    trapezoidal_area_km2 = trapezoidal_area / (pixels_per_km_adjusted ** 2)
    boole_area_km2 = boole_area / (pixels_per_km_adjusted ** 2)

    # Save the original image and the highlighted region as SVG files
    processor.save_image_as_png(processor.image_np, 'Оброблений знімок екрану', "original_image.png")
    processor.save_image_as_png(highlighted_image, 'Сегментований Краматорський район', "segmented_image.png")

    # Print the calculated areas
    print(f"\nК-сть пікселів на 1 км: {pixels_per_km_adjusted}\n")
    print(f"К-сть пікселів в сегментованій області за методом Сімпсона: {simpson_area}")
    print(f"К-сть пікселів в сегментованій області за методом Трапецій: {trapezoidal_area}")
    print(f"К-сть пікселів в сегментованій області за методом Буля:: {boole_area}\n----")
    print(f"Орієнтована площа сегментованої області в км² за методом Сімпсона: {simpson_area_km2:.2f}")
    print(f"Орієнтована площа сегментованої області в км² за методом Трапецій: {trapezoidal_area_km2:.2f}")
    print(f"Орієнтована площа сегментованої області в км² за методом Буля: {boole_area_km2:.2f}\n")

if __name__ == "__main__":
    main()
