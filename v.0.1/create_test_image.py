#!/usr/bin/env python3
"""
Test Image Generator - Creates sample images for testing the circle editor
"""

import cv2
import numpy as np


def create_test_image(width=800, height=600, filename="test_image.jpg"):
    """Create a colorful test image with various elements"""
    
    # Create base image with gradient background
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Gradient background
    for i in range(height):
        intensity = int(255 * (i / height))
        image[i, :] = [intensity // 2, intensity // 3, 200 - intensity // 2]
    
    # Add colored rectangles
    cv2.rectangle(image, (50, 50), (250, 200), (0, 255, 0), -1)
    cv2.rectangle(image, (300, 50), (500, 200), (255, 0, 0), -1)
    cv2.rectangle(image, (550, 50), (750, 200), (0, 0, 255), -1)
    
    # Add circles
    cv2.circle(image, (150, 350), 60, (255, 255, 0), -1)
    cv2.circle(image, (400, 350), 60, (0, 255, 255), -1)
    cv2.circle(image, (650, 350), 60, (255, 0, 255), -1)
    
    # Add text
    cv2.putText(image, "TEST IMAGE", (250, 500),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    # Add some "sensitive" information areas to blur
    cv2.putText(image, "Secret Data 123-45-6789", (50, 550),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add patterns
    for i in range(10):
        x = 50 + i * 70
        y = 450
        cv2.line(image, (x, y), (x + 50, y - 30), (255, 255, 255), 2)
    
    # Save image
    cv2.imwrite(filename, image)
    print(f"Test image created: {filename}")
    print(f"Size: {width}x{height}")
    print("\nSuggestions for testing:")
    print("1. Mark the green rectangle with HIGHLIGHT mode")
    print("2. Mark 'Secret Data' text with BLUR mode")
    print("3. Mark the yellow circle with PIXELATE mode")
    print("4. Mark the blue rectangle with GRAYSCALE mode")
    
    return image


if __name__ == "__main__":
    create_test_image()
    print("\nRun the editor with:")
    print("  python image_circle_editor.py test_image.jpg")
    print("  python advanced_circle_editor.py test_image.jpg")
