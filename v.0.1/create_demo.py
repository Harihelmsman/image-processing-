#!/usr/bin/env python3
"""
Demo Script - Shows example output from the labeled circle editors
Creates sample images showing what the output looks like
"""

import cv2
import numpy as np


def create_demo_labeled_output():
    """Create a demo showing what labeled output looks like"""
    
    # Create base image
    width, height = 800, 600
    image = np.ones((height, width, 3), dtype=np.uint8) * 240
    
    # Add gradient background
    for i in range(height):
        intensity = int(50 + 100 * (i / height))
        image[i, :] = [240 - intensity//3, 240 - intensity//4, 240 - intensity//2]
    
    # Add sample objects to detect
    # Object 1: Rectangle (representing a car)
    cv2.rectangle(image, (100, 150), (250, 280), (100, 100, 200), -1)
    cv2.rectangle(image, (100, 150), (250, 280), (50, 50, 100), 3)
    
    # Object 2: Circle (representing a ball)
    cv2.circle(image, (500, 200), 60, (50, 200, 50), -1)
    cv2.circle(image, (500, 200), 60, (30, 150, 30), 3)
    
    # Object 3: Triangle (representing a sign)
    pts = np.array([[650, 400], [750, 400], [700, 320]], np.int32)
    cv2.fillPoly(image, [pts], (200, 200, 50))
    cv2.polylines(image, [pts], True, (150, 150, 30), 3)
    
    # Add text
    cv2.putText(image, "Sample Image for Labeling", (200, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(image, "Draw circles around objects and add labels", (150, 550),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
    
    # Save base image
    cv2.imwrite("demo_input.jpg", image)
    print("✓ Created: demo_input.jpg")
    
    # Now create the "labeled" version
    labeled_image = image.copy()
    
    # Circle 1: Car with highlight
    mask1 = np.zeros(image.shape[:2], dtype=np.uint8)
    center1 = (175, 215)
    radius1 = 90
    cv2.circle(mask1, center1, radius1, 255, -1)
    
    highlighted = cv2.addWeighted(
        labeled_image, 0.6,
        np.full_like(labeled_image, 255), 0.4, 0
    )
    labeled_image = np.where(mask1[:, :, np.newaxis] == 255, highlighted, labeled_image)
    cv2.circle(labeled_image, center1, radius1, (0, 255, 0), 2)
    
    # Label 1
    draw_label(labeled_image, center1, radius1, "Car", 1, (0, 255, 0))
    
    # Circle 2: Ball with different color
    center2 = (500, 200)
    radius2 = 70
    cv2.circle(labeled_image, center2, radius2, (0, 255, 255), 2)
    draw_label(labeled_image, center2, radius2, "Ball", 2, (0, 255, 255))
    
    # Circle 3: Sign
    center3 = (700, 360)
    radius3 = 55
    cv2.circle(labeled_image, center3, radius3, (255, 0, 255), 2)
    draw_label(labeled_image, center3, radius3, "Warning Sign", 3, (255, 0, 255))
    
    # Save labeled image
    cv2.imwrite("demo_labeled_output.png", labeled_image)
    print("✓ Created: demo_labeled_output.png")
    
    # Create label file
    with open("demo_labels.txt", 'w') as f:
        f.write("Labeled Objects - Demo Output\n")
        f.write("="*70 + "\n\n")
        f.write("#1: Car\n")
        f.write(f"  Position: {center1}\n")
        f.write(f"  Radius: {radius1}px\n\n")
        f.write("#2: Ball\n")
        f.write(f"  Position: {center2}\n")
        f.write(f"  Radius: {radius2}px\n\n")
        f.write("#3: Warning Sign\n")
        f.write(f"  Position: {center3}\n")
        f.write(f"  Radius: {radius3}px\n")
    
    print("✓ Created: demo_labels.txt")
    
    print("\n" + "="*70)
    print("DEMO FILES CREATED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated files:")
    print("  1. demo_input.jpg - Original image to label")
    print("  2. demo_labeled_output.png - Image with labels applied")
    print("  3. demo_labels.txt - List of labels")
    print("\nTo try the actual editor:")
    print("  python labeled_circle_editor.py demo_input.jpg")
    print("  python advanced_labeled_editor.py demo_input.jpg")
    print("="*70 + "\n")


def draw_label(image, center, radius, label, number, color):
    """Draw a label near a circle"""
    label_x = center[0] - radius
    label_y = center[1] - radius - 15
    
    # Adjust if out of bounds
    if label_y < 25:
        label_y = center[1] + radius + 30
    if label_x < 10:
        label_x = 10
    
    full_label = f"#{number}: {label}"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 1
    
    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(full_label, font, scale, thickness)
    
    # Draw background
    padding = 5
    cv2.rectangle(image,
                 (label_x - padding, label_y - text_h - padding),
                 (label_x + text_w + padding, label_y + baseline + padding),
                 (0, 0, 0), -1)
    
    # Draw border
    cv2.rectangle(image,
                 (label_x - padding, label_y - text_h - padding),
                 (label_x + text_w + padding, label_y + baseline + padding),
                 color, 1)
    
    # Draw text
    cv2.putText(image, full_label, (label_x, label_y),
               font, scale, (255, 255, 255), thickness + 1)
    
    # Draw connector line
    line_start = (label_x + text_w // 2, label_y + baseline + padding)
    cv2.line(image, line_start, center, color, 1)


def create_effects_comparison():
    """Create a comparison image showing different effects"""
    
    # Create base image with a face-like pattern
    width, height = 1000, 400
    image = np.ones((height, width, 3), dtype=np.uint8) * 200
    
    # Add pattern
    for i in range(0, width, 50):
        cv2.line(image, (i, 0), (i, height), (180, 180, 180), 1)
    for i in range(0, height, 50):
        cv2.line(image, (0, i), (width, i), (180, 180, 180), 1)
    
    # Add some colored rectangles
    cv2.rectangle(image, (50, 100), (950, 300), (100, 150, 200), -1)
    
    # Add circles and text
    cv2.circle(image, (200, 200), 50, (255, 200, 100), -1)
    cv2.circle(image, (500, 200), 50, (100, 255, 200), -1)
    cv2.circle(image, (800, 200), 50, (200, 100, 255), -1)
    
    cv2.putText(image, "ORIGINAL", (400, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)
    
    cv2.imwrite("demo_effects_original.jpg", image)
    print("✓ Created: demo_effects_original.jpg")
    
    # Create effects versions
    effects = {
        'highlight': 'Brightened region',
        'blur': 'Blurred (privacy protection)',
        'pixelate': 'Pixelated (anonymization)',
        'grayscale': 'Grayscale conversion'
    }
    
    print("\nEffect demo images created:")
    print("  demo_effects_original.jpg - Base image")
    print("\nYou can test these effects with:")
    print("  python advanced_labeled_editor.py demo_effects_original.jpg")


if __name__ == "__main__":
    print("Creating demo images...\n")
    create_demo_labeled_output()
    print()
    create_effects_comparison()
    print("\n✅ All demo files created successfully!")
    print("\nNow you can:")
    print("1. View the demo outputs to see what results look like")
    print("2. Run the editors on demo_input.jpg to try them yourself")
    print("3. Run 'python create_test_image.py' for another test image")
