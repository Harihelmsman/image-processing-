#!/usr/bin/env python3
"""
Demo: Real-time Label Typing Visualization
Shows how labels appear above circles while typing
"""

import cv2
import numpy as np


def create_typing_demo():
    """Create visual demo of real-time label typing"""
    
    width, height = 900, 700
    
    # Create 3 frames showing the progression
    frames = []
    labels_progress = ["", "C", "Car"]
    
    for idx, label in enumerate(labels_progress):
        # Create base image
        image = np.ones((height, width, 3), dtype=np.uint8) * 240
        
        # Add gradient
        for i in range(height):
            intensity = int(50 + 100 * (i / height))
            image[i, :] = [240 - intensity//3, 240 - intensity//4, 240 - intensity//2]
        
        # Draw a sample car shape
        cv2.rectangle(image, (300, 250), (600, 450), (100, 100, 200), -1)
        cv2.rectangle(image, (300, 250), (600, 450), (50, 50, 100), 3)
        
        # Add wheels
        cv2.circle(image, (360, 450), 40, (40, 40, 40), -1)
        cv2.circle(image, (540, 450), 40, (40, 40, 40), -1)
        
        # Draw the circle around the car
        center = (450, 350)
        radius = 150
        cv2.circle(image, center, radius, (0, 255, 255), 3)
        
        # Draw title
        title = ["1. Draw Circle", "2. Start Typing", "3. Keep Typing"][idx]
        cv2.putText(image, title, (250, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (50, 50, 50), 2)
        
        # Draw the label as user types (real-time)
        if label:
            draw_realtime_label(image, center, radius, label)
        
        # Add instruction text
        if idx == 0:
            instruction = "Circle drawn - ready for label"
        elif idx == 1:
            instruction = "Label appears as you type: 'C'"
        else:
            instruction = "Label updates in real-time: 'Car'"
        
        cv2.putText(image, instruction, (150, height - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        
        # Draw input indicator
        input_box_text = f"Typing: {label}_"
        cv2.rectangle(image, (50, height - 120), (width - 50, height - 80), (0, 0, 0), -1)
        cv2.rectangle(image, (50, height - 120), (width - 50, height - 80), (0, 255, 255), 2)
        cv2.putText(image, input_box_text, (70, height - 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        frames.append(image)
    
    # Save individual frames
    for idx, frame in enumerate(frames):
        cv2.imwrite(f"demo_typing_step{idx + 1}.png", frame)
        print(f"✓ Created: demo_typing_step{idx + 1}.png")
    
    # Create side-by-side comparison
    combined = np.hstack(frames)
    combined = cv2.resize(combined, (1800, 700))
    cv2.imwrite("demo_realtime_typing_comparison.png", combined)
    print("✓ Created: demo_realtime_typing_comparison.png")
    
    # Create final result
    final_image = np.ones((height, width, 3), dtype=np.uint8) * 240
    for i in range(height):
        intensity = int(50 + 100 * (i / height))
        final_image[i, :] = [240 - intensity//3, 240 - intensity//4, 240 - intensity//2]
    
    # Draw car
    cv2.rectangle(final_image, (300, 250), (600, 450), (100, 100, 200), -1)
    cv2.rectangle(final_image, (300, 250), (600, 450), (50, 50, 100), 3)
    cv2.circle(final_image, (360, 450), 40, (40, 40, 40), -1)
    cv2.circle(final_image, (540, 450), 40, (40, 40, 40), -1)
    
    # Draw final labeled circle
    center = (450, 350)
    radius = 150
    cv2.circle(final_image, center, radius, (0, 255, 0), 2)
    
    # Draw final label (after pressing ENTER)
    draw_final_label(final_image, center, radius, "Car", 1)
    
    cv2.putText(final_image, "4. Press ENTER - Label Saved!", (200, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (50, 50, 50), 2)
    cv2.putText(final_image, "Label is now permanently attached to circle", (150, height - 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 0), 2)
    
    cv2.imwrite("demo_typing_step4.png", final_image)
    print("✓ Created: demo_typing_step4.png")
    
    print("\n" + "="*70)
    print("REAL-TIME TYPING DEMO CREATED!")
    print("="*70)
    print("\nKey Feature:")
    print("  ✅ Label text appears ABOVE the circle while you type")
    print("  ✅ Updates in real-time with each keystroke")
    print("  ✅ Visual feedback shows exactly what will be saved")
    print("\nFiles created:")
    print("  1. demo_typing_step1.png - Circle drawn")
    print("  2. demo_typing_step2.png - Start typing (shows 'C')")
    print("  3. demo_typing_step3.png - Continue typing (shows 'Car')")
    print("  4. demo_typing_step4.png - Final saved label")
    print("  5. demo_realtime_typing_comparison.png - All steps side-by-side")
    print("="*70 + "\n")


def draw_realtime_label(image, center, radius, label):
    """Draw label as it's being typed (cyan/yellow highlight)"""
    label_x = center[0] - radius
    label_y = center[1] - radius - 15
    
    if label_y < 25:
        label_y = center[1] + radius + 30
    if label_x < 10:
        label_x = 10
    
    # Show with cursor
    display_label = label + "_"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    
    (text_w, text_h), baseline = cv2.getTextSize(display_label, font, scale, thickness)
    
    padding = 6
    
    # Bright background (typing state)
    cv2.rectangle(image,
                 (label_x - padding, label_y - text_h - padding),
                 (label_x + text_w + padding, label_y + baseline + padding),
                 (0, 100, 100), -1)
    
    # Bright border
    cv2.rectangle(image,
                 (label_x - padding, label_y - text_h - padding),
                 (label_x + text_w + padding, label_y + baseline + padding),
                 (0, 255, 255), 2)
    
    # White text
    cv2.putText(image, display_label, (label_x, label_y),
               font, scale, (255, 255, 255), thickness)
    
    # Connector line
    line_start = (label_x + text_w // 2, label_y + baseline + padding)
    cv2.line(image, line_start, center, (0, 255, 255), 2)


def draw_final_label(image, center, radius, label, number):
    """Draw final saved label (green, permanent)"""
    label_x = center[0] - radius
    label_y = center[1] - radius - 15
    
    if label_y < 25:
        label_y = center[1] + radius + 30
    if label_x < 10:
        label_x = 10
    
    full_label = f"#{number}: {label}"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    
    (text_w, text_h), baseline = cv2.getTextSize(full_label, font, scale, thickness)
    
    padding = 5
    
    # Dark background (saved state)
    cv2.rectangle(image,
                 (label_x - padding, label_y - text_h - padding),
                 (label_x + text_w + padding, label_y + baseline + padding),
                 (0, 0, 0), -1)
    
    # Green border
    cv2.rectangle(image,
                 (label_x - padding, label_y - text_h - padding),
                 (label_x + text_w + padding, label_y + baseline + padding),
                 (0, 255, 0), 1)
    
    # White text
    cv2.putText(image, full_label, (label_x, label_y),
               font, scale, (255, 255, 255), thickness)
    
    # Connector line
    line_start = (label_x + text_w // 2, label_y + baseline + padding)
    cv2.line(image, line_start, center, (0, 255, 0), 1)


if __name__ == "__main__":
    print("Creating real-time typing demonstration...\n")
    create_typing_demo()
    print("\nYou can now see how labels appear in real-time!")
    print("Run the actual editor to experience it:")
    print("  python labeled_circle_editor.py demo_input.jpg")
