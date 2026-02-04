#!/usr/bin/env python3
"""
Advanced Image Circle Editor - Multiple editing effects
Optimized for performance with various circle-based effects
"""

import cv2
import numpy as np
from pathlib import Path
from enum import Enum
import argparse


class EditMode(Enum):
    """Available editing modes"""
    HIGHLIGHT = "highlight"
    BLUR = "blur"
    PIXELATE = "pixelate"
    DARKEN = "darken"
    GRAYSCALE = "grayscale"
    INVERT = "invert"
    OUTLINE = "outline"


class AdvancedCircleEditor:
    """Advanced image editor with multiple circular region effects"""
    
    def __init__(self, image_path):
        self.original_image = cv2.imread(str(image_path))
        if self.original_image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Scale image if too large for display
        self._scale_image()
        
        self.display_image = self.scaled_image.copy()
        self.output_image = self.scaled_image.copy()
        self.circles = []
        self.drawing = False
        self.center = None
        self.current_radius = 0
        
        # Current editing mode
        self.current_mode = EditMode.HIGHLIGHT
        
        # Effect parameters
        self.blur_kernel = 25
        self.pixelate_size = 10
        self.highlight_alpha = 0.4
        
        # Colors for different modes
        self.mode_colors = {
            EditMode.HIGHLIGHT: (0, 255, 0),    # Green
            EditMode.BLUR: (255, 0, 0),         # Blue
            EditMode.PIXELATE: (0, 0, 255),     # Red
            EditMode.DARKEN: (128, 128, 128),   # Gray
            EditMode.GRAYSCALE: (200, 200, 200),# Light gray
            EditMode.INVERT: (255, 255, 0),     # Cyan
            EditMode.OUTLINE: (0, 255, 255),    # Yellow
        }
        
        # Window setup
        self.window_name = "Advanced Circle Editor"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
    
    def _scale_image(self):
        """Scale image for display if needed"""
        max_height = 900
        max_width = 1400
        
        height, width = self.original_image.shape[:2]
        self.scale_factor = 1.0
        
        if height > max_height or width > max_width:
            scale_h = max_height / height
            scale_w = max_width / width
            self.scale_factor = min(scale_h, scale_w)
            
            new_width = int(width * self.scale_factor)
            new_height = int(height * self.scale_factor)
            
            self.scaled_image = cv2.resize(
                self.original_image, 
                (new_width, new_height),
                interpolation=cv2.INTER_AREA
            )
        else:
            self.scaled_image = self.original_image.copy()
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.center = (x, y)
            self.current_radius = 0
            
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current_radius = int(np.sqrt((x - self.center[0])**2 + 
                                             (y - self.center[1])**2))
            self._update_display()
            
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            if self.current_radius > 5:
                self.circles.append({
                    'center': self.center,
                    'radius': self.current_radius,
                    'mode': self.current_mode
                })
                self._apply_all_effects()
            self._update_display()
    
    def _apply_effect(self, image, circle):
        """Apply specific effect to a circular region"""
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, circle['center'], circle['radius'], 255, -1)
        
        mode = circle['mode']
        
        if mode == EditMode.HIGHLIGHT:
            # Brighten region
            highlighted = cv2.addWeighted(
                image, 1 - self.highlight_alpha,
                np.full_like(image, 255), self.highlight_alpha, 0
            )
            image = np.where(mask[:, :, np.newaxis] == 255, highlighted, image)
        
        elif mode == EditMode.BLUR:
            # Gaussian blur
            blurred = cv2.GaussianBlur(image, (self.blur_kernel, self.blur_kernel), 0)
            image = np.where(mask[:, :, np.newaxis] == 255, blurred, image)
        
        elif mode == EditMode.PIXELATE:
            # Pixelate effect
            h, w = image.shape[:2]
            temp = cv2.resize(
                image, 
                (w // self.pixelate_size, h // self.pixelate_size),
                interpolation=cv2.INTER_LINEAR
            )
            pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
            image = np.where(mask[:, :, np.newaxis] == 255, pixelated, image)
        
        elif mode == EditMode.DARKEN:
            # Darken region
            darkened = cv2.addWeighted(image, 0.5, np.zeros_like(image), 0.5, 0)
            image = np.where(mask[:, :, np.newaxis] == 255, darkened, image)
        
        elif mode == EditMode.GRAYSCALE:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            image = np.where(mask[:, :, np.newaxis] == 255, gray_bgr, image)
        
        elif mode == EditMode.INVERT:
            # Invert colors
            inverted = cv2.bitwise_not(image)
            image = np.where(mask[:, :, np.newaxis] == 255, inverted, image)
        
        elif mode == EditMode.OUTLINE:
            # Just outline, no fill effect
            pass
        
        return image
    
    def _apply_all_effects(self):
        """Apply all circle effects to output image"""
        self.output_image = self.scaled_image.copy()
        
        for circle in self.circles:
            self.output_image = self._apply_effect(self.output_image, circle)
            
            # Draw circle border
            color = self.mode_colors[circle['mode']]
            cv2.circle(self.output_image, circle['center'], 
                      circle['radius'], color, 2)
    
    def _update_display(self):
        """Update display with UI elements"""
        self.display_image = self.output_image.copy()
        
        # Draw current circle being drawn
        if self.drawing and self.current_radius > 0:
            color = self.mode_colors[self.current_mode]
            cv2.circle(self.display_image, self.center, 
                      self.current_radius, color, 2)
        
        # Add UI overlay
        self._draw_ui()
        
        cv2.imshow(self.window_name, self.display_image)
    
    def _draw_ui(self):
        """Draw user interface overlay"""
        overlay = self.display_image.copy()
        
        # Draw mode indicator
        mode_text = f"Mode: {self.current_mode.value.upper()}"
        cv2.rectangle(overlay, (10, 10), (300, 60), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, 10), (300, 60), 
                     self.mode_colors[self.current_mode], 2)
        cv2.putText(overlay, mode_text, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw controls
        controls = [
            "1-7: Change mode | S: Save | C: Clear | U: Undo | Q: Quit",
            f"Circles marked: {len(self.circles)}"
        ]
        
        y_pos = self.display_image.shape[0] - 60
        for idx, text in enumerate(controls):
            cv2.putText(overlay, text, (10, y_pos + idx * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Blend overlay
        cv2.addWeighted(overlay, 0.7, self.display_image, 0.3, 0, self.display_image)
    
    def change_mode(self, mode_number):
        """Change editing mode"""
        modes = list(EditMode)
        if 0 <= mode_number < len(modes):
            self.current_mode = modes[mode_number]
            print(f"Mode changed to: {self.current_mode.value}")
            self._update_display()
    
    def save_output(self, path):
        """Save output image"""
        # Scale back to original size if needed
        if self.scale_factor != 1.0:
            original_height, original_width = self.original_image.shape[:2]
            final_output = cv2.resize(
                self.output_image,
                (original_width, original_height),
                interpolation=cv2.INTER_LANCZOS4
            )
        else:
            final_output = self.output_image
        
        cv2.imwrite(str(path), final_output)
        print(f"Saved to: {path}")
    
    def run(self):
        """Main application loop"""
        print("\n" + "="*60)
        print("ADVANCED CIRCLE EDITOR")
        print("="*60)
        print("\nEditing Modes:")
        for idx, mode in enumerate(EditMode, 1):
            print(f"  {idx}: {mode.value.upper()}")
        print("\nControls:")
        print("  - Click and drag to mark circular regions")
        print("  - 1-7: Switch editing mode")
        print("  - S: Save output image")
        print("  - C: Clear all circles")
        print("  - U: Undo last circle")
        print("  - Q: Quit application")
        print("="*60 + "\n")
        
        self._update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_output("output_advanced.png")
            elif key == ord('c'):
                self.circles.clear()
                self.output_image = self.scaled_image.copy()
                self._update_display()
                print("Cleared all circles")
            elif key == ord('u'):
                if self.circles:
                    self.circles.pop()
                    self._apply_all_effects()
                    self._update_display()
                    print("Undone last circle")
            elif ord('1') <= key <= ord('7'):
                self.change_mode(key - ord('1'))
        
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Advanced Image Circle Editor with multiple effects"
    )
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--output", "-o", type=str, default="output_advanced.png",
                       help="Output image path")
    
    args = parser.parse_args()
    
    try:
        editor = AdvancedCircleEditor(args.image)
        editor.run()
        
        if editor.circles:
            editor.save_output(args.output)
        else:
            print("No circles marked.")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
