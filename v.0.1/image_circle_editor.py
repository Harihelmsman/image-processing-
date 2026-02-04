#!/usr/bin/env python3
"""
Image Circle Editor - Manual highlight and mark circles for editing
Optimized version with efficient image processing
"""

import cv2
import numpy as np
from pathlib import Path
import argparse


class ImageCircleEditor:
    """Interactive image editor for marking and highlighting circular regions"""
    
    def __init__(self, image_path):
        self.original_image = cv2.imread(str(image_path))
        if self.original_image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.display_image = self.original_image.copy()
        self.output_image = self.original_image.copy()
        self.circles = []
        self.drawing = False
        self.center = None
        self.current_radius = 0
        
        # Configuration
        self.circle_color = (0, 255, 0)  # Green
        self.circle_thickness = 2
        self.highlight_alpha = 0.3
        
        # Window setup
        self.window_name = "Image Circle Editor"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for circle drawing"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start drawing circle
            self.drawing = True
            self.center = (x, y)
            self.current_radius = 0
            
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # Update circle radius while dragging
            self.current_radius = int(np.sqrt((x - self.center[0])**2 + 
                                             (y - self.center[1])**2))
            self._update_display()
            
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            # Finish drawing circle
            self.drawing = False
            if self.current_radius > 5:  # Minimum radius threshold
                self.circles.append({
                    'center': self.center,
                    'radius': self.current_radius
                })
                self._apply_edits()
            self._update_display()
    
    def _update_display(self):
        """Update display with current circles"""
        self.display_image = self.original_image.copy()
        
        # Draw all saved circles
        for circle in self.circles:
            cv2.circle(self.display_image, circle['center'], 
                      circle['radius'], self.circle_color, self.circle_thickness)
        
        # Draw current circle being drawn
        if self.drawing and self.current_radius > 0:
            cv2.circle(self.display_image, self.center, 
                      self.current_radius, (255, 0, 0), self.circle_thickness)
        
        cv2.imshow(self.window_name, self.display_image)
    
    def _apply_edits(self):
        """Apply highlighting effects to marked circles"""
        self.output_image = self.original_image.copy()
        
        for circle in self.circles:
            # Create mask for circle
            mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
            cv2.circle(mask, circle['center'], circle['radius'], 255, -1)
            
            # Apply highlight effect (brighten the region)
            highlighted = cv2.addWeighted(
                self.original_image, 1 - self.highlight_alpha,
                np.full_like(self.original_image, 255), self.highlight_alpha, 0
            )
            
            # Blend highlighted region with original
            self.output_image = np.where(
                mask[:, :, np.newaxis] == 255,
                highlighted,
                self.output_image
            )
            
            # Draw circle border
            cv2.circle(self.output_image, circle['center'], 
                      circle['radius'], self.circle_color, self.circle_thickness)
    
    def _add_instructions(self, image):
        """Add instruction text to image"""
        instructions = [
            "Click and drag to draw circles",
            "Press 's' to save output",
            "Press 'c' to clear all circles",
            "Press 'u' to undo last circle",
            "Press 'h' to toggle highlights",
            "Press 'q' to quit"
        ]
        
        y_offset = 30
        for idx, text in enumerate(instructions):
            cv2.putText(image, text, (10, y_offset + idx * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, text, (10, y_offset + idx * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        
        return image
    
    def run(self):
        """Main application loop"""
        print("Image Circle Editor Started")
        print("=" * 50)
        print("Controls:")
        print("  - Click and drag to draw circles")
        print("  - Press 's' to save output")
        print("  - Press 'c' to clear all circles")
        print("  - Press 'u' to undo last circle")
        print("  - Press 'q' to quit")
        print("=" * 50)
        
        self._update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                # Quit application
                break
            
            elif key == ord('s'):
                # Save output image
                output_path = self._save_output()
                print(f"Output saved to: {output_path}")
            
            elif key == ord('c'):
                # Clear all circles
                self.circles.clear()
                self.output_image = self.original_image.copy()
                self._update_display()
                print("Cleared all circles")
            
            elif key == ord('u'):
                # Undo last circle
                if self.circles:
                    self.circles.pop()
                    self._apply_edits()
                    self._update_display()
                    print("Undone last circle")
            
            elif key == ord('h'):
                # Toggle highlight effect
                self.highlight_alpha = 0.0 if self.highlight_alpha > 0 else 0.3
                self._apply_edits()
                self._update_display()
                print(f"Highlight alpha: {self.highlight_alpha}")
        
        cv2.destroyAllWindows()
    
    def _save_output(self):
        """Save the edited output image"""
        output_path = Path("output_edited.png")
        counter = 1
        while output_path.exists():
            output_path = Path(f"output_edited_{counter}.png")
            counter += 1
        
        cv2.imwrite(str(output_path), self.output_image)
        return output_path
    
    def get_output(self):
        """Return the edited output image"""
        return self.output_image


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Image Circle Editor - Mark and highlight circular regions"
    )
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--output", "-o", type=str, default="output_edited.png",
                       help="Output image path (default: output_edited.png)")
    
    args = parser.parse_args()
    
    try:
        editor = ImageCircleEditor(args.image)
        editor.run()
        
        # Save final output
        if editor.circles:
            cv2.imwrite(args.output, editor.get_output())
            print(f"\nFinal output saved to: {args.output}")
        else:
            print("\nNo circles marked. Output not saved.")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
