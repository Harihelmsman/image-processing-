#!/usr/bin/env python3
"""
Image Circle Editor with Labels - Mark circles and add text labels
Optimized version with manual labeling support
"""

import cv2
import numpy as np
from pathlib import Path
import argparse


class LabeledCircleEditor:
    """Interactive image editor for marking circles with text labels"""
    
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
        self.current_label = ""
        self.label_input_mode = False
        
        # Configuration
        self.circle_color = (0, 255, 0)  # Green
        self.circle_thickness = 2
        self.highlight_alpha = 0.3
        
        # Label settings
        self.label_font = cv2.FONT_HERSHEY_SIMPLEX
        self.label_scale = 0.6
        self.label_thickness = 2
        self.label_bg_color = (0, 0, 0)
        self.label_text_color = (255, 255, 255)
        
        # Window setup
        self.window_name = "Labeled Circle Editor"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        print("\n" + "="*60)
        print("LABELED CIRCLE EDITOR")
        print("="*60)
        print("\nInstructions:")
        print("1. Click and drag to draw a circle")
        print("2. Release mouse to finish circle")
        print("3. Type a label for the circle")
        print("4. Press ENTER to confirm label")
        print("5. Press ESC to skip label")
        print("\nControls:")
        print("  S - Save output image")
        print("  C - Clear all circles")
        print("  U - Undo last circle")
        print("  L - List all labels")
        print("  E - Edit label of last circle")
        print("  Q - Quit")
        print("="*60 + "\n")
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for circle drawing"""
        if self.label_input_mode:
            return  # Ignore mouse during label input
        
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
                # Enter label input mode
                self._enter_label_input_mode()
    
    def _enter_label_input_mode(self):
        """Enter mode to input label text"""
        self.label_input_mode = True
        self.current_label = ""
        print("\n→ Circle drawn! Enter label (or press ESC to skip):")
        self._update_display_with_input()
    
    def _exit_label_input_mode(self, save=True):
        """Exit label input mode"""
        self.label_input_mode = False
        
        if save and self.current_label.strip():
            # Save circle with label
            self.circles.append({
                'center': self.center,
                'radius': self.current_radius,
                'label': self.current_label.strip()
            })
            print(f"✓ Added: '{self.current_label.strip()}'")
        elif save and not self.current_label.strip():
            # Save without label
            self.circles.append({
                'center': self.center,
                'radius': self.current_radius,
                'label': ""
            })
            print("✓ Added circle without label")
        else:
            print("✗ Circle cancelled")
        
        self.current_label = ""
        self._apply_edits()
        self._update_display()
    
    def _update_display(self):
        """Update display with current circles and labels"""
        self.display_image = self.original_image.copy()
        
        # Draw all saved circles with labels
        for idx, circle in enumerate(self.circles):
            # Draw circle
            cv2.circle(self.display_image, circle['center'], 
                      circle['radius'], self.circle_color, self.circle_thickness)
            
            # Draw label if exists
            if circle['label']:
                self._draw_label(self.display_image, circle['center'], 
                               circle['radius'], circle['label'], idx + 1)
        
        # Draw current circle being drawn
        if self.drawing and self.current_radius > 0:
            cv2.circle(self.display_image, self.center, 
                      self.current_radius, (255, 0, 0), self.circle_thickness)
        
        cv2.imshow(self.window_name, self.display_image)
    
    def _update_display_with_input(self):
        """Update display during label input"""
        self.display_image = self.original_image.copy()
        
        # Draw all existing circles
        for idx, circle in enumerate(self.circles):
            cv2.circle(self.display_image, circle['center'], 
                      circle['radius'], self.circle_color, self.circle_thickness)
            if circle['label']:
                self._draw_label(self.display_image, circle['center'], 
                               circle['radius'], circle['label'], idx + 1)
        
        # Draw current circle being labeled
        cv2.circle(self.display_image, self.center, 
                  self.current_radius, (0, 255, 255), self.circle_thickness)
        
        # Draw the label text ABOVE the circle in real-time while typing
        if self.current_label:
            self._draw_typing_label(self.display_image, self.center, 
                                   self.current_radius, self.current_label)
        
        # Draw input box at bottom (optional, can be removed if you want)
        self._draw_input_box()
        
        cv2.imshow(self.window_name, self.display_image)
    
    def _draw_input_box(self):
        """Draw label input box at bottom of screen"""
        h, w = self.display_image.shape[:2]
        box_height = 80
        box_y = h - box_height
        
        # Semi-transparent background
        overlay = self.display_image.copy()
        cv2.rectangle(overlay, (0, box_y), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, self.display_image, 0.3, 0, self.display_image)
        
        # Border
        cv2.rectangle(self.display_image, (0, box_y), (w, h), (0, 255, 255), 2)
        
        # Prompt text
        prompt = "Enter label:"
        cv2.putText(self.display_image, prompt, (10, box_y + 30),
                   self.label_font, 0.7, (255, 255, 255), 2)
        
        # Input text with cursor
        input_text = self.current_label + "_"
        cv2.putText(self.display_image, input_text, (10, box_y + 60),
                   self.label_font, 0.8, (0, 255, 255), 2)
    
    def _draw_label(self, image, center, radius, label, number):
        """Draw label near circle"""
        # Position label above circle
        label_x = center[0] - radius
        label_y = center[1] - radius - 10
        
        # Ensure label is within image bounds
        if label_y < 20:
            label_y = center[1] + radius + 25
        if label_x < 10:
            label_x = 10
        
        # Format label with number
        full_label = f"#{number}: {label}"
        
        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(
            full_label, self.label_font, self.label_scale, self.label_thickness
        )
        
        # Draw background rectangle
        padding = 5
        cv2.rectangle(image,
                     (label_x - padding, label_y - text_h - padding),
                     (label_x + text_w + padding, label_y + baseline + padding),
                     self.label_bg_color, -1)
        
        # Draw border
        cv2.rectangle(image,
                     (label_x - padding, label_y - text_h - padding),
                     (label_x + text_w + padding, label_y + baseline + padding),
                     self.circle_color, 1)
        
        # Draw text
        cv2.putText(image, full_label, (label_x, label_y),
                   self.label_font, self.label_scale, 
                   self.label_text_color, self.label_thickness)
        
        # Draw line from label to circle
        line_start = (label_x + text_w // 2, label_y + baseline + padding)
        cv2.line(image, line_start, center, self.circle_color, 1)
    
    def _draw_typing_label(self, image, center, radius, label):
        """Draw label text above circle in real-time while typing"""
        # Position label above circle
        label_x = center[0] - radius
        label_y = center[1] - radius - 10
        
        # Ensure label is within image bounds
        if label_y < 20:
            label_y = center[1] + radius + 25
        if label_x < 10:
            label_x = 10
        
        # Show current text with cursor
        display_label = label + "_"
        
        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(
            display_label, self.label_font, self.label_scale + 0.1, self.label_thickness
        )
        
        # Draw background rectangle with highlight color
        padding = 5
        cv2.rectangle(image,
                     (label_x - padding, label_y - text_h - padding),
                     (label_x + text_w + padding, label_y + baseline + padding),
                     (0, 100, 100), -1)  # Dark cyan background while typing
        
        # Draw bright border
        cv2.rectangle(image,
                     (label_x - padding, label_y - text_h - padding),
                     (label_x + text_w + padding, label_y + baseline + padding),
                     (0, 255, 255), 2)  # Bright cyan border
        
        # Draw text in bright color
        cv2.putText(image, display_label, (label_x, label_y),
                   self.label_font, self.label_scale + 0.1, 
                   (255, 255, 255), self.label_thickness + 1)
        
        # Draw line from label to circle
        line_start = (label_x + text_w // 2, label_y + baseline + padding)
        cv2.line(image, line_start, center, (0, 255, 255), 2)
    
    def _apply_edits(self):
        """Apply highlighting effects to marked circles"""
        self.output_image = self.original_image.copy()
        
        for idx, circle in enumerate(self.circles):
            # Create mask for circle
            mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
            cv2.circle(mask, circle['center'], circle['radius'], 255, -1)
            
            # Apply highlight effect
            highlighted = cv2.addWeighted(
                self.original_image, 1 - self.highlight_alpha,
                np.full_like(self.original_image, 255), self.highlight_alpha, 0
            )
            
            # Blend highlighted region
            self.output_image = np.where(
                mask[:, :, np.newaxis] == 255,
                highlighted,
                self.output_image
            )
            
            # Draw circle border
            cv2.circle(self.output_image, circle['center'], 
                      circle['radius'], self.circle_color, self.circle_thickness)
            
            # Draw label
            if circle['label']:
                self._draw_label(self.output_image, circle['center'], 
                               circle['radius'], circle['label'], idx + 1)
    
    def _list_labels(self):
        """Print all labels"""
        print("\n" + "="*60)
        print("LABELED OBJECTS")
        print("="*60)
        if not self.circles:
            print("No circles marked yet.")
        else:
            for idx, circle in enumerate(self.circles, 1):
                label = circle['label'] if circle['label'] else "(no label)"
                center = circle['center']
                radius = circle['radius']
                print(f"#{idx}: {label}")
                print(f"     Position: ({center[0]}, {center[1]}), Radius: {radius}px")
        print("="*60 + "\n")
    
    def _edit_last_label(self):
        """Edit label of the last circle"""
        if not self.circles:
            print("No circles to edit!")
            return
        
        last_circle = self.circles[-1]
        current_label = last_circle['label']
        
        print(f"\nCurrent label: '{current_label}'")
        print("Enter new label (or press ESC to cancel):")
        
        self.current_label = current_label
        self.label_input_mode = True
        self._update_display_with_input()
        
        # Wait for input
        while self.label_input_mode:
            key = cv2.waitKey(0) & 0xFF
            
            if key == 27:  # ESC
                print("✗ Edit cancelled")
                self.label_input_mode = False
                self.current_label = ""
            elif key == 13:  # ENTER
                last_circle['label'] = self.current_label.strip()
                print(f"✓ Updated to: '{self.current_label.strip()}'")
                self.label_input_mode = False
                self.current_label = ""
                self._apply_edits()
            elif key == 8:  # BACKSPACE
                self.current_label = self.current_label[:-1]
                self._update_display_with_input()
            elif 32 <= key <= 126:  # Printable characters
                self.current_label += chr(key)
                self._update_display_with_input()
        
        self._update_display()
    
    def run(self):
        """Main application loop"""
        self._update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            # Handle label input mode
            if self.label_input_mode:
                if key == 27:  # ESC - skip label
                    self._exit_label_input_mode(save=True)
                elif key == 13:  # ENTER - confirm label
                    self._exit_label_input_mode(save=True)
                elif key == 8:  # BACKSPACE
                    self.current_label = self.current_label[:-1]
                    self._update_display_with_input()
                elif 32 <= key <= 126:  # Printable characters
                    self.current_label += chr(key)
                    self._update_display_with_input()
                continue
            
            # Normal mode controls
            if key == ord('q'):
                break
            
            elif key == ord('s'):
                output_path = self._save_output()
                print(f"✓ Saved to: {output_path}")
            
            elif key == ord('c'):
                self.circles.clear()
                self.output_image = self.original_image.copy()
                self._update_display()
                print("✓ Cleared all circles")
            
            elif key == ord('u'):
                if self.circles:
                    removed = self.circles.pop()
                    label = removed['label'] if removed['label'] else "(unlabeled)"
                    print(f"✓ Removed: {label}")
                    self._apply_edits()
                    self._update_display()
            
            elif key == ord('l'):
                self._list_labels()
            
            elif key == ord('e'):
                self._edit_last_label()
        
        cv2.destroyAllWindows()
    
    def _save_output(self):
        """Save the edited output image"""
        output_path = Path("labeled_output.png")
        counter = 1
        while output_path.exists():
            output_path = Path(f"labeled_output_{counter}.png")
            counter += 1
        
        cv2.imwrite(str(output_path), self.output_image)
        
        # Also save labels to text file
        labels_path = output_path.with_suffix('.txt')
        with open(labels_path, 'w') as f:
            f.write("Labeled Objects\n")
            f.write("="*50 + "\n\n")
            for idx, circle in enumerate(self.circles, 1):
                label = circle['label'] if circle['label'] else "(no label)"
                f.write(f"#{idx}: {label}\n")
                f.write(f"  Position: {circle['center']}\n")
                f.write(f"  Radius: {circle['radius']}px\n\n")
        
        print(f"✓ Labels saved to: {labels_path}")
        
        return output_path
    
    def get_output(self):
        """Return the edited output image and labels"""
        return self.output_image, self.circles


def main():
    parser = argparse.ArgumentParser(
        description="Labeled Circle Editor - Mark and label objects in images"
    )
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--output", "-o", type=str, default="labeled_output.png",
                       help="Output image path")
    
    args = parser.parse_args()
    
    try:
        editor = LabeledCircleEditor(args.image)
        editor.run()
        
        if editor.circles:
            cv2.imwrite(args.output, editor.get_output()[0])
            print(f"\n✓ Final output saved to: {args.output}")
        else:
            print("\nNo circles marked.")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
