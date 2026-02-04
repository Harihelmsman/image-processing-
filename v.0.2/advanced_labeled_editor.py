#!/usr/bin/env python3
"""
Advanced Labeled Circle Editor - Multiple effects with text labels
Optimized with comprehensive labeling support
"""

import cv2
import numpy as np
from pathlib import Path
from enum import Enum
import argparse
import json

from openpyxl import Workbook


class EditMode(Enum):
    """Available editing modes"""
    HIGHLIGHT = "highlight"
    BLUR = "blur"
    PIXELATE = "pixelate"
    DARKEN = "darken"
    GRAYSCALE = "grayscale"
    INVERT = "invert"
    OUTLINE = "outline"


class AdvancedLabeledEditor:
    """Advanced editor with multiple effects and labeling"""
    
    def __init__(self, image_path):
        self.original_image = cv2.imread(str(image_path))
        if self.original_image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self._scale_image()
        
        self.display_image = self.scaled_image.copy()
        self.output_image = self.scaled_image.copy()
        self.circles = []
        self.drawing = False
        self.center = None
        self.current_radius = 0
        self.current_label = ""
        self.label_input_mode = False
        
        # Current editing mode
        self.current_mode = EditMode.HIGHLIGHT
        
        # Effect parameters
        self.blur_kernel = 25
        self.pixelate_size = 10
        self.highlight_alpha = 0.4
        
        # Label settings
        self.label_font = cv2.FONT_HERSHEY_SIMPLEX
        self.label_scale = 0.5
        self.label_thickness = 1
        self.show_labels = True
        
        # Mode colors
        self.mode_colors = {
            EditMode.HIGHLIGHT: (0, 255, 0),
            EditMode.BLUR: (255, 0, 0),
            EditMode.PIXELATE: (0, 0, 255),
            EditMode.DARKEN: (128, 128, 128),
            EditMode.GRAYSCALE: (200, 200, 200),
            EditMode.INVERT: (255, 255, 0),
            EditMode.OUTLINE: (0, 255, 255),
        }
        
        # Window setup
        self.window_name = "Advanced Labeled Editor"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        self._print_instructions()
    
    def _print_instructions(self):
        """Print usage instructions"""
        print("\n" + "="*70)
        print("ADVANCED LABELED CIRCLE EDITOR")
        print("="*70)
        print("\nWorkflow:")
        print("  1. Select editing mode (1-7)")
        print("  2. Click and drag to draw circle")
        print("  3. Type label for the object")
        print("  4. Press ENTER to confirm")
        print("\nEditing Modes:")
        for idx, mode in enumerate(EditMode, 1):
            print(f"  {idx}: {mode.value.upper()}")
        print("\nControls:")
        print("  1-7       - Switch editing mode")
        print("  S         - Save output image and labels")
        print("  C         - Clear all circles")
        print("  U         - Undo last circle")
        print("  L         - List all labeled objects")
        print("  E         - Edit label of last circle")
        print("  T         - Toggle label visibility")
        print("  ENTER     - Confirm label (during input)")
        print("  ESC       - Skip label (during input)")
        print("  Q         - Quit application")
        print("="*70 + "\n")
    
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
        if self.label_input_mode:
            return
        
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
                self._enter_label_input_mode()
    
    def _enter_label_input_mode(self):
        """Enter label input mode"""
        self.label_input_mode = True
        self.current_label = ""
        print(f"\n→ Circle drawn in {self.current_mode.value.upper()} mode")
        print("  Enter label (or ESC to skip):")
        self._update_display_with_input()
    
    def _exit_label_input_mode(self, save=True):
        """Exit label input mode"""
        self.label_input_mode = False
        
        if save:
            label = self.current_label.strip()
            self.circles.append({
                'center': self.center,
                'radius': self.current_radius,
                'mode': self.current_mode,
                'label': label
            })
            
            if label:
                print(f"  ✓ Added: '{label}' [{self.current_mode.value}]")
            else:
                print(f"  ✓ Added unlabeled [{self.current_mode.value}]")
            
            self._apply_all_effects()
        else:
            print("  ✗ Cancelled")
        
        self.current_label = ""
        self._update_display()
    
    def _apply_effect(self, image, circle):
        """Apply specific effect to circular region"""
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, circle['center'], circle['radius'], 255, -1)
        
        mode = circle['mode']
        
        if mode == EditMode.HIGHLIGHT:
            highlighted = cv2.addWeighted(
                image, 1 - self.highlight_alpha,
                np.full_like(image, 255), self.highlight_alpha, 0
            )
            image = np.where(mask[:, :, np.newaxis] == 255, highlighted, image)
        
        elif mode == EditMode.BLUR:
            blurred = cv2.GaussianBlur(image, (self.blur_kernel, self.blur_kernel), 0)
            image = np.where(mask[:, :, np.newaxis] == 255, blurred, image)
        
        elif mode == EditMode.PIXELATE:
            h, w = image.shape[:2]
            temp = cv2.resize(
                image,
                (w // self.pixelate_size, h // self.pixelate_size),
                interpolation=cv2.INTER_LINEAR
            )
            pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
            image = np.where(mask[:, :, np.newaxis] == 255, pixelated, image)
        
        elif mode == EditMode.DARKEN:
            darkened = cv2.addWeighted(image, 0.5, np.zeros_like(image), 0.5, 0)
            image = np.where(mask[:, :, np.newaxis] == 255, darkened, image)
        
        elif mode == EditMode.GRAYSCALE:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            image = np.where(mask[:, :, np.newaxis] == 255, gray_bgr, image)
        
        elif mode == EditMode.INVERT:
            inverted = cv2.bitwise_not(image)
            image = np.where(mask[:, :, np.newaxis] == 255, inverted, image)
        
        return image
    
    def _apply_all_effects(self):
        """Apply all effects"""
        self.output_image = self.scaled_image.copy()
        
        for circle in self.circles:
            self.output_image = self._apply_effect(self.output_image, circle)
            
            # Draw circle border
            color = self.mode_colors[circle['mode']]
            cv2.circle(self.output_image, circle['center'],
                      circle['radius'], color, 2)
    
    def _draw_label(self, image, circle, number):
        """Draw label for a circle"""
        if not self.show_labels or not circle['label']:
            return
        
        center = circle['center']
        radius = circle['radius']
        label = circle['label']
        
        # Position label
        label_x = center[0] - radius
        label_y = center[1] - radius - 15
        
        # Adjust if out of bounds
        if label_y < 25:
            label_y = center[1] + radius + 30
        if label_x < 10:
            label_x = 10
        
        # Format label
        mode_short = circle['mode'].value[:3].upper()
        full_label = f"#{number} [{mode_short}] {label}"
        
        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(
            full_label, self.label_font, self.label_scale, self.label_thickness
        )
        
        # Draw background
        padding = 4
        cv2.rectangle(image,
                     (label_x - padding, label_y - text_h - padding),
                     (label_x + text_w + padding, label_y + baseline + padding),
                     (0, 0, 0), -1)
        
        # Draw border
        color = self.mode_colors[circle['mode']]
        cv2.rectangle(image,
                     (label_x - padding, label_y - text_h - padding),
                     (label_x + text_w + padding, label_y + baseline + padding),
                     color, 1)
        
        # Draw text
        cv2.putText(image, full_label, (label_x, label_y),
                   self.label_font, self.label_scale,
                   (255, 255, 255), self.label_thickness)
        
        # Draw connector line
        line_start = (label_x + text_w // 2, label_y + baseline + padding)
        cv2.line(image, line_start, center, color, 1)
    
    def _draw_typing_label(self, image, center, radius, label, mode):
        """Draw label text above circle in real-time while typing"""
        # Position label
        label_x = center[0] - radius
        label_y = center[1] - radius - 15
        
        # Adjust if out of bounds
        if label_y < 25:
            label_y = center[1] + radius + 30
        if label_x < 10:
            label_x = 10
        
        # Show current text with cursor
        mode_short = mode.value[:3].upper()
        display_label = f"[{mode_short}] {label}_"
        
        # Get text size with larger font
        (text_w, text_h), baseline = cv2.getTextSize(
            display_label, self.label_font, self.label_scale + 0.2, self.label_thickness + 1
        )
        
        # Draw background with mode color (semi-transparent effect)
        padding = 6
        color = self.mode_colors[mode]
        
        # Dark background
        cv2.rectangle(image,
                     (label_x - padding, label_y - text_h - padding),
                     (label_x + text_w + padding, label_y + baseline + padding),
                     (0, 0, 0), -1)
        
        # Bright colored border (thicker while typing)
        cv2.rectangle(image,
                     (label_x - padding, label_y - text_h - padding),
                     (label_x + text_w + padding, label_y + baseline + padding),
                     color, 2)
        
        # Draw text in bright white
        cv2.putText(image, display_label, (label_x, label_y),
                   self.label_font, self.label_scale + 0.2,
                   (255, 255, 255), self.label_thickness + 1)
        
        # Draw thicker connector line
        line_start = (label_x + text_w // 2, label_y + baseline + padding)
        cv2.line(image, line_start, center, color, 2)
    
    def _update_display(self):
        """Update display"""
        self.display_image = self.output_image.copy()
        
        # Draw labels
        for idx, circle in enumerate(self.circles):
            self._draw_label(self.display_image, circle, idx + 1)
        
        # Draw current circle
        if self.drawing and self.current_radius > 0:
            color = self.mode_colors[self.current_mode]
            cv2.circle(self.display_image, self.center,
                      self.current_radius, color, 2)
        
        # Draw UI
        self._draw_ui()
        cv2.imshow(self.window_name, self.display_image)
    
    def _update_display_with_input(self):
        """Update display during label input"""
        self.display_image = self.output_image.copy()
        
        # Draw existing labels
        for idx, circle in enumerate(self.circles):
            self._draw_label(self.display_image, circle, idx + 1)
        
        # Draw current circle
        color = self.mode_colors[self.current_mode]
        cv2.circle(self.display_image, self.center,
                  self.current_radius, color, 3)
        
        # Draw the label text ABOVE the circle in real-time while typing
        if self.current_label:
            self._draw_typing_label(self.display_image, self.center,
                                   self.current_radius, self.current_label,
                                   self.current_mode)
        
        # Draw input box at bottom
        self._draw_input_box()
        cv2.imshow(self.window_name, self.display_image)
    
    def _draw_input_box(self):
        """Draw label input box"""
        h, w = self.display_image.shape[:2]
        box_height = 70
        box_y = h - box_height
        
        # Background
        overlay = self.display_image.copy()
        cv2.rectangle(overlay, (0, box_y), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, self.display_image, 0.2, 0, self.display_image)
        
        # Border
        color = self.mode_colors[self.current_mode]
        cv2.rectangle(self.display_image, (0, box_y), (w, h), color, 2)
        
        # Text
        mode_text = f"[{self.current_mode.value.upper()}]"
        cv2.putText(self.display_image, mode_text, (10, box_y + 25),
                   self.label_font, 0.6, color, 2)
        
        prompt = "Label:"
        cv2.putText(self.display_image, prompt, (150, box_y + 25),
                   self.label_font, 0.6, (255, 255, 255), 1)
        
        input_text = self.current_label + "_"
        cv2.putText(self.display_image, input_text, (10, box_y + 55),
                   self.label_font, 0.7, (0, 255, 255), 2)
    
    def _draw_ui(self):
        """Draw UI overlay"""
        overlay = self.display_image.copy()
        
        # Mode indicator
        mode_text = f"Mode: {self.current_mode.value.upper()}"
        cv2.rectangle(overlay, (10, 10), (300, 55), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, 10), (300, 55),
                     self.mode_colors[self.current_mode], 2)
        cv2.putText(overlay, mode_text, (20, 38),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Status
        status = f"Objects: {len(self.circles)} | Labels: {'ON' if self.show_labels else 'OFF'}"
        h = self.display_image.shape[0]
        cv2.putText(overlay, status, (10, h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.addWeighted(overlay, 0.7, self.display_image, 0.3, 0, self.display_image)
    
    def _list_labels(self):
        """List all labels"""
        print("\n" + "="*70)
        print("LABELED OBJECTS")
        print("="*70)
        if not self.circles:
            print("No objects marked yet.")
        else:
            for idx, circle in enumerate(self.circles, 1):
                label = circle['label'] if circle['label'] else "(no label)"
                mode = circle['mode'].value
                print(f"#{idx}: {label}")
                print(f"     Mode: {mode}, Position: {circle['center']}, "
                      f"Radius: {circle['radius']}px")
        print("="*70 + "\n")
    
    def _edit_last_label(self):
        """Edit last circle's label"""
        if not self.circles:
            print("No objects to edit!")
            return
        
        last_circle = self.circles[-1]
        current_label = last_circle['label']
        
        print(f"\nCurrent label: '{current_label}'")
        print("Enter new label (ESC to cancel):")
        
        self.current_label = current_label
        self.label_input_mode = True
        self._update_display_with_input()
        
        while self.label_input_mode:
            key = cv2.waitKey(0) & 0xFF
            
            if key == 27:  # ESC
                print("  ✗ Edit cancelled")
                self.label_input_mode = False
                self.current_label = ""
            elif key == 13:  # ENTER
                last_circle['label'] = self.current_label.strip()
                print(f"  ✓ Updated to: '{self.current_label.strip()}'")
                self.label_input_mode = False
                self.current_label = ""
                self._apply_all_effects()
            elif key == 8:  # BACKSPACE
                self.current_label = self.current_label[:-1]
                self._update_display_with_input()
            elif 32 <= key <= 126:
                self.current_label += chr(key)
                self._update_display_with_input()
        
        self._update_display()
    def _save_excel_summary(self, image_path):
        """Create Excel summary file"""
        try:
            wb = Workbook()
            ws = wb.active
            ws.title = "Label Summary"

            # Headers
            ws.append(["Image Name", "Number of Labels", "Label Names"])

            # Prepare data
            image_name = Path(image_path).name
            labels = [c['label'] for c in self.circles if c['label']]
            label_names = ", ".join(labels) if labels else "(No labels)"
            label_count = len(labels)

            ws.append([image_name, label_count, label_names])

            excel_path = Path(image_path).with_suffix(".xlsx")
            wb.save(str(excel_path))

            print("[OK] Excel summary saved: " + str(excel_path))
        except Exception as e:
            print("[ERROR] Failed to save Excel file: " + str(e))
            import traceback
            traceback.print_exc()
    def save_output(self, image_path, labels_path=None):
        """Save output image and labels"""
        # Save image
        if self.scale_factor != 1.0:
            h, w = self.original_image.shape[:2]
            final = cv2.resize(self.output_image, (w, h),
                              interpolation=cv2.INTER_LANCZOS4)
        else:
            final = self.output_image
        
        cv2.imwrite(str(image_path), final)
        print(f"✓ Image saved: {image_path}")
        
        # Save labels
        if labels_path is None:
            labels_path = Path(image_path).with_suffix('.json')
        
        labels_data = {
            'image': str(image_path),
            'objects': []
        }
        
        for idx, circle in enumerate(self.circles, 1):
            labels_data['objects'].append({
                'id': idx,
                'label': circle['label'],
                'mode': circle['mode'].value,
                'center': circle['center'],
                'radius': circle['radius']
            })
        
        with open(labels_path, 'w') as f:
            json.dump(labels_data, f, indent=2)
        
        print(f"✓ Labels saved: {labels_path}")
        
        # Also save as text
        txt_path = Path(labels_path).with_suffix('.txt')
        with open(txt_path, 'w') as f:
            f.write("Labeled Objects\n")
            f.write("="*70 + "\n\n")
            for idx, circle in enumerate(self.circles, 1):
                label = circle['label'] if circle['label'] else "(no label)"
                f.write(f"#{idx}: {label}\n")
                f.write(f"  Mode: {circle['mode'].value}\n")
                f.write(f"  Position: {circle['center']}\n")
                f.write(f"  Radius: {circle['radius']}px\n\n")
        
        print(f"✓ Text file saved: {txt_path}")
        self._save_excel_summary(image_path)
        print("All data saved successfully.")
    
    def run(self):
        """Main loop"""
        self._update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if self.label_input_mode:
                if key == 27:  # ESC
                    self._exit_label_input_mode(save=True)
                elif key == 13:  # ENTER
                    self._exit_label_input_mode(save=True)
                elif key == 8:  # BACKSPACE
                    self.current_label = self.current_label[:-1]
                    self._update_display_with_input()
                elif 32 <= key <= 126:
                    self.current_label += chr(key)
                    self._update_display_with_input()
                continue
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_output("labeled_advanced_output.png")
            elif key == ord('c'):
                self.circles.clear()
                self.output_image = self.scaled_image.copy()
                self._update_display()
                print("✓ Cleared all")
            elif key == ord('u'):
                if self.circles:
                    removed = self.circles.pop()
                    label = removed['label'] if removed['label'] else "(unlabeled)"
                    print(f"✓ Removed: {label}")
                    self._apply_all_effects()
                    self._update_display()
            elif key == ord('l'):
                self._list_labels()
            elif key == ord('e'):
                self._edit_last_label()
            elif key == ord('t'):
                self.show_labels = not self.show_labels
                print(f"✓ Labels: {'ON' if self.show_labels else 'OFF'}")
                self._update_display()
            elif ord('1') <= key <= ord('7'):
                modes = list(EditMode)
                self.current_mode = modes[key - ord('1')]
                print(f"✓ Mode: {self.current_mode.value.upper()}")
                self._update_display()
        
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Advanced Labeled Circle Editor"
    )
    parser.add_argument("image", type=str, help="Input image path")
    parser.add_argument("--output", "-o", type=str,
                       default="labeled_advanced_output.png",
                       help="Output image path")
    
    args = parser.parse_args()
    
    try:
        editor = AdvancedLabeledEditor(args.image)
        editor.run()
        
        if editor.circles:
            editor.save_output(args.output)
        else:
            print("No objects marked.")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
