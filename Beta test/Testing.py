#!/usr/bin/env python3
"""
Batch Advanced Labeled Editor - Process entire folders of images
Navigate with A (previous) and D (next) keys
Zoom with mouse wheel
Saves output to separate folder with labels visible
"""

import cv2
import numpy as np
from pathlib import Path
from enum import Enum
import argparse
import json
from datetime import datetime


class EditMode(Enum):
    """Available editing modes"""
    HIGHLIGHT = "highlight"
    BLUR = "blur"
    PIXELATE = "pixelate"
    DARKEN = "darken"
    GRAYSCALE = "grayscale"
    INVERT = "invert"
    OUTLINE = "outline"


# ── Typography constants ────────────────────────────────────────────────────────
FONT       = cv2.FONT_HERSHEY_DUPLEX   # smoother than SIMPLEX
AA         = cv2.LINE_AA               # anti-aliased drawing
CANVAS_W   = 1440
CANVAS_H   = 900
CANVAS_BG  = (18, 18, 18)             # near-black
CANVAS_PAD = 40                        # pixels kept clear around the image


class BatchLabeledEditor:
    """Batch editor for processing multiple images in a folder"""

    def __init__(self, input_folder, output_folder=None):
        self.input_folder = Path(input_folder)

        if output_folder:
            self.output_folder = Path(output_folder)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_folder = self.input_folder.parent / f"labeled_output_{timestamp}"

        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.image_files = self._load_image_files()
        if not self.image_files:
            raise ValueError(f"No images found in {input_folder}")

        self.current_index = 0
        self.total_images = len(self.image_files)

        # Editor state
        self.original_image = None
        self.scaled_image = None
        self.display_image = None
        self.output_image = None
        self.circles = []
        self.drawing = False
        self.center = None
        self.current_radius = 0
        self.current_label = ""
        self.label_input_mode = False

        # Canvas / zoom / pan
        self.canvas_w = CANVAS_W
        self.canvas_h = CANVAS_H
        self.zoom_level = 1.0
        self.min_zoom = 0.5
        self.max_zoom = 8.0
        self.pan_x = 0
        self.pan_y = 0
        self.is_panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0

        # Current editing mode
        self.current_mode = EditMode.HIGHLIGHT

        # Effect parameters
        self.blur_kernel = 25
        self.pixelate_size = 10
        self.highlight_alpha = 0.4

        # Label settings
        self.label_scale = 0.55
        self.label_thickness = 1
        self.show_labels = True

        self.mode_colors = {
            EditMode.HIGHLIGHT: (0, 255, 0),
            EditMode.BLUR:      (255, 80, 80),
            EditMode.PIXELATE:  (80, 80, 255),
            EditMode.DARKEN:    (160, 160, 160),
            EditMode.GRAYSCALE: (210, 210, 210),
            EditMode.INVERT:    (255, 255, 0),
            EditMode.OUTLINE:   (0, 220, 255),
        }

        self.saved_status = {}
        self.image_states = {}

        self.window_name = "Batch Labeled Editor"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, CANVAS_W, CANVAS_H)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        self._load_current_image()
        self._print_instructions()

    # ── File helpers ────────────────────────────────────────────────────────────

    def _load_image_files(self):
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        files = []
        for ext in extensions:
            files.extend(self.input_folder.glob(f'*{ext}'))
        return sorted(files)

    # ── Image loading ───────────────────────────────────────────────────────────

    def _load_current_image(self):
        current_file = self.image_files[self.current_index]
        self.original_image = cv2.imread(str(current_file))
        if self.original_image is None:
            print(f"Error loading: {current_file}")
            return False

        self._scale_image()

        # Reset view
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0

        # Restore state from JSON if available
        json_path = self.output_folder / current_file.with_suffix('.json').name
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                self.circles = []
                for obj in data.get('objects', []):
                    mode = EditMode(obj.get('mode', 'highlight'))
                    self.circles.append({
                        'center': tuple(obj['center']),
                        'radius': obj['radius'],
                        'mode':   mode,
                        'label':  obj.get('label', '')
                    })
                print(f"\n✓ Loaded: {current_file.name} ({self.current_index+1}/{self.total_images})"
                      f" — RESTORED ({len(self.circles)} objects)")
            except Exception as e:
                print(f"Error loading JSON: {e}")
                self.circles = []
        elif current_file.name in self.image_states:
            self.circles = self.image_states[current_file.name]['circles'].copy()
            print(f"\n✓ Loaded: {current_file.name} ({self.current_index+1}/{self.total_images}) — from memory")
        else:
            self.circles = []
            print(f"\nLoaded: {current_file.name} ({self.current_index+1}/{self.total_images})")

        self.drawing = False
        self.label_input_mode = False
        self.current_label = ""

        self.output_image = self.scaled_image.copy()
        if self.circles:
            self._apply_all_effects()
        self.display_image = self.output_image.copy()
        return True

    def _scale_image(self):
        """Scale image to fit inside the canvas with padding."""
        max_w = self.canvas_w - CANVAS_PAD * 2
        max_h = self.canvas_h - CANVAS_PAD * 2
        h, w = self.original_image.shape[:2]
        self.scale_factor = 1.0

        if h > max_h or w > max_w:
            self.scale_factor = min(max_h / h, max_w / w)
            new_w = int(w * self.scale_factor)
            new_h = int(h * self.scale_factor)
            self.scaled_image = cv2.resize(
                self.original_image, (new_w, new_h),
                interpolation=cv2.INTER_AREA
            )
        else:
            self.scaled_image = self.original_image.copy()

    # ── Coordinate transforms ───────────────────────────────────────────────────

    def _img_offset(self):
        """Top-left corner of the (zoomed) image on the canvas."""
        img_h, img_w = self.scaled_image.shape[:2]
        zoomed_w = int(img_w * self.zoom_level)
        zoomed_h = int(img_h * self.zoom_level)
        ox = (self.canvas_w - zoomed_w) // 2 + self.pan_x
        oy = (self.canvas_h - zoomed_h) // 2 + self.pan_y
        return ox, oy

    def _screen_to_image_coords(self, sx, sy):
        ox, oy = self._img_offset()
        return int((sx - ox) / self.zoom_level), int((sy - oy) / self.zoom_level)

    def _image_to_screen_coords(self, ix, iy):
        ox, oy = self._img_offset()
        return int(ix * self.zoom_level + ox), int(iy * self.zoom_level + oy)

    # ── Mouse callback ──────────────────────────────────────────────────────────

    def _mouse_callback(self, event, x, y, flags, param):
        # Zoom with mouse wheel
        if event == cv2.EVENT_MOUSEWHEEL:
            factor = 1.1 if flags > 0 else 0.9
            new_zoom = max(self.min_zoom, min(self.max_zoom, self.zoom_level * factor))
            ix, iy = self._screen_to_image_coords(x, y)
            self.zoom_level = new_zoom
            img_h, img_w = self.scaled_image.shape[:2]
            zw = int(img_w * self.zoom_level)
            zh = int(img_h * self.zoom_level)
            self.pan_x = int(x - ix * self.zoom_level - (self.canvas_w - zw) // 2)
            self.pan_y = int(y - iy * self.zoom_level - (self.canvas_h - zh) // 2)
            self._update_display()
            return

        # Pan with right click
        if event == cv2.EVENT_RBUTTONDOWN:
            self.is_panning = True
            self.pan_start_x = x - self.pan_x
            self.pan_start_y = y - self.pan_y
            return
        if event == cv2.EVENT_RBUTTONUP:
            self.is_panning = False
            return
        if event == cv2.EVENT_MOUSEMOVE and self.is_panning:
            self.pan_x = x - self.pan_start_x
            self.pan_y = y - self.pan_start_y
            self._update_display()
            return

        if self.label_input_mode or self.is_panning:
            return

        # Draw circles with left click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.center = self._screen_to_image_coords(x, y)
            self.current_radius = 0
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            ix, iy = self._screen_to_image_coords(x, y)
            self.current_radius = int(np.hypot(ix - self.center[0], iy - self.center[1]))
            self._update_display()
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            if self.current_radius > 5:
                self._enter_label_input_mode()

    # ── Label input ─────────────────────────────────────────────────────────────

    def _enter_label_input_mode(self):
        self.label_input_mode = True
        self.current_label = ""
        print(f"\n→ Circle drawn [{self.current_mode.value.upper()}]  — type label, ENTER to confirm, ESC to skip:")
        self._update_display_with_input()

    def _exit_label_input_mode(self, save=True):
        self.label_input_mode = False
        if save:
            label = self.current_label.strip()
            self.circles.append({
                'center': self.center,
                'radius': self.current_radius,
                'mode':   self.current_mode,
                'label':  label
            })
            tag = f"'{label}'" if label else "(unlabeled)"
            print(f"  ✓ Added {tag} [{self.current_mode.value}]")
            self._apply_all_effects()
        else:
            print("  ✗ Cancelled")
        self.current_label = ""
        self._update_display()

    # ── Effect application ──────────────────────────────────────────────────────

    def _apply_effect(self, image, circle):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, circle['center'], circle['radius'], 255, -1)
        mode = circle['mode']

        if mode == EditMode.HIGHLIGHT:
            lit = cv2.addWeighted(image, 1 - self.highlight_alpha,
                                  np.full_like(image, 255), self.highlight_alpha, 0)
            image = np.where(mask[:, :, np.newaxis] == 255, lit, image)

        elif mode == EditMode.BLUR:
            blurred = cv2.GaussianBlur(image, (self.blur_kernel, self.blur_kernel), 0)
            image = np.where(mask[:, :, np.newaxis] == 255, blurred, image)

        elif mode == EditMode.PIXELATE:
            h, w = image.shape[:2]
            small = cv2.resize(image, (w // self.pixelate_size, h // self.pixelate_size),
                               interpolation=cv2.INTER_NEAREST)
            pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            image = np.where(mask[:, :, np.newaxis] == 255, pixelated, image)

        elif mode == EditMode.DARKEN:
            darkened = cv2.addWeighted(image, 0.5, np.zeros_like(image), 0.5, 0)
            image = np.where(mask[:, :, np.newaxis] == 255, darkened, image)

        elif mode == EditMode.GRAYSCALE:
            gray_bgr = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            image = np.where(mask[:, :, np.newaxis] == 255, gray_bgr, image)

        elif mode == EditMode.INVERT:
            inverted = cv2.bitwise_not(image)
            image = np.where(mask[:, :, np.newaxis] == 255, inverted, image)

        return image

    def _apply_all_effects(self):
        self.output_image = self.scaled_image.copy()
        for circle in self.circles:
            self.output_image = self._apply_effect(self.output_image, circle)
            color = self.mode_colors[circle['mode']]
            cv2.circle(self.output_image, circle['center'], circle['radius'], color, 2, AA)

    # ── Label drawing ───────────────────────────────────────────────────────────

    def _draw_label(self, image, circle, number):
        if not self.show_labels or not circle['label']:
            return
        center, radius, label = circle['center'], circle['radius'], circle['label']
        mode_short = circle['mode'].value[:3].upper()
        full_label = f"#{number} [{mode_short}] {label}"

        (tw, th), bl = cv2.getTextSize(full_label, FONT, self.label_scale, self.label_thickness)
        pad = 5
        total_w, total_h = tw + 2 * pad, th + bl + 2 * pad
        img_h, img_w = image.shape[:2]

        positions = [
            (center[0] - radius,            center[1] - radius - total_h - 10),
            (center[0] - radius,            center[1] + radius + 20),
            (center[0] - radius - total_w - 10, center[1] - total_h // 2),
            (center[0] + radius + 10,       center[1] - total_h // 2),
            (center[0] + radius + 10,       center[1] - radius - total_h - 10),
            (center[0] + radius + 10,       center[1] + radius + 20),
            (center[0] - radius - total_w - 10, center[1] - radius - total_h - 10),
            (center[0] - radius - total_w - 10, center[1] + radius + 20),
        ]

        lx, ly = None, None
        for px, py in positions:
            if (px >= pad and py >= th + pad and
                    px + total_w <= img_w - pad and
                    py + bl + pad <= img_h - pad):
                lx, ly = px, py + th
                break

        if lx is None:
            lx = max(pad, min(center[0] - radius, img_w - total_w - pad))
            ly = max(th + pad, min(center[1] - radius - 10, img_h - bl - pad))
            if ly <= th + pad:
                ly = center[1] + radius + th + 10

        lx = int(max(pad, min(lx, img_w - total_w - pad)))
        ly = int(max(th + pad, min(ly, img_h - bl - pad)))

        color = self.mode_colors[circle['mode']]
        # Background
        cv2.rectangle(image, (lx - pad, ly - th - pad), (lx + tw + pad, ly + bl + pad),
                      (10, 10, 10), -1)
        cv2.rectangle(image, (lx - pad, ly - th - pad), (lx + tw + pad, ly + bl + pad),
                      color, 1, AA)
        cv2.putText(image, full_label, (lx, ly), FONT, self.label_scale,
                    (255, 255, 255), self.label_thickness, AA)
        cv2.line(image, (lx + tw // 2, ly + bl + pad), center, color, 1, AA)

    def _draw_typing_label(self, image, center, radius, label, mode):
        mode_short = mode.value[:3].upper()
        display = f"[{mode_short}] {label}_"
        sc = self.label_scale + 0.15
        th_w = self.label_thickness + 1

        (tw, th), bl = cv2.getTextSize(display, FONT, sc, th_w)
        pad = 6
        total_w = tw + 2 * pad
        img_h, img_w = image.shape[:2]

        positions = [
            (center[0] - radius,        center[1] - radius - th - 2 * pad - 10),
            (center[0] - radius,        center[1] + radius + 20),
            (center[0] - radius - total_w - 10, center[1] - (th + 2 * pad) // 2),
            (center[0] + radius + 10,   center[1] - (th + 2 * pad) // 2),
        ]

        lx, ly = None, None
        for px, py in positions:
            if (px >= pad and py >= th + pad and
                    px + total_w <= img_w - pad and
                    py + bl + pad <= img_h - pad):
                lx, ly = px, py + th
                break

        if lx is None:
            lx = max(pad, min(center[0] - radius, img_w - total_w - pad))
            ly = max(th + pad, min(center[1] - radius - 10, img_h - bl - pad))
            if ly <= th + pad:
                ly = center[1] + radius + th + 20

        lx = int(max(pad, min(lx, img_w - total_w - pad)))
        ly = int(max(th + pad, min(ly, img_h - bl - pad)))
        color = self.mode_colors[mode]

        cv2.rectangle(image, (lx - pad, ly - th - pad), (lx + tw + pad, ly + bl + pad),
                      (10, 10, 10), -1)
        cv2.rectangle(image, (lx - pad, ly - th - pad), (lx + tw + pad, ly + bl + pad),
                      color, 2, AA)
        cv2.putText(image, display, (lx, ly), FONT, sc, (255, 255, 255), th_w, AA)
        cv2.line(image, (lx + tw // 2, ly + bl + pad), center, color, 2, AA)

    # ── Canvas / zoom rendering ─────────────────────────────────────────────────

    def _get_zoomed_view(self, image):
        """Render image centred on a dark canvas, respecting zoom + pan."""
        img_h, img_w = image.shape[:2]
        zw = max(1, int(img_w * self.zoom_level))
        zh = max(1, int(img_h * self.zoom_level))

        interp = cv2.INTER_LINEAR if self.zoom_level >= 1.0 else cv2.INTER_AREA
        zoomed = cv2.resize(image, (zw, zh), interpolation=interp)

        # Dark canvas
        canvas = np.full((self.canvas_h, self.canvas_w, 3), CANVAS_BG, dtype=np.uint8)

        # Image top-left on canvas (centred + user pan)
        ox = (self.canvas_w - zw) // 2 + self.pan_x
        oy = (self.canvas_h - zh) // 2 + self.pan_y

        # Clipping
        src_x = max(0, -ox);  src_y = max(0, -oy)
        dst_x = max(0,  ox);  dst_y = max(0,  oy)
        cw = min(zw - src_x, self.canvas_w - dst_x)
        ch = min(zh - src_y, self.canvas_h - dst_y)

        if cw > 0 and ch > 0:
            canvas[dst_y:dst_y+ch, dst_x:dst_x+cw] = zoomed[src_y:src_y+ch, src_x:src_x+cw]

        # Subtle border around the image area on canvas
        bx1 = max(0, ox - 1);  by1 = max(0, oy - 1)
        bx2 = min(self.canvas_w - 1, ox + zw);  by2 = min(self.canvas_h - 1, oy + zh)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (45, 45, 45), 1)

        return canvas

    # ── Display update ──────────────────────────────────────────────────────────

    def _update_display(self):
        temp = self.output_image.copy()
        for idx, circle in enumerate(self.circles):
            self._draw_label(temp, circle, idx + 1)
        if self.drawing and self.current_radius > 0:
            cv2.circle(temp, self.center, self.current_radius,
                       self.mode_colors[self.current_mode], 2, AA)
        self.display_image = self._get_zoomed_view(temp)
        self._draw_ui()
        cv2.imshow(self.window_name, self.display_image)

    def _update_display_with_input(self):
        temp = self.output_image.copy()
        for idx, circle in enumerate(self.circles):
            self._draw_label(temp, circle, idx + 1)
        color = self.mode_colors[self.current_mode]
        cv2.circle(temp, self.center, self.current_radius, color, 3, AA)
        if self.current_label:
            self._draw_typing_label(temp, self.center, self.current_radius,
                                    self.current_label, self.current_mode)
        self.display_image = self._get_zoomed_view(temp)
        self._draw_input_box()
        cv2.imshow(self.window_name, self.display_image)

    # ── UI overlays ─────────────────────────────────────────────────────────────

    def _draw_input_box(self):
        h, w = self.display_image.shape[:2]
        box_h = 64
        by = h - box_h
        color = self.mode_colors[self.current_mode]

        overlay = self.display_image.copy()
        cv2.rectangle(overlay, (0, by), (w, h), (12, 12, 12), -1)
        cv2.addWeighted(overlay, 0.88, self.display_image, 0.12, 0, self.display_image)
        cv2.rectangle(self.display_image, (0, by), (w, h), color, 2, AA)

        cv2.putText(self.display_image,
                    f"[{self.current_mode.value.upper()}]",
                    (12, by + 22), FONT, 0.52, color, 1, AA)
        cv2.putText(self.display_image, "Label:",
                    (160, by + 22), FONT, 0.52, (200, 200, 200), 1, AA)
        cv2.putText(self.display_image, self.current_label + "_",
                    (12, by + 50), FONT, 0.65, (0, 240, 240), 1, AA)

    def _draw_ui(self):
        overlay = self.display_image.copy()
        h, w = self.display_image.shape[:2]
        bar_h = 88

        cv2.rectangle(overlay, (0, 0), (w, bar_h), (12, 12, 12), -1)

        color = self.mode_colors[self.current_mode]
        cv2.putText(overlay, f"Mode: {self.current_mode.value.upper()}",
                    (14, 24), FONT, 0.58, color, 1, AA)
        cv2.putText(overlay, f"Zoom: {self.zoom_level:.2f}x",
                    (14, 48), FONT, 0.48, (100, 190, 255), 1, AA)

        current_file = self.image_files[self.current_index]
        cv2.putText(overlay, f"{self.current_index+1} / {self.total_images}",
                    (14, 72), FONT, 0.46, (170, 170, 170), 1, AA)

        fname = current_file.name if len(current_file.name) <= 44 else current_file.name[:41] + "..."
        cv2.putText(overlay, fname, (200, 24), FONT, 0.48, (190, 190, 190), 1, AA)
        cv2.putText(overlay, f"Objects: {len(self.circles)}",
                    (200, 48), FONT, 0.46, (170, 170, 170), 1, AA)

        # Status badge
        is_saved = self.saved_status.get(current_file.name, False)
        has_edits = current_file.name in self.image_states or len(self.circles) > 0
        status_text  = "SAVED"   if is_saved   else ("EDITED" if has_edits else "NO EDITS")
        status_color = (60, 200, 60) if is_saved else ((40, 160, 255) if has_edits else (90, 90, 90))
        cv2.putText(overlay, status_text, (w - 160, 24), FONT, 0.56, status_color, 1, AA)

        # Bottom hint bar
        hint = "Wheel:Zoom  RClick:Pan  A/D:Navigate  S:Save  R:Reset  C:Clear  U:Undo  Q:Quit"
        cv2.putText(overlay, hint, (12, h - 10), FONT, 0.38, (120, 120, 120), 1, AA)

        cv2.addWeighted(overlay, 0.82, self.display_image, 0.18, 0, self.display_image)

    # ── Navigation ──────────────────────────────────────────────────────────────

    def _previous_image(self):
        if self.current_index > 0:
            if self.circles:
                self.save_current(auto_save=True)
            self.current_index -= 1
            self._load_current_image()
            self._update_display()
        else:
            print("Already at first image")

    def _next_image(self):
        if self.current_index < self.total_images - 1:
            if self.circles:
                self.save_current(auto_save=True)
            self.current_index += 1
            self._load_current_image()
            self._update_display()
        else:
            print("Already at last image")

    # ── Misc commands ───────────────────────────────────────────────────────────

    def _list_labels(self):
        print("\n" + "="*70)
        print(f"OBJECTS — {self.image_files[self.current_index].name}")
        print("="*70)
        if not self.circles:
            print("No objects marked yet.")
        else:
            for i, c in enumerate(self.circles, 1):
                lbl = c['label'] or "(no label)"
                print(f"  #{i}: {lbl}  [{c['mode'].value}]  "
                      f"pos={c['center']}  r={c['radius']}px")
        print("="*70 + "\n")

    def _edit_last_label(self):
        if not self.circles:
            print("No objects to edit!")
            return
        last = self.circles[-1]
        print(f"\nEditing label: '{last['label']}'  (ENTER=confirm, ESC=cancel)")
        self.current_label = last['label']
        self.label_input_mode = True
        self._update_display_with_input()

        while self.label_input_mode:
            key = cv2.waitKey(0) & 0xFF
            if key == 27:
                print("  ✗ Edit cancelled")
                self.label_input_mode = False
                self.current_label = ""
            elif key == 13:
                last['label'] = self.current_label.strip()
                print(f"  ✓ Updated to: '{last['label']}'")
                self.label_input_mode = False
                self.current_label = ""
                self._apply_all_effects()
            elif key == 8:
                self.current_label = self.current_label[:-1]
                self._update_display_with_input()
            elif 32 <= key <= 126:
                self.current_label += chr(key)
                self._update_display_with_input()

        self._update_display()

    # ── Save ────────────────────────────────────────────────────────────────────

    def save_current(self, auto_save=False):
        if not self.circles:
            if not auto_save:
                print("No objects to save")
            return

        current_file = self.image_files[self.current_index]
        self.image_states[current_file.name] = {
            'circles': [c.copy() for c in self.circles]
        }

        out_img  = self.output_folder / current_file.name
        out_json = out_img.with_suffix('.json')

        # Build full-resolution output
        if self.scale_factor != 1.0:
            final = self.original_image.copy()
            for circle in self.circles:
                sc = {
                    'center': (int(circle['center'][0] / self.scale_factor),
                               int(circle['center'][1] / self.scale_factor)),
                    'radius': int(circle['radius'] / self.scale_factor),
                    'mode':   circle['mode'],
                    'label':  circle['label']
                }
                final = self._apply_effect(final, sc)
                cv2.circle(final, sc['center'], sc['radius'],
                           self.mode_colors[sc['mode']], 3, AA)
            for idx, circle in enumerate(self.circles, 1):
                sc = {
                    'center': (int(circle['center'][0] / self.scale_factor),
                               int(circle['center'][1] / self.scale_factor)),
                    'radius': int(circle['radius'] / self.scale_factor),
                    'mode':   circle['mode'],
                    'label':  circle['label']
                }
                self._draw_label(final, sc, idx)
        else:
            final = self.output_image.copy()
            for idx, circle in enumerate(self.circles, 1):
                self._draw_label(final, circle, idx)

        cv2.imwrite(str(out_img), final)

        labels_data = {
            'source_image': current_file.name,
            'timestamp':    datetime.now().isoformat(),
            'objects':      []
        }
        for idx, c in enumerate(self.circles, 1):
            labels_data['objects'].append({
                'id':     idx,
                'label':  c['label'],
                'mode':   c['mode'].value,
                'center': list(c['center']),
                'radius': c['radius']
            })
        with open(out_json, 'w') as f:
            json.dump(labels_data, f, indent=2)

        self.saved_status[current_file.name] = True

        if auto_save:
            print(f"    ✓ Auto-saved {len(self.circles)} objects")
        else:
            print(f"\n✓ Saved: {out_img.name}  ({len(self.circles)} objects)")
            print(f"  Image  → {out_img}")
            print(f"  Labels → {out_json}")

    # ── Summary ─────────────────────────────────────────────────────────────────

    def generate_summary(self):
        excel_path = self.output_folder / "processing_summary.xlsx"
        try:
            import openpyxl
            from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Processing Summary"

            hfill  = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
            hfont  = Font(bold=True, color="FFFFFF", size=12)
            border = Border(left=Side(style='thin'), right=Side(style='thin'),
                            top=Side(style='thin'),  bottom=Side(style='thin'))

            for col, hdr in enumerate(["Image Name", "Number of Mistakes", "Error Names"], 1):
                cell = ws.cell(row=1, column=col)
                cell.value = hdr; cell.fill = hfill; cell.font = hfont
                cell.alignment = Alignment(horizontal='center', vertical='center')
                cell.border = border

            row = 2
            for img_file in sorted(self.image_files):
                if img_file.name not in self.saved_status:
                    continue
                jp = self.output_folder / img_file.with_suffix('.json').name
                if not jp.exists():
                    continue
                with open(jp) as f:
                    data = json.load(f)
                ws.cell(row=row, column=1).value = img_file.name
                ws.cell(row=row, column=1).border = border
                ws.cell(row=row, column=2).value = len(data['objects'])
                ws.cell(row=row, column=2).alignment = Alignment(horizontal='center')
                ws.cell(row=row, column=2).border = border
                labels = [o['label'] for o in data['objects'] if o['label']]
                ws.cell(row=row, column=3).value = ", ".join(labels) if labels else "(no labels)"
                ws.cell(row=row, column=3).border = border
                row += 1

            row += 1
            sfill = PatternFill(start_color="E7E6E6", end_color="E7E6E6", fill_type="solid")
            sfont = Font(bold=True, size=11)
            ws.cell(row=row, column=1).value = "SUMMARY"
            ws.cell(row=row, column=1).font = sfont
            ws.cell(row=row, column=1).fill = sfill
            ws.merge_cells(f'A{row}:C{row}')
            row += 1
            ws.cell(row=row, column=1).value = "Total Images Processed"
            ws.cell(row=row, column=2).value = len(self.saved_status)
            ws.cell(row=row, column=1).font = Font(bold=True)
            row += 1
            total_obj = sum(
                len(json.load(open(self.output_folder / f.with_suffix('.json').name))['objects'])
                for f in self.image_files
                if f.name in self.saved_status and
                   (self.output_folder / f.with_suffix('.json').name).exists()
            )
            ws.cell(row=row, column=1).value = "Total Objects Labeled"
            ws.cell(row=row, column=2).value = total_obj
            ws.cell(row=row, column=1).font = Font(bold=True)

            ws.column_dimensions['A'].width = 30
            ws.column_dimensions['B'].width = 18
            ws.column_dimensions['C'].width = 60
            wb.save(str(excel_path))
            print(f"✓ Excel summary: {excel_path}")

        except ImportError:
            print("⚠  openpyxl not installed — pip install openpyxl")
        except Exception as e:
            print(f"⚠  Could not create Excel: {e}")

    # ── Instructions ─────────────────────────────────────────────────────────────

    def _print_instructions(self):
        print("\n" + "="*70)
        print("BATCH LABELED EDITOR")
        print("="*70)
        print(f"  Input : {self.input_folder}")
        print(f"  Output: {self.output_folder}")
        print(f"  Images: {self.total_images}")
        print()
        print("  Wheel        Zoom in/out        R    Reset zoom")
        print("  Right-click  Pan                A/D  Prev/Next image")
        print("  Left-drag    Draw circle        S    Save to disk")
        print("  1–7          Switch mode        C    Clear all  U  Undo")
        print("  L            List objects       E    Edit last label")
        print("  T            Toggle labels      Q    Quit & save summary")
        print("="*70 + "\n")

    # ── Main loop ────────────────────────────────────────────────────────────────

    def run(self):
        self._update_display()

        while True:
            key = cv2.waitKey(1) & 0xFF

            if self.label_input_mode:
                if key == 27:
                    self._exit_label_input_mode(save=True)
                elif key == 13:
                    self._exit_label_input_mode(save=True)
                elif key == 8:
                    self.current_label = self.current_label[:-1]
                    self._update_display_with_input()
                elif 32 <= key <= 126:
                    self.current_label += chr(key)
                    self._update_display_with_input()
                continue

            if   key == ord('a'): self._previous_image()
            elif key == ord('d'): self._next_image()
            elif key == ord('r'):
                self.zoom_level = 1.0; self.pan_x = 0; self.pan_y = 0
                self._update_display(); print("Zoom reset")
            elif key == ord('s'): self.save_current(); self._update_display()
            elif key == ord('S'): self.save_current(); self._next_image()
            elif key == ord('c'):
                self.circles.clear()
                self.output_image = self.scaled_image.copy()
                self._update_display(); print("✓ Cleared all")
            elif key == ord('u'):
                if self.circles:
                    removed = self.circles.pop()
                    print(f"✓ Removed: {removed['label'] or '(unlabeled)'}")
                    self._apply_all_effects(); self._update_display()
            elif key == ord('l'): self._list_labels()
            elif key == ord('e'): self._edit_last_label()
            elif key == ord('t'):
                self.show_labels = not self.show_labels
                print(f"✓ Labels: {'ON' if self.show_labels else 'OFF'}")
                self._update_display()
            elif ord('1') <= key <= ord('7'):
                self.current_mode = list(EditMode)[key - ord('1')]
                print(f"✓ Mode: {self.current_mode.value.upper()}")
                self._update_display()
            elif key == ord('q'):
                break

        cv2.destroyAllWindows()
        if self.saved_status:
            self.generate_summary()
            print(f"\n✅ Done — {len(self.saved_status)}/{self.total_images} images saved")
        else:
            print("\n⚠  No images were saved")


# ── Entry point ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch Labeled Editor with Zoom")
    parser.add_argument("input_folder")
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    try:
        BatchLabeledEditor(args.input_folder, args.output).run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback; traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
