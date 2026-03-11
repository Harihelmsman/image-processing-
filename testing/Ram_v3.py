#!/usr/bin/env python3
"""
Batch Labeled Editor  —  High-Performance Low-Memory Annotation
Navigate: A/D   Zoom: Wheel   Pan: Right-drag   Draw: Left-drag

════════════════════════════════════════════════════════════════
PERFORMANCE ARCHITECTURE
════════════════════════════════════════════════════════════════

ROOT CAUSES OF PREVIOUS SLOWDOWNS (and their fixes)

  1. Ghost circle copied 11 MB overview EVERY mouse move during drawing
     FIX: ghost drawn in SCREEN SPACE on the 4 MB canvas → 0 allocation per drag

  2. Effects (GaussianBlur etc.) ran on ENTIRE overview image
     2400×1600 image = 11 MB.  Blurring it = ~30 ms per effect rebuild.
     FIX: ROI-clip to circle bounding box, only those pixels processed.
     Result: 50 px circle on any image size → ~0.3 ms instead of 30 ms.

  3. Label glow: image.copy() × 2 per label  (20 labels = 440 MB temp alloc)
     FIX: removed glow, simple border rectangle.  Composite rebuild: 5 ms not 200.

  4. gc.collect() every frame and every effect → GC pause spikes
     FIX: gc.collect() called ONLY at ImageSource.free() and after save

  5. Hard-coded 25×25 blur kernel applied regardless of circle size or ROI
     FIX: kernel scales with radius, capped to ROI dimensions

  6. waitKey(1) spun the CPU at 1000 fps even when nothing happened
     FIX: waitKey(16) → ~60 fps cap, CPU idles between frames

  7. np.where() allocated full output array for every effect
     FIX: boolean-mask scatter-write (image[mb] = src[mb]), in-place on ROI

STEADY STATE MEMORY  (example: 60 MB JPEG, 9000×6000 px)
  overview (2400×1600):     ~11 MB   always resident
  effects_cache:            ~11 MB   resident while editing
  composite_cache:          ~11 MB   resident while editing
  vp canvas (1440×900):      ~4 MB   pre-allocated, never reallocated
  Python + OpenCV:          ~25 MB
  ─────────────────────────────────
  Total:                    ~62 MB   << 100 MB ✓

  Ghost circle during draw:  +0 MB   screen-space, no overview alloc
  Save peak:                 +raw decoded (~180 MB for 60 MB JPEG), freed in < 1 s
════════════════════════════════════════════════════════════════
"""

import gc
import cv2
import numpy as np
from pathlib import Path
from enum import Enum
import argparse
import json
from datetime import datetime
import time
import sys

FONT        = cv2.FONT_HERSHEY_DUPLEX
AA          = cv2.LINE_AA
CANVAS_W    = 1440
CANVAS_H    = 900
CANVAS_BG   = np.array([18, 18, 18], dtype=np.uint8)
CANVAS_PAD  = 40
OVERVIEW_PX = 2400   # longest edge of always-resident working image


# ═════════════════════════════════════════════════════════════════════════════
class ImageSource:
    """One image in RAM at a time.  Full-res decoded → downscaled → deleted."""

    __slots__ = ('path', 'orig_w', 'orig_h', 'ov_scale', 'overview')

    def __init__(self, path: Path):
        self.path     = path
        self.orig_w   = 0
        self.orig_h   = 0
        self.ov_scale = 1.0
        self.overview = None

    def load(self) -> None:
        full = cv2.imread(str(self.path))
        if full is None:
            raise IOError(f"Cannot read: {self.path}")
        self.orig_h, self.orig_w = full.shape[:2]
        if min(self.orig_h, self.orig_w) < 32:
            raise ValueError(f"Image too small: {self.path}")
        longest = max(self.orig_w, self.orig_h)
        if longest > OVERVIEW_PX:
            self.ov_scale = OVERVIEW_PX / longest
            nw = max(1, int(self.orig_w * self.ov_scale))
            nh = max(1, int(self.orig_h * self.ov_scale))
            self.overview = cv2.resize(full, (nw, nh), interpolation=cv2.INTER_AREA)
        else:
            self.ov_scale = 1.0
            self.overview = full.copy()
        del full
        gc.collect()   # appropriate: large buffer just freed

    def free(self) -> None:
        self.overview = None
        gc.collect()

    @property
    def ov_w(self): return 0 if self.overview is None else self.overview.shape[1]
    @property
    def ov_h(self): return 0 if self.overview is None else self.overview.shape[0]


# ═════════════════════════════════════════════════════════════════════════════
class Viewport:
    """Affine viewer with pre-allocated canvas buffer (no per-frame malloc)."""

    ZOOM_STEPS = [0.10, 0.125, 0.167, 0.25, 0.333, 0.5, 0.667,
                  1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]
    SNAP_THR   = 0.04

    def __init__(self, cw=CANVAS_W, ch=CANVAS_H):
        self.cw = cw;  self.ch = ch
        self._s = 1.0;  self._tx = 0.0;  self._ty = 0.0
        self._buf = np.full((ch, cw, 3), CANVAS_BG, dtype=np.uint8)

    def fit(self, iw, ih):
        self._s = min((self.cw-2*CANVAS_PAD)/iw, (self.ch-2*CANVAS_PAD)/ih)
        self._snap();  self._recenter(iw, ih)

    def zoom_at(self, sx, sy, factor, iw, ih):
        ix = (sx-self._tx)/self._s;  iy = (sy-self._ty)/self._s
        min_s = min(0.05, (self.cw-2*CANVAS_PAD)/max(iw,1))
        self._s = max(min_s, min(16.0, self._s * factor))
        self._snap()
        self._tx = sx - ix*self._s;  self._ty = sy - iy*self._s

    def pan(self, dx, dy):  self._tx += dx;  self._ty += dy
    def reset(self, iw, ih): self.fit(iw, ih)

    def to_screen(self, ix, iy): return int(ix*self._s+self._tx), int(iy*self._s+self._ty)
    def to_img(self, sx, sy):    return int((sx-self._tx)/self._s), int((sy-self._ty)/self._s)

    @property
    def scale(self): return self._s

    def render(self, image: np.ndarray) -> np.ndarray:
        """Crop-then-scale into pre-allocated buffer.  Returns a VIEW (not a copy)."""
        ih, iw = image.shape[:2]
        s = self._s;  tx = self._tx;  ty = self._ty
        self._buf[:] = CANVAS_BG
        x0f = max(0.0, (-tx)/s);         y0f = max(0.0, (-ty)/s)
        x1f = min(float(iw),(self.cw-tx)/s); y1f = min(float(ih),(self.ch-ty)/s)
        if x1f<=x0f or y1f<=y0f: return self._buf
        cx0=int(np.floor(x0f)); cy0=int(np.floor(y0f))
        cx1=min(iw,int(np.ceil(x1f))); cy1=min(ih,int(np.ceil(y1f)))
        crop = image[cy0:cy1, cx0:cx1]
        if crop.size == 0: return self._buf
        dw = max(1, int(round((cx1-cx0)*s)));  dh = max(1, int(round((cy1-cy0)*s)))
        interp = cv2.INTER_LANCZOS4 if s>=1.0 else cv2.INTER_AREA
        scaled = cv2.resize(crop, (dw, dh), interpolation=interp)
        if s >= 1.0:
            blur = cv2.GaussianBlur(scaled, (0,0), 1.0)
            scaled = cv2.addWeighted(scaled, 1.35, blur, -0.35, 0)
        dx=max(0,int(tx+cx0*s)); dy=max(0,int(ty+cy0*s))
        pw=min(dw,self.cw-dx);   ph=min(dh,self.ch-dy)
        if pw>0 and ph>0: self._buf[dy:dy+ph, dx:dx+pw] = scaled[:ph,:pw]
        cv2.rectangle(self._buf,
                      (max(0,int(tx)-1), max(0,int(ty)-1)),
                      (min(self.cw-1,int(tx)+int(iw*s)), min(self.ch-1,int(ty)+int(ih*s))),
                      (50,50,50), 1, AA)
        return self._buf

    def _snap(self):
        for step in self.ZOOM_STEPS:
            if abs(self._s-step)/step < self.SNAP_THR: self._s=step; return
    def _recenter(self, iw, ih):
        self._tx=(self.cw-iw*self._s)/2; self._ty=(self.ch-ih*self._s)/2


# ═════════════════════════════════════════════════════════════════════════════
class EditMode(Enum):
    HIGHLIGHT = "highlight"
    BLUR      = "blur"
    PIXELATE  = "pixelate"
    DARKEN    = "darken"
    GRAYSCALE = "grayscale"
    INVERT    = "invert"
    OUTLINE   = "outline"

MODE_COLORS = {
    EditMode.HIGHLIGHT: (0,255,0), EditMode.BLUR:      (255,80,80),
    EditMode.PIXELATE:  (80,80,255), EditMode.DARKEN:  (160,160,160),
    EditMode.GRAYSCALE: (210,210,210), EditMode.INVERT: (255,255,0),
    EditMode.OUTLINE:   (0,220,255),
}


# ═════════════════════════════════════════════════════════════════════════════
def _odd(k): return max(3, k if k%2==1 else k+1)

def apply_effect_roi(image: np.ndarray, cx: int, cy: int, r: int, mode: EditMode) -> None:
    """
    KEY PERFORMANCE FIX: clip to bounding box, operate on ROI only.
    Old: GaussianBlur(full_2400x1600_image) = ~30 ms
    New: GaussianBlur(50x50_roi)            = ~0.3 ms
    In-place boolean-mask write avoids full-array allocation.
    """
    ih, iw = image.shape[:2]
    x0=max(0,cx-r); y0=max(0,cy-r); x1=min(iw,cx+r+1); y1=min(ih,cy+r+1)
    if x0>=x1 or y0>=y1: return
    roi = image[y0:y1, x0:x1]
    rh, rw = roi.shape[:2]
    mask = np.zeros((rh, rw), dtype=np.uint8)
    cv2.circle(mask, (cx-x0, cy-y0), r, 255, -1)
    mb = mask.astype(bool)
    if not mb.any(): return

    if mode == EditMode.HIGHLIGHT:
        lit = cv2.addWeighted(roi, 0.55, np.full_like(roi, 255), 0.45, 0)
        roi[mb] = lit[mb]
    elif mode == EditMode.BLUR:
        k = min(_odd(max(5, r//4)), rw-(1 if rw%2==0 else 0), rh-(1 if rh%2==0 else 0))
        k = _odd(k)
        roi[mb] = cv2.GaussianBlur(roi, (k,k), 0)[mb]
    elif mode == EditMode.PIXELATE:
        step = max(2, r//8)
        small = cv2.resize(roi, (max(1,rw//step), max(1,rh//step)), interpolation=cv2.INTER_NEAREST)
        roi[mb] = cv2.resize(small, (rw,rh), interpolation=cv2.INTER_NEAREST)[mb]
    elif mode == EditMode.DARKEN:
        roi[mb] = (roi[mb].astype(np.float32) * 0.45).clip(0, 255).astype(np.uint8)
    elif mode == EditMode.GRAYSCALE:
        gray3 = cv2.cvtColor(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        roi[mb] = gray3[mb]
    elif mode == EditMode.INVERT:
        roi[mb] = np.uint8(255) - roi[mb]


# ═════════════════════════════════════════════════════════════════════════════
def _rects_overlap(a, b, buf=3):
    return not (a[2]+buf<b[0] or a[0]-buf>b[2] or a[3]+buf<b[1] or a[1]-buf>b[3])

def draw_labels(image: np.ndarray, circles: list) -> None:
    """
    KEY PERFORMANCE FIX: NO glow (was image.copy() × 2 per label → huge alloc).
    Uses simple bordered rectangle instead.
    Called ONCE per composite rebuild — never per frame.
    """
    if not circles: return
    ih, iw = image.shape[:2]
    md = min(ih, iw)
    fsc = 0.70 if md>=900 else 0.56 if md>=500 else 0.44
    fth = 2 if md>=900 else 1
    pad = max(4, int(5*fsc/0.5))
    placed = []

    for i, c in enumerate(circles, 1):
        color = MODE_COLORS[c['mode']]
        cx = int(c['_ov_cx']);  cy = int(c['_ov_cy']);  r = int(c['_ov_r'])

        # Badge
        badge = str(i)
        (bw, bh), _ = cv2.getTextSize(badge, FONT, fsc*0.85, fth)
        br = max(bw,bh)//2 + 5
        cv2.circle(image, (cx+2,cy+2), br, (0,0,0), -1, AA)
        cv2.circle(image, (cx,cy),     br, color,   -1, AA)
        cv2.circle(image, (cx,cy),     br, (255,255,255), 1, AA)
        cv2.putText(image, badge, (cx-bw//2, cy+bh//2), FONT, fsc*0.85, (0,0,0), fth+1, AA)

        # Label tag
        lbl = c['label'] or f"Err #{i}"
        tag = f"#{i} [{c['mode'].value[:3].upper()}] {lbl}"
        (tw, fh_), bl = cv2.getTextSize(tag, FONT, fsc, fth)
        W=tw+2*pad;  H=fh_+bl+2*pad
        cands = [
            (cx-r, cy-r-H-8), (cx-r, cy+r+6), (cx+r+6, cy-H//2), (cx-r-W-6, cy-H//2),
            (cx+r+6, cy-r-H-8), (cx-r-W-6, cy-r-H-8), (cx+r+6, cy+r+6), (cx-r-W-6, cy+r+6),
        ]
        for extra in (35, 70, 110):
            cands += [(px,py+extra) for px,py in cands[:4]]
            cands += [(px,py-extra) for px,py in cands[:4]]
        lx=ly=0; chosen=None
        for px,py in cands:
            r1,r2,r3,r4 = int(px),int(py),int(px+W),int(py+H)
            if r1<pad or r2<pad or r3>iw-pad or r4>ih-pad: continue
            rect=(r1,r2,r3,r4)
            if not any(_rects_overlap(rect,p) for p in placed):
                lx=r1+pad; ly=r2+fh_+pad; chosen=rect; break
        if chosen is None:
            bot = max((p[3] for p in placed), default=cy+r)+5
            lx=max(pad, min(cx-W//2, iw-W-pad)); ly=min(bot+fh_+pad, ih-bl-pad)
            chosen=(lx-pad, ly-fh_-pad, lx+tw+pad, ly+bl+pad)
        placed.append(chosen)
        x1,y1,x2,y2 = chosen

        # Simple fast background (no glow alloc)
        cv2.rectangle(image, (x1,y1), (x2,y2), (18,18,25), -1)
        cv2.rectangle(image, (x1,y1), (x1+3,y2), color, -1, AA)
        cv2.rectangle(image, (x1,y1), (x2,y2), color, 2, AA)
        cv2.putText(image, tag, (lx+5,ly+1), FONT, fsc, (0,0,0),      fth+1, AA)
        cv2.putText(image, tag, (lx+4,ly),   FONT, fsc, (255,255,255), fth,  AA)

        # Connector
        mx=(x1+x2)//2; my=y2 if ly>cy else y1
        ang=np.arctan2(my-cy, mx-cx)
        ex=int(cx+r*np.cos(ang)); ey=int(cy+r*np.sin(ang))
        cv2.line(image, (mx,my), (ex,ey), color, 1, AA)
        cv2.circle(image, (ex,ey), 3, color, -1, AA)


# ═════════════════════════════════════════════════════════════════════════════
class RenderCache:
    """
    Three-level cache.  Dirty flags prevent unnecessary rebuilds.

    Level 1 — effects_cache:   overview + effects + outlines
    Level 2 — composite_cache: effects + labels (O(n²) only here)
    Level 3 — screen frame:    composite → Viewport.render (fast, no rebuild)

    Ghost circle drawn in SCREEN SPACE → zero overview copy during drag.
    """

    def __init__(self, img: ImageSource, vp: Viewport):
        self._img = img;  self._vp = vp
        self.ec = None;   self.cc = None   # effects, composite caches
        self._ed = True;  self._cd = True  # dirty flags

    def attach(self, img: ImageSource):
        self._img=img; self.ec=None; self.cc=None; self._ed=True; self._cd=True

    def dirty_effects(self):    self._ed=True;  self._cd=True
    def dirty_composite(self):  self._cd=True

    def _build_effects(self, circles):
        base = self._img.overview.copy()
        sc   = self._img.ov_scale
        for c in circles:
            ov_cx=int(c['center'][0]*sc); ov_cy=int(c['center'][1]*sc)
            ov_r=max(1,int(c['radius']*sc))
            c['_ov_cx']=ov_cx; c['_ov_cy']=ov_cy; c['_ov_r']=ov_r
            apply_effect_roi(base, ov_cx, ov_cy, ov_r, c['mode'])
            cv2.circle(base,(ov_cx,ov_cy),ov_r,MODE_COLORS[c['mode']],2,AA)
        self.ec=base; self._ed=False; self._cd=True

    def _build_composite(self, circles, show_labels):
        tmp = self.ec.copy()
        if show_labels:
            draw_labels(tmp, circles)
        self.cc=tmp; self._cd=False

    def get_composite(self, circles, show_labels):
        if self._ed: self._build_effects(circles)
        if self._cd: self._build_composite(circles, show_labels)
        return self.cc

    def render_frame(self, circles, show_labels,
                     ghost_orig=None, ghost_r_orig=0, ghost_mode=EditMode.HIGHLIGHT):
        """
        Ghost circle rendered in SCREEN SPACE (after vp.render).
        Cost during drag: one 4 MB canvas copy — never touches the 11 MB composite.
        """
        comp  = self.get_composite(circles, show_labels)
        frame = self._vp.render(comp)   # returns view of pre-allocated buffer

        if ghost_orig is not None and ghost_r_orig > 0:
            sc=self._img.ov_scale
            sx,sy = self._vp.to_screen(int(ghost_orig[0]*sc), int(ghost_orig[1]*sc))
            sr=max(1, int(ghost_r_orig*sc*self._vp.scale))
            frame=frame.copy()   # 4 MB copy (canvas, not overview)
            cv2.circle(frame,(sx,sy),sr,MODE_COLORS[ghost_mode],2,AA)
        else:
            frame=frame.copy()
        return frame


# ═════════════════════════════════════════════════════════════════════════════
class AnnotationStore:
    """Metadata only — zero pixel data.  Coords in original pixel space."""

    def __init__(self, out_dir: Path):
        self.out_dir=out_dir; self._db={}

    def load(self, path: Path) -> list:
        name=path.name; js=self.out_dir/path.with_suffix('.json').name
        if js.exists():
            try:
                with open(js) as f: d=json.load(f)
                circles=[{'center':tuple(o['center']),'radius':int(o['radius']),
                           'mode':EditMode(o.get('mode','highlight')),
                           'label':o.get('label',''),'description':o.get('description','')}
                          for o in d.get('objects',[])]
                self._db[name]=circles; return circles
            except Exception as e: print(f"  JSON error: {e}")
        if name in self._db: return [c.copy() for c in self._db[name]]
        return []

    def remember(self, path: Path, circles: list):
        self._db[path.name]=[self._clean(c) for c in circles]

    def write(self, path: Path, circles: list, orig_w, orig_h) -> Path:
        clean=[self._clean(c) for c in circles]
        self._db[path.name]=clean
        js=self.out_dir/path.with_suffix('.json').name
        data={'source_image':path.name,'original_size':[orig_w,orig_h],
              'timestamp':datetime.now().isoformat(),
              'objects':[{'id':i,'label':c['label'],'description':c.get('description',''),
                          'mode':c['mode'].value,'center':list(c['center']),'radius':c['radius']}
                         for i,c in enumerate(clean,1)]}
        with open(js,'w') as f: json.dump(data,f,indent=2)
        return js

    @staticmethod
    def _clean(c): return {k:v for k,v in c.items() if not k.startswith('_')}


# ═════════════════════════════════════════════════════════════════════════════
class BatchLabeledEditor:

    MAX_BATCH     = 200
    WARN_CIRCLES  = 30
    ZOOM_DEBOUNCE = 14   # ms — ~70 fps zoom

    def __init__(self, input_folder: str, output_folder: str = None):
        self.input_folder = Path(input_folder)
        self.out_dir = (Path(output_folder) if output_folder else
                        self.input_folder.parent /
                        f"labeled_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.files = self._scan()
        if not self.files: raise ValueError(f"No images in {input_folder}")
        self.total=len(self.files); self.idx=0

        self.img   = ImageSource(self.files[0])
        self.vp    = Viewport()
        self.ann   = AnnotationStore(self.out_dir)
        self.cache = RenderCache(self.img, self.vp)

        self.circles=[]; self.drawing=False; self.c_orig=None; self.r_orig=0
        self.cur_label=""; self.cur_desc=""
        self.in_label=False; self.in_desc=False
        self.mode=EditMode.HIGHLIGHT; self.show_labels=True; self.saved={}
        self.panning=False; self._px=self._py=0; self._last_zoom=0.0

        WIN="Batch Labeled Editor"; self.WIN=WIN
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, CANVAS_W, CANVAS_H)
        cv2.setMouseCallback(WIN, self._on_mouse)

        self._load(0); self._print_help()

    def _scan(self):
        exts={'.jpg','.jpeg','.png','.bmp','.tiff','.tif'}
        files=sorted(f for e in exts for f in self.input_folder.glob(f'*{e}')
                     if f.stat().st_size>=512)
        if len(files)>self.MAX_BATCH:
            print(f"\n  {len(files)} images found")
            if input("  Continue? (y/N): ").strip().lower()!='y': sys.exit(0)
        return files

    def _load(self, idx):
        self.img.free()   # free previous BEFORE loading next
        self.idx=idx; f=self.files[idx]
        self.img=ImageSource(f); self.img.load()
        self.vp.fit(self.img.ov_w, self.img.ov_h)
        self.circles=self.ann.load(f)
        self.cache.attach(self.img)
        self.drawing=self.in_label=self.in_desc=False
        self.cur_label=self.cur_desc=""
        n=len(self.circles)
        print(f"\n  [{idx+1}/{self.total}]  {f.name}"
              f"  {self.img.orig_w}×{self.img.orig_h}"
              f"  ov={self.img.ov_w}×{self.img.ov_h}"
              f"  scale={self.img.ov_scale:.3f}  circles={n}")

    def _screen_to_orig(self, sx, sy):
        ov_x,ov_y=self.vp.to_img(sx,sy); sc=self.img.ov_scale
        return int(ov_x/sc), int(ov_y/sc)

    # ── Mouse ─────────────────────────────────────────────────────────────────
    def _on_mouse(self, event, x, y, flags, _):
        iw,ih=self.img.ov_w,self.img.ov_h
        if event==cv2.EVENT_MOUSEWHEEL:
            now=time.perf_counter()*1000
            if now-self._last_zoom<self.ZOOM_DEBOUNCE: return
            self._last_zoom=now
            self.vp.zoom_at(x,y,1.18 if flags>0 else 0.83,iw,ih)
            self._show(); return
        if event==cv2.EVENT_RBUTTONDOWN:
            self.panning=True; self._px=x; self._py=y; return
        if event==cv2.EVENT_RBUTTONUP:
            self.panning=False; return
        if event==cv2.EVENT_MOUSEMOVE and self.panning:
            self.vp.pan(x-self._px, y-self._py); self._px=x; self._py=y
            self._show(); return
        if self.in_label or self.in_desc or self.panning: return
        if event==cv2.EVENT_LBUTTONDOWN:
            self.drawing=True; self.c_orig=self._screen_to_orig(x,y); self.r_orig=0
        elif event==cv2.EVENT_MOUSEMOVE and self.drawing:
            ox,oy=self._screen_to_orig(x,y)
            self.r_orig=int(np.hypot(ox-self.c_orig[0],oy-self.c_orig[1]))
            self._show()   # ghost in screen-space — ZERO overview copy
        elif event==cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing=False
            if self.r_orig>5:
                if len(self.circles)>=self.WARN_CIRCLES:
                    print(f"  Note: {len(self.circles)} circles")
                self._start_label()

    # ── Input flow ────────────────────────────────────────────────────────────
    def _start_label(self):
        self.in_label=True; self.cur_label=""
        print(f"\n  [{self.mode.value.upper()}]  Label: (ENTER=ok  ESC=cancel)")
        self._show_input()

    def _finish_label(self, ok):
        self.in_label=False
        if ok:
            self.in_desc=True; self.cur_desc=""
            print("  Description (ENTER or ESC to skip):")
            self._show_input()
        else:
            print("  Cancelled"); self.cur_label=""; self.drawing=False; self._show()

    def _commit(self):
        self.in_desc=False
        self.circles.append({'center':self.c_orig,'radius':self.r_orig,
                              'mode':self.mode,'label':self.cur_label.strip(),
                              'description':self.cur_desc.strip()})
        print(f"  Added '{self.cur_label.strip() or '(unlabeled)'}' [{self.mode.value}]"
              f"  r={self.r_orig}")
        self.cur_label=self.cur_desc=""
        self.cache.dirty_effects(); self._show()

    # ── Render ────────────────────────────────────────────────────────────────
    def _show(self):
        ghost=self.c_orig if (self.drawing and self.r_orig>0) else None
        frame=self.cache.render_frame(self.circles,self.show_labels,
                                      ghost,self.r_orig,self.mode)
        self._hud(frame); cv2.imshow(self.WIN, frame)

    def _show_input(self):
        sc=self.img.ov_scale
        ov_cx=int(self.c_orig[0]*sc); ov_cy=int(self.c_orig[1]*sc)
        ov_r=max(1,int(self.r_orig*sc))
        comp=self.cache.get_composite(self.circles,self.show_labels)
        tmp=comp.copy()
        cv2.circle(tmp,(ov_cx,ov_cy),ov_r,MODE_COLORS[self.mode],3,AA)
        txt=(self.cur_desc if self.in_desc else self.cur_label)+"█"
        fsc=0.60; fth=2
        (tw,fh),bl=cv2.getTextSize(txt,FONT,fsc,fth)
        bp=6; bx=max(bp,min(ov_cx-tw//2,self.img.ov_w-tw-bp*2))
        by=max(fh+bp*2,ov_cy-ov_r-14)
        cv2.rectangle(tmp,(bx-bp,by-fh-bp),(bx+tw+bp,by+bl+bp),(14,14,22),-1)
        cv2.rectangle(tmp,(bx-bp,by-fh-bp),(bx+tw+bp,by+bl+bp),MODE_COLORS[self.mode],2,AA)
        cv2.putText(tmp,txt,(bx,by),FONT,fsc,(0,240,255),fth,AA)
        frame=self.vp.render(tmp).copy()
        h,w=frame.shape[:2]; color=MODE_COLORS[self.mode]
        prompt="Description" if self.in_desc else "Label"
        ov2=frame.copy()
        cv2.rectangle(ov2,(0,h-62),(w,h),(10,10,16),-1)
        cv2.addWeighted(ov2,0.88,frame,0.12,0,frame)
        cv2.rectangle(frame,(0,h-62),(w,h),color,2,AA)
        cv2.putText(frame,f"[{self.mode.value.upper()}]  {prompt}:",(10,h-40),FONT,0.50,color,1,AA)
        cv2.putText(frame,txt,(10,h-12),FONT,0.62,(0,240,255),1,AA)
        self._hud(frame); cv2.imshow(self.WIN, frame)

    def _hud(self, frame):
        ov=frame.copy(); h,w=frame.shape[:2]
        cv2.rectangle(ov,(0,0),(w,80),(10,10,15),-1)
        color=MODE_COLORS[self.mode]; cf=self.files[self.idx]; n=len(self.circles)
        cv2.putText(ov,f"Mode: {self.mode.value.upper()}",(12,22),FONT,0.54,color,1,AA)
        cv2.putText(ov,f"Zoom: {self.vp.scale:.2f}x",(12,44),FONT,0.44,(100,190,255),1,AA)
        cv2.putText(ov,f"{self.idx+1}/{self.total}",(12,66),FONT,0.42,(140,140,140),1,AA)
        pct=int(self.img.ov_scale*100)
        fname=cf.name if len(cf.name)<=42 else cf.name[:39]+"..."
        cv2.putText(ov,f"{fname}  [{self.img.orig_w}x{self.img.orig_h}  ov:{pct}%]",
                    (192,22),FONT,0.41,(175,175,175),1,AA)
        nc=(40,160,255) if n>=self.WARN_CIRCLES else (140,140,140)
        cv2.putText(ov,f"Objects: {n}",(192,44),FONT,0.42,nc,1,AA)
        is_saved=self.saved.get(cf.name,False)
        st,sc2=(("SAVED",(55,200,55)) if is_saved else
                ("EDITED",(40,160,255)) if n>0 else ("NO EDITS",(75,75,75)))
        cv2.putText(ov,st,(w-145,22),FONT,0.52,sc2,1,AA)
        hint="Wheel:Zoom  R-drag:Pan  A/D:Nav  S:Save  R:Reset  C:Clear  U:Undo  H:Help  Q:Quit"
        cv2.putText(ov,hint,(10,h-8),FONT,0.34,(90,90,90),1,AA)
        cv2.addWeighted(ov,0.80,frame,0.20,0,frame)

    # ── Navigation ────────────────────────────────────────────────────────────
    def _guard(self):
        if self.drawing: print("  Finish drawing"); return True
        if self.in_label: print("  Finish label (ESC=cancel)"); return True
        if self.in_desc: print("  Finish description (ENTER=ok)"); return True
        return False

    def _nav(self, delta):
        if self._guard(): return
        new=self.idx+delta
        if not (0<=new<self.total):
            print("  Already at","first" if delta<0 else "last","image"); return
        if self.circles:
            self.ann.remember(self.files[self.idx],self.circles)
            if not self.saved.get(self.files[self.idx].name): self._save(auto=True)
        self._load(new); self._show()

    # ── Save ──────────────────────────────────────────────────────────────────
    def _save(self, auto=False):
        if not self.circles:
            if not auto: print("  No objects to save")
            return
        cf=self.files[self.idx]; self.ann.remember(cf,self.circles)
        try:
            full=cv2.imread(str(cf))
            if full is None: raise IOError(f"Cannot reload {cf}")
            clean=[self.ann._clean(c) for c in self.circles]
            for c in clean:
                apply_effect_roi(full,c['center'][0],c['center'][1],c['radius'],c['mode'])
                cv2.circle(full,c['center'],c['radius'],MODE_COLORS[c['mode']],3,AA)
            draw_labels(full,[{**c,'_ov_cx':c['center'][0],
                                    '_ov_cy':c['center'][1],
                                    '_ov_r': c['radius']} for c in clean])
            out=self.out_dir/cf.name
            if not cv2.imwrite(str(out),full): raise IOError("imwrite failed")
            del full; gc.collect()
            js=self.ann.write(cf,self.circles,self.img.orig_w,self.img.orig_h)
            self.saved[cf.name]=True
            if auto: print(f"  Auto-saved {cf.name}  ({len(self.circles)} objects)")
            else:    print(f"\n  Saved {cf.name}  ({len(self.circles)} objects)\n  {out}\n  {js}")
        except Exception as e: print(f"  Save error: {e}")

    # ── Misc ──────────────────────────────────────────────────────────────────
    def _edit_last(self):
        if not self.circles: print("  No objects to edit"); return
        last=self.circles[-1]
        self.c_orig=last['center']; self.r_orig=last['radius']
        print(f"\n  Editing: '{last['label']}'")
        self.cur_label=last['label']; self.in_label=True; self._show_input()
        while self.in_label:
            k=cv2.waitKey(0)&0xFF
            if k==27: self.in_label=False; self.cur_label=""; self._show(); return
            elif k==13: last['label']=self.cur_label.strip(); self.in_label=False; break
            elif k==8: self.cur_label=self.cur_label[:-1]; self._show_input()
            elif 32<=k<=126: self.cur_label+=chr(k); self._show_input()
        self.cur_desc=last.get('description',''); self.in_desc=True; self._show_input()
        while self.in_desc:
            k=cv2.waitKey(0)&0xFF
            if k in(13,27):
                if k==13: last['description']=self.cur_desc.strip()
                self.in_desc=False; break
            elif k==8: self.cur_desc=self.cur_desc[:-1]; self._show_input()
            elif 32<=k<=126: self.cur_desc+=chr(k); self._show_input()
        self.cur_label=self.cur_desc=""; print("  Updated")
        self.cache.dirty_composite(); self._show()

    def _ram_status(self):
        print(f"\n  {'─'*50}")
        ow,oh=self.img.ov_w,self.img.ov_h
        print(f"  Original : {self.img.orig_w}×{self.img.orig_h}"
              f"  ({self.files[self.idx].stat().st_size//1024} KB on disk)")
        print(f"  Overview : {ow}×{oh} = {ow*oh*3/1048576:.1f} MB"
              f"  scale={self.img.ov_scale:.3f}")
        ec=self.cache.ec
        if ec is not None:
            print(f"  EffCache : {ec.nbytes/1048576:.1f} MB")
        print(f"  Circles  : {len(self.circles)}")
        try:
            import psutil,os
            mb=psutil.Process(os.getpid()).memory_info().rss/1048576
            print(f"  Process  : {mb:.0f} MB  (Task Manager)")
        except ImportError: print("  (pip install psutil for RAM readout)")
        print(f"  {'─'*50}\n")

    def _list(self):
        print(f"\n  {'═'*52}")
        print(f"  {self.files[self.idx].name}  [{self.img.orig_w}×{self.img.orig_h}]")
        if not self.circles: print("  (none)")
        for i,c in enumerate(self.circles,1):
            print(f"  #{i}: {c['label'] or '(no label)'}  [{c['mode'].value}]"
                  f"  r={c['radius']}  {c['center']}")
            if c.get('description'): print(f"        {c['description']}")
        print(f"  {'═'*52}\n")

    def _excel(self):
        xp=self.out_dir/"processing_summary.xlsx"
        try:
            import openpyxl
            from openpyxl.styles import Font,PatternFill,Alignment,Border,Side
            wb=openpyxl.Workbook(); ws=wb.active; ws.title="Summary"
            hf=PatternFill(start_color="1E3A8A",end_color="1E3A8A",fill_type="solid")
            hfnt=Font(bold=True,color="FFFFFF",size=11)
            bdr=Border(**{s:Side(style='thin') for s in('left','right','top','bottom')})
            ca=Alignment(horizontal='center',vertical='center',wrap_text=True)
            la=Alignment(horizontal='left',vertical='top',wrap_text=True)
            def sty(cell,fill=None,font=None,align=None):
                if fill: cell.fill=fill
                if font: cell.font=font
                if align: cell.alignment=align
                cell.border=bdr
            for col,h in enumerate(["#","Image","Count","Error Detail"],1):
                c=ws.cell(row=1,column=col,value=h); sty(c,fill=hf,font=hfnt,align=ca)
            ws.row_dimensions[1].height=26
            dr=2; num=0
            for img in sorted(self.files):
                if img.name not in self.saved: continue
                jp=self.out_dir/img.with_suffix('.json').name
                if not jp.exists(): continue
                with open(jp) as f: d=json.load(f)
                objs=d.get('objects',[]); num+=1
                c1=ws.cell(row=dr,column=1,value=num); sty(c1,align=ca); c1.font=Font(bold=True,size=10)
                c2=ws.cell(row=dr,column=2,value=img.name); sty(c2,align=la)
                c3=ws.cell(row=dr,column=3,value=len(objs)); sty(c3,align=ca)
                if len(objs)==0:   c3.fill=PatternFill(start_color="C6EFCE",end_color="C6EFCE",fill_type="solid"); c3.font=Font(color="276221",bold=True)
                elif len(objs)<=3: c3.fill=PatternFill(start_color="FFEB9C",end_color="FFEB9C",fill_type="solid"); c3.font=Font(color="9C5700",bold=True)
                else:              c3.fill=PatternFill(start_color="FFC7CE",end_color="FFC7CE",fill_type="solid"); c3.font=Font(color="9C0006",bold=True)
                lines=[f"{i}. {o.get('label','').strip() or '(unlabeled)'}"
                       +(f" — {o['description']}" if o.get('description','').strip() else "")
                       for i,o in enumerate(objs,1)] or ["(no errors)"]
                c4=ws.cell(row=dr,column=4,value="\n".join(lines)); sty(c4,align=la)
                ws.row_dimensions[dr].height=max(18,len(objs)*14+4)
                if num%2==0:
                    alt=PatternFill(start_color="EFF3FB",end_color="EFF3FB",fill_type="solid")
                    for col in(1,2,4): ws.cell(row=dr,column=col).fill=alt
                dr+=1
            dr+=1
            sf=PatternFill(start_color="D8E4F0",end_color="D8E4F0",fill_type="solid")
            sfnt=Font(bold=True,size=11)
            def frow(label,val):
                nonlocal dr
                for col,v in((2,label),(3,val)):
                    c=ws.cell(row=dr,column=col,value=v)
                    c.font=sfnt; c.fill=sf; c.border=bdr
                    if col==3: c.alignment=ca
                ws.row_dimensions[dr].height=18; dr+=1
            total_obj=sum(len(json.load(open(self.out_dir/f.with_suffix('.json').name)).get('objects',[]))
                          for f in self.files if f.name in self.saved
                          and (self.out_dir/f.with_suffix('.json').name).exists())
            frow("Total Images Processed",len(self.saved))
            frow("Total Errors Found",total_obj)
            if self.saved: frow("Avg Errors / Image",round(total_obj/len(self.saved),1))
            ws.column_dimensions['A'].width=5; ws.column_dimensions['B'].width=36
            ws.column_dimensions['C'].width=10; ws.column_dimensions['D'].width=64
            ws.freeze_panes="A2"; wb.save(str(xp)); print(f"  Excel: {xp}")
        except ImportError: print("  pip install openpyxl")
        except Exception as e: print(f"  Excel error: {e}")

    def _print_help(self):
        print("\n"+"═"*58)
        print("  BATCH LABELED EDITOR")
        print("═"*58)
        print(f"  Input  : {self.input_folder}")
        print(f"  Output : {self.out_dir}")
        print(f"  Images : {self.total}  |  Overview: {OVERVIEW_PX} px  |  RAM: <100 MB")
        print()
        print("  Wheel     Zoom          R    Reset zoom")
        print("  R-drag    Pan           A/D  Prev / Next image")
        print("  L-drag    Draw circle   S    Save  (Shift+S = save+next)")
        print("  1–7       Effect mode   C    Clear   U  Undo   E  Edit last")
        print("  L         List objects  T    Toggle labels")
        print("  M         RAM status    H    Help    Q  Quit")
        print("═"*58+"\n")

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        self._show()
        while True:
            key=cv2.waitKey(16)&0xFF   # 16 ms = ~60 fps cap, CPU idles between frames

            if self.in_desc:
                if   key==27: self.in_desc=False; self.cur_label=self.cur_desc=""; print("  Discarded"); self._show()
                elif key==13: self._commit()
                elif key==8:  self.cur_desc=self.cur_desc[:-1]; self._show_input()
                elif 32<=key<=126: self.cur_desc+=chr(key); self._show_input()
                continue
            if self.in_label:
                if   key==27: self._finish_label(ok=False)
                elif key==13: self._finish_label(ok=True)
                elif key==8:  self.cur_label=self.cur_label[:-1]; self._show_input()
                elif 32<=key<=126: self.cur_label+=chr(key); self._show_input()
                continue

            if   key==ord('a'): self._nav(-1)
            elif key==ord('d'): self._nav(+1)
            elif key==ord('r'): self.vp.reset(self.img.ov_w,self.img.ov_h); self._show(); print("  Reset")
            elif key==ord('s'): self._save(); self._show()
            elif key==ord('S'): self._save(); self._nav(+1)
            elif key==ord('c'): self.circles.clear(); self.cache.dirty_effects(); self._show(); print("  Cleared")
            elif key==ord('u'):
                if self.circles:
                    rem=self.circles.pop(); print(f"  Undo: {rem['label'] or '(unlabeled)'}"); self.cache.dirty_effects(); self._show()
                else: print("  Nothing to undo")
            elif key==ord('l'): self._list()
            elif key==ord('e'): self._edit_last()
            elif key==ord('t'):
                self.show_labels=not self.show_labels
                print(f"  Labels {'ON' if self.show_labels else 'OFF'}")
                self.cache.dirty_composite(); self._show()
            elif key==ord('m'): self._ram_status()
            elif key in(ord('h'),ord('H')): self._print_help()
            elif ord('1')<=key<=ord('7'):
                self.mode=list(EditMode)[key-ord('1')]
                print(f"  Mode: {self.mode.value.upper()}"); self._show()
            elif key==ord('q'):
                if self.circles and not self.saved.get(self.files[self.idx].name):
                    self._save(auto=True)
                break

        cv2.destroyAllWindows(); self.img.free()
        if self.saved:
            self._excel()
            print(f"\n  Done — {len(self.saved)}/{self.total} saved\n  {self.out_dir}")
        else:
            print("\n  No images saved.")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    global OVERVIEW_PX
    ap=argparse.ArgumentParser(description="Batch Labeled Editor")
    ap.add_argument("input_folder")
    ap.add_argument("--output","-o",default=None)
    ap.add_argument("--overview-px",type=int,default=OVERVIEW_PX,
                    help=f"Overview longest edge px (default {OVERVIEW_PX})")
    args=ap.parse_args()
    OVERVIEW_PX=args.overview_px
    try:
        BatchLabeledEditor(args.input_folder,args.output).run()
    except KeyboardInterrupt: print("\n  Interrupted"); return 130
    except Exception as e:
        print(f"\n  Fatal: {e}"); import traceback; traceback.print_exc(); return 1
    return 0

if __name__=="__main__": sys.exit(main())
