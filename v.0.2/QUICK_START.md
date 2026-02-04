# Quick Start Guide - Choosing the Right Editor

## Which Editor Should I Use?

### 🎯 Choose by Your Need:

#### 1. **Just want to highlight areas?**
→ Use `image_circle_editor.py`
- Simplest interface
- Quick highlighting
- No labels needed

#### 2. **Need to label and identify objects?**
→ Use `labeled_circle_editor.py` ⭐ RECOMMENDED FOR LABELING
- Add text labels to circles
- Perfect for dataset creation
- Export labels as text/JSON
- Track what each circle represents

#### 3. **Need different visual effects?**
→ Use `advanced_circle_editor.py`
- 7 different effects (blur, pixelate, etc.)
- Great for privacy protection
- Multiple effect types in one image

#### 4. **Need both effects AND labels?**
→ Use `advanced_labeled_editor.py` ⭐ MOST POWERFUL
- All 7 effects + labeling
- Professional annotation tool
- Complete dataset creation
- JSON export for machine learning

---

## Common Use Cases

### 📊 Dataset Creation / Object Detection Training
**Best Choice:** `labeled_circle_editor.py` or `advanced_labeled_editor.py`

**Why:** You need labeled data with object positions
```bash
python labeled_circle_editor.py training_image.jpg
# Mark objects, add labels, export JSON for ML training
```

### 🔒 Privacy Protection (Blur/Pixelate Faces)
**Best Choice:** `advanced_circle_editor.py`

**Why:** You need blur/pixelate effects
```bash
python advanced_circle_editor.py photo.jpg
# Press '2' for blur, mark faces, save
```

### 📝 Document Annotation
**Best Choice:** `labeled_circle_editor.py`

**Why:** Label important sections with descriptions
```bash
python labeled_circle_editor.py document.pdf.png
# Mark sections, add descriptive labels
```

### 🏥 Medical Image Analysis
**Best Choice:** `advanced_labeled_editor.py`

**Why:** Label regions + highlight abnormalities
```bash
python advanced_labeled_editor.py xray.jpg
# Use different modes for different findings
# Add medical labels to each region
```

### 🚗 Autonomous Vehicle Training Data
**Best Choice:** `advanced_labeled_editor.py`

**Why:** Label objects with different treatments
```bash
python advanced_labeled_editor.py street_view.jpg
# Label: "Car", "Pedestrian", "Traffic Sign"
# Export JSON for training pipeline
```

---

## Feature Comparison Table

| Feature | Basic | Labeled | Advanced | Advanced Labeled |
|---------|-------|---------|----------|------------------|
| Draw circles | ✅ | ✅ | ✅ | ✅ |
| Highlight effect | ✅ | ✅ | ✅ | ✅ |
| Text labels | ❌ | ✅ | ❌ | ✅ |
| Multiple effects | ❌ | ❌ | ✅ | ✅ |
| Label export | ❌ | ✅ (TXT) | ❌ | ✅ (JSON+TXT) |
| Edit labels | ❌ | ✅ | ❌ | ✅ |
| Toggle labels | ❌ | ❌ | ❌ | ✅ |
| Auto-scaling | ❌ | ❌ | ✅ | ✅ |
| Blur effect | ❌ | ❌ | ✅ | ✅ |
| Pixelate effect | ❌ | ❌ | ✅ | ✅ |
| Complexity | Simple | Medium | Medium | Advanced |

---

## Installation (All Versions)

```bash
# Install dependencies
pip install opencv-python numpy

# Test with sample image
python create_test_image.py

# Choose your editor based on needs above
```

---

## Quick Examples

### Example 1: Label 3 Objects
```bash
python labeled_circle_editor.py photo.jpg

# In the app:
# 1. Draw circle around car → type "Red Toyota" → ENTER
# 2. Draw circle around person → type "Pedestrian" → ENTER  
# 3. Draw circle around sign → type "Stop Sign" → ENTER
# 4. Press 'S' to save

# Output:
# - labeled_output.png (image with labels visible)
# - labeled_output.txt (list of labels)
```

### Example 2: Blur Sensitive Data
```bash
python advanced_circle_editor.py document.jpg

# In the app:
# 1. Press '2' to enable blur mode
# 2. Draw circles around SSN, credit card numbers
# 3. Press 'S' to save

# Output:
# - output_advanced.png (blurred version)
```

### Example 3: Create ML Training Data
```bash
python advanced_labeled_editor.py dataset_001.jpg

# In the app:
# 1. Press '1' (highlight mode)
# 2. Mark object → label "cat" → ENTER
# 3. Mark object → label "dog" → ENTER
# 4. Press 'S' to save

# Output:
# - labeled_advanced_output.png (annotated image)
# - labeled_advanced_output.json (structured data)
# - labeled_advanced_output.txt (human readable)
```

---

## Tips for Success

1. **Label Naming Convention:**
   - Use consistent names: "Car" not "car" or "automobile"
   - Be specific: "Red Car" better than "Vehicle"
   - Use underscores for multi-word: "traffic_light"

2. **Circle Drawing:**
   - Draw slightly larger than the object
   - Center the object in the circle
   - Use undo (U) to correct mistakes

3. **Batch Processing:**
   - For 100+ images, consider writing a script
   - Export JSON labels for automated pipelines
   - Keep a naming convention for files

4. **Performance:**
   - Large images auto-scale for smooth editing
   - Output maintains original resolution
   - Close other apps if system slows down

---

## Keyboard Shortcuts Summary

### Common to All:
- `S` - Save
- `C` - Clear all
- `U` - Undo last
- `Q` - Quit

### Labeling Editors Only:
- `L` - List all labels
- `E` - Edit last label
- `ENTER` - Confirm label
- `ESC` - Skip label

### Advanced Editors Only:
- `1-7` - Switch modes
- `T` - Toggle labels (labeled version)

---

## Support & Feedback

If you encounter issues:
1. Check image file format (JPG, PNG supported)
2. Ensure OpenCV is installed correctly
3. Try the test image generator first
4. Review error messages in terminal

For new features or bug reports:
- Document your use case
- Provide sample images (if possible)
- Note your Python and OpenCV versions

---

**Happy Annotating! 🎨**
