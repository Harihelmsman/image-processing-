import cv2
import json
import numpy as np

image_path = "output_image.jpg"
json_path = "2016_mask.json"

# color mapping
CLASS_COLORS = {
    "Person_FG":"FF8000",      # green
    "Person":"FFFF33",       # red
    "Person_Group": "6666FF"     # blue
}





def fit_to_screen(img, max_w=1400, max_h=900):

    h, w = img.shape[:2]

    scale = min(max_w / w, max_h / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h))

    return resized, scale


# load image
img = cv2.imread(image_path)

img, scale = fit_to_screen(img)


# load json
with open(json_path) as f:
    data = json.load(f)

shapes = data["shapes"]

polygons = []


def hex_to_bgr(hex_color):

    hex_color = hex_color.lstrip('#')

    r = int(hex_color[0:2],16)
    g = int(hex_color[2:4],16)
    b = int(hex_color[4:6],16)

    return (b,g,r)


for shape in shapes:

    label = shape["label"]
    gid = shape["group_id"]

    pts = (np.array(shape["points"], dtype=np.float32) * scale).astype(np.int32)

    color = hex_to_bgr(shape["shape_color"])

    polygons.append({
        "label": label,
        "id": gid,
        "points": pts,
        "color": color
    })


hover_index = -1


def mouse_move(event,x,y,flags,param):
    global hover_index

    hover_index = -1

    for i,p in enumerate(polygons):

        inside = cv2.pointPolygonTest(p["points"], (x,y), False)

        if inside >= 0:
            hover_index = i
            break


cv2.namedWindow("viewer", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("viewer", mouse_move)


while True:

    canvas = img.copy()

    for i,p in enumerate(polygons):

        pts = p["points"]
        color = p["color"]

        if i == hover_index:

            overlay = canvas.copy()
            cv2.fillPoly(overlay,[pts],color)

            alpha = 0.4
            canvas = cv2.addWeighted(overlay, alpha, canvas, 1-alpha, 0)

            text = f"{p['label']} Group_Id:{p['id']}"

            x,y = pts[0]

            cv2.putText(canvas,text,(x,y-5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,(0,255,0),2)

        else:

            cv2.polylines(canvas,[pts],True,color,2)


    cv2.imshow("viewer",canvas)

    if cv2.waitKey(30) == 27:
        break


cv2.destroyAllWindows()