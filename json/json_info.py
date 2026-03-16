import json
import base64

with open ('20240812_154016_mask.json') as f:
    data=json.load(f)


image_data= data["imageData"] 

data["imageData"] =None


# new_data=json.dumps(data,indent=2)
# print(new_data)
CLASS_COLORS = {
    "Person_FG":"FF8000",      # green
    "Person":"FFFF33",       # red
    "Person_Group": "6666FF"     # blue
}



with open('20240812_154016_mask.json', "r") as f:
    data = json.load(f)

    id_map = {}
    counter = 1

    for shape in data["shapes"]:

        label = shape["label"]

        old_id = shape["group_id"]

        key = f"{label}_{old_id}"

        if label not in CLASS_COLORS:
            print(label)
            continue
        
        # assign sequential id
        if key not in id_map:
            id_map[key] = counter
            counter += 1

        new_id = id_map[key]

        shape["group_id"] = new_id

        
        # apply color
        if label in CLASS_COLORS:
            shape["shape_color"] = CLASS_COLORS[label]
            print(label ,new_id,CLASS_COLORS[label])
   
    with open('2016_mask.json', "w") as f:
        json.dump(data, f, indent=4)

    print("Processed:'2016_mask.json'", )
# with open('20240812_154016_mask.json', "r") as f:
#     data = json.load(f)

#     for shape in data["shapes"]:

#         label = shape["label"]

#         if label in CLASS_COLORS:
#             shape["shape_color"] = CLASS_COLORS[label]
#             print(label,shape["shape_color"] )
#     with open('2026_mask.json', "w") as f:
#         json.dump(data, f, indent=4)

#     print("Updated:", '2026_mask.json')

# with open ('new_mask.json','w') as f:
#     data=json.dump(data,f,indent=4)


with open('2016_mask.json ') as f:
      data=json.load(f)


for  d in data["shapes"]:
    print(d["label"],d["group_id"],d["shape_color"])




