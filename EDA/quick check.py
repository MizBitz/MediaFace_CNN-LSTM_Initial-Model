# quick check
import os
root = "dataset_dynamic_3d"
bad = []
for cls in ("open", "closed"):
    folder = os.path.join(root, cls)
    if not os.path.isdir(folder): continue
    for f in os.listdir(folder):
        if f.lower().endswith(".jpg") and not f.endswith(("_dyn_L.jpg","_dyn_R.jpg","_dyn_Combined.jpg")):
            bad.append(os.path.join(folder, f))
print("Unknown lens files:", len(bad))
for p in bad: print(p)

# delete after review
for p in bad: os.remove(p)