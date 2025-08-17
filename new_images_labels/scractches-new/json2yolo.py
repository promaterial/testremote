import os
import json
import glob
import shutil

# 1. 自动统计所有类别并编号
def get_label_map(json_dir):
    label_set = set()
    for json_file in glob.glob(os.path.join(json_dir, "*.json")):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for shape in data.get("shapes", []):
                label_set.add(shape["label"])
    label_list = sorted(list(label_set))
    return {label: idx for idx, label in enumerate(label_list)}

# 2. 单个json转yolo txt
def convert_json_to_txt(json_path, label_map, img_out_dir, label_out_dir, new_base):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    h, w = data["imageHeight"], data["imageWidth"]
    lines = []
    for shape in data.get("shapes", []):
        label = shape["label"]
        points = shape["points"]
        if shape.get("shape_type", "") == "rectangle" and len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = abs(x2 - x1) / w
            bh = abs(y2 - y1) / h
            cls_id = label_map[label]
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    # 保存txt
    txt_path = os.path.join(label_out_dir, new_base + ".txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    # 同步重命名图片
    img_path = os.path.join(os.path.dirname(json_path), data["imagePath"])
    if os.path.exists(img_path):
        ext = os.path.splitext(data["imagePath"])[1]
        new_img_path = os.path.join(img_out_dir, new_base + ext)
        shutil.copy(img_path, new_img_path)

if __name__ == "__main__":
    json_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(json_dir, "scratches_new")
    img_out_dir = os.path.join(out_dir, "images")
    label_out_dir = os.path.join(out_dir, "labels")
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(label_out_dir, exist_ok=True)
    # 手动指定类别映射
    label_map = {
        "bubble": 0,
        "chipping": 1,
        "crack": 2,
        "scratches": 3,
        "sc": 4
    }
    print("类别映射：", label_map)
    # 批量转换并重命名
    json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    for idx, json_file in enumerate(json_files, 1):
        new_base = f"scratches_new_{idx}"
        convert_json_to_txt(json_file, label_map, img_out_dir, label_out_dir, new_base)
    print("全部转换完成，图片和YOLO格式txt分别保存在", img_out_dir, "和", label_out_dir) 