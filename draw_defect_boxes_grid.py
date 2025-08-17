import os
import cv2
import numpy as np
import glob

# 可自定义类别名
CLASS_NAMES = {
    0: "bubble",
    1: "chipping",
    2: "crack",
    3: "scratches",
    4: "sc"
    # ... 如有更多类别请补充
}

def draw_boxes_on_image(img_path, label_path, class_names):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)  # 支持中文路径
    h, w = img.shape[:2]
    if not os.path.exists(label_path):
        return img, []
    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    classes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, cx, cy, bw, bh = map(float, parts)
        classes.append(int(cls))
        # 反归一化
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        # 画红色框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # 写类别
        label = class_names.get(int(cls), str(int(cls)))
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return img, classes

def collect_by_class_and_source(img_dir, label_dir, class_names):
    img_paths = glob.glob(os.path.join(img_dir, "*.png")) + glob.glob(os.path.join(img_dir, "*.jpg"))
    # {类别: {"old": [img, ...], "new": [img, ...]}}
    class_to_imgs = {cls: {"old": [], "new": []} for cls in class_names.keys()}
    for img_path in img_paths:
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, base + ".txt")
        img, classes = draw_boxes_on_image(img_path, label_path, class_names)
        source = "new" if "new" in base.lower() else "old"
        for cls in set(classes):
            class_to_imgs[cls][source].append(img)
    return class_to_imgs

def save_grid_images_by_source(class_to_imgs, out_dir, grid_shape=(2, 5), resize_shape=None):
    os.makedirs(out_dir, exist_ok=True)
    for cls, src_imgs in class_to_imgs.items():
        cls_name = CLASS_NAMES.get(cls, str(cls))
        for source in ["old", "new"]:
            imgs = src_imgs[source]
            if not imgs:
                continue
            for i in range(0, len(imgs), grid_shape[0]*grid_shape[1]):
                group = imgs[i:i+grid_shape[0]*grid_shape[1]]
                # 统一尺寸
                if resize_shape is None:
                    h, w = group[0].shape[:2]
                else:
                    h, w = resize_shape
                group = [cv2.resize(img, (w, h)) for img in group]
                # 补齐空白
                if len(group) < grid_shape[0]*grid_shape[1]:
                    for _ in range(grid_shape[0]*grid_shape[1] - len(group)):
                        group.append(np.ones((h, w, 3), dtype=np.uint8)*255)
                # 拼接
                rows = []
                for r in range(grid_shape[0]):
                    row_imgs = group[r*grid_shape[1]:(r+1)*grid_shape[1]]
                    rows.append(np.hstack(row_imgs))
                grid_img = np.vstack(rows)
                save_path = os.path.join(
                    out_dir, f"{cls_name}_{source}_{i//(grid_shape[0]*grid_shape[1])+1}.jpg"
                )
                cv2.imencode('.jpg', grid_img)[1].tofile(save_path)
    print(f"拼接输出完成，结果保存在{out_dir}")

# 主程序调用
if __name__ == "__main__":
    # 处理train
    class_to_imgs = collect_by_class_and_source(
        img_dir=r"defect_images/defect_images/train/images",
        label_dir=r"defect_images/defect_images/train/labels",
        class_names=CLASS_NAMES
    )
    save_grid_images_by_source(class_to_imgs, out_dir="output/train_grid", grid_shape=(2, 5), resize_shape=None)

    # 处理valid
    class_to_imgs = collect_by_class_and_source(
        img_dir=r"defect_images/defect_images/valid/images",
        label_dir=r"defect_images/defect_images/valid/labels",
        class_names=CLASS_NAMES
    )
    save_grid_images_by_source(class_to_imgs, out_dir="output/valid_grid", grid_shape=(2, 5), resize_shape=None) 