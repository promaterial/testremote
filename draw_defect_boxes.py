import os
import cv2
import numpy as np
import glob

# 可自定义类别名
CLASS_NAMES = {
    0: "bubble",
    1: "chipping",
    2: "crack",
    3:'scratches',
    4: 'sc'
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

def process_folder(img_dir, label_dir, out_dir, class_names):
    os.makedirs(out_dir, exist_ok=True)
    img_paths = glob.glob(os.path.join(img_dir, "*.png")) + glob.glob(os.path.join(img_dir, "*.jpg"))
    for img_path in img_paths:
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, base + ".txt")
        img, classes = draw_boxes_on_image(img_path, label_path, class_names)
        for cls in set(classes):
            save_dir = os.path.join(out_dir, class_names.get(cls, str(cls)))
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, base + ".jpg")
            cv2.imencode('.jpg', img)[1].tofile(save_path)  # 支持中文路径

if __name__ == "__main__":
    # 处理train
    process_folder(
        img_dir=r"defect_images/defect_images/train/images",
        label_dir=r"defect_images/defect_images/train/labels",
        out_dir="output/train",
        class_names=CLASS_NAMES
    )
    # 处理valid
    process_folder(
        img_dir=r"defect_images/defect_images/valid/images",
        label_dir=r"defect_images/defect_images/valid/labels",
        out_dir="output/valid",
        class_names=CLASS_NAMES
    )
    print("处理完成，结果保存在output文件夹下。") 