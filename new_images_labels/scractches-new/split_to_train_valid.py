import os
import shutil
import random

# 源目录
img_dir = os.path.join(os.path.dirname(__file__), 'scratches_new', 'images')
label_dir = os.path.join(os.path.dirname(__file__), 'scratches_new', 'labels')

# 目标目录
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../defect_images/defect_images'))
train_img_dir = os.path.join(root, 'train', 'images')
train_label_dir = os.path.join(root, 'train', 'labels')
valid_img_dir = os.path.join(root, 'valid', 'images')
valid_label_dir = os.path.join(root, 'valid', 'labels')

os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(valid_img_dir, exist_ok=True)
os.makedirs(valid_label_dir, exist_ok=True)

# 获取所有图片文件名（不带扩展名）
img_files = [f for f in os.listdir(img_dir) if f.lower().endswith('.png')]
bases = [os.path.splitext(f)[0] for f in img_files]

# 随机划分
random.seed(42)
random.shuffle(bases)
n = len(bases)
split = int(n * 0.8)
train_bases = bases[:split]
valid_bases = bases[split:]

for base in train_bases:
    shutil.copy(os.path.join(img_dir, base + '.png'), os.path.join(train_img_dir, base + '.png'))
    shutil.copy(os.path.join(label_dir, base + '.txt'), os.path.join(train_label_dir, base + '.txt'))
for base in valid_bases:
    shutil.copy(os.path.join(img_dir, base + '.png'), os.path.join(valid_img_dir, base + '.png'))
    shutil.copy(os.path.join(label_dir, base + '.txt'), os.path.join(valid_label_dir, base + '.txt'))

print(f"划分完成，训练集{len(train_bases)}，验证集{len(valid_bases)}。") 