import json
import os
import torch
print(torch.cuda.is_available())
'''
任务：实例分割，labelme的json文件, 转txt文件
Ultralytics YOLO format
<class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>
'''

# 类别映射表，定义每个类别对应的ID
label_to_class_id = {
    "rebar exposure": 0,
    "spalling": 1,
    # 根据需要添加更多类别
}


# json转txt
def convert_labelme_json_to_yolo(json_file, output_dir):
    with open(json_file, 'r') as f:
        labelme_data = json.load(f)

    # 获取文件名（不含扩展名）
    file_name = os.path.splitext(os.path.basename(json_file))[0]

    # 输出的txt文件路径
    txt_file_path = os.path.join(output_dir, f"{file_name}.txt")

    # 从JSON文件中获取图片的实际尺寸
    img_width = labelme_data['imageWidth']
    img_height = labelme_data['imageHeight']

    with open(txt_file_path, 'w') as txt_file:
        for shape in labelme_data['shapes']:
            label = shape['label']
            points = shape['points']

            # 根据类别映射表获取类别ID，如果类别不在映射表中，跳过该标签
            class_id = label_to_class_id.get(label)
            if class_id is None:
                print(f"Warning: Label '{label}' not found in class mapping. Skipping.")
                continue

            # 将点的坐标归一化到0-1范围
            normalized_points = [(x / img_width, y / img_height) for x, y in points]

            # 写入类别ID
            txt_file.write(f"{class_id}")

            # 写入多边形掩膜的所有归一化顶点坐标
            for point in normalized_points:
                txt_file.write(f" {point[0]:.6f} {point[1]:.6f}")
            txt_file.write("\n")


if __name__ == "__main__":
    json_dir = "D:\BaiduNetdiskDownload/20221120_datasets - new/20221120_datasets - new\Annotations"  # 替换为LabelMe标注的JSON文件目录
    output_dir = "D:\BaiduNetdiskDownload/20221120_datasets - new/20221120_datasets - new/txt"  # 输出的YOLO格式txt文件目录

    # 创建输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 批量处理所有json文件
    for json_file in os.listdir(json_dir):
        if json_file.endswith(".json"):
            json_path = os.path.join(json_dir, json_file)
            convert_labelme_json_to_yolo(json_path, output_dir)