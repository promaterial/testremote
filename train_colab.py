# from ultralytics import YOLO
#
# if __name__ == '__main__':
#     model = YOLO(r'ultralytics/cfg/models/v8/yolov8-seg.yaml')
#     model.train(data=r'D:\waibao\yolo11\ultralytics-main\ultralytics\cfg\datasets\mydata.yaml',
#                 imgsz=640,
#                 epochs=100,
#                 single_cls=True,
#                 batch=16,
#                 workers=10,
#                 device='0',
#                 )
#


from ultralytics import YOLO
if __name__ == '__main__':

# 加载预训练的模型
# model = YOLO("yolo11m-seg.yaml").load("weights/yolo11m-seg.pt")
    model = YOLO(r'ultralytics/cfg/models/11/yolo11.yaml')

    # 定义训练参数，添加默认值、范围和中文注释
    train_params = {
        'data': r"/content/drive/MyDrive/Steel_surface_defect_detection/ultralytics/cfg/datasets/mydata.yaml",  # 数据集配置文件路径，需要自定义修改
        'epochs': 500,  # 总训练轮次，默认值 100，范围 >= 1
        'imgsz': 640,  # 输入图像大小，默认值 640，范围 >= 32
        'batch': 16,  # 批次大小，默认值 16，范围 >= 1
        'save': True,  # 是否保存训练结果和模型，默认值 True
        'save_period': -1,  # 模型保存频率，默认值 -1，表示只保存最终结果
        'cache': False,  # 是否缓存数据集，默认值 False
        'device': 0,  # 训练设备，默认值 None，支持 "cpu", "gpu"(device=0,1), "mps"
        'workers': 8,  # 数据加载线程数，默认值 8，影响数据预处理速度
        'project': "runs",   # 项目主目录为 runs
        'name': "newdatav2-yolo11yaml",  # 子目录为 train_colab.py
        'exist_ok': False,  # 是否覆盖已有项目/名称目录，默认值 False
        'optimizer': 'auto',  # 优化器，默认值 'auto'，支持 'SGD', 'Adam', 'AdamW'
        'verbose': True,  # 是否启用详细日志输出，默认值 False
        'seed': 0,  # 随机种子，确保结果的可重复性，默认值 0
        'deterministic': True,  # 是否强制使用确定性算法，默认值 True
        'single_cls': False,  # 是否将多类别数据集视为单一类别，默认值 False
        'rect': False,  # 是否启用矩形训练（优化批次图像大小），默认值 False
        'cos_lr': False,  # 是否使用余弦学习率调度器，默认值 False
        'close_mosaic': 10,  # 在最后 N 轮次中禁用 Mosaic 数据增强，默认值 10
        'resume': False,  # 是否从上次保存的检查点继续训练，默认值 False
        'amp': True,  # 是否启用自动混合精度（AMP）训练，默认值 True
        'fraction': 1.0,  # 使用数据集的比例，默认值 1.0
        'profile': False,  # 是否启用 ONNX 或 TensorRT 模型优化分析，默认值 False
        'freeze': None,  # 冻结模型的前 N 层，默认值 None
        'lr0': 0.01,  # 初始学习率，默认值 0.01，范围 >= 0
        'lrf': 0.01,  # 最终学习率与初始学习率的比值，默认值 0.01
        'momentum': 0.937,  # SGD 或 Adam 的动量因子，默认值 0.937，范围 [0, 1]
        'weight_decay': 0.0005,  # 权重衰减，防止过拟合，默认值 0.0005
        'warmup_epochs': 3.0,  # 预热学习率的轮次，默认值 3.0
        'warmup_momentum': 0.8,  # 预热阶段的初始动量，默认值 0.8
        'warmup_bias_lr': 0.1,  # 预热阶段的偏置学习率，默认值 0.1
        'box': 7.5,  # 边框损失的权重，默认值 7.5
        'cls': 0.5,  # 分类损失的权重，默认值 0.5
        'dfl': 1.5,  # 分布焦点损失的权重，默认值 1.5
        'pose': 12.0,  # 姿态损失的权重，默认值 12.0
        'kobj': 1.0,  # 关键点目标损失的权重，默认值 1.0
        'label_smoothing': 0.0,  # 标签平滑处理，默认值 0.0
        'nbs': 64,  # 归一化批次大小，默认值 64
        'overlap_mask': True,  # 是否在训练期间启用掩码重叠，默认值 True
        'mask_ratio': 4,  # 掩码下采样比例，默认值 4
        'dropout': 0.0,  # 随机失活率，用于防止过拟合，默认值 0.0
        'val': True,  # 是否在训练期间启用验证，默认值 True
        'plots': True,  # 是否生成训练曲线和验证指标图，默认值 True

        # 数据增强相关参数
        'hsv_h': 0.2,  # 色相变化范围 (0.0 - 1.0)，默认值 0.015
        'hsv_s': 0.7,  # 饱和度变化范围 (0.0 - 1.0)，默认值 0.7
        'hsv_v': 0.4,  # 亮度变化范围 (0.0 - 1.0)，默认值 0.4
        'degrees': 30.0,  # 旋转角度范围 (-180 - 180)，默认值 0.0
        'translate': 0.1,  # 平移范围 (0.0 - 1.0)，默认值 0.1
        'scale': 0.5,  # 缩放比例范围 (>= 0.0)，默认值 0.5
        'shear': 0.0,  # 剪切角度范围 (-180 - 180)，默认值 0.0
        'perspective': 0.0,  # 透视变化范围 (0.0 - 0.001)，默认值 0.0
        'flipud': 0.0,  # 上下翻转概率 (0.0 - 1.0)，默认值 0.0
        'fliplr': 0.5,  # 左右翻转概率 (0.0 - 1.0)，默认值 0.5
        'bgr': 0.0,  # BGR 色彩顺序调整概率 (0.0 - 1.0)，默认值 0.0
        'mosaic': 0.5,  # Mosaic 数据增强 (0.0 - 1.0)，默认值 1.0
        'mixup': 0.0,  # Mixup 数据增强 (0.0 - 1.0)，默认值 0.0
        'copy_paste': 0.0,  # Copy-Paste 数据增强 (0.0 - 1.0)，默认值 0.0
        'copy_paste_mode': 'flip',  # Copy-Paste 增强模式 ('flip' 或 'mixup')，默认值 'flip'
        'auto_augment': 'randaugment',  # 自动增强策略 ('randaugment', 'autoaugment', 'augmix')，默认值 'randaugment'
        'erasing': 0.4,  # 随机擦除增强比例 (0.0 - 0.9)，默认值 0.4
        'crop_fraction': 1.0,  # 裁剪比例 (0.1 - 1.0)，默认值 1.0

    }

    # 进行训练
    results = model.train(**train_params)

    # ===================== 统计模型速度与复杂度指标 =====================
    import torch
    import time
    import os
    from pathlib import Path
    try:
        from thop import profile
    except ImportError:
        profile = None
        print("未安装thop库，无法统计FLOPs。可通过 pip install thop 安装。")

    # 1. 统计参数量
    model_torch = model.model if hasattr(model, 'model') else model
    total_params = sum(p.numel() for p in model_torch.parameters())

    # 2. 统计FLOPs（如有thop）
    dummy_input = torch.randn(1, 3, train_params['imgsz'], train_params['imgsz']).to(model_torch.device)
    flops = None
    if profile is not None:
        try:
            flops, _ = profile(model_torch, inputs=(dummy_input,), verbose=False)
            flops = flops / 1e9  # 转为GFLOPs
        except Exception as e:
            print(f"FLOPs统计失败: {e}")

    # 3. 推理速度（FPS/Latency）
    model_torch.eval()
    with torch.no_grad():
        warmup = 10
        repeat = 50
        # 预热
        for _ in range(warmup):
            _ = model_torch(dummy_input)
        # 正式计时
        start = time.time()
        for _ in range(repeat):
            _ = model_torch(dummy_input)
        end = time.time()
        latency = (end - start) / repeat * 1000  # ms
        fps = 1000.0 / latency if latency > 0 else 0

    # 4. 模型文件大小
    last_weight = None
    if hasattr(results, 'save_dir'):
        # Ultralytics 3.x/8.x
        weight_dir = Path(results.save_dir)
        for f in weight_dir.glob('weights/*.pt'):
            last_weight = f
            break
    if last_weight is None:
        # 兼容性查找
        for root, dirs, files in os.walk('runs'):
            for file in files:
                if file.endswith('.pt'):
                    last_weight = os.path.join(root, file)
                    break
    model_size = os.path.getsize(last_weight) / 1024 / 1024 if last_weight else None

    # 5. 显存占用（可选）
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        gpu_mem = None

    # 6. mAP（Ultralytics会自动输出）

    # 7. 结果输出与保存
    log_str = f"\n模型统计信息：\n" \
             f"参数量: {total_params:,}\n" \
             f"FLOPs: {flops:.2f} GFLOPs\n" if flops is not None else "FLOPs: 未统计\n" \
             f"推理时延: {latency:.2f} ms\n" \
             f"FPS: {fps:.2f}\n" \
             f"模型文件大小: {model_size:.2f} MB\n" if model_size is not None else "模型文件大小: 未找到\n" \
             f"最大显存占用: {gpu_mem:.2f} MB\n" if gpu_mem is not None else "最大显存占用: CPU\n"
    print(log_str)
    # 保存到日志文件
    with open('model_speed_log.txt', 'a', encoding='utf-8') as f:
        f.write(log_str + '\n')