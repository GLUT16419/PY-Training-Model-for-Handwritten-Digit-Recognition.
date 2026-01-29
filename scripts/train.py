import argparse
import os
import sys
import subprocess
from datetime import datetime

# 添加项目根目录到Python搜索路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def get_optimal_batch_size():
    """
    根据GPU内存自动计算最佳批处理大小
    Returns:
        int: 最佳批处理大小
    """
    try:
        import torch
        if torch.cuda.is_available():
            # 获取GPU内存信息
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_memory = memory_info.total / (1024 ** 3)  # 转换为GB
            pynvml.nvmlShutdown()
            
            print(f"[Auto Batch Size] Detected GPU memory: {total_memory:.2f} GB")
            
            # 根据GPU内存确定批处理大小，控制在16GB内存以内
            if total_memory >= 16:
                return 256  # 16GB GPU使用256，确保内存占用合理
            elif total_memory >= 8:
                return 168  # 8GB GPU使用128
            elif total_memory >= 4:
                return 128   # 4GB GPU使用64
            elif total_memory >= 2:
                return 96   # 2GB GPU使用32
            else:
                return 96
        else:
            # CPU模式下使用较小的批处理大小
            print("[Auto Batch Size] No GPU detected, using CPU batch size")
            return 32
    except Exception as e:
        print(f"[Auto Batch Size] Error detecting GPU: {e}, using default batch size")
        return 64

# 延迟导入，减少启动时间
def main():
    """训练脚本主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Train MNIST model')
    parser.add_argument('--model', type=str, default='enhanced', choices=['lightweight', 'enhanced', 'tiny', 'advanced', 'stm32', 'ultralight'],
                        help='Model architecture to use')
    parser.add_argument('--use-depthwise', action='store_true',
                        help='Use depthwise separable convolution for stm32 model')
    parser.add_argument('--epochs', type=int, default=80,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=-1,
                        help='Batch size for training (default: auto-detect based on GPU memory)')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--data-dir', type=str, default='data/MNIST',
                        help='Directory for MNIST data')
    parser.add_argument('--custom-data-dir', type=str, default=None,
                        help='Directory for custom hand-drawn data')
    parser.add_argument('--custom-ratio', type=float, default=0.3,
                        help='Ratio of custom data in mixed dataset')
    parser.add_argument('--model-save-dir', type=str, default='models/trained',
                        help='Directory to save trained models')
    parser.add_argument('--history-save-dir', type=str, default='reports',
                        help='Directory to save training history plots')
    parser.add_argument('--two-stage', action='store_true',
                        help='Enable two-stage training: pre-train on MNIST then fine-tune on mixed data')
    parser.add_argument('--pretrain-epochs', type=int, default=5,
                        help='Number of epochs for pre-training on MNIST')
    parser.add_argument('--finetune-epochs', type=int, default=10,
                        help='Number of epochs for fine-tuning on mixed data')
    
    args = parser.parse_args()
    
    # 自动计算批处理大小（如果未指定）
    if args.batch_size == -1:
        args.batch_size = get_optimal_batch_size()
        print(f'[Auto Batch Size] Using automatically determined batch size: {args.batch_size}')
    
    # 添加时间戳前缀
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 创建日志目录
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日志文件名
    log_filename = f'train_{timestamp}.log'
    log_path = os.path.join(log_dir, log_filename)
    
    # 打开日志文件
    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f'[{timestamp}] 开始MNIST模型训练...\n')
        log_file.write(f'[{timestamp}] 模型架构: {args.model}\n')
        log_file.write(f'[{timestamp}] 训练轮次: {args.epochs}\n')
        log_file.write(f'[{timestamp}] 批处理大小: {args.batch_size}\n')
        log_file.write(f'[{timestamp}] 验证分割比例: {args.val_split}\n')
        if args.custom_data_dir:
            log_file.write(f'[{timestamp}] 自定义数据目录: {args.custom_data_dir}\n')
            log_file.write(f'[{timestamp}] 自定义数据比例: {args.custom_ratio}\n')
        log_file.write(f'[{timestamp}] 日志文件: {log_path}\n\n')
    
    print(f'[{timestamp}] 开始MNIST模型训练...')
    print(f'[{timestamp}] 模型架构: {args.model}')
    print(f'[{timestamp}] 训练轮次: {args.epochs}')
    print(f'[{timestamp}] 批处理大小: {args.batch_size}')
    print(f'[{timestamp}] 验证分割比例: {args.val_split}')
    if args.custom_data_dir:
        print(f'[{timestamp}] 自定义数据目录: {args.custom_data_dir}')
        print(f'[{timestamp}] 自定义数据比例: {args.custom_ratio}')
    print(f'[{timestamp}] 日志文件: {log_path}')
    
    # 延迟导入，减少启动时间
    from src.data_preprocessing import MNISTDataProcessor
    from src.training.model_architectures import get_model
    from src.training import ModelTrainer
    from src.utils import plot_training_history
    from config import config
    
    # 使用配置文件中的值，如果命令行参数未指定
    if args.epochs == 100:  # 默认值
        args.epochs = config.EPOCHS
    if args.batch_size == 64:  # 默认值
        args.batch_size = config.BATCH_SIZE
    if args.data_dir == 'data/MNIST':  # 默认值
        args.data_dir = config.DATA_DIR
    if args.model_save_dir == 'models/trained':  # 默认值
        args.model_save_dir = config.MODEL_SAVE_DIR
    if args.history_save_dir == 'reports':  # 默认值
        args.history_save_dir = config.REPORTS_DIR
    
    # 写入配置到日志文件
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f'[{timestamp}] 使用配置:\n')
        log_file.write(f'[{timestamp}]   训练轮次: {args.epochs}\n')
        log_file.write(f'[{timestamp}]   批处理大小: {args.batch_size}\n')
        log_file.write(f'[{timestamp}]   数据目录: {args.data_dir}\n')
        log_file.write(f'[{timestamp}]   模型保存目录: {args.model_save_dir}\n')
        log_file.write(f'[{timestamp}]   历史保存目录: {args.history_save_dir}\n\n')
    
    print(f'[{timestamp}] 使用配置:')
    print(f'[{timestamp}]   训练轮次: {args.epochs}')
    print(f'[{timestamp}]   批处理大小: {args.batch_size}')
    print(f'[{timestamp}]   数据目录: {args.data_dir}')
    
    # 创建带有时间戳的训练文件夹
    train_folder_name = f'train_{timestamp}'
    train_folder_path = os.path.join(args.model_save_dir, train_folder_name)
    
    # 写入训练文件夹信息到日志
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f'[{timestamp}] 创建训练文件夹: {train_folder_path}\n\n')
    
    print(f'\n[{timestamp}] 创建训练文件夹: {train_folder_path}')
    os.makedirs(train_folder_path, exist_ok=True)
    
    # 1. 加载和预处理数据
    print(f'\n[{timestamp}] 1. 加载和预处理数据...')
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f'[{timestamp}] 1. 加载和预处理数据...\n')
    
    processor = MNISTDataProcessor(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split
    )

    # 两阶段训练策略
    if args.two_stage and args.custom_data_dir:
        print(f'\n[{timestamp}] ======================================')
        print(f'[{timestamp}] 启用两阶段训练!')
        print(f'[{timestamp}] 第一阶段: 在MNIST数据集上预训练')
        print(f'[{timestamp}] 第二阶段: 在混合数据集上微调')
        print(f'[{timestamp}] ======================================')
        
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f'\n[{timestamp}] ======================================\n')
            log_file.write(f'[{timestamp}] 启用两阶段训练!\n')
            log_file.write(f'[{timestamp}] 第一阶段: 在MNIST数据集上预训练\n')
            log_file.write(f'[{timestamp}] 第二阶段: 在混合数据集上微调\n')
            log_file.write(f'[{timestamp}] ======================================\n\n')
        
        # 加载 MNIST 数据集用于预训练
        print(f'\n[{timestamp}] 加载MNIST数据集用于预训练...')
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f'[{timestamp}] 加载MNIST数据集用于预训练...\n')
        
        mnist_train_loader, mnist_val_loader, test_loader = processor.load_data()
        
        input_shape, num_classes = processor.get_data_shape()
        
        print(f'[{timestamp}] 输入形状: {input_shape}')
        print(f'[{timestamp}] 类别数量: {num_classes}')
        print(f'[{timestamp}] MNIST训练批次: {len(mnist_train_loader)}')
        print(f'[{timestamp}] MNIST验证批次: {len(mnist_val_loader)}')
        print(f'[{timestamp}] 测试批次: {len(test_loader)}')
        
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f'[{timestamp}] 输入形状: {input_shape}\n')
            log_file.write(f'[{timestamp}] 类别数量: {num_classes}\n')
            log_file.write(f'[{timestamp}] MNIST训练批次: {len(mnist_train_loader)}\n')
            log_file.write(f'[{timestamp}] MNIST验证批次: {len(mnist_val_loader)}\n')
            log_file.write(f'[{timestamp}] 测试批次: {len(test_loader)}\n\n')
        
        # 2. 创建模型
        print(f'\n[{timestamp}] 2. 创建模型...')
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f'[{timestamp}] 2. 创建模型...\n')
        
        model = get_model(
            model_name=args.model,
            input_shape=input_shape,
            num_classes=num_classes,
            use_depthwise=args.use_depthwise
        )
        
        print(f'[{timestamp}] 模型创建完成: {model.__class__.__name__}')
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f'[{timestamp}] 模型创建完成: {model.__class__.__name__}\n\n')
        
        # 3. 第一阶段：在 MNIST 上预训练
        print(f'\n[{timestamp}] 3. 第一阶段: 在MNIST数据集上预训练...')
        print(f'[{timestamp}] 预训练轮次: {args.pretrain_epochs}')
        
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f'[{timestamp}] 3. 第一阶段: 在MNIST数据集上预训练...\n')
            log_file.write(f'[{timestamp}] 预训练轮次: {args.pretrain_epochs}\n\n')
        
        # 创建预训练日志文件
        pretrain_log_filename = f'pretrain_{timestamp}.log'
        pretrain_log_path = os.path.join(log_dir, pretrain_log_filename)
        
        trainer = ModelTrainer(model, log_path=pretrain_log_path)
        
        pretrained_model = trainer.train(
            train_loader=mnist_train_loader,
            val_loader=mnist_val_loader,
            custom_val_loader=None,
            epochs=args.pretrain_epochs,
            model_save_path=train_folder_path
        )
        
        # 保存预训练模型
        pretrain_model_path = os.path.join(train_folder_path, f'pretrained_model_{timestamp}.pth')
        import torch
        torch.save(pretrained_model.state_dict(), pretrain_model_path)
        print(f'[{timestamp}] 预训练模型保存到: {pretrain_model_path}')
        
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f'[{timestamp}] 预训练模型保存到: {pretrain_model_path}\n')
            log_file.write(f'[{timestamp}] 预训练日志文件: {pretrain_log_path}\n\n')
        
        # 4. 第二阶段：在混合数据上微调
        print(f'\n[{timestamp}] 4. 第二阶段: 加载混合数据集用于微调...')
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f'[{timestamp}] 4. 第二阶段: 加载混合数据集用于微调...\n')
        
        train_loader, val_loader, test_loader, custom_val_loader = processor.load_mixed_data(
            custom_data_dir=args.custom_data_dir,
            custom_ratio=args.custom_ratio
        )
        
        print(f'[{timestamp}] 混合训练批次: {len(train_loader)}')
        print(f'[{timestamp}] 混合验证批次: {len(val_loader)}')
        if custom_val_loader:
            print(f'[{timestamp}] 自定义验证批次: {len(custom_val_loader)}')
        
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f'[{timestamp}] 混合训练批次: {len(train_loader)}\n')
            log_file.write(f'[{timestamp}] 混合验证批次: {len(val_loader)}\n')
            if custom_val_loader:
                log_file.write(f'[{timestamp}] 自定义验证批次: {len(custom_val_loader)}\n')
            log_file.write('\n')
        
        # 微调训练
        print(f'\n[{timestamp}] 5. 在混合数据集上微调...')
        print(f'[{timestamp}] 微调轮次: {args.epochs - args.pretrain_epochs}')
        
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f'[{timestamp}] 5. 在混合数据集上微调...\n')
            log_file.write(f'[{timestamp}] 微调轮次: {args.epochs - args.pretrain_epochs}\n\n')
        
        # 创建微调日志文件
        finetune_log_filename = f'finetune_{timestamp}.log'
        finetune_log_path = os.path.join(log_dir, finetune_log_filename)
        
        # 继续使用同一个训练器，模型已经是预训练的
        trainer = ModelTrainer(model, log_path=finetune_log_path)
        
        trained_model = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            custom_val_loader=custom_val_loader,
            epochs=args.epochs - args.pretrain_epochs,
            model_save_path=train_folder_path
        )
        
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f'[{timestamp}] 微调日志文件: {finetune_log_path}\n\n')
        
    else:
        # 传统训练策略
        # 根据是否提供自定义数据目录选择加载方法
        if args.custom_data_dir:
            print(f'[{timestamp}] 加载混合数据集 (MNIST + 自定义数据)...')
            with open(log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f'[{timestamp}] 加载混合数据集 (MNIST + 自定义数据)...\n')
            
            train_loader, val_loader, test_loader, custom_val_loader = processor.load_mixed_data(
                custom_data_dir=args.custom_data_dir,
                custom_ratio=args.custom_ratio
            )
        else:
            print(f'[{timestamp}] 加载MNIST数据集...')
            with open(log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f'[{timestamp}] 加载MNIST数据集...\n')
            
            train_loader, val_loader, test_loader = processor.load_data()
            custom_val_loader = None
        
        input_shape, num_classes = processor.get_data_shape()
        
        print(f'[{timestamp}] 输入形状: {input_shape}')
        print(f'[{timestamp}] 类别数量: {num_classes}')
        print(f'[{timestamp}] 训练批次: {len(train_loader)}')
        print(f'[{timestamp}] 验证批次: {len(val_loader)}')
        print(f'[{timestamp}] 测试批次: {len(test_loader)}')
        if custom_val_loader:
            print(f'[{timestamp}] 自定义验证批次: {len(custom_val_loader)}')
        
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f'[{timestamp}] 输入形状: {input_shape}\n')
            log_file.write(f'[{timestamp}] 类别数量: {num_classes}\n')
            log_file.write(f'[{timestamp}] 训练批次: {len(train_loader)}\n')
            log_file.write(f'[{timestamp}] 验证批次: {len(val_loader)}\n')
            log_file.write(f'[{timestamp}] 测试批次: {len(test_loader)}\n')
            if custom_val_loader:
                log_file.write(f'[{timestamp}] 自定义验证批次: {len(custom_val_loader)}\n')
            log_file.write('\n')
        
        # 2. 创建模型
        print(f'\n[{timestamp}] 2. 创建模型...')
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f'[{timestamp}] 2. 创建模型...\n')
        
        model = get_model(
            model_name=args.model,
            input_shape=input_shape,
            num_classes=num_classes,
            use_depthwise=args.use_depthwise
        )
        
        print(f'[{timestamp}] 模型创建完成: {model.__class__.__name__}')
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f'[{timestamp}] 模型创建完成: {model.__class__.__name__}\n\n')
        
        # 3. 训练模型
        print(f'\n[{timestamp}] 3. 训练模型...')
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f'[{timestamp}] 3. 训练模型...\n\n')
        
        # 创建训练日志文件
        train_log_filename = f'training_{timestamp}.log'
        train_log_path = os.path.join(log_dir, train_log_filename)
        
        trainer = ModelTrainer(model, log_path=train_log_path)
        
        trained_model = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            custom_val_loader=custom_val_loader,
            epochs=args.epochs,
            model_save_path=train_folder_path
        )
    
    # 4. 保存训练历史
    print(f'\n[{timestamp}] 4. 保存训练历史...')
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f'[{timestamp}] 4. 保存训练历史...\n')
    
    history = trainer.get_train_history()
    
    # 创建历史保存目录
    os.makedirs(args.history_save_dir, exist_ok=True)
    
    # 绘制并保存训练历史
    history_plot_path = os.path.join(args.history_save_dir, f'training_history_{args.model}.png')
    plot_training_history(history, save_path=history_plot_path)
    
    print(f'[{timestamp}] 训练历史图表保存到: {history_plot_path}')
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f'[{timestamp}] 训练历史图表保存到: {history_plot_path}\n\n')
    
    # 5. 生成评估报告
    print(f'\n[{timestamp}] 5. 生成评估报告...')
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f'[{timestamp}] 5. 生成评估报告...\n')
    
    try:
        # 运行报告生成脚本
        report_cmd = f'python scripts/generate_report.py --model {args.model}'
        print(f'[{timestamp}] 运行: {report_cmd}')
        
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f'[{timestamp}] 运行: {report_cmd}\n')
        
        # 使用check=True确保命令执行失败时会抛出异常
        result = subprocess.run(report_cmd, shell=True, capture_output=True, text=True, check=True)
        print(f'[{timestamp}] 报告生成输出:')
        print(result.stdout)
        if result.stderr:
            print(f'[{timestamp}] 报告生成错误:')
            print(result.stderr)
        
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f'[{timestamp}] 报告生成输出:\n{result.stdout}\n')
            if result.stderr:
                log_file.write(f'[{timestamp}] 报告生成错误:\n{result.stderr}\n')
    except subprocess.CalledProcessError as e:
        print(f'[{timestamp}] 生成报告时出错: {e}')
        print(f'[{timestamp}] 输出: {e.stdout}')
        print(f'[{timestamp}] 错误: {e.stderr}')
        
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f'[{timestamp}] 生成报告时出错: {e}\n')
            log_file.write(f'[{timestamp}] 输出: {e.stdout}\n')
            log_file.write(f'[{timestamp}] 错误: {e.stderr}\n')
    
    # 写入训练完成信息到日志
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f'\n[{timestamp}] 训练完成成功!\n')
        log_file.write(f'[{timestamp}] 训练文件夹: {train_folder_path}\n')
        log_file.write(f'[{timestamp}] 最佳模型保存在: {train_folder_path}\n')
        log_file.write(f'[{timestamp}] 训练历史: {history_plot_path}\n')
        log_file.write(f'[{timestamp}] 评估报告: reports/\n')
        log_file.write(f'[{timestamp}] 日志文件: {log_path}\n')
    
    print(f'\n[{timestamp}] 训练完成成功!')
    print(f'[{timestamp}] 训练文件夹: {train_folder_path}')
    print(f'[{timestamp}] 最佳模型保存在: {train_folder_path}')
    print(f'[{timestamp}] 训练历史: {history_plot_path}')
    print(f'[{timestamp}] 评估报告: reports/')
    print(f'[{timestamp}] 日志文件: {log_path}')

if __name__ == '__main__':
    main()