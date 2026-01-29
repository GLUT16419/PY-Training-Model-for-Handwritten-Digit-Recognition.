import argparse
import os
import sys
import torch

# 添加项目根目录到Python搜索路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing.data_processor import MNISTDataProcessor
from src.training.model_architectures import get_model
from src.evaluation import ModelEvaluator
from src.deployment.model_quantizer import ModelQuantizer, get_quantizable_model

def main():
    """评估脚本主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Evaluate MNIST model')
    parser.add_argument('--model', type=str, default='lightweight', choices=['lightweight', 'enhanced', 'tiny', 'advanced', 'stm32', 'ultralight'],
                        help='Model architecture to use')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model weights')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--data-dir', type=str, default='data/MNIST',
                        help='Directory for MNIST data')
    parser.add_argument('--custom-data-dir', type=str, default='data/custom/unified_dataset',
                        help='Directory for custom data')
    parser.add_argument('--eval-save-dir', type=str, default='reports',
                        help='Directory to save evaluation results')
    parser.add_argument('--quantize', action='store_true',
                        help='Quantize model to INT8 format')
    parser.add_argument('--quantization-type', type=str, default='ptq', choices=['ptq', 'qat'],
                        help='Quantization type: PTQ (Post-training Quantization) or QAT (Quantization-Aware Training)')
    parser.add_argument('--use-depthwise', action='store_true',
                        help='Use depthwise separable convolution for stm32 model')
    
    args = parser.parse_args()
    
    print('开始MNIST模型评估...')
    print(f'模型架构: {args.model}')
    print(f'模型路径: {args.model_path}')
    print(f'批大小: {args.batch_size}')
    print(f'Quantize model: {args.quantize}')
    if args.quantize:
        print(f'Quantization type: {args.quantization_type}')
    print(f'Use depthwise convolution: {args.use_depthwise}')
    
    # 1. 加载数据
    print('\n1. 加载数据...')
    processor = MNISTDataProcessor(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
    
    train_loader, val_loader, test_loader = processor.load_data()
    input_shape, num_classes = processor.get_data_shape()
    
    print(f'输入形状: {input_shape}')
    print(f'类别数量: {num_classes}')
    print(f'测试批次数: {len(test_loader)}')
    
    # 加载自定义数据验证集
    custom_val_loader = None
    if os.path.exists(args.custom_data_dir):
        print(f'\n加载自定义数据验证集...')
        # 使用 load_mixed_data 来获取自定义数据验证加载器
        _, _, _, custom_val_loader = processor.load_mixed_data(args.custom_data_dir)
        if custom_val_loader:
            print(f'自定义数据验证批次数: {len(custom_val_loader)}')
    
    # 2. 创建和加载模型
    print('\n2. 创建和加载模型...')
    
    if args.quantize and args.model == 'stm32':
        # 使用可量化模型
        model = get_quantizable_model(
            model_name=args.model,
            input_shape=input_shape,
            num_classes=num_classes,
            use_depthwise=args.use_depthwise
        )
    else:
        # 使用普通模型
        model = get_model(
            model_name=args.model,
            input_shape=input_shape,
            num_classes=num_classes,
            use_depthwise=args.use_depthwise
        )
    
    # 加载模型权重
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.eval()
    
    # 3. 量化模型
    if args.quantize:
        print('\n3. 量化模型...')
        quantizer = ModelQuantizer()
        
        if args.quantization_type == 'ptq':
            # 使用PTQ量化
            model = quantizer.quantize_ptq(model, val_loader)
        else:
            # 使用QAT量化
            model = quantizer.quantize_qat(model, train_loader, val_loader, epochs=5)
        
        print('模型量化完成！')
    
    print(f'模型加载完成: {model.__class__.__name__}')
    
    # 4. 评估模型
    print('\n4. 评估模型...')
    evaluator = ModelEvaluator(model)
    
    # 运行 MNIST 数据集评估
    print('\n评估 MNIST 数据集:')
    mnist_results = evaluator.run_evaluation(
        data_loader=test_loader,
        save_dir=args.eval_save_dir,
        dataset_name='MNIST'
    )
    
    # 运行自定义数据集评估
    if custom_val_loader:
        print('\n评估自定义数据集:')
        custom_results = evaluator.run_evaluation(
            data_loader=custom_val_loader,
            save_dir=args.eval_save_dir,
            dataset_name='Custom'
        )
    
    # 5. 比较量化前后性能（如果启用了量化）
    if args.quantize:
        print('\n5. 比较量化前后性能...')
        # 创建浮点模型用于比较
        float_model = get_model(
            model_name=args.model,
            input_shape=input_shape,
            num_classes=num_classes,
            use_depthwise=args.use_depthwise
        )
        float_model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
        float_model.eval()
        
        # 评估浮点模型
        float_evaluator = ModelEvaluator(float_model)
        float_results = float_evaluator.run_evaluation(
            data_loader=test_loader,
            save_dir=None,
            dataset_name='MNIST'
        )
        
        print(f'浮点模型准确率: {float_results["accuracy"]:.2f}%')
        print(f'量化模型准确率: {mnist_results["accuracy"]:.2f}%')
        print(f'准确率差异: {abs(float_results["accuracy"] - mnist_results["accuracy"]):.2f}%')
    
    print('\n评估完成！')
    print(f'评估结果保存在: {args.eval_save_dir}')

if __name__ == '__main__':
    main()