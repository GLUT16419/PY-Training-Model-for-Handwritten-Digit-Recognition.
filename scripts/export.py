import argparse
import os
import sys
import torch
import glob

# 添加项目根目录到Python搜索路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.model_architectures import get_model
from src.deployment import ModelExporter
from src.deployment.model_quantizer import ModelQuantizer, get_quantizable_model

def get_model_accuracies(model_path):
    """
    从模型文件名中提取准确率信息
    Args:
        model_path: 模型文件路径
    Returns:
        mnist_acc: MNIST验证集准确率
        custom_acc: 自定义数据集准确率
    """
    try:
        filename = os.path.basename(model_path)
        parts = filename.split('_')
        if len(parts) >= 4:
            mnist_acc = float(parts[2])
            custom_acc = float(parts[3])
            return mnist_acc, custom_acc
        return 0.0, 0.0
    except Exception as e:
        print(f"Error extracting accuracies from filename: {e}")
        return 0.0, 0.0

def main():
    """导出脚本主函数"""
    try:
        # 解析命令行参数
        parser = argparse.ArgumentParser(description='Export MNIST model')
        parser.add_argument('--model', type=str, default='lightweight', choices=['lightweight', 'tiny', 'simple', 'stm32', 'ultralight'],
                        help='Model architecture to export')
        parser.add_argument('--model-path', type=str, required=True,
                            help='Path to trained model weights')
        parser.add_argument('--export-dir', type=str, default='models/exported',
                            help='Directory to save exported models')
        parser.add_argument('--quantize', action='store_true',
                            help='Quantize model to INT8 format')
        parser.add_argument('--quantization-type', type=str, default='ptq', choices=['ptq', 'qat'],
                            help='Quantization type: PTQ (Post-training Quantization) or QAT (Quantization-Aware Training)')
        parser.add_argument('--use-depthwise', action='store_true',
                            help='Use depthwise separable convolution for stm32 model')
        
        args = parser.parse_args()
        
        print('Starting MNIST model export...')
        print(f'Model architecture: {args.model}')
        print(f'Model path: {args.model_path}')
        print(f'Export directory: {args.export_dir}')
        print(f'Quantize model: {args.quantize}')
        if args.quantize:
            print(f'Quantization type: {args.quantization_type}')
        print(f'Use depthwise convolution: {args.use_depthwise}')
        
        # 验证模型路径是否存在
        if not os.path.exists(args.model_path):
            print(f"Error: Model path does not exist: {args.model_path}")
            return
        
        # 创建导出目录
        os.makedirs(args.export_dir, exist_ok=True)
        
        # 获取模型准确率
        mnist_acc, custom_acc = get_model_accuracies(args.model_path)
        print(f"Model accuracies - MNIST: {mnist_acc:.2f}%, Custom: {custom_acc:.2f}%")
        
        # 1. 创建和加载模型
        print('\n1. Creating and loading model...')
        
        # 定义输入形状 (batch_size, channels, height, width)
        input_shape = (1, 1, 28, 28)  # MNIST图像形状
        num_classes = 10  # MNIST有10个类别
        
        try:
            if args.quantize and args.model == 'stm32':
                # 使用可量化模型
                model = get_quantizable_model(
                    model_name=args.model,
                    input_shape=input_shape[1:],  # 去掉batch_size维度
                    num_classes=num_classes,
                    use_depthwise=args.use_depthwise
                )
            else:
                # 使用普通模型
                model = get_model(
                    model_name=args.model,
                    input_shape=input_shape[1:],  # 去掉batch_size维度
                    num_classes=num_classes,
                    use_depthwise=args.use_depthwise
                )
            
            print(f'Model created: {model.__class__.__name__}')
        except Exception as e:
            print(f"Error creating model: {e}")
            return
        
        # 加载模型权重
        try:
            # 加载模型权重，使用map_location确保在CPU上加载
            state_dict = torch.load(args.model_path, map_location='cpu')
            print(f"Loaded state dict with {len(state_dict)} keys")
            
            # 尝试加载权重
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            print(f'Model loaded successfully: {model.__class__.__name__}')
        except Exception as e:
            print(f"Error loading model weights: {e}")
            print("Attempting to load with compatibility adjustments...")
            # 尝试使用兼容性加载
            try:
                # 创建一个新的模型实例并尝试加载
                if args.quantize and args.model == 'stm32':
                    model = get_quantizable_model(
                        model_name=args.model,
                        input_shape=input_shape[1:],
                        num_classes=num_classes,
                        use_depthwise=args.use_depthwise
                    )
                else:
                    model = get_model(
                        model_name=args.model,
                        input_shape=input_shape[1:],
                        num_classes=num_classes,
                        use_depthwise=args.use_depthwise
                    )
                # 只加载匹配的键
                model_dict = model.state_dict()
                filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
                model_dict.update(filtered_state_dict)
                model.load_state_dict(model_dict, strict=False)
                model.eval()
                print(f'Model loaded with compatibility adjustments: {model.__class__.__name__}')
                print(f"Loaded {len(filtered_state_dict)} out of {len(state_dict)} keys")
            except Exception as e2:
                print(f"Error loading model with compatibility adjustments: {e2}")
                return
        
        # 2. 量化模型
        if args.quantize:
            print('\n2. Quantizing model...')
            try:
                quantizer = ModelQuantizer()
                
                # 加载校准数据
                from src.data_preprocessing.data_processor import MNISTDataProcessor
                processor = MNISTDataProcessor(batch_size=32)
                _, val_loader, _ = processor.load_data()
                
                if args.quantization_type == 'ptq':
                    # 使用PTQ量化
                    model = quantizer.quantize_ptq(model, val_loader)
                else:
                    # 使用QAT量化
                    _, train_loader, _ = processor.load_data()
                    model = quantizer.quantize_qat(model, train_loader, val_loader, epochs=5)
                
                print('Model quantized successfully!')
            except Exception as e:
                print(f"Error quantizing model: {e}")
                return
        
        # 3. 导出模型
        print('\n3. Exporting model...')
        try:
            # 确保模型在CPU上
            model = model.to('cpu')
            exporter = ModelExporter(model, input_shape)
            
            # 修改导出路径，包含准确率信息
            onnx_filename = f'mnist_model_{mnist_acc:.2f}_{custom_acc:.2f}.onnx'
            onnx_path = os.path.join(args.export_dir, onnx_filename)
            
            # 导出为ONNX格式
            print(f'Exporting ONNX model to: {onnx_path}')
            torch.onnx.export(
                model,
                torch.randn(input_shape),
                onnx_path,
                verbose=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # 验证ONNX模型
            print('Verifying ONNX model...')
            is_valid = exporter.verify_onnx_model(onnx_path)
            
            if is_valid:
                # 为STM32准备模型，包含准确率信息
                stm32_dir = os.path.join(args.export_dir, 'stm32')
                os.makedirs(stm32_dir, exist_ok=True)
                
                # 复制ONNX模型到STM32目录，包含准确率信息
                stm32_onnx_filename = f'mnist_model_{mnist_acc:.2f}_{custom_acc:.2f}.onnx'
                stm32_onnx_path = os.path.join(stm32_dir, stm32_onnx_filename)
                
                import shutil
                shutil.copy2(onnx_path, stm32_onnx_path)
                
                # 创建模型信息文件
                model_info = {
                    'model_name': 'MNIST_Digit_Classifier',
                    'input_shape': input_shape,
                    'output_shape': (input_shape[0], 10),  # MNIST有10个类别
                    'mnist_accuracy': f'{mnist_acc:.2f}%',
                    'custom_accuracy': f'{custom_acc:.2f}%',
                    'onnx_path': stm32_onnx_filename
                }
                
                info_path = os.path.join(stm32_dir, f'model_info_{mnist_acc:.2f}_{custom_acc:.2f}.txt')
                with open(info_path, 'w') as f:
                    for key, value in model_info.items():
                        f.write(f'{key}: {value}\n')
                
                # 创建部署说明文件
                deploy_guide = """
STM32部署指南
=============

1. 打开STM32CubeIDE
2. 创建或打开一个STM32F407项目
3. 打开X-Cube-AI插件
4. 导入此目录中的ONNX文件
5. 配置AI模型参数
6. 生成代码
7. 将生成的代码集成到项目中
8. 编译并烧录到STM32F407开发板

注意事项:
- 确保STM32F407有足够的内存来运行模型
- 输入数据需要与训练时的预处理方式一致
- 模型输入尺寸为28x28的灰度图像
"""
                
                guide_path = os.path.join(stm32_dir, 'deploy_guide.txt')
                with open(guide_path, 'w') as f:
                    f.write(deploy_guide)
                
                print('\nExport completed successfully!')
                print(f'ONNX model: {onnx_path}')
                print(f'STM32 deployment files: {stm32_dir}')
                
                # 清理STM32目录，只保留3个最佳模型
                print('Cleaning STM32 directory, keeping only top 3 models...')
                try:
                    # 获取所有STM32模型文件
                    stm32_onnx_files = glob.glob(os.path.join(stm32_dir, 'mnist_model_*.onnx'))
                    
                    # 按准确率排序
                    def get_accuracy_from_filename(filepath):
                        try:
                            parts = os.path.basename(filepath).split('_')
                            if len(parts) >= 4:
                                mnist_acc = float(parts[2])
                                custom_acc = float(parts[3].split('.')[0])
                                # 计算综合评分
                                return mnist_acc * 0.65 + custom_acc * 0.35
                        except:
                            return 0.0
                    
                    stm32_onnx_files.sort(key=get_accuracy_from_filename, reverse=True)
                    
                    # 保留前3个最佳模型，删除其余的
                    if len(stm32_onnx_files) > 3:
                        files_to_remove = stm32_onnx_files[3:]
                        for file_path in files_to_remove:
                            try:
                                os.remove(file_path)
                                print(f'Removed old STM32 model: {os.path.basename(file_path)}')
                                
                                # 同时删除对应的信息文件
                                info_file = file_path.replace('.onnx', '.txt')
                                if os.path.exists(info_file):
                                    os.remove(info_file)
                            except Exception as e:
                                print(f"Error removing file: {e}")
                except Exception as e:
                    print(f"Error cleaning STM32 directory: {e}")
                    
            else:
                print('ONNX model verification failed!')
                
        except Exception as e:
            print(f"Error exporting model: {e}")
            return
            
    except Exception as e:
        print(f"Error in export script: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()