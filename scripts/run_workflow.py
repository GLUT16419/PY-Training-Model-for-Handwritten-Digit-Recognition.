import argparse
import os
import subprocess
import glob
from datetime import datetime
import time

def run_command(cmd, cwd=None):
    """运行命令并实时打印输出，同时生成日志文件"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f'[{timestamp}] 开始执行命令: {cmd}')
    
    # 创建日志目录
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日志文件名
    log_filename = f'command_{timestamp}.log'
    log_path = os.path.join(log_dir, log_filename)
    
    # 打开日志文件
    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f'[{timestamp}] 开始执行命令: {cmd}\n')
        log_file.write(f'[{timestamp}] 工作目录: {cwd or os.getcwd()}\n\n')
    
    start_time = time.time()
    
    try:
        # 使用Popen实现实时输出，设置较小的缓冲区以确保实时性
        process = subprocess.Popen(
            cmd, 
            shell=True, 
            cwd=cwd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # 实时读取输出，逐行打印
        line_count = 0
        with open(log_path, 'a', encoding='utf-8') as log_file:
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                
                # 写入日志文件
                log_file.write(line)
                log_file.flush()
                
                # 终端输出控制：每10行输出一次，或者输出包含关键字的行
                line_count += 1
                if line_count % 10 == 0 or any(keyword in line for keyword in ['Epoch', 'Loss', 'Accuracy', 'Error', '完成', '失败', '开始', '结束']):
                    print(line.rstrip(), flush=True)
        
        # 等待命令完成
        process.wait()
        
        # 计算执行时间
        execution_time = time.time() - start_time
        
        # 写入执行结果到日志
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f'\n[{timestamp}] 命令执行完成! 执行时间: {execution_time:.2f}s\n')
            log_file.write(f'[{timestamp}] 返回代码: {process.returncode}\n')
        
        if process.returncode == 0:
            print(f'[{timestamp}] 命令执行成功! 执行时间: {execution_time:.2f}s')
            print(f'[{timestamp}] 详细日志: {log_path}')
        else:
            print(f'[{timestamp}] 命令执行失败，返回代码: {process.returncode}')
            print(f'[{timestamp}] 详细日志: {log_path}')
        
        return process.returncode
    except KeyboardInterrupt:
        print(f'[{timestamp}] 命令执行被用户中断!')
        if 'process' in locals():
            process.terminate()
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f'\n[{timestamp}] 命令执行被用户中断!\n')
        return -1
    except subprocess.TimeoutExpired:
        print(f'[{timestamp}] 命令执行超时!')
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f'\n[{timestamp}] 命令执行超时!\n')
        return -1
    except Exception as e:
        print(f'[{timestamp}] 执行命令时出错: {e}')
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f'\n[{timestamp}] 执行命令时出错: {e}\n')
        return -1

def find_latest_model():
    """查找最新的训练模型"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f'[{timestamp}] 正在搜索最新的训练模型...')
    
    try:
        # 首先查找训练文件夹中的最佳模型
        train_folders = glob.glob('models/trained/train_*')
        if train_folders:
            # 按修改时间排序，取最新的训练文件夹
            train_folders.sort(key=os.path.getmtime, reverse=True)
            latest_train_folder = train_folders[0]
            print(f'[{timestamp}] 最新训练文件夹: {latest_train_folder}')
            
            # 在最新训练文件夹中查找最佳模型
            model_files = glob.glob(os.path.join(latest_train_folder, 'best_model_*.pth'))
            if model_files:
                # 按准确率排序，取最高的
                def get_accuracy_from_filename(filepath):
                    # 从文件名中提取准确率，处理路径分隔符问题
                    filename = os.path.basename(filepath)
                    parts = filename.split('_')
                    if len(parts) >= 4:
                        try:
                            # 新格式: best_model_{mnist_acc:.2f}_{custom_acc:.2f}_{timestamp}.pth
                            mnist_acc = float(parts[2])
                            custom_acc = float(parts[3])
                            # 计算综合评分
                            return mnist_acc * 0.65 + custom_acc * 0.35
                        except ValueError:
                            # 旧格式兼容
                            try:
                                return float(parts[2])
                            except ValueError:
                                return 0.0
                    return 0.0
                
                model_files.sort(key=get_accuracy_from_filename, reverse=True)
                model_path = model_files[0]
                print(f'[{timestamp}] 找到最佳模型: {model_path}')
                return model_path
        
        # 如果没有找到训练文件夹中的模型，尝试查找根目录中的模型
        model_files = glob.glob('models/trained/*.pth')
        if model_files:
            # 按修改时间排序，取最新的
            model_files.sort(key=os.path.getmtime, reverse=True)
            model_path = model_files[0]
            print(f'[{timestamp}] 在根目录找到模型: {model_path}')
            return model_path
        
        return None
    except Exception as e:
        print(f'[{timestamp}] 查找最新模型时出错: {e}')
        return None

def main():
    """主脚本，运行完整的工作流程"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    try:
        # 解析命令行参数
        parser = argparse.ArgumentParser(description='Run complete MNIST model workflow')
        parser.add_argument('--model', type=str, default='tiny', choices=['lightweight', 'enhanced', 'tiny', 'stm32', 'ultralight'],
                            help='Model architecture to use')
        parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
        parser.add_argument('--batch-size', type=int, default=64,
                            help='Batch size for training')
        parser.add_argument('--use-depthwise', action='store_true', default=True,
                            help='Use depthwise separable convolution for stm32 model')
        parser.add_argument('--quantize', action='store_true',
                            help='Quantize model to INT8 format')
        parser.add_argument('--quantization-type', type=str, default='ptq', choices=['ptq', 'qat'],
                            help='Quantization type: PTQ (Post-training Quantization) or QAT (Quantization-Aware Training)')
        parser.add_argument('--skip-training', action='store_true',
                            help='Skip training step and use existing model')
        
        args = parser.parse_args()
        
        print(f'\n{"=" * 120}')
        print(f'[{timestamp}] 开始完整的MNIST模型工作流程')
        print(f'{"=" * 120}')
        print(f'[{timestamp}] 模型架构: {args.model}')
        print(f'[{timestamp}] 训练轮次: {args.epochs}')
        print(f'[{timestamp}] 批处理大小: {args.batch_size}')
        print(f'[{timestamp}] 使用深度可分离卷积: {args.use_depthwise}')
        print(f'[{timestamp}] 量化模型: {args.quantize}')
        if args.quantize:
            print(f'[{timestamp}] 量化类型: {args.quantization_type}')
        print(f'[{timestamp}] 跳过训练: {args.skip_training}')
        print(f'{"=" * 120}')
        
        # 确保必要的目录存在
        print(f'\n[{timestamp}] 检查并创建必要的目录...')
        os.makedirs('models/trained', exist_ok=True)
        os.makedirs('models/exported', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
        print(f'[{timestamp}] 目录检查完成!')
        
        model_path = None
        
        # 1. 训练模型
        if not args.skip_training:
            print(f'\n{"=" * 120}')
            print(f'[{timestamp}] 步骤 1: 训练模型')
            print(f'{"=" * 120}')
            
            # 构建训练命令
            train_cmd = f'python scripts/train.py --model {args.model}'
            if args.epochs != 100:  # 只有当用户指定了轮次时才传递
                train_cmd += f' --epochs {args.epochs}'
            if args.batch_size != 64:  # 只有当用户指定了批处理大小时才传递
                train_cmd += f' --batch-size {args.batch_size}'
            if args.use_depthwise:
                train_cmd += ' --use-depthwise'
            # 只有当自定义数据集存在时才添加该参数
            if os.path.exists('data/custom/unified_dataset'):
                train_cmd += ' --custom-data-dir "data/custom/unified_dataset"'
                train_cmd += ' --custom-ratio 0.3'
                train_cmd += ' --two-stage'
            
            print(f'[{timestamp}] 执行训练命令: {train_cmd}')
            return_code = run_command(train_cmd)
            
            if return_code != 0:
                print(f'\n{"=" * 120}')
                print(f'[{timestamp}] 训练失败! 退出...')
                print(f'{"=" * 120}')
                return
            else:
                print(f'\n{"=" * 120}')
                print(f'[{timestamp}] 训练完成!')
                print(f'{"=" * 120}')
        
        # 2. 查找模型
        print(f'\n{"=" * 120}')
        print(f'[{timestamp}] 步骤 2: 查找训练好的模型')
        print(f'{"=" * 120}')
        
        # 找到最新训练的模型
        model_path = find_latest_model()
        
        if not model_path:
            print(f'\n{"=" * 120}')
            print(f'[{timestamp}] 错误: 没有找到训练好的模型!')
            print(f'[{timestamp}] 请检查训练是否成功完成.')
            print(f'[{timestamp}] 退出...')
            print(f'{"=" * 120}')
            return
        
        print(f'[{timestamp}] 使用模型: {model_path}')
        
        # 3. 评估模型
        print(f'\n{"=" * 120}')
        print(f'[{timestamp}] 步骤 3: 评估模型')
        print(f'{"=" * 120}')
        
        # 构建评估命令
        eval_cmd = f'python scripts/evaluate.py --model {args.model} --model-path "{model_path}" --batch-size {args.batch_size}'
        if args.use_depthwise:
            eval_cmd += ' --use-depthwise'
        if args.quantize:
            eval_cmd += f' --quantize --quantization-type {args.quantization_type}'
        # 只有当自定义数据集存在时才添加该参数
        if os.path.exists('data/custom/unified_dataset'):
            eval_cmd += ' --custom-data-dir "data/custom/unified_dataset"'
        
        print(f'[{timestamp}] 执行评估命令: {eval_cmd}')
        return_code = run_command(eval_cmd)
        
        if return_code != 0:
            print(f'\n{"=" * 120}')
            print(f'[{timestamp}] 评估失败! 退出...')
            print(f'{"=" * 120}')
            return
        else:
            print(f'\n{"=" * 120}')
            print(f'[{timestamp}] 评估完成!')
            print(f'{"=" * 120}')
        
        # 4. 生成详细报告
        print(f'\n{"=" * 120}')
        print(f'[{timestamp}] 步骤 4: 生成详细评估报告')
        print(f'{"=" * 120}')
        
        # 构建报告生成命令
        report_cmd = f'python scripts/generate_report.py --model {args.model}'
        print(f'[{timestamp}] 执行报告生成命令: {report_cmd}')
        return_code = run_command(report_cmd)
        
        # 检查是否只是pynvml警告
        if return_code != 0:
            # 读取日志文件检查错误类型
            log_files = glob.glob('logs/command_*.log')
            if log_files:
                log_files.sort(key=os.path.getmtime, reverse=True)
                latest_log = log_files[0]
                try:
                    with open(latest_log, 'r', encoding='utf-8') as f:
                        log_content = f.read()
                    # 检查是否只是pynvml警告
                    if 'pynvml package is deprecated' in log_content and 'Error' not in log_content:
                        print(f'\n[{timestamp}] 报告生成只有pynvml警告，继续执行...')
                    else:
                        print(f'\n[{timestamp}] 报告生成失败，但继续执行...')
                except Exception as e:
                    print(f'\n[{timestamp}] 报告生成失败，但继续执行...')
            else:
                print(f'\n[{timestamp}] 报告生成失败，但继续执行...')
        else:
            print(f'\n{"=" * 120}')
            print(f'[{timestamp}] 报告生成完成!')
            print(f'{"=" * 120}')
        
        # 5. 导出模型
        print(f'\n{"=" * 120}')
        print(f'[{timestamp}] 步骤 5: 导出模型')
        print(f'{"=" * 120}')
        
        # 构建导出命令
        export_cmd = f'python scripts/export.py --model {args.model} --model-path "{model_path}"'
        if args.use_depthwise:
            export_cmd += ' --use-depthwise'
        if args.quantize:
            export_cmd += f' --quantize --quantization-type {args.quantization_type}'
        
        print(f'[{timestamp}] 执行导出命令: {export_cmd}')
        return_code = run_command(export_cmd)
        
        if return_code != 0:
            print(f'\n{"=" * 120}')
            print(f'[{timestamp}] 导出失败! 退出...')
            print(f'{"=" * 120}')
            return
        else:
            print(f'\n{"=" * 120}')
            print(f'[{timestamp}] 导出完成!')
            print(f'{"=" * 120}')
        
        # 6. 工作流完成
        print(f'\n{"=" * 120}')
        print(f'[{timestamp}] 完整工作流执行成功!')
        print(f'{"=" * 120}')
        print(f'[{timestamp}] 最佳模型: {os.path.basename(model_path)}')
        print(f'[{timestamp}] 训练结果: reports/training_history_{args.model}.png')
        print(f'[{timestamp}] 评估报告: reports/')
        print(f'[{timestamp}] 导出模型: models/exported/')
        print(f'[{timestamp}] STM32部署文件: models/exported/stm32/')
        print(f'{"=" * 120}')
        
    except Exception as e:
        print(f'\n{"=" * 120}')
        print(f'[{timestamp}] 工作流执行出错: {e}')
        print(f'{"=" * 120}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    import time
    main()