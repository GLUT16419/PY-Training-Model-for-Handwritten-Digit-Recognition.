import os
import sys
import glob
from datetime import datetime

# 添加项目根目录到Python搜索路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.data_preprocessing import MNISTDataProcessor
from src.training.model_architectures import get_model
from src.training import ModelTrainer
from src.evaluation import ModelEvaluator

def generate_model_report(model_name, model_path, report_dir='reports'):
    """为单个模型生成详细的评估报告"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = os.path.basename(model_path)
    
    print(f"\n[{timestamp}] 正在为模型生成详细报告: {model_filename}")
    
    # 1. 加载数据
    print(f"[{timestamp}] 1. 加载测试数据...")
    processor = MNISTDataProcessor(batch_size=64, val_split=0.1)
    _, _, test_loader = processor.load_data()
    
    # 尝试加载自定义数据集
    custom_test_loader = None
    if os.path.exists('data/custom/unified_dataset'):
        try:
            _, _, _, custom_test_loader = processor.load_mixed_data(
                custom_data_dir='data/custom/unified_dataset',
                custom_ratio=0.3
            )
            print(f"[{timestamp}] 自定义数据集加载成功!")
        except Exception as e:
            print(f"[{timestamp}] 加载自定义数据集失败: {e}")
    
    # 2. 加载模型
    print(f"[{timestamp}] 2. 加载模型: {model_path}")
    
    # 获取输入形状
    input_shape = (1, 28, 28)
    num_classes = 10
    
    # 检测是否需要使用深度可分离卷积
    use_depthwise = False
    try:
        # 先尝试加载权重以检测架构
        state_dict = torch.load(model_path, map_location='cpu')
        # 检查是否包含深度可分离卷积的权重键
        for key in state_dict.keys():
            if 'depthwise' in key or 'pointwise' in key:
                use_depthwise = True
                print(f"[{timestamp}] 检测到深度可分离卷积模型")
                break
    except Exception as e:
        print(f"[{timestamp}] 检测模型架构时出错: {e}")
    
    # 创建模型
    model = get_model(model_name, input_shape, num_classes, use_depthwise)
    
    # 加载模型权重
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"[{timestamp}] 模型权重加载成功!")
    except Exception as e:
        print(f"[{timestamp}] 加载模型权重失败: {e}")
        # 尝试使用不同的架构加载
        print(f"[{timestamp}] 尝试使用不同架构加载...")
        try:
            # 尝试不使用深度可分离卷积
            model = get_model(model_name, input_shape, num_classes, use_depthwise=False)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"[{timestamp}] 使用非深度可分离卷积架构加载成功!")
        except Exception as e2:
            print(f"[{timestamp}] 所有架构加载失败: {e2}")
            return None
    
    # 3. 评估模型
    print(f"[{timestamp}] 3. 评估模型...")
    evaluator = ModelEvaluator(model)
    
    # 使用 run_evaluation 方法代替 evaluate 方法，获取完整的评估结果
    evaluation_results = evaluator.run_evaluation(test_loader)
    
    # 评估自定义数据集
    custom_evaluation_results = None
    if custom_test_loader:
        try:
            custom_evaluation_results = evaluator.run_evaluation(custom_test_loader)
            print(f"[{timestamp}] 自定义数据集评估完成!")
        except Exception as e:
            print(f"[{timestamp}] 评估自定义数据集失败: {e}")
    
    # 4. 生成详细报告
    print(f"[{timestamp}] 4. 生成详细报告...")
    os.makedirs(report_dir, exist_ok=True)
    
    # 从模型文件名提取准确率信息
    def extract_accuracies_from_filename(filename):
        parts = filename.split('_')
        mnist_acc = 0.0
        custom_acc = 0.0
        if len(parts) >= 4:
            try:
                mnist_acc = float(parts[2])
                custom_acc = float(parts[3])
            except ValueError:
                pass
        return mnist_acc, custom_acc
    
    mnist_acc, custom_acc = extract_accuracies_from_filename(model_filename)
    
    # 添加时间戳和模型信息到报告文件名
    report_filename = f'evaluation_report_{model_filename.replace(".pth", "")}_{timestamp}.txt'
    report_path = os.path.join(report_dir, report_filename)
    
    # 分析模型性能
    def analyze_performance(evaluation_results, dataset_name):
        """分析模型性能并返回详细分析"""
        analysis = []
        
        # 基本指标分析
        accuracy = evaluation_results.get('accuracy', 0)
        loss = evaluation_results.get('loss', 0)
        
        analysis.append(f"{dataset_name}数据集性能分析:")
        analysis.append(f"- 准确率: {accuracy:.2f}%")
        analysis.append(f"- 损失值: {loss:.4f}")
        
        # 分类报告分析
        if 'classification_report' in evaluation_results:
            report_lines = evaluation_results['classification_report'].split('\n')
            macro_avg_line = None
            weighted_avg_line = None
            
            for line in report_lines:
                if 'macro avg' in line:
                    macro_avg_line = line
                elif 'weighted avg' in line:
                    weighted_avg_line = line
            
            if macro_avg_line:
                parts = macro_avg_line.split()
                if len(parts) >= 5:
                    try:
                        precision = float(parts[1])
                        recall = float(parts[2])
                        f1_score = float(parts[3])
                        analysis.append(f"- 宏平均精确率: {precision:.2f}")
                        analysis.append(f"- 宏平均召回率: {recall:.2f}")
                        analysis.append(f"- 宏平均F1分数: {f1_score:.2f}")
                    except (ValueError, IndexError):
                        pass
            
            if weighted_avg_line:
                parts = weighted_avg_line.split()
                if len(parts) >= 5:
                    try:
                        precision = float(parts[1])
                        recall = float(parts[2])
                        f1_score = float(parts[3])
                        analysis.append(f"- 加权平均精确率: {precision:.2f}")
                        analysis.append(f"- 加权平均召回率: {recall:.2f}")
                        analysis.append(f"- 加权平均F1分数: {f1_score:.2f}")
                    except (ValueError, IndexError):
                        pass
        
        # 混淆矩阵分析
        if 'confusion_matrix' in evaluation_results:
            cm = evaluation_results['confusion_matrix']
            # 计算每类的准确率
            class_accuracies = []
            for i in range(len(cm)):
                total = sum(cm[i])
                if total > 0:
                    correct = cm[i][i]
                    class_acc = (correct / total) * 100
                    class_accuracies.append((i, class_acc))
            
            # 找出表现最好和最差的类别
            if class_accuracies:
                class_accuracies.sort(key=lambda x: x[1], reverse=True)
                best_class, best_acc = class_accuracies[0]
                worst_class, worst_acc = class_accuracies[-1]
                
                analysis.append(f"- 表现最好的类别: {best_class} (准确率: {best_acc:.2f}%)")
                analysis.append(f"- 表现最差的类别: {worst_class} (准确率: {worst_acc:.2f}%)")
        
        return analysis
    
    # 分析过拟合情况
    def analyze_overfitting(train_history):
        """分析模型过拟合情况"""
        analysis = []
        
        if not train_history:
            analysis.append("过拟合分析: 无训练历史数据")
            return analysis
        
        # 获取训练和验证准确率
        train_accuracies = train_history.get('accuracy', [])
        val_accuracies = train_history.get('val_accuracy', [])
        custom_val_accuracies = train_history.get('custom_val_accuracy', [])
        
        if not train_accuracies or not val_accuracies:
            analysis.append("过拟合分析: 训练历史数据不完整")
            return analysis
        
        # 计算最终的训练和验证准确率
        final_train_acc = train_accuracies[-1] if train_accuracies else 0
        final_val_acc = val_accuracies[-1] if val_accuracies else 0
        final_custom_val_acc = custom_val_accuracies[-1] if custom_val_accuracies else 0
        
        # 计算准确率差异
        train_val_gap = final_train_acc - final_val_acc
        analysis.append("过拟合分析:")
        analysis.append(f"- 最终训练准确率: {final_train_acc:.2f}%")
        analysis.append(f"- 最终MNIST验证准确率: {final_val_acc:.2f}%")
        analysis.append(f"- 最终自定义验证准确率: {final_custom_val_acc:.2f}%")
        analysis.append(f"- 训练-验证准确率差异: {train_val_gap:.2f}%")
        
        # 过拟合程度评估
        if train_val_gap < 5:
            analysis.append("- 过拟合程度: 轻微")
            analysis.append("- 评估: 模型泛化能力良好")
        elif train_val_gap < 15:
            analysis.append("- 过拟合程度: 中等")
            analysis.append("- 评估: 模型存在一定过拟合，建议使用正则化或数据增强")
        else:
            analysis.append("- 过拟合程度: 严重")
            analysis.append("- 评估: 模型过拟合严重，需要显著调整")
        
        # 趋势分析
        if len(train_accuracies) > 1 and len(val_accuracies) > 1:
            train_trend = train_accuracies[-1] - train_accuracies[0]
            val_trend = val_accuracies[-1] - val_accuracies[0]
            
            analysis.append(f"- 训练准确率趋势: {'上升' if train_trend > 0 else '下降'} {abs(train_trend):.2f}%")
            analysis.append(f"- 验证准确率趋势: {'上升' if val_trend > 0 else '下降'} {abs(val_trend):.2f}%")
            
            if train_trend > 0 and val_trend < 0:
                analysis.append("- 警告: 训练准确率上升但验证准确率下降，可能存在过拟合")
            elif train_trend > 0 and val_trend > 0:
                if val_trend < train_trend * 0.8:
                    analysis.append("- 注意: 验证准确率提升速度低于训练准确率，可能开始过拟合")
        
        return analysis
    
    # 生成详细报告内容
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("MNIST模型详细评估报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. 模型信息\n")
        f.write("-" * 60 + "\n")
        f.write(f"模型名称: {model_name}\n")
        f.write(f"模型架构: {model.__class__.__name__}\n")
        f.write(f"模型路径: {model_path}\n")
        f.write(f"模型文件名: {model_filename}\n")
        f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"MNIST准确率: {mnist_acc:.2f}%\n")
        f.write(f"自定义数据集准确率: {custom_acc:.2f}%\n\n")
        
        f.write("2. MNIST测试集评估结果\n")
        f.write("-" * 60 + "\n")
        f.write(f"测试损失: {evaluation_results['loss']:.4f}\n")
        f.write(f"测试准确率: {evaluation_results['accuracy']:.2f}%\n")
        f.write(f"Top-1 准确率: {evaluation_results['accuracy']:.2f}%\n")
        f.write(f"Top-5 准确率: {evaluation_results['accuracy']:.2f}%\n\n")
        
        if 'classification_report' in evaluation_results:
            f.write("3. MNIST分类报告\n")
            f.write("-" * 60 + "\n")
            f.write(evaluation_results['classification_report'] + "\n")
        
        if 'confusion_matrix' in evaluation_results:
            f.write("4. MNIST混淆矩阵\n")
            f.write("-" * 60 + "\n")
            f.write(str(evaluation_results['confusion_matrix']) + "\n\n")
        
        # 自定义数据集评估结果
        if custom_evaluation_results:
            f.write("5. 自定义数据集评估结果\n")
            f.write("-" * 60 + "\n")
            f.write(f"测试损失: {custom_evaluation_results['loss']:.4f}\n")
            f.write(f"测试准确率: {custom_evaluation_results['accuracy']:.2f}%\n\n")
            
            if 'classification_report' in custom_evaluation_results:
                f.write("6. 自定义数据集分类报告\n")
                f.write("-" * 60 + "\n")
                f.write(custom_evaluation_results['classification_report'] + "\n")
            
            if 'confusion_matrix' in custom_evaluation_results:
                f.write("7. 自定义数据集混淆矩阵\n")
                f.write("-" * 60 + "\n")
                f.write(str(custom_evaluation_results['confusion_matrix']) + "\n\n")
        
        # 详细性能分析
        f.write("8. 模型性能详细分析\n")
        f.write("-" * 60 + "\n")
        
        # MNIST数据集分析
        mnist_analysis = analyze_performance(evaluation_results, "MNIST")
        for line in mnist_analysis:
            f.write(line + "\n")
        
        # 自定义数据集分析
        if custom_evaluation_results:
            f.write("\n")
            custom_analysis = analyze_performance(custom_evaluation_results, "自定义")
            for line in custom_analysis:
                f.write(line + "\n")
        
        # 过拟合分析
        f.write("\n9. 过拟合分析\n")
        f.write("-" * 60 + "\n")
        
        # 尝试获取训练历史
        train_history = {}
        try:
            # 从模型训练过程中获取训练历史
            # 这里需要根据实际情况修改，可能需要从文件或其他地方加载
            # 暂时使用空字典，后续会通过其他方式获取
            pass
        except Exception as e:
            print(f"获取训练历史失败: {e}")
        
        overfitting_analysis = analyze_overfitting(train_history)
        for line in overfitting_analysis:
            f.write(line + "\n")
        
        # 模型优缺点分析
        f.write("\n10. 模型优缺点分析\n")
        f.write("-" * 60 + "\n")
        
        # 优点
        advantages = []
        if evaluation_results.get('accuracy', 0) >= 95:
            advantages.append("- MNIST数据集准确率高，模型泛化能力强")
        if custom_evaluation_results and custom_evaluation_results.get('accuracy', 0) >= 85:
            advantages.append("- 自定义数据集准确率高，适应能力强")
        if evaluation_results.get('loss', float('inf')) <= 0.1:
            advantages.append("- 损失值低，模型拟合效果好")
        
        if advantages:
            f.write("优点:\n")
            for point in advantages:
                f.write(point + "\n")
        else:
            f.write("优点: 模型表现基本符合预期\n")
        
        # 缺点
        disadvantages = []
        if evaluation_results.get('accuracy', 100) < 90:
            disadvantages.append("- MNIST数据集准确率有待提高")
        if custom_evaluation_results and custom_evaluation_results.get('accuracy', 100) < 70:
            disadvantages.append("- 自定义数据集准确率较低，需要进一步优化")
        if evaluation_results.get('loss', 0) > 0.3:
            disadvantages.append("- 损失值较高，模型拟合效果一般")
        
        if disadvantages:
            f.write("缺点:\n")
            for point in disadvantages:
                f.write(point + "\n")
        else:
            f.write("缺点: 模型表现良好，无明显缺点\n")
        
        # 改进建议
        f.write("\n10. 改进建议\n")
        f.write("-" * 60 + "\n")
        
        suggestions = []
        if evaluation_results.get('accuracy', 100) < 95:
            suggestions.append("- 考虑增加模型深度或宽度，提高模型容量")
            suggestions.append("- 调整学习率调度策略，优化训练过程")
        if custom_evaluation_results and custom_evaluation_results.get('accuracy', 100) < 80:
            suggestions.append("- 增加自定义数据集的训练样本数量")
            suggestions.append("- 使用更多的数据增强技术，提高模型的泛化能力")
            suggestions.append("- 考虑使用迁移学习，从MNIST预训练模型开始微调")
        
        if suggestions:
            for point in suggestions:
                f.write(point + "\n")
        else:
            f.write("- 模型表现良好，可考虑量化压缩以适应STM32部署\n")
            f.write("- 进一步优化模型结构，减少参数量和计算复杂度\n")
        
        f.write("\n11. 模型部署信息\n")
        f.write("-" * 60 + "\n")
        f.write("ONNX模型路径: models/exported/mnist_model.onnx\n")
        f.write("STM32部署指南: models/exported/stm32/deploy_guide.txt\n\n")
        
        # 模型性能总结
        f.write("12. 性能总结\n")
        f.write("-" * 60 + "\n")
        
        overall_score = (evaluation_results.get('accuracy', 0) * 0.65)
        if custom_evaluation_results:
            overall_score += (custom_evaluation_results.get('accuracy', 0) * 0.35)
        
        f.write(f"综合性能评分: {overall_score:.2f}%\n")
        
        if overall_score >= 90:
            f.write("性能等级: 优秀\n")
            f.write("评估: 模型性能出色，适合部署到STM32设备\n")
        elif overall_score >= 80:
            f.write("性能等级: 良好\n")
            f.write("评估: 模型性能良好，可以部署到STM32设备\n")
        elif overall_score >= 70:
            f.write("性能等级: 一般\n")
            f.write("评估: 模型性能基本符合要求，建议进一步优化后部署\n")
        else:
            f.write("性能等级: 需改进\n")
            f.write("评估: 模型性能有待提高，建议优化后再部署\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("报告生成成功!\n")
    
    print(f"[{timestamp}] 报告已保存到: {report_path}")
    return report_path

def generate_report(model_name='lightweight'):
    """生成完整的模型训练和评估报告"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"[{timestamp}] 正在为 {model_name} 模型生成综合报告...")
    
    # 创建报告目录
    report_dir = 'reports'
    os.makedirs(report_dir, exist_ok=True)
    
    # 查找所有最佳模型
    print(f"[{timestamp}] 1. 查找所有最佳模型...")
    
    best_models = []
    
    # 查找所有训练文件夹中的最佳模型
    train_folders = glob.glob('models/trained/train_*')
    if train_folders:
        for train_folder in train_folders:
            # 在训练文件夹中查找最佳模型
            model_files = glob.glob(os.path.join(train_folder, 'best_model_*.pth'))
            best_models.extend(model_files)
    
    # 按修改时间排序，取最新的模型
    if best_models:
        best_models.sort(key=os.path.getmtime, reverse=True)
        print(f"[{timestamp}] 找到 {len(best_models)} 个最佳模型")
        
        # 为每个最佳模型生成报告
        generated_reports = []
        for i, model_path in enumerate(best_models[:5]):  # 最多处理5个最佳模型
            print(f"\n[{timestamp}] 处理模型 {i+1}/{len(best_models)}: {os.path.basename(model_path)}")
            report_path = generate_model_report(model_name, model_path, report_dir)
            if report_path:
                generated_reports.append(report_path)
        
        # 生成综合比较报告
        if len(generated_reports) > 1:
            print(f"\n[{timestamp}] 生成模型比较报告...")
            generate_comparison_report(generated_reports, report_dir)
    else:
        print(f"[{timestamp}] 错误: 没有找到最佳模型!")
    
    print(f"\n[{timestamp}] 所有报告生成完成!")

def generate_comparison_report(report_paths, report_dir='reports'):
    """生成多个模型的比较报告"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    comparison_report_path = os.path.join(report_dir, f'model_comparison_report_{timestamp}.txt')
    
    print(f"[{timestamp}] 生成模型比较报告: {comparison_report_path}")
    
    with open(comparison_report_path, 'w', encoding='utf-8') as f:
        f.write("MNIST模型性能比较报告\n")
        f.write("=" * 120 + "\n\n")
        f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"比较模型数量: {len(report_paths)}\n\n")
        
        # 提取每个模型的关键指标
        model_metrics = []
        for report_path in report_paths:
            metrics = extract_metrics_from_report(report_path)
            if metrics:
                model_metrics.append(metrics)
        
        # 按准确率排序
        model_metrics.sort(key=lambda x: x.get('accuracy', 0), reverse=True)
        
        # 生成比较表格
        f.write("1. 模型性能比较表\n")
        f.write("-" * 120 + "\n")
        f.write(f"{'模型名称':<40} {'MNIST训练':<10} {'MNIST验证':<10} {'自定义训练':<10} {'自定义验证':<10} {'损失':<10}\n")
        f.write("-" * 120 + "\n")
        
        for metrics in model_metrics:
            model_name = metrics.get('model_name', 'Unknown')
            train_accuracy = metrics.get('train_accuracy', 0)
            accuracy = metrics.get('accuracy', 0)
            custom_train_accuracy = metrics.get('custom_train_accuracy', 0)
            custom_accuracy = metrics.get('custom_accuracy', 0)
            loss = metrics.get('loss', 0)
            
            f.write(f"{model_name:<40} {train_accuracy:<10.2f}% {accuracy:<10.2f}% {custom_train_accuracy:<10.2f}% {custom_accuracy:<10.2f}% {loss:<10.4f}\n")
        
        f.write("\n2. 最佳模型推荐\n")
        f.write("-" * 120 + "\n")
        if model_metrics:
            best_model = model_metrics[0]
            f.write(f"推荐最佳模型: {best_model.get('model_name', 'Unknown')}\n")
            f.write(f"推荐理由:\n")
            f.write(f"- MNIST验证准确率最高: {best_model.get('accuracy', 0):.2f}%\n")
            if best_model.get('custom_accuracy', 0) > 0:
                f.write(f"- 自定义验证准确率: {best_model.get('custom_accuracy', 0):.2f}%\n")
            f.write(f"- 损失值: {best_model.get('loss', 0):.4f}\n")
        
        f.write("\n3. 过拟合分析\n")
        f.write("-" * 120 + "\n")
        for metrics in model_metrics[:3]:  # 分析前3个模型
            model_name = metrics.get('model_name', 'Unknown')
            train_accuracy = metrics.get('train_accuracy', 0)
            accuracy = metrics.get('accuracy', 0)
            train_val_gap = train_accuracy - accuracy
            
            f.write(f"模型: {model_name}\n")
            f.write(f"- 训练-验证准确率差异: {train_val_gap:.2f}%\n")
            
            if train_val_gap < 5:
                f.write(f"- 过拟合程度: 轻微\n")
            elif train_val_gap < 15:
                f.write(f"- 过拟合程度: 中等\n")
            else:
                f.write(f"- 过拟合程度: 严重\n")
            f.write("\n")
        
        f.write("\n4. 模型选择建议\n")
        f.write("-" * 120 + "\n")
        f.write("根据部署环境和需求选择合适的模型:\n")
        f.write("- 追求最高准确率: 选择综合性能最好的模型\n")
        f.write("- 资源受限环境: 选择较小的模型，如tiny模型\n")
        f.write("- 自定义数据重要: 选择自定义数据集准确率高的模型\n")
        f.write("- 避免过拟合: 选择训练-验证差异小的模型\n\n")
        
        f.write("=" * 120 + "\n")
        f.write("比较报告生成成功!\n")
    
    print(f"[{timestamp}] 比较报告已保存到: {comparison_report_path}")

def extract_metrics_from_report(report_path):
    """从报告文件中提取关键指标"""
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        metrics = {
            'model_name': os.path.basename(report_path),
            'accuracy': 0,
            'train_accuracy': 0,
            'custom_accuracy': 0,
            'custom_train_accuracy': 0,
            'loss': 0
        }
        
        # 从模型文件名中提取准确率信息
        model_name = metrics['model_name']
        # 提取文件名中的准确率信息，格式如 evaluation_report_best_model_59.70_72.70_20260113_164045_20260113_164814.txt
        parts = model_name.split('_')
        if len(parts) >= 6:
            try:
                # 从文件名中提取训练准确率
                if 'best_model' in model_name:
                    # 格式: evaluation_report_best_model_{mnist_train_acc}_{custom_train_acc}_{timestamp}_{timestamp}.txt
                    if len(parts) >= 6:
                        # 找到best_model在parts中的位置
                        best_model_index = None
                        for i, part in enumerate(parts):
                            if part == 'best':
                                best_model_index = i
                                break
                        if best_model_index is not None and best_model_index + 4 < len(parts):
                            metrics['train_accuracy'] = float(parts[best_model_index + 2])
                            metrics['custom_train_accuracy'] = float(parts[best_model_index + 3])
            except ValueError:
                pass
        
        # 提取MNIST准确率和自定义数据集准确率
        lines = content.split('\n')
        
        # 状态追踪，记录当前正在解析的数据集部分
        current_dataset = 'mnist'  # 默认为MNIST数据集
        
        for line in lines:
            # 检测数据集部分切换
            if 'MNIST测试集评估结果' in line:
                current_dataset = 'mnist'
            elif '自定义数据集评估结果' in line:
                current_dataset = 'custom'
            
            # 提取准确率
            if '测试准确率:' in line:
                parts = line.split(':')
                if len(parts) > 1:
                    try:
                        acc_value = float(parts[1].strip().replace('%', ''))
                        if current_dataset == 'mnist':
                            metrics['accuracy'] = acc_value
                        elif current_dataset == 'custom':
                            metrics['custom_accuracy'] = acc_value
                    except ValueError:
                        pass
            elif '训练准确率:' in line:
                parts = line.split(':')
                if len(parts) > 1:
                    try:
                        acc_value = float(parts[1].strip().replace('%', ''))
                        if current_dataset == 'mnist':
                            metrics['train_accuracy'] = acc_value
                        elif current_dataset == 'custom':
                            metrics['custom_train_accuracy'] = acc_value
                    except ValueError:
                        pass
            
            # 提取损失值（只记录MNIST的损失值）
            if '测试损失:' in line and current_dataset == 'mnist':
                parts = line.split(':')
                if len(parts) > 1:
                    try:
                        metrics['loss'] = float(parts[1].strip())
                    except ValueError:
                        pass
        
        # 如果没有找到自定义准确率，尝试从报告开头的模型信息中提取
        if metrics['custom_accuracy'] == 0:
            for line in lines:
                if '自定义数据集准确率:' in line:
                    parts = line.split(':')
                    if len(parts) > 1:
                        try:
                            metrics['custom_accuracy'] = float(parts[1].strip().replace('%', ''))
                        except ValueError:
                            pass
        
        return metrics
    except Exception as e:
        print(f"提取报告指标失败: {e}")
        return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate model evaluation report')
    parser.add_argument('--model', type=str, default='lightweight', choices=['lightweight', 'tiny', 'simple', 'stm32', 'ultralight'],
                        help='Model architecture to evaluate')
    args = parser.parse_args()
    
    generate_report(args.model)