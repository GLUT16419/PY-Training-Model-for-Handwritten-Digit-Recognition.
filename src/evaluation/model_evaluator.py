import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
from datetime import datetime

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model, device=None):
        """
        初始化评估器
        Args:
            model: 模型实例
            device: 设备 (cuda 或 cpu)
        """
        try:
            self.model = model
            self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model.eval()
            print(f"Model evaluator initialized on {self.device}")
        except Exception as e:
            print(f"Error initializing ModelEvaluator: {e}")
            raise
    
    def evaluate(self, data_loader):
        """
        评估模型
        Args:
            data_loader: 数据加载器
        Returns:
            accuracy: 准确率
            loss: 平均损失
            predictions: 预测结果
            targets: 真实标签
        """
        try:
            criterion = torch.nn.CrossEntropyLoss()
            
            loss = 0.0
            correct = 0
            total = 0
            predictions = []
            targets_list = []
            
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(data_loader):
                    try:
                        # 数据移至设备
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        
                        # 前向传播
                        outputs = self.model(inputs)
                        batch_loss = criterion(outputs, targets)
                        
                        # 计算损失和准确率
                        loss += batch_loss.item()
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()
                        
                        # 保存预测结果和真实标签
                        predictions.extend(predicted.cpu().numpy())
                        targets_list.extend(targets.cpu().numpy())
                    except Exception as e:
                        print(f"Error processing batch {batch_idx}: {e}")
                        continue
            
            if total == 0:
                print("Warning: No samples were processed during evaluation!")
                return 0.0, float('inf'), np.array([]), np.array([])
            
            accuracy = 100. * correct / total
            avg_loss = loss / len(data_loader) if len(data_loader) > 0 else float('inf')
            
            return accuracy, avg_loss, np.array(predictions), np.array(targets_list)
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return 0.0, float('inf'), np.array([]), np.array([])
    
    def generate_confusion_matrix(self, predictions, targets, class_names=None):
        """
        生成混淆矩阵
        Args:
            predictions: 预测结果
            targets: 真实标签
            class_names: 类别名称列表
        Returns:
            cm: 混淆矩阵
        """
        try:
            if class_names is None:
                class_names = [str(i) for i in range(10)]  # MNIST默认类别名称
            
            if len(predictions) == 0 or len(targets) == 0:
                print("Warning: Empty predictions or targets, cannot generate confusion matrix!")
                return np.zeros((10, 10), dtype=int)
            
            cm = confusion_matrix(targets, predictions)
            return cm
        except Exception as e:
            print(f"Error generating confusion matrix: {e}")
            return np.zeros((10, 10), dtype=int)
    
    def plot_confusion_matrix(self, cm, class_names=None, save_path=None):
        """
        绘制混淆矩阵
        Args:
            cm: 混淆矩阵
            class_names: 类别名称列表
            save_path: 保存路径
        """
        try:
            if class_names is None:
                class_names = [str(i) for i in range(10)]  # MNIST默认类别名称
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            
            if save_path:
                try:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    plt.savefig(save_path)
                    print(f'Confusion matrix saved to: {save_path}')
                except Exception as e:
                    print(f"Error saving confusion matrix: {e}")
            
            plt.close()
        except Exception as e:
            print(f"Error plotting confusion matrix: {e}")
    
    def generate_classification_report(self, predictions, targets, class_names=None):
        """
        生成分类报告
        Args:
            predictions: 预测结果
            targets: 真实标签
            class_names: 类别名称列表
        Returns:
            report: 分类报告
        """
        try:
            if class_names is None:
                class_names = [str(i) for i in range(10)]  # MNIST默认类别名称
            
            if len(predictions) == 0 or len(targets) == 0:
                print("Warning: Empty predictions or targets, cannot generate classification report!")
                return "No classification report available (empty data)"
            
            report = classification_report(targets, predictions, target_names=class_names)
            return report
        except Exception as e:
            print(f"Error generating classification report: {e}")
            return f"Error generating classification report: {e}"
    
    def save_evaluation_report(self, accuracy, loss, report, cm, save_dir='reports', dataset_name='MNIST'):
        """
        保存评估报告
        Args:
            accuracy: 准确率
            loss: 损失
            report: 分类报告
            cm: 混淆矩阵
            save_dir: 保存目录
            dataset_name: 数据集名称
        """
        # 如果 save_dir 为 None，直接返回，不保存报告
        if save_dir is None:
            return
        
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            # 生成时间戳，确保文件名唯一
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 保存文本报告
            report_path = os.path.join(save_dir, f'evaluation_report_{dataset_name}_{timestamp}.txt')
            try:
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(f'评估报告\n')
                    f.write('=' * 50 + '\n')
                    f.write(f'时间戳: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
                    f.write(f'数据集: {dataset_name}\n')
                    f.write(f'准确率: {accuracy:.2f}%\n')
                    f.write(f'平均损失: {loss:.4f}\n')
                    f.write('=' * 50 + '\n')
                    f.write('分类报告:\n')
                    f.write(report + '\n')
                    f.write('=' * 50 + '\n')
                    f.write('混淆矩阵:\n')
                    f.write(str(cm) + '\n')
                
                print(f'评估报告已保存到: {report_path}')
            except Exception as e:
                print(f"Error saving evaluation report: {e}")
            
            # 保存混淆矩阵图像
            cm_path = os.path.join(save_dir, f'confusion_matrix_{dataset_name}_{timestamp}.png')
            self.plot_confusion_matrix(cm, save_path=cm_path)
        except Exception as e:
            print(f"Error in save_evaluation_report: {e}")
    
    def run_evaluation(self, data_loader, save_dir='reports', dataset_name='MNIST'):
        """
        运行完整的评估流程
        Args:
            data_loader: 数据加载器
            save_dir: 保存目录
            dataset_name: 数据集名称
        Returns:
            evaluation_results: 评估结果
        """
        try:
            print(f'正在 {self.device} 上评估模型...')
            
            # 评估模型
            accuracy, loss, predictions, targets = self.evaluate(data_loader)
            
            # 生成混淆矩阵
            cm = self.generate_confusion_matrix(predictions, targets)
            
            # 生成分类报告
            report = self.generate_classification_report(predictions, targets)
            
            # 保存评估报告
            self.save_evaluation_report(accuracy, loss, report, cm, save_dir, dataset_name)
            
            # 打印评估结果
            print('=' * 50)
            print(f'评估结果 ({dataset_name}):')
            print(f'准确率: {accuracy:.2f}%')
            print(f'平均损失: {loss:.4f}')
            print('=' * 50)
            
            evaluation_results = {
                'accuracy': accuracy,
                'loss': loss,
                'predictions': predictions,
                'targets': targets,
                'confusion_matrix': cm,
                'classification_report': report
            }
            
            return evaluation_results
        except Exception as e:
            print(f"Error in run_evaluation: {e}")
            # 返回默认值
            return {
                'accuracy': 0.0,
                'loss': float('inf'),
                'predictions': np.array([]),
                'targets': np.array([]),
                'confusion_matrix': np.zeros((10, 10), dtype=int),
                'classification_report': f"Error in evaluation: {e}"
            }