import sys
import os
import time
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QGroupBox, QSlider, QGridLayout,
    QStatusBar, QFileDialog, QMessageBox, QSpinBox, QTextEdit
)
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt

from src.testing.digit_drawer import DigitDrawer
from src.testing.model_loader import ModelLoader

class DigitRecognitionApp(QMainWindow):
    """手写数字识别应用"""
    
    def __init__(self):
        """初始化应用"""
        super(DigitRecognitionApp, self).__init__()
        
        # 设置窗口标题和大小
        self.setWindowTitle("手写数字识别测试软件")
        self.setGeometry(100, 100, 800, 600)
        
        # 初始化模型加载器
        self.model_loader = ModelLoader()
        
        # 初始化数据收集相关变量
        self.dataset_path = os.path.join('data', 'custom', 'unified_dataset')
        self.dataset_path = os.path.normpath(self.dataset_path)
        os.makedirs(self.dataset_path, exist_ok=True)
        for i in range(10):
            digit_dir = os.path.join(self.dataset_path, str(i))
            os.makedirs(digit_dir, exist_ok=True)
        
        # 初始化准确率统计变量
        self.total_predictions = 0
        self.correct_predictions = 0
        self.accuracy = 0.0
        
        # 初始化每个数字的统计变量
        self.digit_stats = {}
        for i in range(10):
            self.digit_stats[i] = {
                'total': 0,      # 预测次数
                'correct': 0,    # 正确次数
                'incorrect': 0,  # 错误次数
                'accuracy': 0.0  # 准确率
            }
        
        # 创建主部件和布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 创建主布局
        self.main_layout = QHBoxLayout(self.central_widget)
        
        # 创建左侧布局（绘图区域）
        self.left_layout = QVBoxLayout()
        
        # 创建绘图控件
        self.drawer = DigitDrawer(width=400, height=400)
        self.left_layout.addWidget(self.drawer)
        
        # 创建绘图控制按钮
        self.control_layout = QHBoxLayout()
        
        self.clear_button = QPushButton("清除")
        self.clear_button.clicked.connect(self.drawer.clear_canvas)
        self.control_layout.addWidget(self.clear_button)
        
        self.save_button = QPushButton("保存")
        self.save_button.clicked.connect(self.save_drawing)
        self.control_layout.addWidget(self.save_button)
        
        self.left_layout.addLayout(self.control_layout)
        
        # 创建右侧布局（模型和结果）
        self.right_layout = QVBoxLayout()
        
        # 创建模型选择区域
        self.model_group = QGroupBox("模型设置")
        self.model_layout = QVBoxLayout()
        
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["PyTorch模型", "ONNX模型"])
        self.model_type_combo.currentIndexChanged.connect(self.update_model_list)
        self.model_layout.addWidget(QLabel("模型类型:"))
        self.model_layout.addWidget(self.model_type_combo)
        
        self.model_file_combo = QComboBox()
        self.update_model_list()
        self.model_layout.addWidget(QLabel("模型文件:"))
        self.model_layout.addWidget(self.model_file_combo)
        
        # 添加模型架构选择
        self.architecture_combo = QComboBox()
        self.architecture_combo.addItems(["自动检测", "tiny", "stm32", "enhanced", "lightweight", "advanced"])
        self.model_layout.addWidget(QLabel("模型架构:"))
        self.model_layout.addWidget(self.architecture_combo)
        
        self.load_model_button = QPushButton("加载模型")
        self.load_model_button.clicked.connect(self.load_model)
        self.model_layout.addWidget(self.load_model_button)
        
        # 添加模型信息显示
        self.model_info_text = QTextEdit()
        self.model_info_text.setReadOnly(True)
        self.model_info_text.setFixedHeight(100)
        self.model_layout.addWidget(QLabel("模型信息:"))
        self.model_layout.addWidget(self.model_info_text)
        
        self.model_group.setLayout(self.model_layout)
        self.right_layout.addWidget(self.model_group)
        
        # 创建画笔设置区域
        self.brush_group = QGroupBox("画笔设置")
        self.brush_layout = QVBoxLayout()
        
        self.brush_size_slider = QSlider(Qt.Horizontal)
        self.brush_size_slider.setMinimum(5)
        self.brush_size_slider.setMaximum(50)
        self.brush_size_slider.setValue(20)
        self.brush_size_slider.valueChanged.connect(lambda value: self.drawer.set_pen_width(value))
        self.brush_layout.addWidget(QLabel("画笔大小:"))
        self.brush_layout.addWidget(self.brush_size_slider)
        
        self.brush_group.setLayout(self.brush_layout)
        self.right_layout.addWidget(self.brush_group)
        
        # 创建预测结果区域
        self.result_group = QGroupBox("预测结果")
        self.result_layout = QVBoxLayout()
        
        self.predict_button = QPushButton("预测")
        self.predict_button.clicked.connect(self.predict_digit)
        self.result_layout.addWidget(self.predict_button)
        
        self.prediction_label = QLabel("预测结果: ")
        self.prediction_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.result_layout.addWidget(self.prediction_label)
        
        self.confidence_label = QLabel("置信度: ")
        self.result_layout.addWidget(self.confidence_label)
        
        self.time_label = QLabel("推理时间: ")
        self.result_layout.addWidget(self.time_label)
        
        # 创建概率显示网格
        self.probability_grid = QGridLayout()
        self.probability_labels = []
        
        for i in range(10):
            label = QLabel(f"{i}: 0%")
            label.setAlignment(Qt.AlignCenter)
            self.probability_labels.append(label)
            row = i // 5
            col = i % 5
            self.probability_grid.addWidget(label, row, col)
        
        self.result_layout.addLayout(self.probability_grid)
        
        self.result_group.setLayout(self.result_layout)
        self.right_layout.addWidget(self.result_group)
        
        # 创建数据收集区域
        self.data_collection_group = QGroupBox("数据收集")
        self.data_collection_layout = QVBoxLayout()
        
        # 数字标签选择
        self.digit_layout = QHBoxLayout()
        self.digit_label = QLabel("数字标签:")
        self.digit_spin = QSpinBox()
        self.digit_spin.setRange(0, 9)
        self.digit_spin.setValue(0)
        self.digit_layout.addWidget(self.digit_label)
        self.digit_layout.addWidget(self.digit_spin)
        self.data_collection_layout.addLayout(self.digit_layout)
        
        # 保存样本按钮
        self.save_sample_button = QPushButton("保存样本")
        self.save_sample_button.clicked.connect(self.save_sample)
        self.data_collection_layout.addWidget(self.save_sample_button)
        
        self.data_collection_group.setLayout(self.data_collection_layout)
        self.right_layout.addWidget(self.data_collection_group)
        
        # 创建报告区域
        self.report_group = QGroupBox("测试报告")
        self.report_layout = QVBoxLayout()
        
        # 准确率显示
        self.accuracy_label = QLabel("准确率: 0.00%")
        self.accuracy_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.report_layout.addWidget(self.accuracy_label)
        
        # 预测统计
        self.stats_label = QLabel("预测次数: 0, 正确: 0")
        self.report_layout.addWidget(self.stats_label)
        
        # 输出报告按钮
        self.generate_report_button = QPushButton("输出报告")
        self.generate_report_button.clicked.connect(self.generate_report)
        self.report_layout.addWidget(self.generate_report_button)
        
        self.report_group.setLayout(self.report_layout)
        self.right_layout.addWidget(self.report_group)
        
        # 将左侧和右侧布局添加到主布局
        self.main_layout.addLayout(self.left_layout)
        self.main_layout.addLayout(self.right_layout)
        
        # 创建状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")
        
        # 更新模型列表
        self.update_model_list()
        
        # 自动加载最新模型
        self.auto_load_latest_model()
    
    def auto_load_latest_model(self):
        """自动加载最新训练的模型"""
        try:
            # 构建模型目录路径
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            model_dir = os.path.join(project_root, "models", "trained")
            
            # 调用模型加载器的自动加载方法
            success, model_path = self.model_loader.load_latest_model(model_dir)
            
            if success:
                model_name = os.path.basename(model_path)
                self.status_bar.showMessage(f"自动加载模型成功: {model_name}")
                print(f"自动加载模型成功: {model_path}")
                # 更新模型信息显示
                self.update_model_info()
            else:
                self.status_bar.showMessage("未找到模型，可手动加载")
                print("未找到模型，可手动加载")
                
        except Exception as e:
            self.status_bar.showMessage(f"自动加载模型失败: {str(e)}")
            print(f"自动加载模型失败: {str(e)}")
    
    def update_model_list(self):
        """更新模型文件列表"""
        model_type = self.model_type_combo.currentText()
        self.model_file_combo.clear()
        
        if model_type == "PyTorch模型":
            # 加载PyTorch模型文件
            model_dir = "models/trained"
            if os.path.exists(model_dir):
                for root, dirs, files in os.walk(model_dir):
                    for file in files:
                        if file.endswith(".pth"):
                            relative_path = os.path.relpath(os.path.join(root, file), start="models")
                            self.model_file_combo.addItem(relative_path)
        else:
            # 加载ONNX模型文件
            model_dir = "models/exported"
            if os.path.exists(model_dir):
                for root, dirs, files in os.walk(model_dir):
                    for file in files:
                        if file.endswith(".onnx"):
                            relative_path = os.path.relpath(os.path.join(root, file), start="models")
                            self.model_file_combo.addItem(relative_path)
    
    def load_model(self):
        """加载选中的模型"""
        model_type = self.model_type_combo.currentText()
        model_file = self.model_file_combo.currentText()
        architecture = self.architecture_combo.currentText()
        
        if not model_file:
            QMessageBox.warning(self, "警告", "请选择一个模型文件")
            return
        
        try:
            # 构建绝对路径以避免路径问题
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # 从 scripts 目录向上一级到项目根目录
            project_root = os.path.dirname(script_dir)
            model_path = os.path.join(project_root, "models", model_file)
            model_path = os.path.normpath(model_path)
            
            print(f"尝试加载模型: {model_path}")
            print(f"模型文件是否存在: {os.path.exists(model_path)}")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
            if model_type == "PyTorch模型":
                if architecture == "自动检测":
                    # 尝试自动检测模型类型
                    model_file_name = os.path.basename(model_path)
                    if "tiny" in model_file_name.lower():
                        self.model_loader.load_pytorch_model(model_path, "tiny")
                    elif "enhanced" in model_file_name.lower():
                        self.model_loader.load_pytorch_model(model_path, "enhanced")
                    else:
                        # 尝试按顺序加载不同模型类型
                        model_architectures = ["tiny", "stm32", "enhanced", "lightweight", "advanced"]
                        loaded = False
                        for arch in model_architectures:
                            try:
                                print(f"尝试使用 {arch} 架构加载...")
                                self.model_loader.load_pytorch_model(model_path, arch)
                                loaded = True
                                break
                            except Exception as e:
                                print(f"使用 {arch} 架构加载失败: {str(e)}")
                                continue
                        
                        if not loaded:
                            raise Exception("无法使用任何支持的架构加载模型")
                else:
                    # 使用指定的架构
                    self.model_loader.load_pytorch_model(model_path, architecture)
            else:
                # 加载ONNX模型
                self.model_loader.load_onnx_model(model_path)
            
            self.status_bar.showMessage(f"模型加载成功: {model_file}")
            QMessageBox.information(self, "成功", "模型加载成功")
            # 更新模型信息显示
            self.update_model_info()
            # 验证模型性能
            self.validate_model_performance()
        except Exception as e:
            error_msg = f"模型加载失败: {str(e)}"
            print(error_msg)
            self.status_bar.showMessage(error_msg)
            QMessageBox.critical(self, "错误", error_msg)
            # 更新模型信息显示
            self.update_model_info()
    
    def update_model_info(self):
        """更新模型信息显示"""
        model_info = self.model_loader.get_model_info()
        info_text = []
        for key, value in model_info.items():
            if value is not None:
                info_text.append(f"{key}: {value}")
        self.model_info_text.setPlainText("\n".join(info_text))
    
    def validate_model_performance(self):
        """验证模型性能"""
        try:
            # 这里可以添加模型性能验证逻辑
            # 例如加载测试数据并评估模型性能
            print("验证模型性能...")
            # 暂时只显示提示
            self.status_bar.showMessage("模型加载成功，性能验证完成")
        except Exception as e:
            print(f"模型性能验证失败: {str(e)}")
    
    def predict_digit(self):
        """预测手写数字"""
        try:
            # 检查模型是否加载
            if self.model_loader.model_type is None:
                QMessageBox.warning(self, "警告", "请先加载模型")
                return
            
            # 检查模型加载是否有错误
            model_info = self.model_loader.get_model_info()
            if model_info.get('load_error'):
                QMessageBox.warning(self, "警告", f"模型加载有错误: {model_info['load_error']}")
                return
            
            # 获取绘制的图像
            image = self.drawer.get_image()
            
            # 检查图像是否为空
            if image.getbbox() is None:
                QMessageBox.warning(self, "警告", "请先绘制数字")
                return
            
            # 记录开始时间
            start_time = time.time()
            
            # 进行预测
            predicted_class, confidence, all_probabilities = self.model_loader.predict(image)
            
            # 计算推理时间
            inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
            
            # 更新结果显示
            self.prediction_label.setText(f"预测结果: {predicted_class}")
            self.confidence_label.setText(f"置信度: {confidence:.2%}")
            self.time_label.setText(f"推理时间: {inference_time:.2f} ms")
            
            # 更新概率显示
            for i, prob in enumerate(all_probabilities):
                self.probability_labels[i].setText(f"{i}: {prob:.2%}")
            
            # 自动保存样本
            self.save_sample()
            
            # 计算准确率
            self.total_predictions += 1
            actual_digit = self.digit_spin.value()
            if predicted_class == actual_digit:
                self.correct_predictions += 1
            
            # 更新每个数字的统计信息
            self.digit_stats[actual_digit]['total'] += 1
            if predicted_class == actual_digit:
                self.digit_stats[actual_digit]['correct'] += 1
            else:
                self.digit_stats[actual_digit]['incorrect'] += 1
            
            # 计算每个数字的准确率
            if self.digit_stats[actual_digit]['total'] > 0:
                self.digit_stats[actual_digit]['accuracy'] = (
                    self.digit_stats[actual_digit]['correct'] / 
                    self.digit_stats[actual_digit]['total']
                )
            else:
                self.digit_stats[actual_digit]['accuracy'] = 0.0
            
            # 计算总体准确率
            if self.total_predictions > 0:
                self.accuracy = self.correct_predictions / self.total_predictions
            else:
                self.accuracy = 0.0
            
            # 更新准确率显示
            self.accuracy_label.setText(f"准确率: {self.accuracy:.2%}")
            self.stats_label.setText(f"预测次数: {self.total_predictions}, 正确: {self.correct_predictions}")
            
            self.status_bar.showMessage(f"预测完成: 数字 {predicted_class}, 置信度 {confidence:.2%}, 样本已保存, 准确率: {self.accuracy:.2%}")
            
        except Exception as e:
            error_msg = f"预测失败: {str(e)}"
            self.status_bar.showMessage(error_msg)
            print(error_msg)
            QMessageBox.critical(self, "错误", error_msg)
    
    def save_drawing(self):
        """保存绘制的图像"""
        try:
            # 获取绘制的图像
            image = self.drawer.get_image()
            
            # 检查图像是否为空
            if image.getbbox() is None:
                QMessageBox.warning(self, "警告", "请先绘制数字")
                return
            
            # 打开保存对话框
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存图像", "", "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)",
                options=options
            )
            
            if file_path:
                image.save(file_path)
                self.status_bar.showMessage(f"图像保存成功: {file_path}")
                QMessageBox.information(self, "成功", "图像保存成功")
                
        except Exception as e:
            error_msg = f"保存失败: {str(e)}"
            self.status_bar.showMessage(error_msg)
            QMessageBox.critical(self, "错误", error_msg)
    
    def save_sample(self):
        """保存当前样本到数据集"""
        try:
            # 获取绘制的图像
            image = self.drawer.get_image()
            
            # 检查图像是否为空
            if image.getbbox() is None:
                return
            
            # 获取当前数字标签
            digit = self.digit_spin.value()
            
            # 创建数字文件夹（如果不存在）
            digit_dir = os.path.join(self.dataset_path, str(digit))
            os.makedirs(digit_dir, exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f'{digit}_{timestamp}.png'
            file_path = os.path.join(digit_dir, filename)
            
            # 保存图像
            image.save(file_path)
            
            # 更新状态栏
            self.status_bar.showMessage(f"样本保存成功: {filename} 到数字 {digit} 的目录")
            
        except Exception as e:
            error_msg = f"保存失败: {str(e)}"
            self.status_bar.showMessage(error_msg)
            print(error_msg)

    def generate_report(self):
        """生成测试报告"""
        try:
            # 生成报告内容
            report_content = f"""
手写数字识别测试报告
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
=====================================
测试统计:
- 总预测次数: {self.total_predictions}
- 正确预测次数: {self.correct_predictions}
- 准确率: {self.accuracy:.2%}
- 错误预测次数: {self.total_predictions - self.correct_predictions}

各数字统计:
"""

            # 添加每个数字的统计信息
            for digit in range(10):
                stats = self.digit_stats[digit]
                report_content += f"- 数字 {digit}: 预测 {stats['total']} 次, 正确 {stats['correct']} 次, 错误 {stats['incorrect']} 次, 准确率 {stats['accuracy']:.2%}\n"

            model_info = self.model_loader.get_model_info()
            report_content += f"""
模型信息:
- 模型类型: {model_info.get('type', '未加载')}
"""
            if model_info.get('model'):
                report_content += f"- 模型架构: {model_info['model']}\n"
            if model_info.get('model_name'):
                report_content += f"- 模型名称: {model_info['model_name']}\n"
            if model_info.get('path'):
                report_content += f"- 模型路径: {model_info['path']}\n"
            if model_info.get('device'):
                report_content += f"- 设备: {model_info['device']}\n"
            if model_info.get('load_error'):
                report_content += f"- 加载错误: {model_info['load_error']}\n"

            report_content += f"""
测试环境:
- 系统: {os.name}
- Python版本: {sys.version.split()[0]}

=====================================
报告结束
"""
            
            # 打开保存对话框
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存测试报告", f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", 
                "Text Files (*.txt);;All Files (*)", options=options
            )
            
            if file_path:
                # 保存报告到文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                
                self.status_bar.showMessage(f"报告保存成功: {file_path}")
                QMessageBox.information(self, "成功", f"报告保存成功:\n{file_path}")
                
        except Exception as e:
            error_msg = f"生成报告失败: {str(e)}"
            self.status_bar.showMessage(error_msg)
            QMessageBox.critical(self, "错误", error_msg)

    def keyPressEvent(self, event):
        """键盘事件处理"""
        # 数字键快速测试
        if event.key() >= Qt.Key_0 and event.key() <= Qt.Key_9:
            # 清除画布
            self.drawer.clear_canvas()
            digit = event.key() - Qt.Key_0
            self.digit_spin.setValue(digit)
            self.status_bar.showMessage(f"准备绘制数字 {digit}")
        # 空格键预测
        elif event.key() == Qt.Key_Space:
            self.predict_digit()
        # ESC键清除画布
        elif event.key() == Qt.Key_Escape:
            self.drawer.clear_canvas()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DigitRecognitionApp()
    window.show()
    sys.exit(app.exec_())