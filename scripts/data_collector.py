import sys
import os
import time
import csv
import json
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QGroupBox, QSlider, QGridLayout,
    QStatusBar, QFileDialog, QMessageBox, QSpinBox, QProgressBar
)
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt

from src.testing.digit_drawer import DigitDrawer

class DataCollectorApp(QMainWindow):
    """手写数字数据收集应用"""
    
    def __init__(self):
        """初始化应用"""
        super(DataCollectorApp, self).__init__()
        
        # 设置窗口标题和大小
        self.setWindowTitle("手写数字数据收集器")
        self.setGeometry(100, 100, 800, 600)
        
        # 配置文件路径
        self.config_file = os.path.join('config', 'data_collector_config.json')
        
        # 初始化数据收集相关变量
        self.dataset_path = None
        self.current_digit = 0
        self.sample_count = 0
        self.samples_per_digit = {}
        for i in range(10):
            self.samples_per_digit[i] = 0
        
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
        
        self.save_button = QPushButton("保存样本")
        self.save_button.clicked.connect(self.save_sample)
        self.control_layout.addWidget(self.save_button)
        
        self.left_layout.addLayout(self.control_layout)
        
        # 创建右侧布局（数据收集和导出）
        self.right_layout = QVBoxLayout()
        
        # 创建数据收集区域
        self.data_group = QGroupBox("数据收集设置")
        self.data_layout = QVBoxLayout()
        
        # 数字标签选择
        self.digit_layout = QHBoxLayout()
        self.digit_label = QLabel("数字标签:")
        self.digit_spin = QSpinBox()
        self.digit_spin.setRange(0, 9)
        self.digit_spin.setValue(0)
        self.digit_spin.valueChanged.connect(self.on_digit_changed)
        self.digit_layout.addWidget(self.digit_label)
        self.digit_layout.addWidget(self.digit_spin)
        self.data_layout.addLayout(self.digit_layout)
        
        # 数据集路径设置
        self.path_layout = QHBoxLayout()
        self.path_label = QLabel("数据集路径:")
        self.path_button = QPushButton("选择路径")
        self.path_button.clicked.connect(self.select_dataset_path)
        self.path_layout.addWidget(self.path_label)
        self.path_layout.addWidget(self.path_button)
        self.data_layout.addLayout(self.path_layout)
        
        # 样本统计
        self.stats_layout = QVBoxLayout()
        self.stats_label = QLabel("样本统计:")
        self.stats_layout.addWidget(self.stats_label)
        
        self.sample_grid = QGridLayout()
        self.sample_labels = []
        for i in range(10):
            label = QLabel(f"{i}: 0")
            label.setAlignment(Qt.AlignCenter)
            self.sample_labels.append(label)
            row = i // 5
            col = i % 5
            self.sample_grid.addWidget(label, row, col)
        
        self.stats_layout.addLayout(self.sample_grid)
        self.data_layout.addLayout(self.stats_layout)
        
        self.data_group.setLayout(self.data_layout)
        self.right_layout.addWidget(self.data_group)
        
        # 创建导出区域
        self.export_group = QGroupBox("数据集导出")
        self.export_layout = QVBoxLayout()
        
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(["PNG + CSV", "NumPy数组", "PyTorch数据集"])
        self.export_layout.addWidget(QLabel("导出格式:"))
        self.export_layout.addWidget(self.export_format_combo)
        
        self.export_button = QPushButton("导出数据集")
        self.export_button.clicked.connect(self.export_dataset)
        self.export_layout.addWidget(self.export_button)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.export_layout.addWidget(self.progress_bar)
        
        self.export_group.setLayout(self.export_layout)
        self.right_layout.addWidget(self.export_group)
        
        # 将左侧和右侧布局添加到主布局
        self.main_layout.addLayout(self.left_layout)
        self.main_layout.addLayout(self.right_layout)
        
        # 创建状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")
        
        # 初始化配置和数据集路径
        self.load_config()
        if not self.dataset_path:
            self.create_default_dataset_path()
        else:
            # 确保文件夹结构存在
            for i in range(10):
                digit_dir = os.path.join(self.dataset_path, str(i))
                os.makedirs(digit_dir, exist_ok=True)
            self.status_bar.showMessage(f"加载上次使用的数据集路径: {self.dataset_path}")
            # 更新样本统计
            self.update_sample_stats()
    
    def load_config(self):
        """加载配置文件"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    if 'dataset_path' in config and os.path.exists(config['dataset_path']):
                        self.dataset_path = config['dataset_path']
            except Exception as e:
                self.status_bar.showMessage(f"加载配置失败: {str(e)}")
    
    def save_config(self):
        """保存配置文件"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            config = {
                'dataset_path': self.dataset_path,
                'last_used': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.status_bar.showMessage(f"保存配置失败: {str(e)}")
    
    def create_default_dataset_path(self):
        """创建默认数据集路径"""
        default_path = os.path.join('data', 'custom', 'unified_dataset')
        self.dataset_path = os.path.normpath(default_path)
        os.makedirs(self.dataset_path, exist_ok=True)
        for i in range(10):
            digit_dir = os.path.join(self.dataset_path, str(i))
            os.makedirs(digit_dir, exist_ok=True)
        self.status_bar.showMessage(f"默认数据集路径: {self.dataset_path}")
        # 保存配置
        self.save_config()
    
    def select_dataset_path(self):
        """选择数据集路径"""
        options = QFileDialog.Options()
        folder_path = QFileDialog.getExistingDirectory(
            self, "选择数据集文件夹", "", options=options
        )
        
        if folder_path:
            self.dataset_path = folder_path
            # 确保文件夹结构存在
            for i in range(10):
                digit_dir = os.path.join(self.dataset_path, str(i))
                os.makedirs(digit_dir, exist_ok=True)
            self.status_bar.showMessage(f"数据集路径: {self.dataset_path}")
            # 更新样本统计
            self.update_sample_stats()
            # 保存配置
            self.save_config()
    
    def on_digit_changed(self):
        """当数字标签改变时"""
        self.current_digit = self.digit_spin.value()
        self.status_bar.showMessage(f"准备绘制数字 {self.current_digit}")
        self.drawer.clear_canvas()
    
    def save_sample(self):
        """保存当前样本"""
        try:
            # 检查数据集路径
            if not self.dataset_path:
                QMessageBox.warning(self, "警告", "请先选择数据集路径")
                return
            
            # 获取绘制的图像
            image = self.drawer.get_image()
            
            # 检查图像是否为空
            if image.getbbox() is None:
                QMessageBox.warning(self, "警告", "请先绘制数字")
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
            
            # 更新统计信息
            self.samples_per_digit[digit] += 1
            self.sample_count += 1
            
            # 更新UI
            self.update_sample_stats()
            self.status_bar.showMessage(f"样本保存成功: {filename}")
            
            # 清除画布，准备下一个样本
            self.drawer.clear_canvas()
            
        except Exception as e:
            self.status_bar.showMessage(f"保存失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")
    
    def update_sample_stats(self):
        """更新样本统计信息"""
        # 重新计算每个数字的样本数量
        for i in range(10):
            digit_dir = os.path.join(self.dataset_path, str(i))
            if os.path.exists(digit_dir):
                count = len([f for f in os.listdir(digit_dir) if f.endswith('.png')])
                self.samples_per_digit[i] = count
                self.sample_labels[i].setText(f"{i}: {count}")
        
        total = sum(self.samples_per_digit.values())
        self.sample_count = total
        self.status_bar.showMessage(f"总样本数: {total}")
    
    def export_dataset(self):
        """导出数据集"""
        try:
            # 检查数据集路径
            if not self.dataset_path:
                QMessageBox.warning(self, "警告", "请先选择数据集路径")
                return
            
            # 检查是否有样本
            total_samples = sum(self.samples_per_digit.values())
            if total_samples == 0:
                QMessageBox.warning(self, "警告", "数据集中没有样本")
                return
            
            # 显示进度条
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.progress_bar.setMaximum(total_samples)
            
            # 获取导出格式
            export_format = self.export_format_combo.currentText()
            
            # 导出为PNG + CSV
            if export_format == "PNG + CSV":
                self.export_png_csv()
            # 导出为NumPy数组
            elif export_format == "NumPy数组":
                self.export_numpy()
            # 导出为PyTorch数据集
            elif export_format == "PyTorch数据集":
                self.export_pytorch()
            
            # 隐藏进度条
            self.progress_bar.setVisible(False)
            self.status_bar.showMessage("数据集导出成功")
            QMessageBox.information(self, "成功", "数据集导出成功")
            
        except Exception as e:
            self.progress_bar.setVisible(False)
            self.status_bar.showMessage(f"导出失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")
    
    def export_png_csv(self):
        """导出为PNG + CSV格式"""
        # 创建CSV文件
        csv_path = os.path.join(self.dataset_path, 'labels.csv')
        
        with open(csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['filename', 'label'])
            
            # 遍历所有数字文件夹
            count = 0
            for digit in range(10):
                digit_dir = os.path.join(self.dataset_path, str(digit))
                if os.path.exists(digit_dir):
                    for filename in os.listdir(digit_dir):
                        if filename.endswith('.png'):
                            csv_writer.writerow([os.path.join(str(digit), filename), digit])
                            count += 1
                            self.progress_bar.setValue(count)
                            QApplication.processEvents()
    
    def export_numpy(self):
        """导出为NumPy数组格式"""
        import numpy as np
        from PIL import Image
        
        # 收集所有图像和标签
        images = []
        labels = []
        
        count = 0
        for digit in range(10):
            digit_dir = os.path.join(self.dataset_path, str(digit))
            if os.path.exists(digit_dir):
                for filename in os.listdir(digit_dir):
                    if filename.endswith('.png'):
                        # 加载图像
                        img_path = os.path.join(digit_dir, filename)
                        img = Image.open(img_path).convert('L')
                        img = img.resize((28, 28))
                        img_array = np.array(img, dtype=np.float32) / 255.0
                        images.append(img_array)
                        labels.append(digit)
                        
                        count += 1
                        self.progress_bar.setValue(count)
                        QApplication.processEvents()
        
        # 转换为NumPy数组
        images = np.array(images)
        labels = np.array(labels)
        
        # 保存为.npy文件
        np.save(os.path.join(self.dataset_path, 'images.npy'), images)
        np.save(os.path.join(self.dataset_path, 'labels.npy'), labels)
    
    def export_pytorch(self):
        """导出为PyTorch数据集格式"""
        import torch
        from torch.utils.data import Dataset, DataLoader
        from PIL import Image
        import numpy as np
        
        # 创建自定义数据集类
        class CustomMNISTDataset(Dataset):
            def __init__(self, root_dir):
                self.root_dir = root_dir
                self.images = []
                self.labels = []
                
                # 收集所有图像和标签
                for digit in range(10):
                    digit_dir = os.path.join(root_dir, str(digit))
                    if os.path.exists(digit_dir):
                        for filename in os.listdir(digit_dir):
                            if filename.endswith('.png'):
                                img_path = os.path.join(digit_dir, filename)
                                self.images.append(img_path)
                                self.labels.append(digit)
            
            def __len__(self):
                return len(self.images)
            
            def __getitem__(self, idx):
                img_path = self.images[idx]
                label = self.labels[idx]
                
                # 加载和预处理图像
                img = Image.open(img_path).convert('L')
                img = img.resize((28, 28))
                img = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
                img = img.unsqueeze(0)  # 添加通道维度
                
                return img, label
        
        # 测试数据集
        dataset = CustomMNISTDataset(self.dataset_path)
        print(f"数据集大小: {len(dataset)}")
        
        # 保存数据集信息
        info_path = os.path.join(self.dataset_path, 'dataset_info.txt')
        with open(info_path, 'w') as f:
            f.write(f"Dataset size: {len(dataset)}\n")
            f.write("Classes: 0-9\n")
            f.write("Image size: 28x28\n")
            f.write(f"Export date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def keyPressEvent(self, event):
        """键盘事件处理"""
        # 数字键快速切换数字
        if event.key() >= Qt.Key_0 and event.key() <= Qt.Key_9:
            digit = event.key() - Qt.Key_0
            self.digit_spin.setValue(digit)
            self.status_bar.showMessage(f"准备绘制数字 {digit}")
            self.drawer.clear_canvas()
        # 空格键保存样本
        elif event.key() == Qt.Key_Space:
            self.save_sample()
        # ESC键清除画布
        elif event.key() == Qt.Key_Escape:
            self.drawer.clear_canvas()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DataCollectorApp()
    window.show()
    sys.exit(app.exec_())
