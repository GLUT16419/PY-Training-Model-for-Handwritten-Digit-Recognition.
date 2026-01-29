from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtGui import QPainter, QPen, QImage
from PyQt5.QtCore import Qt, QPoint
from PIL import Image
import numpy as np

class DigitDrawer(QWidget):
    """手写数字绘制控件"""
    
    def __init__(self, parent=None, width=400, height=400):
        """
        初始化绘图控件
        Args:
            parent: 父控件
            width: 画布宽度
            height: 画布高度
        """
        super(DigitDrawer, self).__init__(parent)
        
        # 设置画布大小
        self.width = width
        self.height = height
        self.setFixedSize(width, height)
        
        # 初始化画布
        self.image = QImage(width, height, QImage.Format_RGB32)
        self.clear_canvas()
        
        # 绘图状态
        self.drawing = False
        self.last_point = QPoint()
        
        # 画笔设置
        self.pen_width = 20
        self.pen_color = Qt.black
        
    def clear_canvas(self):
        """清除画布"""
        self.image.fill(Qt.white)
        self.update()
    
    def set_pen_width(self, width):
        """
        设置画笔宽度
        Args:
            width: 画笔宽度
        """
        self.pen_width = width
    
    def set_pen_color(self, color):
        """
        设置画笔颜色
        Args:
            color: 画笔颜色
        """
        self.pen_color = color
    
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        if event.buttons() & Qt.LeftButton and self.drawing:
            painter = QPainter(self.image)
            pen = QPen(self.pen_color, self.pen_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()
    
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if event.button() == Qt.LeftButton:
            self.drawing = False
    
    def paintEvent(self, event):
        """绘制事件"""
        painter = QPainter(self)
        painter.drawImage(self.rect(), self.image, self.image.rect())
        
        # 绘制网格辅助线
        painter.setPen(QPen(Qt.lightGray, 1, Qt.DashLine))
        
        # 绘制垂直网格线
        grid_size = 40  # 网格大小
        for x in range(0, self.width, grid_size):
            painter.drawLine(x, 0, x, self.height)
        
        # 绘制水平网格线
        for y in range(0, self.height, grid_size):
            painter.drawLine(0, y, self.width, y)
        
        # 绘制中心参考框（28x28像素的区域）
        # 计算中心位置
        center_x = (self.width - 280) // 2  # 放大10倍显示
        center_y = (self.height - 280) // 2
        
        # 绘制中心框
        painter.setPen(QPen(Qt.blue, 2, Qt.SolidLine))
        painter.drawRect(center_x, center_y, 280, 280)
        
        # 添加文字提示
        painter.setPen(QPen(Qt.black, 1))
        painter.drawText(10, 20, "请在蓝色框内绘制数字，大小适中")
    
    def get_image(self):
        """
        获取绘制的图像
        Returns:
            PIL.Image: 绘制的图像
        """
        # 将QImage转换为PIL.Image
        image = Image.new('RGB', (self.width, self.height), color='white')
        for x in range(self.width):
            for y in range(self.height):
                pixel = self.image.pixel(x, y)
                r = (pixel >> 16) & 0xFF
                g = (pixel >> 8) & 0xFF
                b = pixel & 0xFF
                image.putpixel((x, y), (r, g, b))
        
        return image
    
    def get_resized_image(self, size=28):
        """
        获取调整大小后的图像
        Args:
            size: 目标大小
        Returns:
            PIL.Image: 调整大小后的图像
        """
        image = self.get_image()
        # 调整大小
        image = image.resize((size, size), Image.Resampling.LANCZOS)
        return image
    
    def save_image(self, file_path):
        """
        保存图像
        Args:
            file_path: 保存路径
        """
        image = self.get_image()
        image.save(file_path)