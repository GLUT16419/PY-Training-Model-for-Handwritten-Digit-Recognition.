import torch
import os
import numpy as np

class ModelExporter:
    """模型导出器"""
    
    def __init__(self, model, input_shape=(1, 1, 28, 28)):
        """
        初始化导出器
        Args:
            model: 模型实例
            input_shape: 输入数据形状 (batch_size, channels, height, width)
        """
        self.model = model
        self.input_shape = input_shape
        self.model.eval()
    
    def export_to_onnx(self, export_path='models/exported/mnist_model.onnx'):
        """
        导出模型为ONNX格式
        Args:
            export_path: 导出路径
        Returns:
            export_path: 导出路径
        """
        # 创建导出目录
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        
        # 创建示例输入
        dummy_input = torch.randn(self.input_shape)
        
        # 导出模型
        torch.onnx.export(
            self.model,
            dummy_input,
            export_path,
            verbose=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f'Model exported to ONNX format: {export_path}')
        return export_path
    
    def prepare_for_stm32(self, onnx_path, output_dir='models/exported/stm32'):
        """
        为STM32部署准备模型
        Args:
            onnx_path: ONNX模型路径
            output_dir: 输出目录
        Returns:
            output_dir: 输出目录
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 复制ONNX模型到输出目录
        import shutil
        stm32_onnx_path = os.path.join(output_dir, 'mnist_model.onnx')
        shutil.copy2(onnx_path, stm32_onnx_path)
        
        # 创建模型信息文件
        model_info = {
            'model_name': 'MNIST_Digit_Classifier',
            'input_shape': self.input_shape,
            'output_shape': (self.input_shape[0], 10),  # MNIST有10个类别
            'onnx_path': 'mnist_model.onnx',
            'description': 'Lightweight MNIST digit classifier for STM32F407'
        }
        
        # 保存模型信息
        info_path = os.path.join(output_dir, 'model_info.txt')
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
4. 导入此目录中的mnist_model.onnx文件
5. 配置AI模型参数
6. 生成代码
7. 将生成的代码集成到项目中
8. 编译并烧录到STM32F407开发板

注意事项:
- 确保STM32F407有足够的内存来运行模型
- 输入数据需要与训练时的预处理方式一致
- 模型输入尺寸为28x28的灰度图像
"""
        
        guide_path = os.path.join(output_dir, 'deploy_guide.txt')
        with open(guide_path, 'w') as f:
            f.write(deploy_guide)
        
        print(f'Model prepared for STM32 deployment: {output_dir}')
        return output_dir
    
    def verify_onnx_model(self, onnx_path):
        """
        验证ONNX模型是否正确
        Args:
            onnx_path: ONNX模型路径
        Returns:
            is_valid: 模型是否有效
        """
        try:
            import onnx
            import onnxruntime as rt
            
            # 加载ONNX模型
            model = onnx.load(onnx_path)
            
            # 检查模型结构
            onnx.checker.check_model(model)
            print('ONNX model structure is valid.')
            
            # 创建ONNX运行时会话
            sess = rt.InferenceSession(onnx_path)
            
            # 获取输入和输出信息
            input_name = sess.get_inputs()[0].name
            output_name = sess.get_outputs()[0].name
            
            print(f'Input name: {input_name}')
            print(f'Output name: {output_name}')
            
            # 创建测试输入
            test_input = np.random.randn(*self.input_shape).astype(np.float32)
            
            # 运行推理
            outputs = sess.run([output_name], {input_name: test_input})
            print(f'Inference successful! Output shape: {outputs[0].shape}')
            
            return True
            
        except Exception as e:
            print(f'Error verifying ONNX model: {e}')
            return False
    
    def run_export_pipeline(self, export_dir='models/exported'):
        """
        运行完整的导出流程
        Args:
            export_dir: 导出目录
        Returns:
            export_results: 导出结果
        """
        print('Running model export pipeline...')
        
        # 导出为ONNX格式
        onnx_path = self.export_to_onnx(os.path.join(export_dir, 'mnist_model.onnx'))
        
        # 验证ONNX模型
        is_valid = self.verify_onnx_model(onnx_path)
        
        if is_valid:
            # 为STM32准备模型
            stm32_dir = self.prepare_for_stm32(onnx_path, os.path.join(export_dir, 'stm32'))
            
            export_results = {
                'onnx_path': onnx_path,
                'stm32_dir': stm32_dir,
                'is_valid': is_valid
            }
            
            print('Model export pipeline completed successfully!')
            print(f'ONNX model: {onnx_path}')
            print(f'STM32 deployment files: {stm32_dir}')
            
            return export_results
        else:
            print('Model export pipeline failed!')
            return None