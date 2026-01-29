import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import warnings
from datetime import datetime
import glob

# 抑制弃用警告
warnings.filterwarnings("ignore", category=FutureWarning)

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=10, min_delta=0.001):
        """
        初始化早停机制
        Args:
            patience: 容忍模型没有改善的轮数
            min_delta: 认为模型有改善的最小变化值
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_accuracy):
        """
        检查是否应该早停
        Args:
            val_accuracy: 当前验证准确率
        Returns:
            early_stop: 是否应该早停
        """
        score = val_accuracy
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f'早停计数器: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, model, device=None, log_path=None):
        """
        初始化训练器
        Args:
            model: 模型实例
            device: 设备 (cuda 或 cpu)
            log_path: 日志文件路径
        """
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.log_path = log_path
        
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        # 改用AdamW优化器，学习率设置为0.001，添加L2正则化
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.001)
        
        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        # 训练历史
        self.train_history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'custom_val_loss': [],
            'custom_val_accuracy': []
        }
    
    def log(self, message):
        """
        记录日志
        Args:
            message: 日志消息
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_message = f'[{timestamp}] {message}'
        
        # 打印到终端
        print(log_message)
        
        # 写入日志文件
        if self.log_path:
            try:
                with open(self.log_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(log_message + '\n')
                    log_file.flush()
            except Exception as e:
                print(f'日志写入失败: {e}')
    
    def train(self, train_loader, val_loader, custom_val_loader=None, epochs=20, model_save_path='models/trained'):
        """
        训练模型
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            custom_val_loader: 自定义数据验证加载器
            epochs: 训练轮数
            model_save_path: 模型保存路径
        Returns:
            best_model: 性能最好的模型
        """
        try:
            # 创建模型保存目录
            os.makedirs(model_save_path, exist_ok=True)
            
            best_val_accuracy = 0.0
            best_custom_val_accuracy = 0.0
            best_model = None
            
            # 初始化最佳模型列表，最多保存3个
            best_models = []
            
            # 初始化早停机制
            early_stopping = EarlyStopping(patience=8, min_delta=0.001)
            
            # 初始化余弦退火学习率调度器，带有预热
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=8,  # 第一次重启的迭代次数
                T_mult=2,  # 每次重启后T_0的乘数
                eta_min=0.000001,  # 最小学习率
                last_epoch=-1
            )
            
            self.log(f"正在 {self.device} 上训练...")
            
            # 打印训练配置
            self.log("=" * 100)
            self.log("训练配置:")
            self.log(f"模型: {self.model.__class__.__name__}")
            self.log(f"训练轮数: {epochs}")
            self.log(f"批处理大小: {train_loader.batch_size}")
            self.log(f"学习率: {self.optimizer.param_groups[0]['lr']}")
            self.log(f"训练样本数: {len(train_loader.dataset)}")
            self.log(f"验证样本数: {len(val_loader.dataset)}")
            if custom_val_loader:
                self.log(f"自定义验证样本数: {len(custom_val_loader.dataset)}")
            self.log("=" * 100)
            
            # 检查是否使用GPU
            if torch.cuda.is_available():
                import pynvml
                try:
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    self.log(f"GPU: {pynvml.nvmlDeviceGetName(handle)}")
                    self.log(f"GPU Memory Total: {pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1024**3:.2f} GB")
                    gpu_available = True
                except Exception as e:
                    self.log(f"Error initializing GPU monitoring: {e}")
                    gpu_available = False
            else:
                gpu_available = False
            
            # 记录训练开始时间
            training_start_time = time.time()
            
            for epoch in range(epochs):
                start_time = time.time()
                
                # 训练阶段
                self.model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                # 计算训练预计完成时间
                elapsed_time = time.time() - training_start_time
                epoch_time_avg = elapsed_time / (epoch + 1) if epoch > 0 else 0
                remaining_time = epoch_time_avg * (epochs - epoch - 1)
                
                self.log(f"\n{'=' * 100}")
                self.log(f"开始轮次 {epoch+1}/{epochs}")
                self.log(f"预计剩余时间: {time.strftime('%H:%M:%S', time.gmtime(remaining_time))}")
                self.log(f"{'=' * 100}")
                
                # 优化tqdm配置，减少更新频率
                with tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', ncols=100, mininterval=1.0, maxinterval=2.0, leave=True) as pbar:
                    for batch_idx, (inputs, targets) in enumerate(pbar):
                        try:
                            # 数据移至设备
                            inputs, targets = inputs.to(self.device), targets.to(self.device)
                            
                            # 梯度清零
                            self.optimizer.zero_grad()
                            
                            # 使用混合精度训练
                            if self.scaler:
                                with torch.cuda.amp.autocast():
                                    # 前向传播
                                    outputs = self.model(inputs)
                                    loss = self.criterion(outputs, targets)
                                
                                # 反向传播
                                self.scaler.scale(loss).backward()
                                
                                # 添加梯度裁剪，防止梯度爆炸
                                self.scaler.unscale_(self.optimizer)
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                                
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
                                # 标准训练
                                # 前向传播
                                outputs = self.model(inputs)
                                loss = self.criterion(outputs, targets)
                                
                                # 反向传播
                                loss.backward()
                                
                                # 添加梯度裁剪，防止梯度爆炸
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                                
                                self.optimizer.step()
                            
                            # 更新学习率
                            self.scheduler.step(epoch + batch_idx / len(train_loader))
                            
                            # 计算损失和准确率
                            train_loss += loss.item()
                            _, predicted = outputs.max(1)
                            train_total += targets.size(0)
                            train_correct += predicted.eq(targets).sum().item()
                            
                            # 更新进度条，每10个批次更新一次，减少终端输出频率
                            if batch_idx % 10 == 0:
                                current_loss = train_loss / (batch_idx + 1)
                                current_accuracy = 100. * train_correct / train_total
                                current_lr = self.optimizer.param_groups[0]["lr"]
                                
                                if gpu_available:
                                    try:
                                        # 获取GPU使用情况
                                        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                                        gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                                        
                                        pbar.set_postfix({
                                            '损失': f'{current_loss:.3f}',
                                            '准确率': f'{current_accuracy:.2f}%',
                                            'GPU': f'{gpu_utilization}%',
                                            '内存': f'{memory_info.used / 1024**2:.0f}MB',
                                            '学习率': f'{current_lr:.6f}'
                                        }, refresh=False)
                                    except Exception as e:
                                        pbar.set_postfix({
                                            '损失': f'{current_loss:.3f}',
                                            '准确率': f'{current_accuracy:.2f}%',
                                            '学习率': f'{current_lr:.6f}'
                                        }, refresh=False)
                                else:
                                    pbar.set_postfix({
                                        '损失': f'{current_loss:.3f}',
                                        '准确率': f'{current_accuracy:.2f}%',
                                        '学习率': f'{current_lr:.6f}'
                                    }, refresh=False)
                            
                            # 释放批处理相关的内存
                            del inputs, targets, outputs, loss
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                                torch.cuda.empty_cache()
                        except Exception as e:
                            self.log(f"Error processing batch {batch_idx}: {e}")
                            continue
            
                # 验证阶段
                val_loss, val_accuracy = self.evaluate(val_loader)
                
                # 自定义数据验证
                custom_val_loss = 0.0
                custom_val_accuracy = 0.0
                if custom_val_loader:
                    custom_val_loss, custom_val_accuracy = self.evaluate(custom_val_loader)
                else:
                    # 如果没有自定义验证数据，使用MNIST验证准确率
                    custom_val_accuracy = val_accuracy
                
                # 记录历史
                train_accuracy = 100. * train_correct / train_total
                self.train_history['loss'].append(train_loss / len(train_loader))
                self.train_history['accuracy'].append(train_accuracy)
                self.train_history['val_loss'].append(val_loss)
                self.train_history['val_accuracy'].append(val_accuracy)
                
                if custom_val_loader:
                    self.train_history['custom_val_loss'].append(custom_val_loss)
                    self.train_history['custom_val_accuracy'].append(custom_val_accuracy)
                else:
                    self.train_history['custom_val_loss'].append(val_loss)
                    self.train_history['custom_val_accuracy'].append(val_accuracy)
                
                # 计算 epoch 时间
                epoch_time = time.time() - start_time
                
                # 打印详细信息
                self.log(f"\n{'=' * 100}")
                self.log(f"轮次 {epoch+1}/{epochs} 详细摘要")
                self.log(f"{'=' * 100}")
                self.log(f"执行时间: {epoch_time:.2f}s")
                self.log(f"训练损失: {train_loss/len(train_loader):.4f} | 训练准确率: {train_accuracy:.2f}%")
                self.log(f"验证损失: {val_loss:.4f} | 验证准确率: {val_accuracy:.2f}%")
                if custom_val_loader:
                    self.log(f"自定义验证损失: {custom_val_loss:.4f} | 自定义验证准确率: {custom_val_accuracy:.2f}%")
                self.log(f"当前学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
                if gpu_available:
                    try:
                        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                        self.log(f"GPU利用率: {gpu_utilization}% | GPU内存: {memory_info.used / 1024**2:.0f}MB / {memory_info.total / 1024**2:.0f}MB")
                    except Exception as e:
                        self.log(f"Error getting GPU info: {e}")
                self.log(f"{'=' * 100}")
                
                # 计算综合评分（MNIST准确率占65%，自定义数据集准确率占35%）
                combined_score = val_accuracy * 0.65 + custom_val_accuracy * 0.35
                
                # 保存最佳模型
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_custom_val_accuracy = custom_val_accuracy
                    best_model = self.model.state_dict()
                    
                    # 生成时间戳
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    
                    # 保存最佳模型，文件名包含MNIST准确率、自定义数据集准确率和时间戳
                    best_model_filename = f'best_model_{val_accuracy:.2f}_{custom_val_accuracy:.2f}_{timestamp}.pth'
                    best_model_path = os.path.join(model_save_path, best_model_filename)
                    try:
                        torch.save(best_model, best_model_path)
                        self.log(f'\n最佳模型已保存: {best_model_filename}')
                        self.log(f'MNIST准确率: {val_accuracy:.2f}% | 自定义数据集准确率: {custom_val_accuracy:.2f}%')
                        
                        # 添加到最佳模型列表
                        best_models.append({
                            'path': best_model_path,
                            'score': combined_score,
                            'mnist_acc': val_accuracy,
                            'custom_acc': custom_val_accuracy
                        })
                        
                        # 按综合评分排序
                        best_models.sort(key=lambda x: x['score'], reverse=True)
                        
                        # 保留前3个最佳模型，删除其余的
                        if len(best_models) > 3:
                            models_to_remove = best_models[3:]
                            best_models = best_models[:3]
                            
                            # 删除多余的模型文件
                            for model_info in models_to_remove:
                                try:
                                    if os.path.exists(model_info['path']):
                                        os.remove(model_info['path'])
                                        self.log(f'删除旧模型: {os.path.basename(model_info["path"])}')
                                except Exception as e:
                                    self.log(f"Error removing old model: {e}")
                    except Exception as e:
                        self.log(f"Error saving best model: {e}")
                
                # 检查早停
                if early_stopping(val_accuracy):
                    self.log('\n触发早停!')
                    break
            
            # 清理其他模型文件，只保留3个最佳模型和最终模型
            self.log("\n清理模型文件，只保留3个最佳模型...")
            try:
                # 获取所有模型文件
                all_model_files = glob.glob(os.path.join(model_save_path, '*.pth'))
                
                # 提取最佳模型路径
                best_model_paths = [model['path'] for model in best_models]
                
                # 删除非最佳模型文件
                for model_file in all_model_files:
                    if model_file not in best_model_paths and not model_file.endswith('final_model.pth'):
                        try:
                            os.remove(model_file)
                            self.log(f'删除非最佳模型: {os.path.basename(model_file)}')
                        except Exception as e:
                            self.log(f"Error removing model file: {e}")
            except Exception as e:
                self.log(f"Error cleaning model files: {e}")
            
            # 生成时间戳
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 保存最终模型，文件名包含时间戳
            final_model_filename = f'final_model_{timestamp}.pth'
            final_model_path = os.path.join(model_save_path, final_model_filename)
            try:
                torch.save(self.model.state_dict(), final_model_path)
                self.log(f'\n最终模型已保存: {final_model_path}')
            except Exception as e:
                self.log(f"Error saving final model: {e}")
            
            # 加载最佳模型
            if best_model:
                try:
                    self.model.load_state_dict(best_model)
                except Exception as e:
                    self.log(f"Error loading best model: {e}")
            
            # 清理GPU资源
            if torch.cuda.is_available():
                try:
                    if 'pynvml' in locals():
                        pynvml.nvmlShutdown()
                    torch.cuda.empty_cache()
                except Exception as e:
                    self.log(f"Error cleaning GPU resources: {e}")
            
            # 打印训练总结
            total_training_time = time.time() - training_start_time
            self.log(f"\n{'=' * 100}")
            self.log(f"训练完成!")
            self.log(f"{'=' * 100}")
            self.log(f"总训练时间: {time.strftime('%H:%M:%S', time.gmtime(total_training_time))}")
            self.log(f"最佳验证准确率: {best_val_accuracy:.2f}%")
            if custom_val_loader:
                self.log(f"最佳自定义验证准确率: {best_custom_val_accuracy:.2f}%")
            self.log(f"保存的最佳模型数量: {len(best_models)}")
            self.log(f"{'=' * 100}")
            
            return self.model
        except Exception as e:
            self.log(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
            return self.model
    
    def evaluate(self, data_loader):
        """
        评估模型
        Args:
            data_loader: 数据加载器
        Returns:
            loss: 平均损失
            accuracy: 准确率
        """
        self.model.eval()
        loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                # 数据移至设备
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # 使用混合精度评估
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        # 前向传播
                        outputs = self.model(inputs)
                        batch_loss = self.criterion(outputs, targets)
                else:
                    # 标准评估
                    outputs = self.model(inputs)
                    batch_loss = self.criterion(outputs, targets)
                
                # 计算损失和准确率
                loss += batch_loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # 释放内存
                del inputs, targets, outputs, batch_loss
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
        
        avg_loss = loss / len(data_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def get_train_history(self):
        """
        获取训练历史
        Returns:
            train_history: 训练历史字典
        """
        return self.train_history