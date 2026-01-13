"""
GPU Worker 模型加载测试

测试 GPUWorker 类的模型加载功能。
"""

import os
import sys
import pytest
import argparse
from unittest.mock import patch, MagicMock, call

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from slora.server.router.gpu_worker import GPUWorker


class TestGPUWorkerModelLoading:
    """测试 GPUWorker 模型加载"""
    
    def test_load_llama_model(self):
        """测试加载 Llama 模型"""
        args = argparse.Namespace(
            model_dir="/fake/llama/path",
            max_total_token_num=1000,
            mem_adapter_size=100,
            load_way='HF',
            mode=[],
            dummy=False
        )
        
        # Mock 模型配置（Llama 模型，无 num_key_value_heads）
        mock_model_cfg = {
            "model_type": "llama",
            "num_attention_heads": 32,
            "hidden_size": 4096,
            "num_hidden_layers": 32
        }
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('slora.server.router.gpu_worker.get_model_config', return_value=mock_model_cfg), \
             patch('slora.server.router.gpu_worker.LlamaTpPartModel') as mock_llama_model:
            
            # 创建 Worker 并加载模型
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            worker._load_model()
            
            # 验证 LlamaTpPartModel 被调用
            mock_llama_model.assert_called_once()
            
            # 验证调用参数
            call_args = mock_llama_model.call_args
            assert call_args[1]['tp_rank'] == 0
            assert call_args[1]['world_size'] == 1
            assert call_args[1]['weight_dir'] == "/fake/llama/path"
            assert call_args[1]['max_total_token_num'] == 1000
            assert call_args[1]['mem_adapter_size'] == 100
            
            # 验证模型被存储
            assert worker.model is not None
    
    def test_load_llama2_model(self):
        """测试加载 Llama2 模型（带 GQA）"""
        args = argparse.Namespace(
            model_dir="/fake/llama2/path",
            max_total_token_num=2000,
            mem_adapter_size=200,
            load_way='HF',
            mode=[],
            dummy=False
        )
        
        # Mock 模型配置（Llama2 模型，有 num_key_value_heads）
        mock_model_cfg = {
            "model_type": "llama",
            "num_attention_heads": 32,
            "num_key_value_heads": 8,  # GQA 特征
            "hidden_size": 4096,
            "num_hidden_layers": 32
        }
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('slora.server.router.gpu_worker.get_model_config', return_value=mock_model_cfg), \
             patch('slora.server.router.gpu_worker.Llama2TpPartModel') as mock_llama2_model:
            
            # 创建 Worker 并加载模型
            worker = GPUWorker(worker_id=1, gpu_id=1, args=args)
            worker._load_model()
            
            # 验证 Llama2TpPartModel 被调用
            mock_llama2_model.assert_called_once()
            
            # 验证调用参数
            call_args = mock_llama2_model.call_args
            assert call_args[1]['tp_rank'] == 0
            assert call_args[1]['world_size'] == 1
            assert call_args[1]['weight_dir'] == "/fake/llama2/path"
            assert call_args[1]['max_total_token_num'] == 2000
            
            # 验证模型被存储
            assert worker.model is not None
    
    def test_unsupported_model_type_error(self):
        """测试不支持的模型类型抛出异常"""
        args = argparse.Namespace(
            model_dir="/fake/unsupported/path",
            max_total_token_num=1000,
            mem_adapter_size=100,
            load_way='HF',
            mode=[],
            dummy=False
        )
        
        # Mock 不支持的模型配置
        mock_model_cfg = {
            "model_type": "gpt2",  # 不支持的类型
            "num_attention_heads": 12,
            "hidden_size": 768
        }
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('slora.server.router.gpu_worker.get_model_config', return_value=mock_model_cfg):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            # 验证抛出异常
            with pytest.raises(Exception, match="Unsupported model type"):
                worker._load_model()
    
    def test_model_loading_with_optional_args(self):
        """测试使用可选参数加载模型"""
        # 最小参数集
        args = argparse.Namespace(
            model_dir="/fake/path",
            max_total_token_num=1000
        )
        
        mock_model_cfg = {
            "model_type": "llama",
            "num_attention_heads": 32,
            "hidden_size": 4096
        }
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('slora.server.router.gpu_worker.get_model_config', return_value=mock_model_cfg), \
             patch('slora.server.router.gpu_worker.LlamaTpPartModel') as mock_llama_model:
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            worker._load_model()
            
            # 验证使用默认值
            call_args = mock_llama_model.call_args
            assert call_args[1]['mem_adapter_size'] == 0  # 默认值
            assert call_args[1]['load_way'] == 'HF'  # 默认值
            assert call_args[1]['mode'] == []  # 默认值
            assert call_args[1]['dummy'] == False  # 默认值
    
    def test_model_loading_failure_handling(self):
        """测试模型加载失败的错误处理"""
        args = argparse.Namespace(
            model_dir="/fake/path",
            max_total_token_num=1000,
            dummy=False
        )
        
        mock_model_cfg = {
            "model_type": "llama",
            "num_attention_heads": 32
        }
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('slora.server.router.gpu_worker.get_model_config', return_value=mock_model_cfg), \
             patch('slora.server.router.gpu_worker.LlamaTpPartModel', side_effect=RuntimeError("OOM")):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            # 验证异常被正确传播
            with pytest.raises(RuntimeError, match="OOM"):
                worker._load_model()
    
    def test_data_parallel_mode_parameters(self):
        """测试数据并行模式的参数设置"""
        args = argparse.Namespace(
            model_dir="/fake/path",
            max_total_token_num=1000,
            mem_adapter_size=100,
            load_way='HF',
            mode=[],
            dummy=False
        )
        
        mock_model_cfg = {
            "model_type": "llama",
            "num_attention_heads": 32,
            "hidden_size": 4096
        }
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('slora.server.router.gpu_worker.get_model_config', return_value=mock_model_cfg), \
             patch('slora.server.router.gpu_worker.LlamaTpPartModel') as mock_llama_model:
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            worker._load_model()
            
            # 验证数据并行模式的关键参数
            call_args = mock_llama_model.call_args
            # tp_rank=0, world_size=1 表示单 GPU 模式（无张量并行）
            assert call_args[1]['tp_rank'] == 0, "数据并行模式应该使用 tp_rank=0"
            assert call_args[1]['world_size'] == 1, "数据并行模式应该使用 world_size=1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
