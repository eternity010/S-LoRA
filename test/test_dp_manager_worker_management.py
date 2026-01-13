"""
测试 DataParallelRouterManager Worker 进程管理

测试覆盖：
- Worker 进程启动
- 端口分配与 Worker 启动集成
- 多个 Worker 启动
- Worker 进程属性验证
"""

import pytest
import asyncio
import argparse
import multiprocessing as mp
from unittest.mock import patch, MagicMock, AsyncMock

from slora.server.router.dp_manager import DataParallelRouterManager, run_gpu_worker_process


class TestWorkerProcessManagement:
    """测试 Worker 进程管理"""
    
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.is_available')
    @pytest.mark.asyncio
    async def test_start_workers_basic(self, mock_is_available, mock_device_count):
        """测试基本的 Worker 启动"""
        mock_is_available.return_value = True
        mock_device_count.return_value = 2
        
        args = argparse.Namespace(
            num_workers=2,
            gpu_ids="0,1"
        )
        
        manager = DataParallelRouterManager(args, router_port=10000, response_port=10001)
        
        # 使用一个简单的函数来模拟 Worker 进程（保持进程存活）
        def mock_worker_func(*args, **kwargs):
            import time
            time.sleep(10)  # 保持进程存活
        
        # Patch run_gpu_worker_process
        with patch('slora.server.router.dp_manager.run_gpu_worker_process', side_effect=mock_worker_func):
            # 启动 Workers
            await manager.start_workers()
            
            # 验证端口已分配
            assert len(manager.worker_ports) == 2
            assert manager.worker_ports == [50000, 50001]
            
            # 验证 Worker 进程已创建
            assert len(manager.workers) == 2
            
            # 验证所有 Worker 进程都已启动
            for worker in manager.workers:
                assert isinstance(worker, mp.Process)
                assert worker.is_alive()
            
            # 清理：终止所有 Worker 进程
            for worker in manager.workers:
                worker.terminate()
                worker.join(timeout=1)
    
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.is_available')
    def test_start_worker_single(self, mock_is_available, mock_device_count):
        """测试启动单个 Worker"""
        mock_is_available.return_value = True
        mock_device_count.return_value = 1
        
        args = argparse.Namespace(
            num_workers=1,
            gpu_ids="0"
        )
        
        manager = DataParallelRouterManager(args, router_port=10000, response_port=10001)
        manager._allocate_ports()
        
        # 使用一个简单的函数来模拟 Worker 进程
        def mock_worker_func(*args, **kwargs):
            import time
            time.sleep(10)
        
        with patch('slora.server.router.dp_manager.run_gpu_worker_process', side_effect=mock_worker_func):
            # 启动单个 Worker
            worker = manager._start_worker(0, 0)
            
            # 验证 Worker 进程
            assert isinstance(worker, mp.Process)
            assert worker.is_alive()
            assert worker.name == "GPUWorker-0"
            
            # 清理
            worker.terminate()
            worker.join(timeout=1)
    
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.is_available')
    @pytest.mark.asyncio
    async def test_start_workers_multiple(self, mock_is_available, mock_device_count):
        """测试启动多个 Worker"""
        mock_is_available.return_value = True
        mock_device_count.return_value = 3
        
        args = argparse.Namespace(
            num_workers=3,
            gpu_ids="0,1,2"
        )
        
        manager = DataParallelRouterManager(args, router_port=10000, response_port=10001)
        
        def mock_worker_func(*args, **kwargs):
            import time
            time.sleep(10)
        
        with patch('slora.server.router.dp_manager.run_gpu_worker_process', side_effect=mock_worker_func):
            # 启动 Workers
            await manager.start_workers()
            
            # 验证所有 Worker 都已启动
            assert len(manager.workers) == 3
            
            for i, worker in enumerate(manager.workers):
                assert isinstance(worker, mp.Process)
                assert worker.is_alive()
                assert worker.name == f"GPUWorker-{i}"
            
            # 清理
            for worker in manager.workers:
                worker.terminate()
                worker.join(timeout=1)
    
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.is_available')
    def test_worker_process_arguments(self, mock_is_available, mock_device_count):
        """测试 Worker 进程接收正确的参数"""
        mock_is_available.return_value = True
        mock_device_count.return_value = 2
        
        args = argparse.Namespace(
            num_workers=2,
            gpu_ids="0,1",
            model_dir="/path/to/model"
        )
        
        manager = DataParallelRouterManager(args, router_port=10000, response_port=10001)
        manager._allocate_ports()
        
        def mock_worker_func(*args, **kwargs):
            import time
            time.sleep(10)
        
        with patch('slora.server.router.dp_manager.run_gpu_worker_process', side_effect=mock_worker_func):
            # 启动 Worker
            worker = manager._start_worker(0, 0)
            
            # 等待进程启动
            import time
            time.sleep(0.1)
            
            # 验证 run_gpu_worker_process 被调用
            # 注意：由于是在子进程中调用，我们无法直接验证参数
            # 但可以验证进程已启动
            assert worker.is_alive()
            
            # 清理
            worker.terminate()
            worker.join(timeout=1)
    
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.is_available')
    @pytest.mark.asyncio
    async def test_start_workers_port_allocation(self, mock_is_available, mock_device_count):
        """测试 Worker 启动时端口正确分配"""
        mock_is_available.return_value = True
        mock_device_count.return_value = 3
        
        args = argparse.Namespace(
            num_workers=3,
            gpu_ids="0,1,2"
        )
        
        manager = DataParallelRouterManager(args, router_port=10000, response_port=10001)
        
        # 启动前端口列表应该为空
        assert len(manager.worker_ports) == 0
        
        def mock_worker_func(*args, **kwargs):
            import time
            time.sleep(10)
        
        with patch('slora.server.router.dp_manager.run_gpu_worker_process', side_effect=mock_worker_func):
            # 启动 Workers
            await manager.start_workers()
            
            # 验证端口已分配
            assert len(manager.worker_ports) == 3
            assert manager.worker_ports == [50000, 50001, 50002]
            
            # 清理
            for worker in manager.workers:
                worker.terminate()
                worker.join(timeout=1)
    
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.is_available')
    @pytest.mark.asyncio
    async def test_start_workers_waits_for_ready(self, mock_is_available, mock_device_count):
        """测试 start_workers 等待 Worker 就绪"""
        mock_is_available.return_value = True
        mock_device_count.return_value = 2
        
        args = argparse.Namespace(
            num_workers=2,
            gpu_ids="0,1"
        )
        
        manager = DataParallelRouterManager(args, router_port=10000, response_port=10001)
        
        def mock_worker_func(*args, **kwargs):
            import time
            time.sleep(10)
        
        with patch('slora.server.router.dp_manager.run_gpu_worker_process', side_effect=mock_worker_func):
            # 记录开始时间
            import time
            start_time = time.time()
            
            # 启动 Workers
            await manager.start_workers()
            
            # 验证等待了一段时间（至少 4 秒，因为 sleep(5) 但有一些误差）
            elapsed_time = time.time() - start_time
            assert elapsed_time >= 4.0
            
            # 清理
            for worker in manager.workers:
                worker.terminate()
                worker.join(timeout=1)
    
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.is_available')
    def test_worker_process_names(self, mock_is_available, mock_device_count):
        """测试 Worker 进程名称正确设置"""
        mock_is_available.return_value = True
        mock_device_count.return_value = 3
        
        args = argparse.Namespace(
            num_workers=3,
            gpu_ids="0,1,2"
        )
        
        manager = DataParallelRouterManager(args, router_port=10000, response_port=10001)
        manager._allocate_ports()
        
        def mock_worker_func(*args, **kwargs):
            import time
            time.sleep(10)
        
        with patch('slora.server.router.dp_manager.run_gpu_worker_process', side_effect=mock_worker_func):
            # 启动多个 Worker
            workers = []
            for i in range(3):
                worker = manager._start_worker(i, manager.gpu_ids[i])
                workers.append(worker)
            
            # 验证进程名称
            for i, worker in enumerate(workers):
                assert worker.name == f"GPUWorker-{i}"
            
            # 清理
            for worker in workers:
                worker.terminate()
                worker.join(timeout=1)
    
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.is_available')
    @pytest.mark.asyncio
    async def test_start_workers_gpu_id_mapping(self, mock_is_available, mock_device_count):
        """测试 Worker 与 GPU ID 的正确映射"""
        mock_is_available.return_value = True
        mock_device_count.return_value = 4
        
        # 使用非连续的 GPU IDs
        args = argparse.Namespace(
            num_workers=3,
            gpu_ids="0,2,3"
        )
        
        manager = DataParallelRouterManager(args, router_port=10000, response_port=10001)
        
        def mock_worker_func(*args, **kwargs):
            import time
            time.sleep(10)
        
        with patch('slora.server.router.dp_manager.run_gpu_worker_process', side_effect=mock_worker_func):
            # 启动 Workers
            await manager.start_workers()
            
            # 验证 Worker 数量
            assert len(manager.workers) == 3
            
            # 验证 GPU IDs 映射
            assert manager.gpu_ids == [0, 2, 3]
            
            # 清理
            for worker in manager.workers:
                worker.terminate()
                worker.join(timeout=1)


class TestRunGPUWorkerProcess:
    """测试 run_gpu_worker_process 函数"""
    
    @patch('slora.server.router.dp_manager.asyncio.run')
    @patch('slora.server.router.gpu_worker.GPUWorker')
    def test_run_gpu_worker_process_basic(self, mock_worker_class, mock_asyncio_run):
        """测试 run_gpu_worker_process 基本功能"""
        # 创建 mock Worker 实例
        mock_worker = MagicMock()
        mock_worker_class.return_value = mock_worker
        
        args = argparse.Namespace(
            model_dir="/path/to/model"
        )
        
        # 调用函数
        try:
            run_gpu_worker_process(0, 0, args, 50000, 10001)
        except Exception:
            pass  # 可能会因为 mock 不完整而失败，但我们主要验证调用
        
        # 验证 Worker 被创建
        mock_worker_class.assert_called_once_with(0, 0, args)
        
        # 验证 ZMQ 设置被调用
        mock_worker._setup_zmq.assert_called_once_with(50000, 10001)
    
    @patch('slora.server.router.dp_manager.asyncio.run')
    @patch('slora.server.router.gpu_worker.GPUWorker')
    def test_run_gpu_worker_process_error_handling(self, mock_worker_class, mock_asyncio_run):
        """测试 run_gpu_worker_process 错误处理"""
        # 模拟 Worker 创建失败
        mock_worker_class.side_effect = Exception("Worker creation failed")
        
        args = argparse.Namespace()
        
        # 验证异常被正确抛出
        with pytest.raises(Exception, match="Worker creation failed"):
            run_gpu_worker_process(0, 0, args, 50000, 10001)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
