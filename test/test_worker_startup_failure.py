"""
测试 Worker 启动失败处理

验证当 Worker 启动失败时，系统能够正确检测并处理错误。
"""

import pytest
import asyncio
import argparse
import multiprocessing as mp
from unittest.mock import Mock, patch, MagicMock
import sys


def test_run_gpu_worker_process_exception_handling():
    """
    测试 run_gpu_worker_process 的异常处理
    
    验证：
    1. 捕获所有异常
    2. 记录详细错误日志
    3. 通过 sys.exit(1) 返回非零退出码
    
    Requirements: 7.1, 7.2
    """
    from slora.server.router.dp_manager import run_gpu_worker_process
    
    # 创建模拟的 args
    args = argparse.Namespace(
        model_dir='/fake/model',
        max_total_token_num=1000,
        batch_max_tokens=500,
        running_max_req_size=10,
        lora_dirs=None,
        no_lora=True
    )
    
    # 模拟 GPUWorker 初始化失败（需要 patch 在函数内部导入的位置）
    with patch('slora.server.router.gpu_worker.GPUWorker') as mock_worker_class:
        mock_worker_class.side_effect = RuntimeError("GPU initialization failed")
        
        # 模拟 sys.exit
        with patch('sys.exit') as mock_exit:
            # 调用函数
            run_gpu_worker_process(0, 0, args, 50000, 50001)
            
            # 验证 sys.exit(1) 被调用
            mock_exit.assert_called_once_with(1)


def test_run_gpu_worker_process_keyboard_interrupt():
    """
    测试 run_gpu_worker_process 处理 KeyboardInterrupt
    
    验证：
    1. 捕获 KeyboardInterrupt
    2. 优雅退出（exit code 0）
    
    Requirements: 7.1
    """
    from slora.server.router.dp_manager import run_gpu_worker_process
    
    args = argparse.Namespace(
        model_dir='/fake/model',
        max_total_token_num=1000,
        batch_max_tokens=500,
        running_max_req_size=10,
        lora_dirs=None,
        no_lora=True
    )
    
    # 模拟 GPUWorker 初始化时抛出 KeyboardInterrupt
    with patch('slora.server.router.gpu_worker.GPUWorker') as mock_worker_class:
        mock_worker_class.side_effect = KeyboardInterrupt()
        
        # 模拟 sys.exit
        with patch('sys.exit') as mock_exit:
            # 调用函数
            run_gpu_worker_process(0, 0, args, 50000, 50001)
            
            # 验证 sys.exit(0) 被调用
            mock_exit.assert_called_once_with(0)


@pytest.mark.asyncio
async def test_start_workers_detects_failure():
    """
    测试 start_workers 检测 Worker 启动失败
    
    验证：
    1. 检测 Worker 进程退出
    2. 终止所有 Worker 进程
    3. 终止 Response Merger 进程
    4. 抛出 RuntimeError
    
    Requirements: 7.2
    """
    from slora.server.router.dp_manager import DataParallelRouterManager
    
    # 创建模拟的 args
    args = argparse.Namespace(
        model_dir='/fake/model',
        max_total_token_num=1000,
        batch_max_tokens=500,
        running_max_req_size=10,
        num_workers=2,
        gpu_ids='0,1',
        lora_dirs=None
    )
    
    # 创建 Router Manager
    manager = DataParallelRouterManager(args, 40000, 40001, 40002)
    
    # 模拟 Worker 进程（一个失败，一个成功）
    mock_failed_worker = Mock()
    mock_failed_worker.is_alive.return_value = False
    mock_failed_worker.exitcode = 1
    
    mock_alive_worker = Mock()
    mock_alive_worker.is_alive.return_value = True
    mock_alive_worker.terminate = Mock()
    mock_alive_worker.join = Mock()
    mock_alive_worker.kill = Mock()
    
    # 模拟 Response Merger 进程
    mock_merger = Mock()
    mock_merger.is_alive.return_value = True
    mock_merger.terminate = Mock()
    mock_merger.join = Mock()
    mock_merger.kill = Mock()
    
    # 替换 _start_worker 和 _start_response_merger
    with patch.object(manager, '_start_worker') as mock_start_worker:
        with patch.object(manager, '_start_response_merger') as mock_start_merger:
            # 第一个 Worker 失败，第二个 Worker 成功
            mock_start_worker.side_effect = [mock_failed_worker, mock_alive_worker]
            mock_start_merger.return_value = None
            manager.merger_process = mock_merger
            
            # 调用 start_workers，应该抛出 RuntimeError
            with pytest.raises(RuntimeError) as exc_info:
                await manager.start_workers()
            
            # 验证错误消息
            assert "Worker startup failed" in str(exc_info.value)
            
            # 验证存活的 Worker 被终止
            mock_alive_worker.terminate.assert_called_once()
            
            # 验证 Response Merger 被终止
            mock_merger.terminate.assert_called_once()


@pytest.mark.asyncio
async def test_start_workers_all_success():
    """
    测试 start_workers 所有 Worker 启动成功
    
    验证：
    1. 所有 Worker 进程正常启动
    2. 不抛出异常
    3. 正常返回
    
    Requirements: 1.1, 1.5
    """
    from slora.server.router.dp_manager import DataParallelRouterManager
    
    # 创建模拟的 args
    args = argparse.Namespace(
        model_dir='/fake/model',
        max_total_token_num=1000,
        batch_max_tokens=500,
        running_max_req_size=10,
        num_workers=2,
        gpu_ids='0,1',
        lora_dirs=None
    )
    
    # 创建 Router Manager
    manager = DataParallelRouterManager(args, 40000, 40001, 40002)
    
    # 模拟所有 Worker 进程都成功
    mock_worker1 = Mock()
    mock_worker1.is_alive.return_value = True
    
    mock_worker2 = Mock()
    mock_worker2.is_alive.return_value = True
    
    # 模拟 Response Merger 进程
    mock_merger = Mock()
    mock_merger.is_alive.return_value = True
    
    # 替换 _start_worker 和 _start_response_merger
    with patch.object(manager, '_start_worker') as mock_start_worker:
        with patch.object(manager, '_start_response_merger') as mock_start_merger:
            mock_start_worker.side_effect = [mock_worker1, mock_worker2]
            mock_start_merger.return_value = None
            manager.merger_process = mock_merger
            
            # 调用 start_workers，应该正常返回
            await manager.start_workers()
            
            # 验证所有 Worker 都被启动
            assert len(manager.workers) == 2
            assert mock_start_worker.call_count == 2


@pytest.mark.asyncio
async def test_start_workers_early_failure_detection():
    """
    测试 start_workers 早期检测 Worker 失败
    
    验证：
    1. 在等待期间检测到 Worker 失败
    2. 立即终止所有进程
    3. 不等待完整的 5 秒
    
    Requirements: 7.2
    """
    from slora.server.router.dp_manager import DataParallelRouterManager
    
    # 创建模拟的 args
    args = argparse.Namespace(
        model_dir='/fake/model',
        max_total_token_num=1000,
        batch_max_tokens=500,
        running_max_req_size=10,
        num_workers=1,
        gpu_ids='0',
        lora_dirs=None
    )
    
    # 创建 Router Manager
    manager = DataParallelRouterManager(args, 40000, 40001, 40002)
    
    # 模拟 Worker 进程（第一次检查时存活，第二次检查时失败）
    mock_worker = Mock()
    check_count = [0]
    
    def is_alive_side_effect():
        check_count[0] += 1
        # 第一次检查返回 True，第二次检查返回 False
        return check_count[0] == 1
    
    mock_worker.is_alive.side_effect = is_alive_side_effect
    mock_worker.exitcode = 1
    mock_worker.terminate = Mock()
    mock_worker.join = Mock()
    
    # 模拟 Response Merger 进程
    mock_merger = Mock()
    mock_merger.is_alive.return_value = True
    mock_merger.terminate = Mock()
    mock_merger.join = Mock()
    
    # 替换 _start_worker 和 _start_response_merger
    with patch.object(manager, '_start_worker') as mock_start_worker:
        with patch.object(manager, '_start_response_merger') as mock_start_merger:
            mock_start_worker.return_value = mock_worker
            mock_start_merger.return_value = None
            manager.merger_process = mock_merger
            
            # 调用 start_workers，应该在第二次检查时抛出 RuntimeError
            with pytest.raises(RuntimeError) as exc_info:
                await manager.start_workers()
            
            # 验证错误消息
            assert "Worker startup failed" in str(exc_info.value)
            
            # 验证 is_alive 被调用了至少 2 次（第一次返回 True，第二次返回 False）
            assert mock_worker.is_alive.call_count >= 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
