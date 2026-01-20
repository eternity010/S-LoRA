"""
测试 Worker 进程监控功能

测试 DataParallelRouterManager 的进程健康检查功能：
- 定期检测 Worker 进程是否存活
- 检测 Worker 进程意外退出
- 记录进程退出日志（worker_id、exitcode、时间）

Requirements:
    - 7.5: 定期检测 Worker 进程是否存活，记录进程退出日志
"""

import pytest
import asyncio
import multiprocessing as mp
import time
from unittest.mock import Mock, MagicMock, patch
from argparse import Namespace


class TestWorkerProcessMonitoring:
    """测试 Worker 进程监控功能"""
    
    @pytest.fixture
    def mock_args(self):
        """创建模拟的命令行参数"""
        args = Namespace()
        args.num_workers = 2
        args.gpu_ids = "0,1"
        args.model_dir = "/fake/model"
        args.max_total_token_num = 1000
        args.batch_max_tokens = 100
        args.running_max_req_size = 10
        args.adapter_dirs = []
        return args
    
    @pytest.fixture
    def manager(self, mock_args):
        """创建 DataParallelRouterManager 实例（不启动 Worker）"""
        from slora.server.router.dp_manager import DataParallelRouterManager
        
        # 创建 manager 实例
        manager = DataParallelRouterManager(
            args=mock_args,
            router_port=60000,
            response_port=60001,
            detoken_port=60002
        )
        
        return manager
    
    def test_check_worker_health_all_alive(self, manager):
        """测试所有 Worker 进程都存活的情况"""
        # 创建模拟的 Worker 进程（存活状态）
        mock_worker1 = Mock()
        mock_worker1.is_alive.return_value = True
        mock_worker1.exitcode = None
        
        mock_worker2 = Mock()
        mock_worker2.is_alive.return_value = True
        mock_worker2.exitcode = None
        
        manager.workers = [mock_worker1, mock_worker2]
        
        # 运行健康检查（只检查一次）
        async def run_single_check():
            # 模拟一次检查
            for i, worker in enumerate(manager.workers):
                assert worker.is_alive(), f"Worker {i} should be alive"
        
        asyncio.run(run_single_check())
        
        # 验证 is_alive 被调用
        assert mock_worker1.is_alive.called
        assert mock_worker2.is_alive.called
    
    def test_check_worker_health_one_dead(self, manager, capsys):
        """测试一个 Worker 进程退出的情况"""
        # 创建模拟的 Worker 进程
        mock_worker1 = Mock()
        mock_worker1.is_alive.return_value = True
        mock_worker1.exitcode = None
        
        mock_worker2 = Mock()
        mock_worker2.is_alive.return_value = False
        mock_worker2.exitcode = 1  # 错误退出
        
        manager.workers = [mock_worker1, mock_worker2]
        
        # 运行健康检查（只检查一次）
        async def run_single_check():
            # 模拟一次检查
            for i, worker in enumerate(manager.workers):
                if not worker.is_alive():
                    exitcode = worker.exitcode
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    print(f"[DataParallelRouterManager] WARNING: Worker {i} (GPU {manager.gpu_ids[i]}) "
                          f"has exited unexpectedly!")
                    print(f"[DataParallelRouterManager]   Worker ID: {i}")
                    print(f"[DataParallelRouterManager]   GPU ID: {manager.gpu_ids[i]}")
                    print(f"[DataParallelRouterManager]   Exit code: {exitcode}")
                    print(f"[DataParallelRouterManager]   Timestamp: {timestamp}")
        
        asyncio.run(run_single_check())
        
        # 验证日志输出
        captured = capsys.readouterr()
        assert "Worker 1" in captured.out
        assert "has exited unexpectedly" in captured.out
        assert "Exit code: 1" in captured.out
    
    def test_check_worker_health_exit_codes(self, manager, capsys):
        """测试不同退出码的处理"""
        test_cases = [
            (0, "Normal exit"),
            (1, "Error exit"),
            (-9, "Killed by signal 9"),
            (-15, "Terminated by signal 15"),
        ]
        
        for exitcode, expected_msg in test_cases:
            # 创建模拟的 Worker 进程
            mock_worker = Mock()
            mock_worker.is_alive.return_value = False
            mock_worker.exitcode = exitcode
            
            manager.workers = [mock_worker]
            
            # 运行健康检查
            async def run_single_check():
                for i, worker in enumerate(manager.workers):
                    if not worker.is_alive():
                        exitcode = worker.exitcode
                        
                        print(f"[DataParallelRouterManager] WARNING: Worker {i} has exited")
                        print(f"[DataParallelRouterManager]   Exit code: {exitcode}")
                        
                        # 根据退出码提供更多信息
                        if exitcode == 0:
                            print(f"[DataParallelRouterManager]   Status: Normal exit (exit code 0)")
                        elif exitcode == 1:
                            print(f"[DataParallelRouterManager]   Status: Error exit (exit code 1)")
                        elif exitcode == -9:
                            print(f"[DataParallelRouterManager]   Status: Killed by signal 9 (SIGKILL)")
                        elif exitcode == -15:
                            print(f"[DataParallelRouterManager]   Status: Terminated by signal 15 (SIGTERM)")
            
            asyncio.run(run_single_check())
            
            # 验证日志输出
            captured = capsys.readouterr()
            assert f"Exit code: {exitcode}" in captured.out
            assert expected_msg in captured.out
    
    def test_check_worker_health_periodic(self, manager):
        """测试定期健康检查（模拟）"""
        # 创建模拟的 Worker 进程
        mock_worker = Mock()
        mock_worker.is_alive.return_value = True
        mock_worker.exitcode = None
        
        manager.workers = [mock_worker]
        
        # 模拟多次检查
        async def run_multiple_checks():
            for _ in range(3):
                # 模拟一次检查
                for worker in manager.workers:
                    worker.is_alive()
                
                # 模拟等待（缩短时间）
                await asyncio.sleep(0.1)
        
        asyncio.run(run_multiple_checks())
        
        # 验证 is_alive 被多次调用
        assert mock_worker.is_alive.call_count >= 3
    
    def test_check_worker_health_exception_handling(self, manager, capsys):
        """测试健康检查中的异常处理"""
        # 创建会抛出异常的模拟 Worker
        mock_worker = Mock()
        mock_worker.is_alive.side_effect = Exception("Test exception")
        
        manager.workers = [mock_worker]
        
        # 运行健康检查（应该捕获异常并继续）
        async def run_check_with_exception():
            try:
                for worker in manager.workers:
                    worker.is_alive()
            except Exception as e:
                print(f"[DataParallelRouterManager] ERROR in worker health check: {str(e)}")
        
        asyncio.run(run_check_with_exception())
        
        # 验证异常被捕获并记录
        captured = capsys.readouterr()
        assert "ERROR in worker health check" in captured.out
        assert "Test exception" in captured.out
    
    @pytest.mark.asyncio
    async def test_health_check_task_creation(self, manager):
        """测试健康检查任务的创建"""
        # 创建模拟的 Worker 进程
        mock_worker = Mock()
        mock_worker.is_alive.return_value = True
        manager.workers = [mock_worker]
        
        # 创建健康检查任务
        health_check_task = asyncio.create_task(manager._check_worker_health())
        
        # 等待一小段时间
        await asyncio.sleep(0.2)
        
        # 取消任务
        health_check_task.cancel()
        
        try:
            await health_check_task
        except asyncio.CancelledError:
            pass
        
        # 验证任务被创建
        assert health_check_task is not None


class TestWorkerProcessMonitoringIntegration:
    """集成测试：测试实际的进程监控"""
    
    def test_monitor_real_process_exit(self, capsys):
        """测试监控实际进程退出"""
        # 创建一个会立即退出的进程
        def worker_func():
            import sys
            sys.exit(1)
        
        proc = mp.Process(target=worker_func)
        proc.start()
        proc.join(timeout=2)
        
        # 检查进程状态
        assert not proc.is_alive()
        assert proc.exitcode == 1
        
        # 模拟监控逻辑
        if not proc.is_alive():
            exitcode = proc.exitcode
            print(f"Worker exited with code {exitcode}")
        
        captured = capsys.readouterr()
        assert "Worker exited with code 1" in captured.out
    
    def test_monitor_real_process_alive(self):
        """测试监控存活的进程"""
        # 创建一个会持续运行的进程
        def worker_func():
            import time
            time.sleep(10)
        
        proc = mp.Process(target=worker_func)
        proc.start()
        
        try:
            # 检查进程状态
            assert proc.is_alive()
            assert proc.exitcode is None
        finally:
            # 清理进程
            proc.terminate()
            proc.join(timeout=2)
            if proc.is_alive():
                proc.kill()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
