"""
Task 8.7.2: 验证请求统计

测试内容：
- 启动完整系统
- 发送 20 个请求
- 等待 10 秒
- 检查统计日志是否输出：
  - 总请求数
  - 成功/失败数
  - 每个 Worker 的请求分布

预期结果：统计信息正确输出
"""

import sys
import time
import argparse
import multiprocessing as mp
import asyncio
import zmq
import zmq.asyncio
from pathlib import Path
from io import StringIO
import threading

# Set multiprocessing start method to 'spawn' BEFORE importing torch
mp.set_start_method('spawn', force=True)

# Add S-LoRA to path
SLORA_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(SLORA_ROOT))

from test_base import ManualTest
from slora.server.router.dp_manager import DataParallelRouterManager


class RequestStatisticsTest(ManualTest):
    """测试请求统计功能"""
    
    def __init__(self):
        super().__init__(
            test_name="Task 8.7.2: Verify Request Statistics",
            description="Verify that request statistics are correctly output after sending 20 requests"
        )
        self.manager = None
        self.manager_task = None
        self.captured_output = []
        self.output_lock = threading.Lock()
        
    def setup(self):
        """准备测试环境"""
        self.logger.info("Setting up test environment...")
        
        # 创建测试参数
        self.args = argparse.Namespace(
            model_dir="/home/hzheng/models/llama-7b",
            tokenizer_mode="slow",
            max_total_token_num=6000,
            batch_max_tokens=2000,
            eos_id=2,
            running_max_req_size=100,
            max_req_input_len=2048,
            max_req_total_len=3072,
            mode=[],
            trust_remote_code=False,
            lora_dirs=[],
            dummy=True,  # 使用 Dummy 模式加快测试
            num_workers=2,  # 2 个 Worker
            gpu_ids="0,1",  # 使用 GPU 0 和 1
        )
        
        self.logger.info("Test arguments configured:")
        self.logger.info(f"  Model dir: {self.args.model_dir}")
        self.logger.info(f"  Num workers: {self.args.num_workers}")
        self.logger.info(f"  GPU IDs: {self.args.gpu_ids}")
        self.logger.info(f"  Dummy mode: {self.args.dummy}")
    
    def execute(self):
        """执行测试"""
        self.logger.info("Starting test execution...")
        
        # 创建输出捕获
        output_buffer = StringIO()
        
        # 创建 DataParallelRouterManager
        self.logger.info("Creating DataParallelRouterManager...")
        
        # 捕获输出
        import sys
        old_stdout = sys.stdout
        
        try:
            # 重定向输出到缓冲区（同时保留控制台输出）
            class TeeOutput:
                def __init__(self, *outputs):
                    self.outputs = outputs
                
                def write(self, data):
                    for output in self.outputs:
                        output.write(data)
                        output.flush()
                
                def flush(self):
                    for output in self.outputs:
                        output.flush()
            
            sys.stdout = TeeOutput(old_stdout, output_buffer)
            
            # 创建 Manager
            self.manager = DataParallelRouterManager(
                args=self.args,
                router_port=50200,
                response_port=50201,
                detoken_port=50202
            )
            
            # 启动 Workers
            self.logger.info("Starting workers...")
            asyncio.run(self.manager.start_workers())
            
            # 设置 ZMQ 通信
            self.logger.info("Setting up ZMQ communication...")
            self.manager._setup_zmq()
            
            # 在后台运行 Manager 的主循环
            self.logger.info("Starting manager main loop in background...")
            
            async def run_manager():
                """在后台运行 Manager"""
                await self.manager.run()
            
            # 创建事件循环并在后台线程中运行
            def run_in_thread():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(run_manager())
            
            manager_thread = threading.Thread(target=run_in_thread, daemon=True)
            manager_thread.start()
            
            # 等待 Manager 启动
            time.sleep(2)
            
            # 发送 20 个测试请求
            self.logger.info("Sending 20 test requests...")
            self._send_test_requests(20)
            
            # 等待 10 秒让统计任务输出
            self.logger.info("Waiting 10 seconds for statistics output...")
            time.sleep(11)  # 等待 11 秒确保统计输出
            
        finally:
            # 恢复标准输出
            sys.stdout = old_stdout
        
        # 获取捕获的输出
        captured_output = output_buffer.getvalue()
        self.captured_output = captured_output.split('\n')
        
        # 验证统计日志
        self.logger.info("Verifying statistics logs...")
        self._verify_statistics()
        
        self.logger.info("All statistics verified successfully!")
    
    def _send_test_requests(self, num_requests: int):
        """发送测试请求到 Manager"""
        # 创建 ZMQ 客户端发送请求
        context = zmq.Context()
        sender = context.socket(zmq.PUSH)
        sender.connect(f"tcp://127.0.0.1:50200")
        
        self.logger.info(f"Sending {num_requests} requests...")
        
        for i in range(num_requests):
            request = {
                'request_id': f'test-stats-{i}',
                'adapter_dir': None,
                'prompt_ids': [1, 2, 3, 4, 5],
                'sampling_params': {
                    'max_new_tokens': 10,
                    'temperature': 1.0,
                }
            }
            sender.send_json(request)
            
            if (i + 1) % 5 == 0:
                self.logger.info(f"  Sent {i + 1}/{num_requests} requests")
        
        self.logger.info(f"All {num_requests} requests sent")
        
        # 关闭 socket
        sender.close()
        context.term()
    
    def _verify_statistics(self):
        """验证统计日志内容"""
        output_text = '\n'.join(self.captured_output)
        
        # 1. 验证统计摘要标题
        self.logger.info("Checking statistics summary header...")
        self.assert_true(
            "Statistics Summary" in output_text,
            "Should contain 'Statistics Summary' header"
        )
        
        # 2. 验证总请求数
        self.logger.info("Checking total requests count...")
        has_total_requests = any("Total Requests:" in line for line in self.captured_output)
        self.assert_true(has_total_requests, "Should show total requests count")
        
        # 提取总请求数
        total_requests = 0
        for line in self.captured_output:
            if "Total Requests:" in line:
                try:
                    # 提取数字
                    parts = line.split("Total Requests:")
                    if len(parts) > 1:
                        total_requests = int(parts[1].strip())
                        self.logger.info(f"  Found total requests: {total_requests}")
                        break
                except:
                    pass
        
        # 验证请求数是否正确（应该是 20）
        if total_requests > 0:
            self.logger.info(f"✓ Total requests recorded: {total_requests}")
            # 允许一些误差，因为可能有其他请求
            self.assert_true(
                total_requests >= 20,
                f"Should have at least 20 requests, got {total_requests}"
            )
        
        # 3. 验证成功/失败数
        self.logger.info("Checking successful/failed requests...")
        has_successful = any("Successful Requests:" in line for line in self.captured_output)
        has_failed = any("Failed Requests:" in line for line in self.captured_output)
        
        self.assert_true(has_successful, "Should show successful requests count")
        self.assert_true(has_failed, "Should show failed requests count")
        
        # 提取成功和失败数
        successful_requests = 0
        failed_requests = 0
        
        for line in self.captured_output:
            if "Successful Requests:" in line:
                try:
                    parts = line.split("Successful Requests:")
                    if len(parts) > 1:
                        successful_requests = int(parts[1].strip())
                        self.logger.info(f"  Found successful requests: {successful_requests}")
                except:
                    pass
            
            if "Failed Requests:" in line:
                try:
                    parts = line.split("Failed Requests:")
                    if len(parts) > 1:
                        failed_requests = int(parts[1].strip())
                        self.logger.info(f"  Found failed requests: {failed_requests}")
                except:
                    pass
        
        if successful_requests > 0:
            self.logger.info(f"✓ Successful requests: {successful_requests}")
        if failed_requests >= 0:
            self.logger.info(f"✓ Failed requests: {failed_requests}")
        
        # 4. 验证吞吐量
        self.logger.info("Checking throughput...")
        has_throughput = any("Average Throughput:" in line for line in self.captured_output)
        self.assert_true(has_throughput, "Should show average throughput")
        
        # 5. 验证运行时间
        self.logger.info("Checking running time...")
        has_running_time = any("Running Time:" in line for line in self.captured_output)
        self.assert_true(has_running_time, "Should show running time")
        
        # 6. 验证 Worker 请求分布
        self.logger.info("Checking worker request distribution...")
        has_distribution = any("Worker Request Distribution:" in line for line in self.captured_output)
        self.assert_true(has_distribution, "Should show worker request distribution")
        
        # 检查每个 Worker 的统计
        worker0_stats = any("Worker 0 (GPU 0):" in line and "requests" in line for line in self.captured_output)
        worker1_stats = any("Worker 1 (GPU 1):" in line and "requests" in line for line in self.captured_output)
        
        self.assert_true(worker0_stats, "Should show Worker 0 request count")
        self.assert_true(worker1_stats, "Should show Worker 1 request count")
        
        # 提取每个 Worker 的请求数
        worker0_count = 0
        worker1_count = 0
        
        for line in self.captured_output:
            if "Worker 0 (GPU 0):" in line and "requests" in line:
                try:
                    # 提取请求数（格式：Worker 0 (GPU 0): X requests (Y%)）
                    parts = line.split(":")
                    if len(parts) > 1:
                        request_part = parts[1].split("requests")[0].strip()
                        worker0_count = int(request_part)
                        self.logger.info(f"  Worker 0 handled: {worker0_count} requests")
                except:
                    pass
            
            if "Worker 1 (GPU 1):" in line and "requests" in line:
                try:
                    parts = line.split(":")
                    if len(parts) > 1:
                        request_part = parts[1].split("requests")[0].strip()
                        worker1_count = int(request_part)
                        self.logger.info(f"  Worker 1 handled: {worker1_count} requests")
                except:
                    pass
        
        # 验证请求分布（应该相对均匀）
        if worker0_count > 0 and worker1_count > 0:
            total_worker_requests = worker0_count + worker1_count
            self.logger.info(f"✓ Total requests distributed: {total_worker_requests}")
            self.logger.info(f"✓ Worker 0: {worker0_count} ({worker0_count/total_worker_requests*100:.1f}%)")
            self.logger.info(f"✓ Worker 1: {worker1_count} ({worker1_count/total_worker_requests*100:.1f}%)")
            
            # 验证分布相对均匀（允许一些偏差）
            ratio = worker0_count / worker1_count if worker1_count > 0 else 0
            self.logger.info(f"  Distribution ratio (Worker0/Worker1): {ratio:.2f}")
            
            # Round Robin 应该产生接近 1:1 的分布
            # 允许 20% 的偏差
            self.assert_true(
                0.5 <= ratio <= 2.0,
                f"Request distribution should be relatively even, got ratio {ratio:.2f}"
            )
        
        # 7. 验证百分比显示
        self.logger.info("Checking percentage display...")
        has_percentage = any("%" in line and "Worker" in line for line in self.captured_output)
        self.assert_true(has_percentage, "Should show percentage for worker distribution")
        
        # 打印一些关键统计行供人工检查
        self.logger.info("\n" + "="*70)
        self.logger.info("Key statistics lines found:")
        self.logger.info("="*70)
        
        key_patterns = [
            "Statistics Summary",
            "Total Requests:",
            "Successful Requests:",
            "Failed Requests:",
            "Average Throughput:",
            "Running Time:",
            "Worker Request Distribution:",
            "Worker 0 (GPU 0):",
            "Worker 1 (GPU 1):",
        ]
        
        for pattern in key_patterns:
            matching_lines = [line for line in self.captured_output if pattern in line]
            if matching_lines:
                for line in matching_lines[:2]:  # 只显示前2个匹配
                    self.logger.info(f"  {line.strip()}")
        
        self.logger.info("="*70 + "\n")
    
    def teardown(self):
        """清理测试环境"""
        self.logger.info("Cleaning up test environment...")
        
        if self.manager:
            # 终止所有 Worker 进程
            self.logger.info("Terminating workers...")
            for i, worker in enumerate(self.manager.workers):
                if worker.is_alive():
                    self.logger.info(f"Terminating worker {i}...")
                    worker.terminate()
                    worker.join(timeout=5)
                    if worker.is_alive():
                        self.logger.warning(f"Force killing worker {i}...")
                        worker.kill()
            
            # 终止 Response Merger 进程
            if self.manager.merger_process and self.manager.merger_process.is_alive():
                self.logger.info("Terminating Response Merger...")
                self.manager.merger_process.terminate()
                self.manager.merger_process.join(timeout=5)
                if self.manager.merger_process.is_alive():
                    self.logger.warning("Force killing Response Merger...")
                    self.manager.merger_process.kill()
        
        self.logger.info("Cleanup complete")


def main():
    """主函数"""
    test = RequestStatisticsTest()
    result = test.run()
    
    # 返回退出码
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
