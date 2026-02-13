"""
Property-based tests for batch rank distribution calculation

Tests the correctness of rank distribution calculation in GPUWorker's
_get_state_for_reporter() method.

Feature: rank-aware-routing, Property 1: Batch Rank Distribution Calculation
**Validates: Requirements 1.1, 1.2, 1.3**
"""

import pytest
import sys
import os
import argparse
from unittest.mock import patch
from hypothesis import given, strategies as st, settings, assume

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from slora.server.router.gpu_worker import GPUWorker
from slora.server.io_struct import Req, Batch
from slora.server.sampling_params import SamplingParams


# Strategy for generating adapter ranks (typical LoRA ranks: 4, 8, 16, 32, 64, 128)
adapter_rank_strategy = st.integers(min_value=1, max_value=256)

# Strategy for generating adapter directories
adapter_dir_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='/_-'),
    min_size=1, max_size=30
).filter(lambda x: len(x.strip()) > 0)


class TestBatchRankDistributionPropertyBased:
    """
    Property-based tests for batch rank distribution calculation
    Feature: rank-aware-routing, Property 1: Batch Rank Distribution Calculation
    **Validates: Requirements 1.1, 1.2, 1.3**
    """
    
    @given(
        ranks=st.lists(
            adapter_rank_strategy,
            min_size=1,
            max_size=50
        )
    )
    @settings(max_examples=100)
    def test_batch_rank_distribution_calculation(self, ranks):
        """
        Property 1: Batch Rank Distribution Calculation
        
        For any batch containing requests with known adapter ranks,
        the reported rank distribution (avg_rank, min_rank, max_rank)
        SHALL be correctly calculated as the arithmetic mean, minimum,
        and maximum of the individual adapter ranks respectively.
        
        **Validates: Requirements 1.1, 1.2, 1.3**
        """
        # Create mock args
        args = argparse.Namespace(model_dir='/fake/path', lora_dirs=[])
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('torch.cuda.mem_get_info', return_value=(8000000000, 16000000000)):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            # Create requests with different adapter ranks
            sample_params = SamplingParams(do_sample=False, max_new_tokens=10, stop_sequences=[])
            reqs = []
            adapter_dirs = []
            
            for i, rank in enumerate(ranks):
                adapter_dir = f'lora_{i}'
                adapter_dirs.append(adapter_dir)
                req = Req(
                    adapter_dir=adapter_dir,
                    request_id=f'req_{i}',
                    prompt_ids=[1, 2, 3],
                    sample_params=sample_params
                )
                reqs.append(req)
                # Set up the rank mapping
                worker.lora_ranks[adapter_dir] = rank
            
            # Create batch
            batch = Batch(batch_id=1, reqs=reqs)
            worker.current_batch = batch
            
            # Get state
            state = worker._get_state_for_reporter()
            
            # Calculate expected values
            expected_avg = sum(ranks) / len(ranks)
            expected_min = min(ranks)
            expected_max = max(ranks)
            
            # Verify avg_rank (allow small floating point error)
            assert abs(state['avg_rank'] - expected_avg) < 1e-6, \
                f"Expected avg_rank={expected_avg}, got {state['avg_rank']}"
            
            # Verify min_rank
            assert state['min_rank'] == expected_min, \
                f"Expected min_rank={expected_min}, got {state['min_rank']}"
            
            # Verify max_rank
            assert state['max_rank'] == expected_max, \
                f"Expected max_rank={expected_max}, got {state['max_rank']}"
    
    @given(
        num_requests=st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=100)
    def test_avg_rank_bounds(self, num_requests):
        """
        Property 1: Batch Rank Distribution Calculation - Bounds check
        
        For any batch with valid adapter ranks, the avg_rank SHALL be
        between the min_rank and max_rank (inclusive).
        
        **Validates: Requirements 1.1, 1.2, 1.3**
        """
        # Create mock args
        args = argparse.Namespace(model_dir='/fake/path', lora_dirs=[])
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('torch.cuda.mem_get_info', return_value=(8000000000, 16000000000)):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            # Create requests with random ranks
            sample_params = SamplingParams(do_sample=False, max_new_tokens=10, stop_sequences=[])
            reqs = []
            
            import random
            for i in range(num_requests):
                rank = random.randint(1, 256)
                adapter_dir = f'lora_{i}'
                req = Req(
                    adapter_dir=adapter_dir,
                    request_id=f'req_{i}',
                    prompt_ids=[1, 2, 3],
                    sample_params=sample_params
                )
                reqs.append(req)
                worker.lora_ranks[adapter_dir] = rank
            
            # Create batch
            batch = Batch(batch_id=1, reqs=reqs)
            worker.current_batch = batch
            
            # Get state
            state = worker._get_state_for_reporter()
            
            # avg_rank must be between min_rank and max_rank
            assert state['min_rank'] <= state['avg_rank'] <= state['max_rank'], \
                f"avg_rank={state['avg_rank']} not in range [{state['min_rank']}, {state['max_rank']}]"
    
    @given(
        rank=adapter_rank_strategy
    )
    @settings(max_examples=100)
    def test_single_request_batch(self, rank):
        """
        Property 1: Batch Rank Distribution Calculation - Single request
        
        For any batch containing a single request with rank R,
        avg_rank, min_rank, and max_rank SHALL all equal R.
        
        **Validates: Requirements 1.1, 1.2, 1.3**
        """
        # Create mock args
        args = argparse.Namespace(model_dir='/fake/path', lora_dirs=[])
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('torch.cuda.mem_get_info', return_value=(8000000000, 16000000000)):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            # Create single request
            sample_params = SamplingParams(do_sample=False, max_new_tokens=10, stop_sequences=[])
            adapter_dir = 'lora_single'
            req = Req(
                adapter_dir=adapter_dir,
                request_id='req_single',
                prompt_ids=[1, 2, 3],
                sample_params=sample_params
            )
            worker.lora_ranks[adapter_dir] = rank
            
            # Create batch
            batch = Batch(batch_id=1, reqs=[req])
            worker.current_batch = batch
            
            # Get state
            state = worker._get_state_for_reporter()
            
            # All rank values should equal the single rank
            assert state['avg_rank'] == float(rank), \
                f"Expected avg_rank={rank}, got {state['avg_rank']}"
            assert state['min_rank'] == rank, \
                f"Expected min_rank={rank}, got {state['min_rank']}"
            assert state['max_rank'] == rank, \
                f"Expected max_rank={rank}, got {state['max_rank']}"
    
    def test_empty_batch_returns_zeros(self):
        """
        Property 1: Batch Rank Distribution Calculation - Empty batch
        
        For any Worker with no active batch (current_batch is None),
        avg_rank, min_rank, and max_rank SHALL all be 0.
        
        **Validates: Requirements 1.4**
        """
        # Create mock args
        args = argparse.Namespace(model_dir='/fake/path', lora_dirs=[])
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('torch.cuda.mem_get_info', return_value=(8000000000, 16000000000)):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            # No batch
            worker.current_batch = None
            
            # Get state
            state = worker._get_state_for_reporter()
            
            # All rank values should be 0
            assert state['avg_rank'] == 0.0, \
                f"Expected avg_rank=0.0, got {state['avg_rank']}"
            assert state['min_rank'] == 0, \
                f"Expected min_rank=0, got {state['min_rank']}"
            assert state['max_rank'] == 0, \
                f"Expected max_rank=0, got {state['max_rank']}"
    
    def test_batch_with_empty_reqs_returns_zeros(self):
        """
        Property 1: Batch Rank Distribution Calculation - Batch with no requests
        
        For any Worker with a batch that has no requests (empty reqs list),
        avg_rank, min_rank, and max_rank SHALL all be 0.
        
        **Validates: Requirements 1.4**
        """
        # Create mock args
        args = argparse.Namespace(model_dir='/fake/path', lora_dirs=[])
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('torch.cuda.mem_get_info', return_value=(8000000000, 16000000000)):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            # Batch with no requests
            batch = Batch(batch_id=1, reqs=[])
            worker.current_batch = batch
            
            # Get state
            state = worker._get_state_for_reporter()
            
            # All rank values should be 0
            assert state['avg_rank'] == 0.0, \
                f"Expected avg_rank=0.0, got {state['avg_rank']}"
            assert state['min_rank'] == 0, \
                f"Expected min_rank=0, got {state['min_rank']}"
            assert state['max_rank'] == 0, \
                f"Expected max_rank=0, got {state['max_rank']}"
    
    @given(
        num_unknown=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=100)
    def test_unknown_adapters_use_default_rank(self, num_unknown):
        """
        Property 1: Batch Rank Distribution Calculation - Unknown adapters
        
        For any batch containing requests with unknown adapter directories
        (not in lora_ranks mapping), the default rank of 16 SHALL be used
        in the calculation.
        
        **Validates: Requirements 1.1, 1.2, 1.3**
        """
        # Create mock args
        args = argparse.Namespace(model_dir='/fake/path', lora_dirs=[])
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('torch.cuda.mem_get_info', return_value=(8000000000, 16000000000)):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            # Create requests with unknown adapters
            sample_params = SamplingParams(do_sample=False, max_new_tokens=10, stop_sequences=[])
            reqs = []
            
            for i in range(num_unknown):
                adapter_dir = f'unknown_lora_{i}'
                req = Req(
                    adapter_dir=adapter_dir,
                    request_id=f'req_{i}',
                    prompt_ids=[1, 2, 3],
                    sample_params=sample_params
                )
                reqs.append(req)
                # Intentionally NOT adding to lora_ranks
            
            # Create batch
            batch = Batch(batch_id=1, reqs=reqs)
            worker.current_batch = batch
            
            # Get state
            state = worker._get_state_for_reporter()
            
            # All unknown adapters should use default rank of 16
            assert state['avg_rank'] == 16.0, \
                f"Expected avg_rank=16.0 for unknown adapters, got {state['avg_rank']}"
            assert state['min_rank'] == 16, \
                f"Expected min_rank=16 for unknown adapters, got {state['min_rank']}"
            assert state['max_rank'] == 16, \
                f"Expected max_rank=16 for unknown adapters, got {state['max_rank']}"
    
    @given(
        known_ranks=st.lists(adapter_rank_strategy, min_size=1, max_size=10),
        num_unknown=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=100)
    def test_mixed_known_and_unknown_adapters(self, known_ranks, num_unknown):
        """
        Property 1: Batch Rank Distribution Calculation - Mixed adapters
        
        For any batch containing both known and unknown adapters,
        the rank distribution SHALL be calculated correctly using
        configured ranks for known adapters and default rank (16)
        for unknown adapters.
        
        **Validates: Requirements 1.1, 1.2, 1.3**
        """
        # Create mock args
        args = argparse.Namespace(model_dir='/fake/path', lora_dirs=[])
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.set_device'), \
             patch('torch.cuda.mem_get_info', return_value=(8000000000, 16000000000)):
            
            worker = GPUWorker(worker_id=0, gpu_id=0, args=args)
            
            # Create requests with known adapters
            sample_params = SamplingParams(do_sample=False, max_new_tokens=10, stop_sequences=[])
            reqs = []
            all_ranks = []
            
            for i, rank in enumerate(known_ranks):
                adapter_dir = f'known_lora_{i}'
                req = Req(
                    adapter_dir=adapter_dir,
                    request_id=f'req_known_{i}',
                    prompt_ids=[1, 2, 3],
                    sample_params=sample_params
                )
                reqs.append(req)
                worker.lora_ranks[adapter_dir] = rank
                all_ranks.append(rank)
            
            # Create requests with unknown adapters
            for i in range(num_unknown):
                adapter_dir = f'unknown_lora_{i}'
                req = Req(
                    adapter_dir=adapter_dir,
                    request_id=f'req_unknown_{i}',
                    prompt_ids=[1, 2, 3],
                    sample_params=sample_params
                )
                reqs.append(req)
                all_ranks.append(16)  # Default rank
            
            # Create batch
            batch = Batch(batch_id=1, reqs=reqs)
            worker.current_batch = batch
            
            # Get state
            state = worker._get_state_for_reporter()
            
            # Calculate expected values
            expected_avg = sum(all_ranks) / len(all_ranks)
            expected_min = min(all_ranks)
            expected_max = max(all_ranks)
            
            # Verify
            assert abs(state['avg_rank'] - expected_avg) < 1e-6, \
                f"Expected avg_rank={expected_avg}, got {state['avg_rank']}"
            assert state['min_rank'] == expected_min, \
                f"Expected min_rank={expected_min}, got {state['min_rank']}"
            assert state['max_rank'] == expected_max, \
                f"Expected max_rank={expected_max}, got {state['max_rank']}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
