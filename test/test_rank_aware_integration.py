"""
Integration test for rank-aware routing feature

This test verifies that all components of the rank-aware routing feature
work together correctly.
"""

import argparse
import pytest
from unittest.mock import patch, MagicMock
from slora.server.router.dp_manager import DataParallelRouterManager
from slora.server.router.adapter_aware_router import AdapterAwareRouter
from slora.server.router.worker_state import RoutingConfig


def test_rank_aware_routing_integration():
    """
    Integration test: Verify that rank-aware routing components integrate correctly
    
    This test verifies:
    1. Command-line parameters are correctly parsed
    2. RoutingConfig includes rank-aware parameters
    3. DataParallelRouterManager loads adapter ranks
    4. AdapterAwareRouter receives adapter ranks
    """
    # Create args with all rank-aware parameters
    lora_dirs = ['/path/to/adapter1', '/path/to/adapter2', '/path/to/adapter3']
    args = argparse.Namespace(
        num_workers=2,
        gpu_ids='0,1',
        routing_strategy='adapter-aware',
        routing_w1=2.0,
        routing_w2=0.5,
        routing_w3=0.3,  # Rank mismatch penalty enabled
        default_lora_rank=32,
        max_rank_diff=128,
        lora_dirs=lora_dirs,
        dummy=False
    )
    
    # Mock get_lora_config to return different ranks
    def mock_get_lora_config(lora_dir, dummy):
        if 'adapter1' in lora_dir:
            return {'r': 8}, lora_dir
        elif 'adapter2' in lora_dir:
            return {'r': 16}, lora_dir
        elif 'adapter3' in lora_dir:
            return {'r': 64}, lora_dir
        return {'r': 32}, lora_dir
    
    # Mock GPU detection and other initialization
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.device_count', return_value=2), \
         patch('torch.cuda.get_device_properties'), \
         patch('slora.models.peft.lora_adapter.get_lora_config', side_effect=mock_get_lora_config):
        
        # Create DataParallelRouterManager
        manager = DataParallelRouterManager(
            args, 
            router_port=10000, 
            response_port=10001, 
            detoken_port=10002
        )
        
        # Verify routing config includes rank-aware parameters
        config = manager._get_routing_config()
        assert config.w3 == 0.3
        assert config.default_lora_rank == 32
        assert config.max_rank_diff == 128
        
        # Verify router is AdapterAwareRouter
        assert isinstance(manager.router, AdapterAwareRouter)
        
        # Verify adapter ranks were loaded and passed to router
        assert manager.router.adapter_ranks is not None
        assert len(manager.router.adapter_ranks) == 3
        assert manager.router.adapter_ranks['/path/to/adapter1'] == 8
        assert manager.router.adapter_ranks['/path/to/adapter2'] == 16
        assert manager.router.adapter_ranks['/path/to/adapter3'] == 64
        
        # Verify router config has rank-aware parameters
        assert manager.router.config.w3 == 0.3
        assert manager.router.config.default_lora_rank == 32
        assert manager.router.config.max_rank_diff == 128


def test_rank_aware_routing_backward_compatibility():
    """
    Integration test: Verify backward compatibility when w3=0
    
    This test verifies that when w3=0 (default), the system behaves
    identically to the original adapter-aware routing.
    """
    # Create args without rank-aware parameters (using defaults)
    args = argparse.Namespace(
        num_workers=1,
        gpu_ids='0',
        routing_strategy='adapter-aware',
        lora_dirs=[],
        dummy=False
    )
    
    # Mock GPU detection
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.device_count', return_value=1), \
         patch('torch.cuda.get_device_properties'):
        
        manager = DataParallelRouterManager(
            args, 
            router_port=10000, 
            response_port=10001, 
            detoken_port=10002
        )
        
        # Verify default values ensure backward compatibility
        config = manager._get_routing_config()
        assert config.w3 == 0.0  # Disabled by default
        assert config.default_lora_rank == 16
        assert config.max_rank_diff == 64
        
        # Verify router still works
        assert isinstance(manager.router, AdapterAwareRouter)


def test_cli_parameters_propagation():
    """
    Test that CLI parameters correctly propagate through the system
    """
    # Simulate command-line arguments
    args = argparse.Namespace(
        num_workers=1,
        gpu_ids='0',
        routing_strategy='adapter-aware',
        routing_w1=5.0,
        routing_w2=2.0,
        routing_w3=1.5,
        default_lora_rank=64,
        max_rank_diff=256,
        lora_dirs=[],
        dummy=False
    )
    
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.device_count', return_value=1), \
         patch('torch.cuda.get_device_properties'):
        
        manager = DataParallelRouterManager(
            args, 
            router_port=10000, 
            response_port=10001, 
            detoken_port=10002
        )
        
        # Verify all parameters propagated correctly
        assert manager.router.config.w1 == 5.0
        assert manager.router.config.w2 == 2.0
        assert manager.router.config.w3 == 1.5
        assert manager.router.config.default_lora_rank == 64
        assert manager.router.config.max_rank_diff == 256


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
