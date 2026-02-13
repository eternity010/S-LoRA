"""
Test for _load_adapter_ranks method in DataParallelRouterManager

This test verifies that the _load_adapter_ranks method correctly loads
adapter rank information from lora_dirs configurations.
"""

import argparse
import pytest
from unittest.mock import patch, MagicMock
from slora.server.router.dp_manager import DataParallelRouterManager


def test_load_adapter_ranks_empty_lora_dirs():
    """Test _load_adapter_ranks with empty lora_dirs"""
    # Create args with empty lora_dirs
    args = argparse.Namespace(
        num_workers=1,
        gpu_ids='0',
        routing_strategy='adapter-aware',
        lora_dirs=[],
        dummy=False
    )
    
    # Mock GPU detection and other initialization
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.device_count', return_value=1), \
         patch('torch.cuda.get_device_properties'):
        
        manager = DataParallelRouterManager(args, router_port=10000, response_port=10001, detoken_port=10002)
        
        # Call _load_adapter_ranks
        adapter_ranks = manager._load_adapter_ranks()
        
        # Verify empty result
        assert adapter_ranks == {}


def test_load_adapter_ranks_with_valid_configs():
    """Test _load_adapter_ranks with valid adapter configurations"""
    # Create args with lora_dirs
    lora_dirs = ['/path/to/adapter1', '/path/to/adapter2']
    args = argparse.Namespace(
        num_workers=1,
        gpu_ids='0',
        routing_strategy='adapter-aware',
        lora_dirs=lora_dirs,
        dummy=False
    )
    
    # Mock get_lora_config to return different ranks
    def mock_get_lora_config(lora_dir, dummy):
        if 'adapter1' in lora_dir:
            return {'r': 16}, lora_dir
        elif 'adapter2' in lora_dir:
            return {'r': 32}, lora_dir
        return {'r': 8}, lora_dir
    
    # Mock GPU detection and other initialization
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.device_count', return_value=1), \
         patch('torch.cuda.get_device_properties'), \
         patch('slora.models.peft.lora_adapter.get_lora_config', side_effect=mock_get_lora_config):
        
        manager = DataParallelRouterManager(args, router_port=10000, response_port=10001, detoken_port=10002)
        
        # Call _load_adapter_ranks
        adapter_ranks = manager._load_adapter_ranks()
        
        # Verify results
        assert len(adapter_ranks) == 2
        assert adapter_ranks['/path/to/adapter1'] == 16
        assert adapter_ranks['/path/to/adapter2'] == 32


def test_load_adapter_ranks_with_missing_config():
    """Test _load_adapter_ranks handles missing configurations gracefully"""
    # Create args with lora_dirs
    lora_dirs = ['/path/to/adapter1', '/path/to/adapter2']
    args = argparse.Namespace(
        num_workers=1,
        gpu_ids='0',
        routing_strategy='adapter-aware',
        lora_dirs=lora_dirs,
        dummy=False,
        default_lora_rank=16
    )
    
    # Mock get_lora_config to raise exception for adapter2
    def mock_get_lora_config(lora_dir, dummy):
        if 'adapter1' in lora_dir:
            return {'r': 32}, lora_dir
        elif 'adapter2' in lora_dir:
            raise FileNotFoundError("Config not found")
        return {'r': 8}, lora_dir
    
    # Mock GPU detection and other initialization
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.device_count', return_value=1), \
         patch('torch.cuda.get_device_properties'), \
         patch('slora.models.peft.lora_adapter.get_lora_config', side_effect=mock_get_lora_config):
        
        manager = DataParallelRouterManager(args, router_port=10000, response_port=10001, detoken_port=10002)
        
        # Call _load_adapter_ranks
        adapter_ranks = manager._load_adapter_ranks()
        
        # Verify results - adapter2 should use default rank
        assert len(adapter_ranks) == 2
        assert adapter_ranks['/path/to/adapter1'] == 32
        assert adapter_ranks['/path/to/adapter2'] == 16  # default rank


def test_load_adapter_ranks_with_missing_r_field():
    """Test _load_adapter_ranks handles missing 'r' field in config"""
    # Create args with lora_dirs
    lora_dirs = ['/path/to/adapter1']
    args = argparse.Namespace(
        num_workers=1,
        gpu_ids='0',
        routing_strategy='adapter-aware',
        lora_dirs=lora_dirs,
        dummy=False,
        default_lora_rank=16
    )
    
    # Mock get_lora_config to return config without 'r' field
    def mock_get_lora_config(lora_dir, dummy):
        return {'lora_alpha': 32}, lora_dir  # Missing 'r' field
    
    # Mock GPU detection and other initialization
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.device_count', return_value=1), \
         patch('torch.cuda.get_device_properties'), \
         patch('slora.models.peft.lora_adapter.get_lora_config', side_effect=mock_get_lora_config):
        
        manager = DataParallelRouterManager(args, router_port=10000, response_port=10001, detoken_port=10002)
        
        # Call _load_adapter_ranks
        adapter_ranks = manager._load_adapter_ranks()
        
        # Verify results - should use default rank when 'r' is missing
        assert len(adapter_ranks) == 1
        assert adapter_ranks['/path/to/adapter1'] == 16  # default rank


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


def test_init_loads_and_passes_adapter_ranks_to_router():
    """Test that __init__ loads adapter ranks and passes them to the router"""
    # Create args with lora_dirs
    lora_dirs = ['/path/to/adapter1', '/path/to/adapter2']
    args = argparse.Namespace(
        num_workers=1,
        gpu_ids='0',
        routing_strategy='adapter-aware',
        lora_dirs=lora_dirs,
        dummy=False
    )
    
    # Mock get_lora_config to return different ranks
    def mock_get_lora_config(lora_dir, dummy):
        if 'adapter1' in lora_dir:
            return {'r': 16}, lora_dir
        elif 'adapter2' in lora_dir:
            return {'r': 32}, lora_dir
        return {'r': 8}, lora_dir
    
    # Mock GPU detection and other initialization
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.device_count', return_value=1), \
         patch('torch.cuda.get_device_properties'), \
         patch('slora.models.peft.lora_adapter.get_lora_config', side_effect=mock_get_lora_config):
        
        manager = DataParallelRouterManager(args, router_port=10000, response_port=10001, detoken_port=10002)
        
        # Verify that the router has the adapter ranks set
        from slora.server.router.adapter_aware_router import AdapterAwareRouter
        assert isinstance(manager.router, AdapterAwareRouter)
        
        # Check that adapter_ranks were set on the router
        assert manager.router.adapter_ranks is not None
        assert len(manager.router.adapter_ranks) == 2
        assert manager.router.adapter_ranks['/path/to/adapter1'] == 16
        assert manager.router.adapter_ranks['/path/to/adapter2'] == 32


def test_init_does_not_load_adapter_ranks_for_round_robin():
    """Test that __init__ does not load adapter ranks for round-robin strategy"""
    # Create args with round-robin strategy
    args = argparse.Namespace(
        num_workers=1,
        gpu_ids='0',
        routing_strategy='round-robin',
        lora_dirs=['/path/to/adapter1'],
        dummy=False
    )
    
    # Mock GPU detection
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.device_count', return_value=1), \
         patch('torch.cuda.get_device_properties'):
        
        manager = DataParallelRouterManager(args, router_port=10000, response_port=10001, detoken_port=10002)
        
        # Verify that the router is RoundRobinRouter (not AdapterAwareRouter)
        from slora.server.router.round_robin_router import RoundRobinRouter
        assert isinstance(manager.router, RoundRobinRouter)


def test_get_routing_config_includes_rank_aware_params():
    """Test that _get_routing_config includes w3, default_lora_rank, and max_rank_diff"""
    # Create args with rank-aware routing parameters
    args = argparse.Namespace(
        num_workers=1,
        gpu_ids='0',
        routing_strategy='adapter-aware',
        routing_w1=2.0,
        routing_w2=0.5,
        routing_w3=0.3,
        default_lora_rank=32,
        max_rank_diff=128,
        lora_dirs=[],
        dummy=False
    )
    
    # Mock GPU detection
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.device_count', return_value=1), \
         patch('torch.cuda.get_device_properties'):
        
        manager = DataParallelRouterManager(args, router_port=10000, response_port=10001, detoken_port=10002)
        
        # Get routing config
        config = manager._get_routing_config()
        
        # Verify rank-aware parameters
        assert config.w3 == 0.3
        assert config.default_lora_rank == 32
        assert config.max_rank_diff == 128
        
        # Verify other parameters are still correct
        assert config.w1 == 2.0
        assert config.w2 == 0.5


def test_get_routing_config_uses_defaults_for_rank_aware_params():
    """Test that _get_routing_config uses defaults when rank-aware params not specified"""
    # Create args without rank-aware routing parameters
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
        
        manager = DataParallelRouterManager(args, router_port=10000, response_port=10001, detoken_port=10002)
        
        # Get routing config
        config = manager._get_routing_config()
        
        # Verify default values for rank-aware parameters
        assert config.w3 == 0.0  # Default: disabled
        assert config.default_lora_rank == 16  # Default: 16
        assert config.max_rank_diff == 64  # Default: 64
