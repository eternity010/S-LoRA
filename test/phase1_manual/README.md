# Phase 1 Manual Testing Scripts

This directory contains manual testing scripts for Phase 1 data parallel implementation.

## Purpose

These scripts are designed for step-by-step validation of the data parallel system, from basic components to full integration.

## Test Environment

- **Conda Environment**: `slora`
- **Python Version**: 3.9.25
- **PyTorch**: 2.0.1+cu118
- **Available GPUs**: 3x NVIDIA L40 (49GB each)

## Test Categories

### 1. Component Tests (8.2.x)
- `test_router.py` - Round Robin Router validation
- `test_worker_init.py` - GPU Worker initialization
- `test_manager_init.py` - DataParallelRouterManager initialization

### 2. Process Tests (8.3.x)
- `test_single_worker_startup.py` - Single Worker process startup
- `test_multi_worker_startup.py` - Multiple Worker processes startup
- `test_merger_startup.py` - Response Merger process startup

### 3. Communication Tests (8.4.x)
- `test_manager_to_worker.py` - Manager → Worker communication
- `test_worker_to_merger.py` - Worker → Merger communication
- `test_full_communication.py` - Complete communication chain

### 4. Functional Tests (8.5.x)
- `test_single_request.py` - Single request processing
- `test_serial_requests.py` - Serial request processing
- `test_concurrent_requests.py` - Concurrent request processing
- `test_high_concurrency.py` - High concurrency testing

### 5. Error Handling Tests (8.6.x)
- `test_worker_startup_failure.py` - Worker startup failure handling
- `test_worker_crash.py` - Worker process crash handling
- `test_inference_error.py` - Inference error handling

## Running Tests

Each test script can be run independently:

```bash
# Activate environment
conda activate slora

# Run a specific test
cd /home/hzheng/S-LoRA
python test/phase1_manual/test_router.py

# Or with pytest
pytest test/phase1_manual/test_router.py -v
```

## Test Execution Order

Follow the task order in `tasks.md` for systematic validation:
1. Component tests (simplest, no processes)
2. Process tests (startup validation)
3. Communication tests (message passing)
4. Functional tests (end-to-end)
5. Error handling tests (failure scenarios)

## Notes

- Tests are designed to be run manually for checkpoint validation
- Each test includes detailed logging for debugging
- Tests can be run multiple times safely
- Some tests may require specific GPU configurations
