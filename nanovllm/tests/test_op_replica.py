"""
Unit tests for operator replica functionality.
"""

import unittest
import torch
import threading
import time
from unittest.mock import Mock, patch

from nanovllm.engine.op_replica import (
    OperatorReplica,
    ReplicaManager,
    ReplicaConfig,
    ReplicaMetrics,
    ReplicaAutoScaler
)


class MockAttentionOp:
    """Mock attention operator for testing."""

    def __init__(self, weights):
        self.weights = weights
        self.call_count = 0

    def forward(self, q, k, v):
        self.call_count += 1
        # Simple mock implementation - just return q
        return q


class TestOperatorReplica(unittest.TestCase):
    """Test OperatorReplica functionality."""

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

    def test_replica_creation(self):
        """Test creating an operator replica."""
        device = torch.device("cuda:0")
        weights = {"test_weight": torch.randn(10, 10)}

        def workspace_factory(dev):
            return {"test_workspace": torch.zeros(5, 5, device=dev)}

        replica = OperatorReplica(
            op_class=MockAttentionOp,
            weights_cpu=weights,
            device=device,
            workspace_factory=workspace_factory,
        )

        self.assertEqual(replica.device, device)
        self.assertIsInstance(replica.stream, torch.cuda.Stream)
        self.assertIn("test_weight", replica.weights)
        self.assertIn("test_workspace", replica.workspace)
        self.assertIsInstance(replica.op, MockAttentionOp)

    def test_replica_forward_async(self):
        """Test async forward pass."""
        device = torch.device("cuda:0")
        weights = {}

        replica = OperatorReplica(
            op_class=MockAttentionOp,
            weights_cpu=weights,
            device=device,
        )

        # Create test input on CPU
        q = torch.randn(2, 8, 64)
        k = torch.randn(2, 8, 64)
        v = torch.randn(2, 8, 64)

        output, event = replica.forward_async(q, k, v)

        # Wait for completion
        event.synchronize()

        # Check output
        self.assertEqual(output.shape, q.shape)
        self.assertEqual(replica.op.call_count, 1)

    def test_replica_metrics(self):
        """Test replica metrics tracking."""
        device = torch.device("cuda:0")
        replica = OperatorReplica(
            op_class=MockAttentionOp,
            weights_cpu={},
            device=device,
        )

        initial_busy = replica.metrics.busy
        self.assertFalse(initial_busy)

        # Simulate a forward pass
        q = torch.randn(1, 1, 64)
        output, event = replica.forward_async(q, q, q)
        event.synchronize()

        # Metrics should be updated
        self.assertEqual(replica.metrics.total_requests, 1)


class TestReplicaManager(unittest.TestCase):
    """Test ReplicaManager functionality."""

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

    def test_manager_creation(self):
        """Test creating a replica manager."""
        config = ReplicaConfig(replica_devices=[0])
        manager = ReplicaManager(config)

        self.assertIsInstance(manager.replicas, dict)
        self.assertIsInstance(manager._round_robin_idx, dict)

    def test_create_replicas(self):
        """Test creating replicas through the manager."""
        config = ReplicaConfig(replica_devices=[0])
        manager = ReplicaManager(config)

        weights = {}
        devices = [0]
        num_per_device = 2

        manager.create_replicas(
            op_name="test_op",
            op_class=MockAttentionOp,
            weights_cpu=weights,
            devices=devices,
            num_replicas_per_device=num_per_device,
        )

        self.assertIn("test_op", manager.replicas)
        self.assertEqual(len(manager.replicas["test_op"]), num_per_device)

    def test_get_replica_round_robin(self):
        """Test round-robin replica selection."""
        config = ReplicaConfig(replica_devices=[0])
        manager = ReplicaManager(config)

        # Create 3 replicas
        manager.create_replicas(
            op_name="test_op",
            op_class=MockAttentionOp,
            weights_cpu={},
            devices=[0],
            num_replicas_per_device=3,
        )

        # Get replicas multiple times to test round-robin
        replicas = []
        for i in range(6):
            replica = manager.get_replica("test_op")
            self.assertIsNotNone(replica)
            replicas.append(id(replica))

        # Should cycle through 3 replicas repeatedly
        self.assertEqual(replicas[0], replicas[3])  # First and fourth should be same
        self.assertEqual(replicas[1], replicas[4])  # Second and fifth should be same
        self.assertEqual(replicas[2], replicas[5])  # Third and sixth should be same

    def test_add_remove_replica(self):
        """Test adding and removing replicas."""
        config = ReplicaConfig(replica_devices=[0])
        manager = ReplicaManager(config)

        # Create initial replica
        manager.create_replicas(
            op_name="test_op",
            op_class=MockAttentionOp,
            weights_cpu={},
            devices=[0],
            num_replicas_per_device=1,
        )

        initial_count = len(manager.replicas["test_op"])

        # Add replica
        success = manager.add_replica("test_op", 0)
        self.assertTrue(success)
        self.assertEqual(len(manager.replicas["test_op"]), initial_count + 1)

        # Remove replica
        success = manager.remove_idle_replica("test_op")
        self.assertTrue(success)
        self.assertEqual(len(manager.replicas["test_op"]), initial_count)

    def test_shutdown(self):
        """Test manager shutdown."""
        config = ReplicaConfig(replica_devices=[0])
        manager = ReplicaManager(config)

        manager.create_replicas(
            op_name="test_op",
            op_class=MockAttentionOp,
            weights_cpu={},
            devices=[0],
        )

        # Shutdown should complete without errors
        manager.shutdown()
        self.assertEqual(len(manager.replicas), 0)


class TestReplicaAutoScaler(unittest.TestCase):
    """Test ReplicaAutoScaler functionality."""

    def test_auto_scaler_creation(self):
        """Test creating an auto scaler."""
        config = ReplicaConfig(
            auto_scaling_enabled=True,
            replica_devices=[0],
        )
        manager = ReplicaManager(config)

        self.assertIsNotNone(manager.auto_scaler)
        self.assertIsInstance(manager.auto_scaler, ReplicaAutoScaler)

    def test_queue_length_updates(self):
        """Test updating queue lengths for auto-scaling."""
        config = ReplicaConfig(auto_scaling_enabled=True)
        manager = ReplicaManager(config)

        # Update queue length
        manager.auto_scaler.update_queue_length("attention", 15)

        # Should be stored (though we can't easily test the scaling logic without mocking)


class TestDeadlockPrevention(unittest.TestCase):
    """Test deadlock prevention in replica operations."""

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

    def test_concurrent_access(self):
        """Test that concurrent replica access doesn't deadlock."""
        config = ReplicaConfig(replica_devices=[0])
        manager = ReplicaManager(config)

        manager.create_replicas(
            op_name="test_op",
            op_class=MockAttentionOp,
            weights_cpu={},
            devices=[0],
            num_replicas_per_device=2,
        )

        results = []
        errors = []

        def worker(worker_id):
            try:
                for i in range(10):
                    replica = manager.get_replica("test_op")
                    if replica:
                        q = torch.randn(1, 1, 64)
                        output, event = replica.forward_async(q, q, q)
                        event.synchronize()
                        results.append((worker_id, i))
                    time.sleep(0.001)  # Small delay to encourage interleaving
            except Exception as e:
                errors.append((worker_id, str(e)))

        # Start multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join(timeout=10)
            self.assertFalse(t.is_alive(), f"Thread {t} did not complete")

        # Check results
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        self.assertGreater(len(results), 0, "No results collected")


if __name__ == "__main__":
    unittest.main()


