# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from aiperf.common.enums import URLSelectionStrategy
from aiperf.common.factories import URLSamplingStrategyFactory
from aiperf.timing.url_samplers import RoundRobinURLSampler


class TestRoundRobinURLSampler:
    """Unit tests for RoundRobinURLSampler."""

    def test_init_with_single_url(self):
        """Single URL should work without issues."""
        sampler = RoundRobinURLSampler(urls=["http://localhost:8000"])
        assert sampler.urls == ["http://localhost:8000"]

    def test_init_with_multiple_urls(self):
        """Multiple URLs should be stored correctly."""
        urls = ["http://server1:8000", "http://server2:8000", "http://server3:8000"]
        sampler = RoundRobinURLSampler(urls=urls)
        assert sampler.urls == urls

    def test_init_with_empty_urls_raises(self):
        """Empty URLs list should raise ValueError."""
        with pytest.raises(ValueError, match="URLs list cannot be empty"):
            RoundRobinURLSampler(urls=[])

    def test_next_url_index_single_url(self):
        """Single URL should always return index 0."""
        sampler = RoundRobinURLSampler(urls=["http://localhost:8000"])
        for _ in range(10):
            assert sampler.next_url_index() == 0

    def test_next_url_index_round_robin_order(self):
        """Multiple URLs should cycle in round-robin order."""
        urls = ["http://server1:8000", "http://server2:8000", "http://server3:8000"]
        sampler = RoundRobinURLSampler(urls=urls)

        # First cycle
        assert sampler.next_url_index() == 0
        assert sampler.next_url_index() == 1
        assert sampler.next_url_index() == 2

        # Second cycle (wrap-around)
        assert sampler.next_url_index() == 0
        assert sampler.next_url_index() == 1
        assert sampler.next_url_index() == 2

    def test_next_url_index_wrap_around(self):
        """Index should wrap around correctly at list boundary."""
        urls = ["http://server1:8000", "http://server2:8000"]
        sampler = RoundRobinURLSampler(urls=urls)

        # Collect many indices
        indices = [sampler.next_url_index() for _ in range(10)]

        # Should alternate between 0 and 1
        expected = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        assert indices == expected

    def test_next_url_index_thread_safety(self):
        """Thread-safe access to next_url_index under concurrent access."""
        urls = ["http://server1:8000", "http://server2:8000", "http://server3:8000"]
        sampler = RoundRobinURLSampler(urls=urls)

        results = []
        lock = threading.Lock()
        num_threads = 10
        calls_per_thread = 1000

        def worker():
            thread_results = []
            for _ in range(calls_per_thread):
                idx = sampler.next_url_index()
                thread_results.append(idx)
            with lock:
                results.extend(thread_results)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker) for _ in range(num_threads)]
            for f in futures:
                f.result()

        # All indices should be valid (0, 1, or 2)
        assert all(0 <= idx < 3 for idx in results)

        # Total calls should be correct
        assert len(results) == num_threads * calls_per_thread

        # Distribution should be approximately even (within 10% tolerance)
        count_0 = results.count(0)
        count_1 = results.count(1)
        count_2 = results.count(2)
        expected_per_url = (num_threads * calls_per_thread) / 3
        tolerance = expected_per_url * 0.1

        assert abs(count_0 - expected_per_url) < tolerance
        assert abs(count_1 - expected_per_url) < tolerance
        assert abs(count_2 - expected_per_url) < tolerance

    def test_kwargs_ignored(self):
        """Extra kwargs should be ignored for factory compatibility."""
        sampler = RoundRobinURLSampler(
            urls=["http://localhost:8000"], extra_arg=True, another_arg="test"
        )
        assert sampler.urls == ["http://localhost:8000"]


class TestURLSamplingStrategyFactory:
    """Unit tests for URLSamplingStrategyFactory."""

    def test_factory_creates_round_robin(self):
        """Factory should create RoundRobinURLSampler for ROUND_ROBIN strategy."""
        urls = ["http://server1:8000", "http://server2:8000"]
        sampler = URLSamplingStrategyFactory.create_instance(
            URLSelectionStrategy.ROUND_ROBIN, urls=urls
        )
        assert isinstance(sampler, RoundRobinURLSampler)
        assert sampler.urls == urls

    def test_factory_creates_round_robin_by_string(self):
        """Factory should accept string strategy name."""
        urls = ["http://server1:8000", "http://server2:8000"]
        sampler = URLSamplingStrategyFactory.create_instance("round_robin", urls=urls)
        assert isinstance(sampler, RoundRobinURLSampler)
        assert sampler.urls == urls

    def test_factory_with_enum_value(self):
        """Factory should accept URLSelectionStrategy enum value."""
        urls = ["http://server1:8000"]
        sampler = URLSamplingStrategyFactory.create_instance(
            URLSelectionStrategy.ROUND_ROBIN, urls=urls
        )
        assert isinstance(sampler, RoundRobinURLSampler)
        assert sampler.urls == urls
