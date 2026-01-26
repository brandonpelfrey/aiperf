# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""URL sampling strategies for multi-URL load balancing.

Provides URL selection strategies for distributing requests across multiple
endpoint URLs when multiple `--url` values are provided.
"""

import threading

from aiperf.common.enums import URLSelectionStrategy
from aiperf.common.factories import URLSamplingStrategyFactory


@URLSamplingStrategyFactory.register(URLSelectionStrategy.ROUND_ROBIN)
class RoundRobinURLSampler:
    """Round-robin URL sampler for even distribution across endpoints.

    Distributes requests evenly across URLs in sequential order. Each call to
    `next_url_index()` returns the next URL index in the list, wrapping around
    when the end is reached. Thread-safe for concurrent access.
    """

    def __init__(self, urls: list[str], **kwargs) -> None:
        """Initialize with list of URLs.

        Args:
            urls: List of endpoint URLs to distribute across.
            **kwargs: Ignored additional arguments for factory compatibility.

        Raises:
            ValueError: If urls list is empty.
        """
        if not urls:
            raise ValueError("URLs list cannot be empty")
        self._urls = urls
        self._index = 0
        self._lock = threading.Lock()

    @property
    def urls(self) -> list[str]:
        """The list of URLs being sampled."""
        return self._urls

    def next_url_index(self) -> int:
        """Return the index of the next URL in round-robin order.

        Thread-safe implementation using a lock to ensure atomic
        read-increment-modulo operation.

        Returns:
            Index into the urls list (0 to len(urls)-1).
        """
        with self._lock:
            index = self._index
            self._index = (self._index + 1) % len(self._urls)
            return index
