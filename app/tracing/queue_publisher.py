"""RabbitMQ publisher for completed trajectory notifications."""

from __future__ import annotations

import json
import os
from typing import Any

import pika


class TrajectoryQueuePublisher:
    def __init__(self, url: str | None = None, queue_name: str = "trajectory_ready") -> None:
        self.url = url or os.environ["RABBITMQ_URL"]
        self.queue_name = queue_name

    def publish(self, trace_id: str, task: str, status: str = "completed", extra: dict[str, Any] | None = None) -> None:
        payload = {"trace_id": trace_id, "task": task, "status": status, **(extra or {})}
        connection = pika.BlockingConnection(pika.URLParameters(self.url))
        try:
            channel = connection.channel()
            channel.queue_declare(queue=self.queue_name, durable=True)
            channel.basic_publish(
                exchange="",
                routing_key=self.queue_name,
                body=json.dumps(payload).encode("utf-8"),
                properties=pika.BasicProperties(delivery_mode=2),
            )
        finally:
            connection.close()
