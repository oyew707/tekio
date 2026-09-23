"""
-------------------------------------------------------
[Program Description]
-------------------------------------------------------
Author:  Einstein Oyewole
Email:   eo2233@nyu.edu
-------------------------------------------------------
"""

# Imports
import json
import os
from typing import Any
import pika

# Constants


class TrajectoryQueuePublisher:
    """
    -------------------------------------------------------
    A publisher class used to send messages to a RabbitMQ queue.
    -------------------------------------------------------
    """

    def __init__(
        self, url: str | None = None, queue_name: str = "trajectory_ready"
    ) -> None:
        """
        -------------------------------------------------------
        Initializes the publisher with connection details and target queue.
        -------------------------------------------------------
        Parameters:
            url - The RabbitMQ connection URL. If None, retrieves from RABBITMQ_URL environment variable. (str | None)
            queue_name - The name of the RabbitMQ queue to publish to. (str)
        -------------------------------------------------------
        """
        self.url = url or os.environ["RABBITMQ_URL"]
        self.queue_name = queue_name

    def publish(
        self,
        trace_id: str,
        task: str,
        status: str = "completed",
        extra: dict[str, Any] | None = None,
    ) -> None:
        """
        -------------------------------------------------------
        Publishes a JSON-encoded payload containing task information to the configured queue.
        -------------------------------------------------------
        Parameters:
            trace_id - Unique identifier for the trajectory trace. (str)
            task - Name or identifier of the task being reported. (str)
            status - The current status of the task (e.g., 'completed', 'failed'). (str)
            extra - Optional dictionary containing additional metadata to include in the payload. (dict[str, Any] | None)
        -------------------------------------------------------
        """
        payload = {
            "trace_id": trace_id,
            "task": task,
            "status": status,
            **(extra or {}),
        }
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
