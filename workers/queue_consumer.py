"""
-------------------------------------------------------
consume messages from a RabbitMQ queue.
-------------------------------------------------------
Author:  Einstein Oyewole
Email:   eo2233@nyu.edu
-------------------------------------------------------
"""

# Imports
import json
import os
from rag.store import TipStore
from dotenv import load_dotenv
from pika import BlockingConnection, exceptions, URLParameters
from utils.logger import get_logger
from .trajectory_extractor import parse_thoughts, outcome, format_analysis_extraction
from .storage import consolidate_tips
from .extract_tips import extract_structured_tips

# Constants
load_dotenv()
logger = get_logger(__name__, "info")
QUEUE_NAME = os.environ.get("QUEUE_NAME", "trajectory_ready")
RABBITMQ_URL = os.environ.get("RABBITMQ_URL")
rag_store: TipStore | None = TipStore()


class QueueConsumer:
    """
    -------------------------------------------------------
    A consumer class used to listen to RabbitMQ and process messages.
    -------------------------------------------------------
    """

    def __init__(self):
        """
        -------------------------------------------------------
        Initializes the consumer, setting up connection details.
        -------------------------------------------------------
        """
        # RabbitMQ Setup
        self.rabbitmq_url = (
            RABBITMQ_URL if RABBITMQ_URL else os.environ.get("RABBITMQ_URL")
        )

    def process_message(self, ch, method, properties, body):
        """
        -------------------------------------------------------
        Callback function executed when a message is received from the queue.
        -------------------------------------------------------
        Parameters:
            ch - parameter description (parameter type and constraints)
            method - parameter description (parameter type and constraints)
            properties - parameter description (parameter type and constraints)
            body - parameter description (parameter type and constraints)
        -------------------------------------------------------
        """
        try:
            # 1. Decode the message
            payload = json.loads(body.decode("utf-8"))
            logger.info(
                f" [x] Received message for trace_id: {payload.get('trace_id')}"
            )

            # 2. Trajectory Analysis
            steps, trajectory = parse_thoughts(payload.get("trace_id"))
            analysis = outcome(steps)

            # 3. Extract tips
            tips = extract_structured_tips(
                "general",
                format_analysis_extraction(analysis=analysis, trajectory=trajectory),
            )

            # 4. Write to Postgres
            for tip in tips:
                rag_store.embed_and_upsert(
                    content=tip.content,
                    metadata=tip.model_dump(exclude={"content"}),
                    trace_id=payload.get("trace_id"),
                )

            # 5. Consolidate the tips
            consolidate_tips(rag_store)

            # Acknowledge the message to RabbitMQ
            ch.basic_ack(delivery_tag=method.delivery_tag)

        except json.JSONDecodeError:
            logger.error(f" [!] Error decoding JSON: {body}")
            ch.basic_nack(delivery_tag=method.delivery_tag)  # Reject bad messages
        except Exception as e:
            logger.error(f" [!] An error occurred during message processing: {e}")
            ch.basic_nack(
                delivery_tag=method.delivery_tag
            )  # Reject and potentially re-queue

    def start_consuming(self):
        """
        -------------------------------------------------------
        Connect to RabbitMQ and start listening for messages.
        -------------------------------------------------------
        """
        if not self.rabbitmq_url:
            logger.warning("Error: RABBITMQ_URL environment variable not set.")
            return

        try:
            logger.info(f"Beginning to consume {self.rabbitmq_url}")
            connection = BlockingConnection(URLParameters(self.rabbitmq_url))
            channel = connection.channel()

            # Declare the queue, ensuring it exists and is durable
            channel.queue_declare(queue=QUEUE_NAME, durable=True)

            logger.info(
                f" [*] Waiting for messages on queue '{QUEUE_NAME}'. To exit press CTRL+C"
            )

            # Start consuming, using the process_message method as the callback
            channel.basic_consume(
                queue=QUEUE_NAME, on_message_callback=self.process_message
            )

            channel.start_consuming()

        except exceptions.AMQPConnectionError as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
        except KeyboardInterrupt:
            logger.info("Consumer shutting down...")
        finally:
            logger.info("Closing Connection")
            if "connection" in locals() and connection.is_open:
                connection.close()


if __name__ == "__main__":
    consumer = QueueConsumer()
    consumer.start_consuming()
