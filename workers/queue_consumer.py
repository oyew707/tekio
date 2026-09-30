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
from rag.store import TipStore
from dotenv import load_dotenv
from pika import BlockingConnection, ConnectionParameters, exceptions
from utils.logger import get_logger

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
            print(f" [x] Received message for trace_id: {payload.get('trace_id')}")

            # 2. Process data
            processed_data = [payload]

            # 3. Write to Postgres
            rag_store.embed_and_upsert(
                content=processed_data, metadata={}, trace_id=payload.get("trace_id")
            )

            # Acknowledge the message to RabbitMQ
            ch.basic_ack(delivery_tag=method.delivery_tag)

        except json.JSONDecodeError:
            print(f" [!] Error decoding JSON: {body}")
            ch.basic_nack(delivery_tag=method.delivery_tag)  # Reject bad messages
        except Exception as e:
            print(f" [!] An error occurred during message processing: {e}")
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
            print("Error: RABBITMQ_URL environment variable not set.")
            return

        try:
            connection = BlockingConnection(
                ConnectionParameters(host=self.rabbitmq_url)
            )
            channel = connection.channel()

            # Declare the queue, ensuring it exists and is durable
            channel.queue_declare(queue=QUEUE_NAME, durable=True)

            print(
                f" [*] Waiting for messages on queue '{QUEUE_NAME}'. To exit press CTRL+C"
            )

            # Start consuming, using the process_message method as the callback
            channel.basic_consume(
                queue=QUEUE_NAME, on_message_callback=self.process_message
            )

            channel.start_consuming()

        except exceptions.AMQPConnectionError as e:
            print(f"Failed to connect to RabbitMQ: {e}")
        except KeyboardInterrupt:
            print("Consumer shutting down...")
        finally:
            if "connection" in locals() and connection.is_open:
                connection.close()


if __name__ == "__main__":
    consumer = QueueConsumer()
    consumer.start_consuming()
