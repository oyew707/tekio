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
import time
import psycopg2
from dotenv import load_dotenv
from pika import BlockingConnection, ConnectionParameters
from utils.logger import get_logger

# Constants
load_dotenv()
logger = get_logger(__name__, "info")
QUEUE_NAME = os.environ.get("QUEUE_NAME", "trajectory_ready")
RABBITMQ_URL = os.environ.get("RABBITMQ_URL")


class QueueConsumer:
    """
    A consumer class used to listen to RabbitMQ and process messages.
    """

    def __init__(self):
        """
        Initializes the consumer, setting up connection details.
        """
        # RabbitMQ Setup
        self.rabbitmq_url = (
            RABBITMQ_URL if RABBITMQ_URL else os.environ.get("RABBITMQ_URL")
        )

        # PostgreSQL Setup (Using service names from docker-compose.yml)
        self.db_host = os.environ.get("POSTGRES_HOST", "postgres")
        self.db_name = os.environ.get("POSTGRES_DB", "tekio")
        self.db_user = os.environ.get("POSTGRES_USER", "tekio")
        self.db_password = os.environ.get("POSTGRES_PASSWORD", "tekio")
        self.db_port = 5432

    def _save_to_postgres(self, data: dict[str, Any]) -> None:
        """
        Writes the processed data to the PostgreSQL database.

        NOTE: This method needs to be updated with your specific SQL logic
        and data mapping.
        """
        try:
            with psycopg2.connect(
                host=self.db_host,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password,
                port=self.db_port,
            ) as conn:
                with conn.cursor() as cur:
                    # Placeholder for insertion logic. Replace with your actual SQL.
                    print(
                        f"Processing data for trace_id: {data.get('trace_id')}. Saving to DB..."
                    )

                    # Example: Insert data into a 'trajectories' table
                    # cur.execute(
                    #     "INSERT INTO trajectories (trace_id, task, status, metadata) VALUES (%s, %s, %s, %s)",
                    #     (data.get('trace_id'), data.get('task'), data.get('status'), json.dumps(data.get('extra')))
                    # )
                    conn.commit()
                    print(
                        f"Successfully saved data for trace_id: {data.get('trace_id')}"
                    )

        except psycopg2.Error as e:
            print(f"Database error occurred: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during DB save: {e}")

    def process_message(self, ch, method, properties, body):
        """
        Callback function executed when a message is received from the queue.
        """
        try:
            # 1. Decode the message
            payload = json.loads(body.decode("utf-8"))
            print(f" [x] Received message for trace_id: {payload.get('trace_id')}")

            # 2. Process the data (You can add complex logic here)
            # For demonstration, we just pass the raw payload.
            processed_data = payload

            # 3. Write to Postgres
            self._save_to_postgres(processed_data)

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
        Connect to RabbitMQ and start listening for messages.
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

        except pika.exceptions.AMQPConnectionError as e:
            print(f"Failed to connect to RabbitMQ: {e}")
        except KeyboardInterrupt:
            print("Consumer shutting down...")
        finally:
            if "connection" in locals() and connection.is_open:
                connection.close()


if __name__ == "__main__":
    consumer = QueueConsumer()
    consumer.start_consuming()
