import logging
from logging.handlers import RotatingFileHandler
import sys

import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, List
import requests
import openai
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models

from bytewax.dataflow import Dataflow
from bytewax import operators as op
from bytewax.outputs import FixedPartitionedSink, StatefulSinkPartition
from bytewax.inputs import FixedPartitionedSource, StatefulSourcePartition, DynamicSource, StatelessSourcePartition

# Load environment variables
load_dotenv()

def setup_logging(log_level=logging.DEBUG):
    """Configure logging for both file and console output"""

    # Create logs directory if it doesn't exist
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create logger
    logger = logging.getLogger('news_ingestion')
    logger.setLevel(log_level)

    # Clear any existing handlers
    logger.handlers = []

    # Formatting
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - '
        '[%(filename)s:%(lineno)d] - %(message)s'
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler with rotation
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, 'news_ingestion_bytewax.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# Initialize logger
logger = setup_logging()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

COLLECTION_NAME = "news_headlines_debug_bytewax"
EMBEDDING_MODEL = "text-embedding-ada-002"

openai.api_key = OPENAI_API_KEY

# Initialize Qdrant collection
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
try:
    qdrant.get_collection(COLLECTION_NAME)
except:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=1536,
            distance=models.Distance.COSINE
        )
    )

# --- Alpaca News Source Implementation ---

class AlpacaNewsPartition(StatefulSourcePartition[Dict, Dict]):
    """A stateful partition that fetches Alpaca news since the last processed timestamp."""

    def __init__(self, api_key: str, secret_key: str, resume_state: Optional[Dict]):
        logger.info("Initializing AlpacaNewsPartition")
        self.api_key = api_key
        self.secret_key = secret_key
        self.session = requests.Session()
        # Add headers initialization here
        self.headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.secret_key
        }
        self.buffer = []
        self.poll_interval = timedelta(seconds=60)  # Poll every 60s
        logger.info(f"Initialized AlpacaNewsPartition with poll interval: {self.poll_interval}")
        self.last_awake = datetime.now(timezone.utc)

        if resume_state and "last_processed_timestamp" in resume_state:
            self.last_processed_timestamp = datetime.fromisoformat(resume_state["last_processed_timestamp"])
            logger.info(f"Resuming from timestamp: {self.last_processed_timestamp}")
        else:
            # Start from None (fetch latest)
            logger.info("Starting from latest (no resume state)")
            self.last_processed_timestamp = None

    def next_batch(self) -> List[Dict]:
        try:
            logger.debug("Starting next_batch poll cycle")

            if self.buffer:
                batch, self.buffer = self.buffer, []
                logger.debug(f"Returning buffered batch of {len(batch)} items")
                return batch

            params = {}
            if self.last_processed_timestamp is not None:
                params['since'] = self.last_processed_timestamp.isoformat()
                logger.debug(f"Polling for news since: {self.last_processed_timestamp}")
            else:
                logger.debug("Polling for latest news (no timestamp filter)")

            url = "https://data.alpaca.markets/v1beta1/news"
            logger.debug(f"Making API request to: {url}")

            resp = self.session.get(url, headers=self.headers, params=params)
            if resp.status_code != 200:
                logger.error(f"Failed to fetch news: status={resp.status_code}, response={resp.text}")
                return []

            data = resp.json()
            news = data.get('news', [])

            if news:
                logger.info(f"Fetched {len(news)} new articles")
                self.last_processed_timestamp = datetime.fromisoformat(
                    news[0]['created_at'].replace('Z', '+00:00')
                )
                logger.debug(f"Updated last_processed_timestamp to: {self.last_processed_timestamp}")
            else:
                logger.debug("Poll completed: No new articles found")

            return news

        except Exception as e:
            logger.exception("Error in next_batch")
            return []

    def next_awake(self) -> Optional[datetime]:
        now = datetime.now(timezone.utc)
        if now < self.last_awake + self.poll_interval:
            next_wake = self.last_awake + self.poll_interval
            logger.debug(f"Sleeping until next poll at: {next_wake}")
            return next_wake

        logger.debug("Poll interval elapsed, proceeding with next poll")
        self.last_awake = now
        return None

    def snapshot(self) -> Dict:
        state = {}
        if self.last_processed_timestamp is not None:
            state["last_processed_timestamp"] = self.last_processed_timestamp.isoformat()
        return state

    def close(self):
        self.session.close()


class AlpacaNewsSource(FixedPartitionedSource[Dict, Dict]):
    def list_parts(self) -> List[str]:
        # Single partition
        return ["alpaca_news_part"]

    def build_part(
        self,
        step_id: str,
        for_part: str,
        resume_state: Optional[Dict],
    ) -> StatefulSourcePartition[Dict, Dict]:
        return AlpacaNewsPartition(ALPACA_API_KEY, ALPACA_SECRET_KEY, resume_state)


# --- Cleanup Trigger Source ---
class CleanupTriggerPartition(StatelessSourcePartition[Dict]):
    """Emit a cleanup event every hour."""
    def __init__(self):
        self.interval = timedelta(hours=1)
        self.last_trigger = datetime.now(timezone.utc)

    def next_batch(self) -> List[Dict]:
        now = datetime.now(timezone.utc)
        if (now - self.last_trigger) >= self.interval:
            self.last_trigger = now
            return [{"type": "cleanup_trigger", "timestamp": now.isoformat()}]
        return []

    def next_awake(self) -> Optional[datetime]:
        # Wake up when next trigger is due
        return self.last_trigger + self.interval

class CleanupTriggerSource(DynamicSource[Dict]):
    def build(self, step_id: str, worker_index: int, worker_count: int) -> StatelessSourcePartition[Dict]:
        # Let only the first worker emit cleanup events, others no-op
        if worker_index == 0:
            return CleanupTriggerPartition()
        else:
            # A partition that never emits
            class EmptyPartition(StatelessSourcePartition[Dict]):
                def next_batch(self) -> List[Dict]:
                    return []
            return EmptyPartition()


# --- Qdrant Sink ---
class QdrantSink(StatefulSinkPartition):
    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self.batch = []
        self.batch_size = 10  # Reduced batch size for more frequent writes
        self.last_flush = datetime.now()
        self.flush_interval = timedelta(seconds=5)  # Force flush every 5 seconds
        logger.info("Initialized QdrantSink")

    def write_batch(self, items):
        logger.info(f"Adding {len(items)} items to batch (current batch size: {len(self.batch)})")
        self.batch.extend(items)

        # Flush if batch size reached or time interval exceeded
        if (len(self.batch) >= self.batch_size or
            datetime.now() - self.last_flush > self.flush_interval):
            self.flush()

    def flush(self):
        if self.batch:
            logger.info(f"Flushing batch of {len(self.batch)} items to Qdrant")
            try:
                self.client.upsert(collection_name=COLLECTION_NAME, points=self.batch)
                logger.info("Successfully flushed batch to Qdrant")
                self.batch.clear()
                self.last_flush = datetime.now()
            except Exception as e:
                logger.exception("Error flushing batch to Qdrant")
                raise

    def snapshot(self) -> dict:
        """
        Return the current state of the sink that needs to be preserved.
        This method is required for StatefulSinkPartition.
        """
        return {
            "last_flush": self.last_flush.isoformat(),
            "batch_size": len(self.batch)
        }

    def close(self):
        """
        Clean up resources when the sink is closed.
        """
        self.flush()  # Flush any remaining items
        self.client.close()


class QdrantOutput(FixedPartitionedSink):
    def list_parts(self):
        return ["single"]

    def build_part(self, step_id, for_part, resume_state):
        return QdrantSink()


# --- Embedding and Processing ---
def get_embedding(text: str) -> List[float]:
    try:
        logger.info(f"Getting embedding for text: {text[:100]}...")
        response = openai.Embedding.create(
            model=EMBEDDING_MODEL,
            input=[text]
        )
        logger.info("Successfully obtained embedding")
        return response["data"][0]["embedding"]
    except Exception as e:
        logger.exception("Error getting embedding")
        raise

def prepare_text(item: Dict):
    text = f"Headline: {item['headline']} Summary: {item['summary']}"
    return item, text

def to_qdrant_point(item_with_embed):
    original_item, embedding = item_with_embed
    logger.debug(f"Converting item to Qdrant point: {original_item['headline']}")
    created_at = datetime.fromisoformat(original_item['created_at'].replace('Z', '+00:00'))
    point_id = int(original_item['id'])
    payload = {
        "headline": original_item['headline'],
        "summary": original_item['summary'],
        "source": original_item.get('source', ''),
        "symbols": original_item.get('symbols', []),
        "timestamp": created_at,
        "author": original_item.get('author', ''),
        "url": original_item.get('url', ''),
        "created_at": created_at.timestamp()
    }
    logger.debug(f"Successfully created Qdrant point with ID: {point_id}")
    return {
        "id": point_id,
        "vector": embedding,
        "payload": payload
    }

def run_cleanup(_):
    try:
        days_to_keep = 7
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cutoff_timestamp = cutoff_date.timestamp()

        logger.info(f"Running cleanup for data older than {cutoff_date}")

        result = qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="created_at",
                            range=models.Range(lt=cutoff_timestamp)
                        )
                    ]
                )
            )
        )

        logger.info(f"Cleanup completed: {result}")
        return {"status": "cleanup_done", "cutoff": cutoff_date.isoformat()}
    except Exception as e:
        logger.exception("Error during cleanup")
        raise

# Build the Bytewax Dataflow
flow = Dataflow("news_ingestion_flow")

# News input
news_stream = op.input("alpaca_news_in", flow, AlpacaNewsSource())

def safe_prepare_text(item):
    try:
        return prepare_text(item)
    except Exception as e:
        logger.exception(f"Error preparing text for item: {item}")
        raise

def safe_to_qdrant_point(item):
    try:
        return to_qdrant_point(item)
    except Exception as e:
        logger.exception(f"Error converting to Qdrant point: {item}")
        raise

# Prepare text and compute embeddings
prepared = op.map("prepare_text", news_stream, safe_prepare_text)
embedded = op.map("compute_embedding", prepared, lambda x: (x[0], get_embedding(x[1])))
# after to_qdrant_point map
qdrant_points = op.map("to_qdrant_point", embedded, safe_to_qdrant_point)

# Add a key so the sink knows which partition to use
keyed_points = op.map("add_partition_key", qdrant_points, lambda p: ("single", p))

op.output("qdrant_output", keyed_points, QdrantOutput())

# Cleanup trigger input
cleanup_stream = op.input("cleanup_input", flow, CleanupTriggerSource())
cleanup_done = op.map("cleanup_map", cleanup_stream, run_cleanup)
op.inspect("cleanup_inspect", cleanup_done)

# Run:
# python -m bytewax.run news_ingestion_service_bytewax:flow
