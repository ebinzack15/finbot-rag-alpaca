import os
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, List, Any
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

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

COLLECTION_NAME = "news_headlines_debug"
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
        self.api_key = api_key
        self.secret_key = secret_key
        self.session = requests.Session()
        self.buffer = []
        self.poll_interval = timedelta(seconds=60)  # Poll every 60s
        self.last_awake = datetime.now(timezone.utc)

        if resume_state and "last_processed_timestamp" in resume_state:
            self.last_processed_timestamp = datetime.fromisoformat(resume_state["last_processed_timestamp"])
        else:
            # Start from None (fetch latest)
            self.last_processed_timestamp = None

    def next_batch(self) -> List[Dict]:
        # If we have a buffer of items, return them first
        if self.buffer:
            batch, self.buffer = self.buffer, []
            return batch

        # Fetch new news
        params = {}
        if self.last_processed_timestamp is not None:
            params['since'] = self.last_processed_timestamp.isoformat()

        url = "https://data.alpaca.markets/v1beta1/news"
        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key
        }

        resp = self.session.get(url, headers=headers, params=params)
        if resp.status_code != 200:
            # Could not fetch now, return empty and try again next time
            return []

        data = resp.json()
        news = data.get('news', [])

        if not news:
            # No new news
            return []

        # Update last_processed_timestamp
        # Alpaca returns newest first
        self.last_processed_timestamp = datetime.fromisoformat(news[0]['created_at'].replace('Z', '+00:00'))
        return news

    def next_awake(self) -> Optional[datetime]:
        # Only call next_batch once every poll_interval
        now = datetime.now(timezone.utc)
        if now < self.last_awake + self.poll_interval:
            return self.last_awake + self.poll_interval
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

    def write_batch(self, items):
        self.batch.extend(items)
        if len(self.batch) >= 50:
            self.flush()

    def flush(self):
        if self.batch:
            self.client.upsert(collection_name=COLLECTION_NAME, points=self.batch)
            self.batch.clear()

    def snapshot(self):
        return None

    def close(self):
        self.flush()


class QdrantOutput(FixedPartitionedSink):
    def list_parts(self):
        return ["single"]

    def build_part(self, step_id, for_part, resume_state):
        return QdrantSink()


# --- Embedding and Processing ---
def get_embedding(text: str) -> List[float]:
    response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=[text]
    )
    return response["data"][0]["embedding"]

def prepare_text(item: Dict):
    text = f"Headline: {item['headline']} Summary: {item['summary']}"
    return item, text

def to_qdrant_point(item_with_embed):
    original_item, embedding = item_with_embed
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
    return {
        "id": point_id,
        "vector": embedding,
        "payload": payload
    }

def run_cleanup(_):
    # Remove news older than 7 days
    days_to_keep = 7
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    cutoff_timestamp = cutoff_date.timestamp()
    qdrant.delete(
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
    return {"status": "cleanup_done", "cutoff": cutoff_date.isoformat()}

# Build the Bytewax Dataflow
flow = Dataflow("news_ingestion_flow")

# News input
news_stream = op.input("alpaca_news_in", flow, AlpacaNewsSource())

# Prepare text and compute embeddings
prepared = op.map("prepare_text", news_stream, prepare_text)
embedded = op.map("compute_embedding", prepared, lambda x: (x[0], get_embedding(x[1])))
# after to_qdrant_point map
qdrant_points = op.map("to_qdrant_point", embedded, to_qdrant_point)

# Add a key so the sink knows which partition to use
keyed_points = op.map("add_partition_key", qdrant_points, lambda p: ("single", p))

op.output("qdrant_output", keyed_points, QdrantOutput())

# Cleanup trigger input
cleanup_stream = op.input("cleanup_input", flow, CleanupTriggerSource())
cleanup_done = op.map("cleanup_map", cleanup_stream, run_cleanup)
op.inspect("cleanup_inspect", cleanup_done)

# Run:
# python -m bytewax.run news_ingestion_service_bytewax:flow
