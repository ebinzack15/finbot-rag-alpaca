import logging
from logging.handlers import RotatingFileHandler
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from openai import OpenAI
import asyncio
import aiohttp
from itertools import islice

# Load environment variables from .env file
load_dotenv()

# Configuration
COLLECTION_NAME = "news_headlines"
BATCH_SIZE = 50  # Number of items to process in one batch
EMBEDDING_MODEL = "text-embedding-3-small"  # or "text-embedding-3-large" for better quality

# Environment variables
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

class NewsVectorDB:
    def __init__(self):
        self.last_processed_id = None
        self.last_processed_timestamp = None
        # Setup logging
        self.setup_logging()

        self.logger.info("Initializing NewsVectorDB")
        # Initialize OpenAI client
        self.openai_client = OpenAI()

        # Initialize Qdrant client
        self.qdrant = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )

        # Create collection if it doesn't exist
        self.setup_collection()

    def setup_logging(self):
        """Setup logging configuration"""
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs')

        # Create logger
        self.logger = logging.getLogger('NewsVectorDB')
        self.logger.setLevel(logging.DEBUG)

        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )

        # File handler (rotating file handler to manage log size)
        file_handler = RotatingFileHandler(
            'logs/news_vector_db.log',
            maxBytes=10485760,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)

        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def setup_collection(self):
        """Setup Qdrant collection with appropriate parameters"""
        try:
            self.qdrant.get_collection(COLLECTION_NAME)
            self.logger.info(f"Collection {COLLECTION_NAME} already exists")
        except Exception as e:
            self.logger.info(f"Creating new collection: {COLLECTION_NAME}")
            try:
                self.qdrant.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=1536,
                        distance=Distance.COSINE
                    )
                )

                # Create payload indices
                self.qdrant.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name="timestamp",
                    field_schema="datetime"
                )
                self.qdrant.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name="symbols",
                    field_schema="keyword"
                )
                self.logger.info("Collection and indices created successfully")
            except Exception as e:
                self.logger.error(f"Error creating collection: {str(e)}", exc_info=True)
                raise

    async def fetch_news(self) -> List[Dict]:
        """Fetch news with pagination and since parameter"""
        try:
            params = {}
            if self.last_processed_timestamp:
                params['since'] = self.last_processed_timestamp.isoformat()

            async with aiohttp.ClientSession() as session:
                headers = {
                    "APCA-API-KEY-ID": ALPACA_API_KEY,
                    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
                }

                async with session.get(
                    "https://data.alpaca.markets/v1beta1/news",
                    headers=headers,
                    params=params
                ) as response:
                    data = await response.json()
                    news = data.get('news', [])

                    if news:
                        # Update last processed timestamp
                        self.last_processed_timestamp = datetime.fromisoformat(news[0]['created_at'].replace('Z', '+00:00'))

                    return news
        except Exception as e:
            self.logger.error(f"Error fetching news: {str(e)}", exc_info=True)
            return []

    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts"""
        self.logger.debug(f"Getting embeddings for batch of {len(texts)} texts")
        try:
            response = await asyncio.to_thread(
                self.openai_client.embeddings.create,
                model=EMBEDDING_MODEL,
                input=texts
            )
            self.logger.debug("Successfully obtained embeddings")
            return [item.embedding for item in response.data]
        except Exception as e:
            self.logger.error(f"Error getting embeddings: {str(e)}", exc_info=True)
            return []

    def batch_generator(self, items, batch_size):
        """Generate batches from items"""
        iterator = iter(items)
        while batch := list(islice(iterator, batch_size)):
            yield batch

    async def process_and_store_news_batch(self, news_items: List[Dict]):
        """Process and store news items in batches"""
        self.logger.info(f"Processing batch of {len(news_items)} news items")
        for batch in self.batch_generator(news_items, BATCH_SIZE):
            self.logger.debug(f"Processing sub-batch of {len(batch)} items")
            # Prepare texts and metadata for the batch
            texts = []
            points = []

            for item in batch:
                text = f"Headline: {item['headline']} Summary: {item['summary']}"
                texts.append(text)

                # Convert timestamp string to datetime if needed
                if isinstance(item['created_at'], str):
                    created_at = datetime.strptime(item['created_at'], "%Y-%m-%dT%H:%M:%SZ").timestamp()
                else:
                    created_at = item['created_at']

                metadata = {
                    "headline": item['headline'],
                    "summary": item['summary'],
                    "source": item.get('source', ''),
                    "symbols": item.get('symbols', []),
                    "timestamp": datetime.fromtimestamp(created_at),
                    "author": item.get('author', ''),
                    "url": item.get('url', ''),
                    "created_at": created_at
                }

                # Use the news item's ID directly, or generate a unique integer ID
                point_id = int(item['id'])  # Alpaca provides an ID field
                points.append((point_id, metadata))

            # Get embeddings for the batch
            embeddings = await self.get_embeddings_batch(texts)

            if embeddings:
                # Prepare points for Qdrant
                qdrant_points = [
                    models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=metadata
                    )
                    for (point_id, metadata), embedding in zip(points, embeddings)
                ]

                try:
                    # Store in Qdrant
                    self.qdrant.upsert(
                        collection_name=COLLECTION_NAME,
                        points=qdrant_points
                    )
                    self.logger.info(f"Successfully stored {len(qdrant_points)} points in Qdrant")
                except Exception as e:
                    self.logger.error(f"Error storing points in Qdrant: {str(e)}", exc_info=True)

    async def cleanup_old_news(self, days_to_keep: int = 7):
        """Remove news older than specified days"""
        self.logger.info(f"Starting cleanup of news older than {days_to_keep} days")
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)

            await asyncio.to_thread(
                self.qdrant.delete,
                collection_name=COLLECTION_NAME,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="timestamp",
                                range=models.Range(lt=cutoff_date)
                            )
                        ]
                    )
                )
            )
            self.logger.info(f"Successfully cleaned up news older than {cutoff_date}")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}", exc_info=True)

    async def run_stream(self, interval_seconds: int = 300):  # 5 minutes
        """Run continuous news streaming"""
        self.logger.info("Starting news streaming service")
        while True:
            try:
                self.logger.debug("Starting new iteration of news fetch")
                # Fetch news
                news_items = await self.fetch_news()

                if news_items:
                    self.logger.info(f"Processing {len(news_items)} news items")
                    # Process and store news in batches
                    await self.process_and_store_news_batch(news_items)

                    # Cleanup old news
                    await self.cleanup_old_news()

                    self.logger.info(f"Completed processing at {datetime.now()}")
                else:
                    self.logger.info("No news items received")

                self.logger.debug(f"Waiting {interval_seconds} seconds before next fetch")
                # Wait for next iteration
                await asyncio.sleep(interval_seconds)

            except Exception as e:
                self.logger.error(f"Error in run_stream: {str(e)}", exc_info=True)
                await asyncio.sleep(60)  # Wait a minute before retrying

    def get_recent_news(self, hours: int = 120) -> List[Dict]:
        """Retrieve recent news from the database"""
        self.logger.debug(f"Retrieving news from last {hours} hours")
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            response = self.qdrant.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="timestamp",
                            range=models.Range(gte=cutoff_time)
                        )
                    ]
                ),
                limit=100
            )
            self.logger.info(f"Retrieved {len(response[0])} recent news items")
            return [point.payload for point in response[0]]
        except Exception as e:
            self.logger.error(f"Error retrieving recent news: {str(e)}", exc_info=True)
            return []

# Usage
if __name__ == "__main__":
    news_db = NewsVectorDB()

    # Run the async stream
    asyncio.run(news_db.run_stream())
