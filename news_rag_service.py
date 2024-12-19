import logging
from logging.handlers import RotatingFileHandler
import os
from dotenv import load_dotenv
from typing import List, Dict
import asyncio
from openai import OpenAI
import tiktoken
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, Range
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Configuration
COLLECTION_NAME = "news_headlines"
EMBEDDING_MODEL = "text-embedding-3-small"

class NewsRAG:
    def __init__(self):
        # Setup logging
        self.setup_logging()

        # Initialize clients
        self.setup_clients()

        # System prompt template
        self.SYSTEM_PROMPT = """You are a helpful financial advisor assistant.
        Using the provided recent news context, help answer the user's question.
        Base your response on the news provided and clearly reference specific news items when relevant.
        If the provided news doesn't contain relevant information to answer the question, acknowledge this and provide general advice while being transparent about the limitations.

        Important guidelines:
        - Be clear and concise
        - Reference specific news items when relevant
        - Maintain professional tone
        - Clearly distinguish between facts from news and general advice
        - Include appropriate disclaimers when giving financial advice
        """

    def setup_logging(self):
        """Setup logging configuration"""
        if not os.path.exists('logs'):
            os.makedirs('logs')

        self.logger = logging.getLogger('NewsRAG')
        self.logger.setLevel(logging.DEBUG)

        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )

        file_handler = RotatingFileHandler(
            'logs/news_rag.log',
            maxBytes=10485760,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def setup_clients(self):
        """Initialize OpenAI and Qdrant clients"""
        try:
            self.openai_client = OpenAI()
            self.qdrant_client = QdrantClient(
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY")
            )
            self.logger.info("Successfully initialized clients")
        except Exception as e:
            self.logger.error(f"Error initializing clients: {str(e)}", exc_info=True)
            raise

    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        try:
            response = await asyncio.to_thread(
                self.openai_client.embeddings.create,
                model=EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"Error getting embedding: {str(e)}", exc_info=True)
            raise

    async def search_news(self, query_embedding: List[float], limit: int = 5,
                     time_filter: bool = False) -> List[Dict]:
        """Search for relevant news using the query embedding"""
        try:
            filter_conditions = []

            # Add time filter if requested (e.g., last 7 days)
            if time_filter:
                cutoff_date = datetime.now() - timedelta(days=7)
                filter_conditions.append(
                    FieldCondition(
                        key="timestamp",
                        range=Range(
                            gte=cutoff_date.timestamp()  # Convert to timestamp
                        )
                    )
                )

            search_results = self.qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_embedding,
                limit=limit,
                query_filter=Filter(
                    must=filter_conditions
                ) if filter_conditions else None
            )

            # Print search results with scores
            for hit in search_results:
                print(f"Score: {hit.score}, Payload: {hit.payload}")

            return [hit.payload for hit in search_results]
        except Exception as e:
            self.logger.error(f"Error searching news: {str(e)}", exc_info=True)
            return []
        
    async def check_collection_status(self):
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            print(f"Available collections: {collections}")
            
            # Get collection info
            collection_info = self.qdrant_client.get_collection(COLLECTION_NAME)
            print(f"Collection info: {collection_info}")
            
            # Get collection size
            collection_size = self.qdrant_client.count(
                collection_name=COLLECTION_NAME
            )
            print(f"Number of points in collection: {collection_size}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error checking collection: {str(e)}", exc_info=True)
            return False

    def format_news_context(self, news_items: List[Dict]) -> str:
        """Format news items into a context string"""
        context = "Recent relevant news:\\n\\n"
        for i, news in enumerate(news_items, 1):
            context += f"{i}. Headline: {news['headline']}\\n"
            context += f"   Summary: {news['summary']}\\n"
            context += f"   Source: {news['source']}\\n"
            context += f"   Date: {news['timestamp']}\\n\\n"
        return context

    def num_tokens_from_string(self, string: str, model: str = "gpt-4") -> int:
        """Returns the number of tokens in a text string"""
        encoding = tiktoken.encoding_for_model(model)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    async def get_rag_response(self, user_query: str,
                             max_news_items: int = 5,
                             temperature: float = 0.7) -> Dict:
        """Generate a response using RAG methodology"""
        try:
            # Get query embedding
            query_embedding = await self.get_embedding(user_query)

            # Search for relevant news
            relevant_news = await self.search_news(
                query_embedding,
                limit=max_news_items
            )

            if not relevant_news:
                return {
                    "response": "I apologize, but I couldn't find any recent relevant news to help answer your question. Would you like me to provide general advice instead?",
                    "news_items": []
                }

            # Format context
            context = self.format_news_context(relevant_news)

            # Prepare messages
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"""Context: {context}\\n\\nUser Question: {user_query}

                Please provide a helpful response based on the context provided."""}
            ]

            # Check token count and truncate if necessary
            total_tokens = sum(self.num_tokens_from_string(msg["content"]) for msg in messages)
            if total_tokens > 6000:  # Leave room for response
                self.logger.warning("Context too long, truncating...")
                relevant_news = relevant_news[:3]
                context = self.format_news_context(relevant_news)
                messages[1]["content"] = f"""Context: {context}\\n\\nUser Question: {user_query}

                Please provide a helpful response based on the context provided."""

            # Generate response
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4o",
                messages=messages,
                temperature=temperature,
                max_tokens=1000
            )

            return {
                "response": response.choices[0].message.content,
                "news_items": relevant_news
            }

        except Exception as e:
            self.logger.error(f"Error generating RAG response: {str(e)}", exc_info=True)
            return {
                "response": "I apologize, but I encountered an error while processing your request. Please try again later.",
                "news_items": []
            }

# FastAPI implementation
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()
rag = NewsRAG()

class Query(BaseModel):
    text: str
    max_news_items: int = 5
    temperature: float = 0.7

@app.post("/ask")
async def ask_question(query: Query):
    try:
        result = await rag.get_rag_response(
            query.text,
            max_news_items=query.max_news_items,
            temperature=query.temperature
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Example usage
async def main():
    rag = NewsRAG()

    # Check collection status first
    await rag.check_collection_status()

    queries = [
        "Should I invest in Conagra Brands?",
        "What's the latest news about Tesla's stock performance?",
        "Which is a good stock to invest in right now?",
        # "Should I invest in AI companies right now?",
        # "What are the current trends in cryptocurrency markets?",
    ]

    for query in queries:
        print(f"\\nQuery: {query}")
        result = await rag.get_rag_response(query)
        print(f"Response: {result['response']}")
        print(f"Number of relevant news items: {len(result['news_items'])}")
        print("-" * 80)

if __name__ == "__main__":
    asyncio.run(main())
