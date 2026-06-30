from typing import List, Dict, Any, Optional

from rag.loader import PDFLoader
from rag.cleaner import TextCleaner
from rag.chunker import SemanticChunker
from rag.embeddings import EmbeddingGenerator
from rag.vectordb import VectorDatabase
from rag.retriever import Retriever
from rag.reranker import MMRReranker
from rag.prompts import PromptManager
from rag.llm import LLMManager
from utils.logger import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    def __init__(self):
        self.loader = PDFLoader()
        self.cleaner = TextCleaner()
        self.chunker = SemanticChunker()
        self.embedder = EmbeddingGenerator()
        self.vector_db = VectorDatabase()
        self.reranker = MMRReranker()
        self.retriever = Retriever(
            embedder=self.embedder,
            vector_db=self.vector_db,
            reranker=self.reranker,
        )
        self.prompts = PromptManager()
        self.llm = LLMManager()

    def ingest(self, file_paths: List[str]) -> dict:
        logger.info("Ingesting %d files", len(file_paths))
        pages = self.loader.load_all(file_paths)
        cleaned = self.cleaner.clean(pages)
        chunks = self.chunker.chunk(cleaned)
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_documents(texts)
        self.vector_db.add_chunks(chunks, embeddings=embeddings)
        stats = self.vector_db.get_collection_stats()
        logger.info("Ingestion complete: %d chunks stored", stats["total_chunks"])
        return {
            "pages": len(pages),
            "cleaned_pages": len(cleaned),
            "chunks": len(chunks),
            "total_chunks": stats["total_chunks"],
        }

    def query(self, question: str) -> Dict[str, Any]:
        logger.info("Processing query: %s", question[:80])
        chunks = self.retriever.retrieve(question)
        if not chunks:
            return {
                "answer": "No relevant information could be found in the uploaded documents.",
                "chunks": [],
                "sources": [],
            }

        context = self.prompts.format_chunks_for_context(chunks)
        user_prompt = self.prompts.format_user_prompt(
            retrieved_context=context,
            user_question=question,
        )
        system_prompt = self.prompts.get_system_prompt()

        answer = self.llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        sources = []
        seen = set()
        for c in chunks:
            doc_name = c["metadata"]["document_name"]
            if doc_name not in seen:
                seen.add(doc_name)
                sources.append({
                    "document_name": doc_name,
                    "page_numbers": list({
                        c2["metadata"]["page_number"]
                        for c2 in chunks
                        if c2["metadata"]["document_name"] == doc_name
                    }),
                    "chunk_ids": [
                        c2["metadata"]["chunk_id"]
                        for c2 in chunks
                        if c2["metadata"]["document_name"] == doc_name
                    ],
                })

        return {
            "answer": answer,
            "chunks": chunks,
            "sources": sources,
        }

    def clear_all(self):
        self.vector_db.delete_collection()
        logger.info("Vector database cleared")

    def delete_document(self, document_name: str):
        self.vector_db.delete_document(document_name)
        logger.info("Document deleted: %s", document_name)
