import os
import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")

    # Look for ChromaDB directories
    # TODO: Create list of directories that match specific criteria (directory type and name pattern)
    chroma_dirs = [dir for dir in current_dir.iterdir() if dir.is_dir() and dir.name.startswith("chroma_db_")]

    # TODO: Loop through each discovered directory
    for chroma_dir in chroma_dirs:
        # TODO: Wrap connection attempt in try-except block for error handling
        try:
            # TODO: Initialize database client with directory path and configuration settings
            client = chromadb.PersistentClient(path=str(chroma_dir))

            # TODO: Retrieve list of available collections from the database
            collections = client.list_collections()

            # TODO: Loop through each collection found
            for collection in collections:
                # TODO: Create unique identifier key combining directory and collection names
                key = f"{chroma_dir.name}_{collection.name}".replace(" ", "_").lower()

                # TODO: Build information dictionary containing:
                    # TODO: Store directory path as string
                    # TODO: Store collection name
                    # TODO: Create user-friendly display name
                # TODO: Add collection information to backends dictionary
                backends[key] = {
                    "directory": str(chroma_dir),
                    "collection_name": collection.name,
                    "display_name": f"{chroma_dir.name} / {collection.name}"
                }

                # TODO: Get document count with fallback for unsupported operations
                try:
                    backends[key]["document_count"] = collection.count()
                except Exception:
                    backends[key]["document_count"] = "unknown"

        # TODO: Handle connection or access errors gracefully
        except Exception as e:
            # TODO: Create fallback entry for inaccessible directories
            # TODO: Include error information in display name with truncation
            # TODO: Set appropriate fallback values for missing information
            key = f"{chroma_dir.name}_error"

            backends[key] = {
                "directory": str(chroma_dir),
                "collection_name": "n/a",
                "display_name": f"{chroma_dir.name} (error: {str(e)[:50]}...)",
                "document_count": "error"
            }

    # TODO: Return complete backends dictionary with all discovered collections
    return backends

def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""

    # TODO: Create a chromadb persistentclient
    openai_api_key = os.getenv("CHROMA_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("Missing OpenAI API key for Chroma embedding function")

    embedding_function = OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name="text-embedding-3-small"
    )

    client = chromadb.PersistentClient(path=chroma_dir)

    # TODO: Return the collection with the collection_name
    return client.get_collection(
        name=collection_name,
        embedding_function=embedding_function
    )

def retrieve_documents(collection, query: str, n_results: int = 5,
                       mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    # TODO: Initialize filter variable to None (represents no filtering)
    where_filter = None

    # TODO: Check if filter parameter exists and is not set to "all" or equivalent
    if mission_filter and mission_filter.lower() != "all":
        # TODO: If filter conditions are met, create filter dictionary with appropriate field-value pairs
        where_filter = {"mission": mission_filter}

    # TODO: Execute database query with the following parameters:
        # TODO: Pass search query in the required format
        # TODO: Set maximum number of results to return
        # TODO: Apply conditional filter (None for no filtering, dictionary for specific filtering)
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter
    )

    # TODO: Return query results to caller
    return results

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""
    
    max_words = 300

    # TODO: Initialize list with header text for context section
    context_parts = ["Context:"]

    # Deduplication set
    seen = set()

    # TODO: Loop through paired documents and their metadata using enumeration
    for i, (doc, meta) in enumerate(zip(documents, metadatas)):

        # Deduplicate identical chunks
        if doc in seen:
            continue # duplicate, skip iteration
        seen.add(doc)

        # TODO: Extract mission information from metadata with fallback value
        mission = meta.get("mission", "Unknown Mission")

        # TODO: Clean up mission name formatting (replace underscores, capitalize)
        mission = mission.replace("_", " ").title()

        # TODO: Extract source information from metadata with fallback value
        source = meta.get("source", "Unknown Source")

        # TODO: Extract category information from metadata with fallback value
        category = meta.get("document_category", meta.get("category", "Unknown Category"))

        # TODO: Create formatted source header with index number and extracted information
        header = f"Source {len(seen)} (Mission: {mission}, Category: {category}, Document: {source}):"

        # TODO: Add source header to context parts list
        context_parts.append(header)

        # TODO: Check document length and truncate if necessary
        words = doc.split()
        if len(words) > max_words:
            doc = " ".join(words[:max_words]) + "..."

        # TODO: Add truncated or full document content to context parts list
        context_parts.append(doc)

    # TODO: Join all context parts with newlines and return formatted string
    return "\n\n".join(context_parts)
