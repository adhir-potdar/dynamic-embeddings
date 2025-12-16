#!/usr/bin/env python3
"""
Interactive Q&A Interface for Embeddings Search

This script provides an interactive command-line interface for querying
embeddings created from JSON files. Users can ask questions and get
relevant answers from the embedded data.

Usage:
    python test_interactive_qa.py
    python test_interactive_qa.py --collection reasoning_output
    python test_interactive_qa.py --threshold 0.6 --limit 5
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add src to path for imports
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not installed, try manual loading
    env_path = project_root / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

from dynamic_embeddings.pipelines.embedding_pipeline import EmbeddingPipeline
from dynamic_embeddings.database.connection import DatabaseConnection


def setup_logging(verbose=False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def create_pipeline():
    """Create and return configured pipeline."""
    db_connection = DatabaseConnection(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=int(os.getenv('POSTGRES_PORT', '5432')),
        database=os.getenv('POSTGRES_DB', 'vectordb'),
        username=os.getenv('POSTGRES_USER', 'postgres'),
        password=os.getenv('POSTGRES_PASSWORD', ''),
    )

    return EmbeddingPipeline(
        database_connection=db_connection,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        embedding_model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-large')
    )


def display_search_results(search_info, show_full_text=False):
    """Display search results in a formatted way."""

    if 'error' in search_info:
        print(f"❌ Search Error: {search_info['error']}")
        return

    results = search_info['results']
    total_embeddings = search_info['total_embeddings']
    matched_embeddings = search_info['matched_embeddings']
    similarity_threshold = search_info['similarity_threshold']

    print(f"\n📊 SEARCH RESULTS:")
    print(f"   Query: '{search_info['query_text']}'")
    print(f"   Collection: {search_info['collection_name']}")
    print(f"   Found: {matched_embeddings}/{total_embeddings} embeddings")
    print(f"   Threshold: {similarity_threshold}")
    print("-" * 60)

    if not results:
        print("❌ No results found above the similarity threshold.")
        print(f"💡 Try lowering the threshold (current: {similarity_threshold}) or rephrasing your question.")
        return

    for i, (record, similarity) in enumerate(results, 1):
        print(f"\n📋 Result {i}:")
        print(f"   📈 Similarity Score: {similarity:.4f}")
        print(f"   🏷️  Document: {record.document_id}")
        print(f"   🔧 Strategy: {record.strategy}")
        print(f"   📝 Content Type: {record.content_type}")
        print(f"   📍 Path: {record.path}")
        print(f"   📊 Semantic Density: {record.semantic_density:.3f}")

        if show_full_text:
            print(f"   📄 FULL TEXT:")
            print(f"      {record.text}")
        else:
            # Show preview of text (first 150 characters)
            preview = record.text[:150] + "..." if len(record.text) > 150 else record.text
            print(f"   📄 TEXT PREVIEW:")
            print(f"      {preview}")

        if i < len(results):
            print("   " + "-" * 55)


def show_available_collections(pipeline):
    """Show available collections and their statistics."""
    try:
        collections = pipeline.list_collections()

        if not collections:
            print("❌ No collections found in the database.")
            print("💡 Use the JSON loader script first to create embeddings from your JSON files.")
            return

        print("\n📚 AVAILABLE COLLECTIONS:")
        print("-" * 60)

        for collection in collections:
            print(f"\n📁 Collection: {collection['collection_name']}")
            print(f"   📊 Total Embeddings: {collection['total_embeddings']}")
            print(f"   🔧 Strategies: {list(collection['strategies'].keys())}")
            print(f"   📝 Content Types: {list(collection['content_types'].keys())}")
            print(f"   📈 Avg Semantic Density: {collection.get('avg_semantic_density', 0):.3f}")

    except Exception as e:
        print(f"❌ Error retrieving collections: {e}")


def interactive_qa_session(collection_name, similarity_threshold, limit):
    """Run the interactive Q&A session."""

    print("🚀 INTERACTIVE EMBEDDINGS Q&A SYSTEM")
    print("="*60)

    try:
        # Create pipeline
        pipeline = create_pipeline()

        # Show available collections
        show_available_collections(pipeline)

        print(f"\n🎯 Current Settings:")
        print(f"   Collection: {collection_name or 'ALL COLLECTIONS'}")
        print(f"   Similarity Threshold: {similarity_threshold}")
        print(f"   Result Limit: {limit}")

        print(f"\n💡 Available Commands:")
        print(f"   - Enter any question to search embeddings")
        print(f"   - Type 'help' for more commands")
        print(f"   - Type 'exit' to quit")
        print("-" * 60)

        while True:
            try:
                # Get user input
                question = input(f"\n❓ Enter your question: ").strip()

                if not question:
                    continue

                # Handle special commands
                if question.lower() == 'exit':
                    print("\n👋 Goodbye!")
                    break

                elif question.lower() == 'help':
                    print(f"\n🆘 HELP:")
                    print(f"   exit           - Quit the program")
                    print(f"   help           - Show this help")
                    print(f"   collections    - Show available collections")
                    print(f"   settings       - Show current settings")
                    print(f"   full           - Toggle full text display")
                    print(f"   <question>     - Search for similar content")
                    continue

                elif question.lower() == 'collections':
                    show_available_collections(pipeline)
                    continue

                elif question.lower() == 'settings':
                    print(f"\n⚙️  CURRENT SETTINGS:")
                    print(f"   Collection: {collection_name or 'ALL COLLECTIONS'}")
                    print(f"   Similarity Threshold: {similarity_threshold}")
                    print(f"   Result Limit: {limit}")
                    continue

                # Perform search
                print(f"\n🔍 Searching for: '{question}'")

                search_info = pipeline.search_similar(
                    query_text=question,
                    collection_name=collection_name,
                    limit=limit,
                    similarity_threshold=similarity_threshold
                )

                # Display results
                display_search_results(search_info, show_full_text=False)

                # Ask if user wants to see full text
                if search_info.get('matched_embeddings', 0) > 0:
                    show_full = input(f"\n📄 Show full text for results? (y/n): ").strip().lower()
                    if show_full in ['y', 'yes']:
                        display_search_results(search_info, show_full_text=True)

            except KeyboardInterrupt:
                print(f"\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error during search: {e}")
                print("💡 Please try rephrasing your question or check the logs.")

        # Cleanup
        pipeline.close()

    except Exception as e:
        print(f"\n❌ Failed to initialize Q&A system: {e}")
        print("💡 Make sure the database is running and contains embeddings.")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Q&A for embeddings search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_interactive_qa.py
  python test_interactive_qa.py --collection reasoning_output
  python test_interactive_qa.py --threshold 0.6 --limit 5 --verbose

During the session, you can ask questions like:
  - "What is device type analysis about?"
  - "Show me information about geography and properties"
  - "What are the key insights from the reasoning data?"
        """
    )

    parser.add_argument('--collection', '-c',
                       default=None,
                       help='Collection name to search (default: search all collections)')
    parser.add_argument('--threshold', '-t',
                       type=float,
                       default=float(os.getenv('DEFAULT_SIMILARITY_THRESHOLD', '0.5')),
                       help='Similarity threshold (0.0-1.0, default from env)')
    parser.add_argument('--limit', '-l',
                       type=int,
                       default=int(os.getenv('SEARCH_RESULT_LIMIT', '5')),
                       help='Maximum results to return (default from env)')
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Validate arguments
    if args.threshold < 0.0 or args.threshold > 1.0:
        print("❌ ERROR: Similarity threshold must be between 0.0 and 1.0")
        sys.exit(1)

    if args.limit < 1:
        print("❌ ERROR: Result limit must be at least 1")
        sys.exit(1)

    # Check environment
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ ERROR: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key in .env file or environment")
        sys.exit(1)

    # Run interactive session
    success = interactive_qa_session(args.collection, args.threshold, args.limit)

    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()