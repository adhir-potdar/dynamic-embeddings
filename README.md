# Dynamic Embeddings

A production-ready Python library for intelligent JSON document chunking and embedding with automatic strategy selection using LangChain and PGVector.

## Understanding JSON Chunking

JSON chunking is the process of breaking down large or complex JSON documents into smaller, meaningful pieces that can be effectively embedded and searched in vector databases. Unlike traditional text chunking that operates on character or sentence boundaries, JSON chunking must respect the inherent structure and semantic relationships within hierarchical data.

The challenge with JSON documents lies in their nested nature and diverse content types. A single JSON file might contain configuration data, performance metrics, business entities, and analytical results - each requiring different approaches for optimal searchability. Traditional approaches either lose structural context by flattening the data or create overly large chunks that perform poorly in vector similarity searches.

Our dynamic chunking system addresses these challenges by analyzing JSON structure, content patterns, and domain-specific characteristics to automatically select the most appropriate chunking strategy. This ensures that related information stays together while maintaining optimal chunk sizes for embedding generation and retrieval performance.

## Chunking Strategy Types

### **Flat Decomposition**
Converts nested JSON structures into simple key-value pairs, ideal for configuration files and simple data structures. Each leaf value becomes a separate document with its full path preserved in metadata.

### **Hierarchical Chunking**
Preserves the natural hierarchy of JSON data by creating documents at multiple levels. Parent-child relationships are maintained through metadata, enabling both granular search and hierarchical browsing.

### **Performance Semantic Grouping**
Specifically designed for analytics data, this strategy groups related performance metrics, trends, and explanations together. Perfect for business intelligence data where context between metrics is crucial.

### **Entity Separation**
Identifies and separates distinct business entities (customers, products, regions) into individual documents while preserving their complete attribute sets and relationships.

### **Dimensional Analysis**
Handles multi-dimensional data structures common in OLAP and analytics systems. Creates documents that respect dimensional hierarchies and enable cross-dimensional analysis.

### **Hierarchical Drill-down**
Optimized for analytical data with drill-down capabilities. Maintains the ability to navigate from high-level summaries to detailed breakdowns while preserving analytical context.

## Features

🚀 **Dynamic Strategy Selection** - Automatically selects optimal chunking strategy based on JSON structure and content
📊 **Analytics-Aware** - Built-in support for performance metrics, hierarchical data, and business analytics
🔧 **Production Ready** - Comprehensive configuration management, error handling, and monitoring
🏗️ **Extensible** - Plugin architecture for custom analyzers, rules, and strategies
📈 **Scalable** - Handles files from KB to GB with streaming and batch processing
🔍 **Observable** - Rich logging, metrics, and strategy explanation

## Quick Start

### Setup

For detailed setup instructions, see [SETUP.md](SETUP.md).

### Installation

```bash
pip install dynamic-embeddings
```

### Basic Usage

```python
from dynamic_embeddings import DynamicChunkingEngine
from dynamic_embeddings.config import DatabaseConfig

# Configure connection
db_config = DatabaseConfig(
    host="localhost",
    port=5432,
    database="vectordb",
    username="user",
    password="password"
)

# Initialize engine
engine = DynamicChunkingEngine(db_config=db_config)

# Process JSON data
json_data = {"your": "hierarchical data"}
documents = await engine.process_json(json_data)

# Store in vector database
await engine.store_documents(documents, collection_name="my_collection")
```

### Configuration

Create a `.env` file:

```env
# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=vectordb
POSTGRES_USER=user
POSTGRES_PASSWORD=password

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=text-embedding-3-large

# Engine Configuration
DEFAULT_COLLECTION=documents
CHUNK_SIZE_LIMIT_MB=10
ENABLE_STREAMING=true
LOG_LEVEL=INFO
```

## Supported Strategies

- **Flat Decomposition** - Simple key-value extraction
- **Hierarchical Chunking** - Preserves nested structure relationships
- **Performance Semantic** - Groups by performance metrics and trends
- **Entity Separation** - Separates different business entities
- **Dimensional Analysis** - Handles multi-dimensional data structures
- **Hierarchical Drill-down** - Supports analytical drill-down patterns
- **Multi-level Hybrid** - Combines multiple strategies for complex data

## Architecture

```
JSON Input → Analyzers → Decision Rules → Strategy Selection → Chunking → Vector Storage
     ↓           ↓            ↓              ↓             ↓           ↓
  Raw JSON   Structure &   Rule Engine   Optimal Strategy  Documents  PGVector
             Content      Evaluation                                   + Metadata
             Analysis
```

## Examples

See the `examples/` directory for:
- Ad-Tech analytics processing
- Configuration file chunking
- Business entity extraction
- Time series data handling

## Development

```bash
# Clone repository
git clone https://github.com/adhir-potdar/dynamic-embeddings.git
cd dynamic-embeddings

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for your changes
4. Ensure all tests pass
5. Submit a pull request

## Support

- 📖 Documentation: [Coming Soon]
- 🐛 Issues: [GitHub Issues](https://github.com/adhir-potdar/dynamic-embeddings/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/adhir-potdar/dynamic-embeddings/discussions)