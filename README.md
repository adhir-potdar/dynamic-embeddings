# Dynamic JSON Embeddings

A production-ready Python library for intelligent JSON document processing, chunking, vector embedding generation, and similarity search using OpenAI embeddings and PGVector.

## 🚀 What This Pipeline Does

The **Dynamic JSON Embeddings Pipeline** transforms complex JSON documents into searchable vector embeddings through three intelligent stages:

### **Stage 1: Document Chunking**
**JSON → Structured Text Chunks**

Automatically analyzes your JSON structure and selects the optimal chunking strategy:
- **Flat**: Simple key-value documents
- **Hierarchical**: Nested object structures
- **Semantic**: Content-rich documents
- **Dimensional**: Array/list data
- **Hybrid**: Combines multiple strategies

### **Stage 2: Vector Generation**
**Text Chunks → High-Dimensional Embeddings**

- Uses OpenAI's `text-embedding-3-large` model (1536 dimensions)
- Efficient batch processing with retry logic
- Preserves all metadata from chunking stage

### **Stage 3: Vector Storage**
**Embeddings → Searchable PGVector Database**

- Stores embeddings with rich metadata (25+ fields)
- Enables semantic similarity search
- Supports collection and document management

## 🔄 **Recent Updates (December 2025)**

### Production-Ready Enhancements
- ✅ **Automatic Document Replacement** - Re-process files without duplicate errors
- ✅ **Extended Database Support** - Field lengths increased to 4096 characters for complex hierarchical paths
- ✅ **Token-Aware Batching** - Intelligent OpenAI API usage with tiktoken integration
- ✅ **Streaming Storage** - Memory-efficient processing for large datasets (100+ embeddings per batch)
- ✅ **Cross-Collection Management** - Smart document replacement across collections
- ✅ **Interactive Collection Selection** - Enhanced QA interface with per-query collection targeting

### New Command-Line Tools
- 🆕 **Single File Processing** - `--file` option for individual JSON processing
- 🆕 **Collection-Scoped Search** - Interactive QA now prompts for collection per query
- 🆕 **Replacement Control** - `--no-replace` flag for strict duplicate prevention

## ⚡ Quick Start

### Prerequisites

1. **PostgreSQL with pgvector extension** - [See installation guide below](#database-setup)
2. **OpenAI API key** - Get from [OpenAI API Keys](https://platform.openai.com/api-keys)
3. **Python 3.8+**

### Database Setup

#### 🍎 **macOS (using Homebrew)**
```bash
# Install PostgreSQL 17 and pgvector
brew install postgresql@17
brew install pgvector

# Start PostgreSQL service
brew services start postgresql@17

# Create postgres user with superuser privileges
psql -d postgres -c "CREATE ROLE postgres WITH LOGIN PASSWORD 'your_postgres_password' CREATEDB SUPERUSER;"

# Create database owned by postgres user
createdb -O postgres vectordb

# Enable pgvector extension
psql -U postgres -d vectordb -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Verify installation
psql -U postgres -d vectordb -c "SELECT version();" -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

#### 🐧 **Linux (Ubuntu/Debian)**
```bash
# Install PostgreSQL 17
sudo apt update
sudo apt install wget ca-certificates
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
echo "deb http://apt.postgresql.org/pub/repos/apt/ $(lsb_release -cs)-pgdg main" | sudo tee /etc/apt/sources.list.d/pgdg.list
sudo apt update
sudo apt install postgresql-17 postgresql-contrib-17

# Install pgvector
sudo apt install postgresql-17-pgvector

# Start PostgreSQL service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create postgres user and database
sudo -u postgres psql -c "CREATE ROLE postgres WITH LOGIN PASSWORD 'your_postgres_password' CREATEDB SUPERUSER;"
sudo -u postgres createdb -O postgres vectordb

# Enable pgvector extension
sudo -u postgres psql -U postgres -d vectordb -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Verify installation
sudo -u postgres psql -U postgres -d vectordb -c "SELECT version();" -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

#### 🪟 **Windows**
```powershell
# Install PostgreSQL 17 from official installer
# Download from: https://www.postgresql.org/download/windows/
# Choose PostgreSQL 17.x during installation

# Install pgvector (after PostgreSQL installation)
# Download from: https://github.com/pgvector/pgvector/releases
# Extract and copy files to PostgreSQL installation directory

# Open Command Prompt as Administrator and run:
cd "C:\Program Files\PostgreSQL\17\bin"

# Create postgres user and database
psql -U postgres -c "CREATE ROLE postgres WITH LOGIN PASSWORD 'your_postgres_password' CREATEDB SUPERUSER;"
createdb -U postgres -O postgres vectordb

# Enable pgvector extension
psql -U postgres -d vectordb -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Verify installation
psql -U postgres -d vectordb -c "SELECT version();" -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

#### 🐳 **Docker (All Platforms)**
```bash
# Quick setup with Docker (PostgreSQL 17 + pgvector)
docker run --name pgvector-db \
  -e POSTGRES_PASSWORD=your_postgres_password \
  -e POSTGRES_DB=vectordb \
  -e POSTGRES_USER=postgres \
  -p 5432:5432 \
  -d pgvector/pgvector:pg17

# Enable pgvector extension
docker exec -it pgvector-db psql -U postgres -d vectordb -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Verify installation
docker exec -it pgvector-db psql -U postgres -d vectordb -c "SELECT version();" -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

#### 📚 **Detailed Installation Links**
- **PostgreSQL**: [Official Download Page](https://www.postgresql.org/download/)
- **PGVector**: [GitHub Repository](https://github.com/pgvector/pgvector)
- **macOS Guide**: [PostgreSQL on macOS](https://postgresapp.com/) or [Homebrew Guide](https://wiki.postgresql.org/wiki/Homebrew)
- **Linux Guide**: [PostgreSQL on Linux](https://www.postgresql.org/download/linux/)
- **Windows Guide**: [PostgreSQL on Windows](https://www.postgresql.org/download/windows/)
- **Docker Guide**: [PGVector Docker Hub](https://hub.docker.com/r/pgvector/pgvector)

### Installation

```bash
# Clone repository
git clone https://github.com/adhir-potdar/dynamic-embeddings.git
cd dynamic-embeddings

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Environment Setup

Create a `.env` file:

```env
# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=vectordb
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_EMBEDDING_MODEL=text-embedding-3-large

# Engine Configuration (Optional)
DEFAULT_COLLECTION=documents
CHUNK_SIZE_LIMIT_MB=10
ENABLE_STREAMING=true
LOG_LEVEL=INFO

# Similarity Search Configuration
DEFAULT_SIMILARITY_THRESHOLD=0.5
SEARCH_RESULT_LIMIT=10
```

### 🎯 Starting Points

#### **Option 1: Complete Pipeline Demo (Recommended)**
```bash
python examples/vector_embeddings_pipeline_demo.py
```
**Shows:** Database setup, JSON processing, similarity search, collection management

#### **Option 2: JSON File Processing**
```bash
# Process entire folder of JSON files
python test_json_embeddings_loader.py /path/to/json_folder --collection my_data

# Process single JSON file
python test_json_embeddings_loader.py --file /path/to/file.json --collection my_data

# Process with automatic replacement (default behavior)
python test_json_embeddings_loader.py --file file.json --collection devices

# Process without replacement (fail on duplicates)
python test_json_embeddings_loader.py --file file.json --collection devices --no-replace
```

#### **Option 3: Interactive Q&A Search**
```bash
# Launch interactive search interface
python test_interactive_qa.py

# With custom settings
python test_interactive_qa.py --threshold 0.6 --limit 5

# During session, you'll be prompted:
# Collection: devices
# Enter your question: What is device type analysis about?
```

#### **Option 4: Collection Management**
```bash
# List all collections and their statistics
./test_collection_manager.py --list

# Search within specific collection
./test_collection_manager.py --collection devices --search "device type"

# Delete specific collection
./test_collection_manager.py --delete devices
```

#### **Option 5: Command Line Interface (Legacy)**
```bash
# Full demonstration
python cli_demo.py --full-demo

# Quick commands
python cli_demo.py --setup-db              # Setup database
python cli_demo.py --process-sample        # Process sample data
python cli_demo.py --search "query text"   # Search embeddings
python cli_demo.py --stats                 # Show statistics
```

#### **Option 6: Document Chunking Only**
```bash
python examples/jsondoc_embeddingchunk_textconvertor_demo.py
```
**Shows:** JSON chunking strategies without requiring database/embeddings

### Basic Usage in Code

```python
from dynamic_embeddings import EmbeddingPipeline, DatabaseConnection

# Setup database connection
db_connection = DatabaseConnection(
    host='localhost',
    database='vectordb',
    username='postgres',
    password='your_password'
)

# Initialize complete pipeline
pipeline = EmbeddingPipeline(
    database_connection=db_connection,
    openai_api_key='your_openai_key'
)

# Setup database (first time only)
pipeline.setup_database()

# Process your JSON data
result = pipeline.process_json_data(
    json_data=your_json_data,
    collection_name="my_collection",
    document_id="doc_001"
)

print(f"Generated {result['total_embeddings']} embeddings")

# Search for similar content (uses environment defaults if parameters not specified)
results = pipeline.search_similar(
    query_text="your search query",
    collection_name="my_collection",
    limit=10,  # Optional: uses SEARCH_RESULT_LIMIT from .env if None
    similarity_threshold=0.7  # Optional: uses DEFAULT_SIMILARITY_THRESHOLD from .env if None
)

for record, similarity in results:
    print(f"Similarity: {similarity:.3f}")
    print(f"Text: {record.text[:100]}...")
```

## 🏗️ Understanding JSON Chunking

JSON chunking breaks down complex JSON documents into meaningful, searchable pieces while preserving structure and semantic relationships. Unlike traditional text chunking, our system:

- **Respects JSON hierarchy** - Maintains parent-child relationships
- **Preserves context** - Keeps related information together
- **Adapts automatically** - Selects optimal strategy per document
- **Handles any structure** - From flat objects to deep hierarchies

### Chunking Strategy Examples

**Hierarchical Strategy** (nested objects):
```json
{
  "campaign": {
    "id": "camp_123",
    "performance": {
      "impressions": 50000,
      "clicks": 1200
    }
  }
}
```
→ Creates chunks that preserve the campaign→performance relationship

**Dimensional Strategy** (arrays):
```json
{
  "products": [
    {"name": "Widget A", "price": 10.99},
    {"name": "Widget B", "price": 15.99}
  ]
}
```
→ Creates individual product chunks while maintaining collection context

## 🎛️ Advanced Features

### Custom Configuration
```python
# Use custom chunking configuration
processor = DocumentProcessor(config_file="custom_config.json")

# Or import additional configurations
from dynamic_embeddings.config import register_custom_config
register_custom_config("ecommerce", "configs/ecommerce.json")
```

### Collection Management
```python
# List all collections
collections = pipeline.list_collections()

# Get detailed statistics
stats = pipeline.get_collection_stats("my_collection")

# Delete collections
deleted_count = pipeline.delete_collection("old_collection")
```

### Advanced Search
```python
# Search with filters
results = pipeline.search_similar(
    query_text="budget optimization",
    collection_name="campaigns",
    filters={
        "strategy": "hierarchical",
        "confidence": [0.8, 0.9, 1.0]
    },
    similarity_threshold=0.6,
    limit=5
)
```

## 📊 Use Cases

### **Ad-Tech Analytics**
Campaign data, audience insights, performance metrics, budget tracking

### **E-commerce**
Product catalogs, customer behavior, inventory data, transaction records

### **Financial Services**
Transaction data, risk metrics, customer profiles, compliance documents

### **Healthcare**
Patient records, clinical trials, drug databases, medical devices

### **Any Complex JSON Data**
Configuration files, API responses, analytical reports, business entities

## 🔧 Architecture

```
JSON Input → Document Chunking → Vector Generation → PGVector Storage
    ↓              ↓                    ↓                  ↓
Raw JSON    Strategy Selection    OpenAI Embeddings    Searchable DB
           Text Conversion        1536 Dimensions      Rich Metadata
           Quality Validation     Batch Processing     Similarity Search
```

## 📁 Project Structure

```
dynamic-embeddings/
├── src/dynamic_embeddings/
│   ├── pipelines/           # Complete pipeline orchestration
│   ├── services/           # Embedding and vector storage services
│   ├── database/           # PGVector schema and connections
│   ├── processors/         # Document chunking and text conversion
│   ├── strategies/         # Chunking strategy implementations
│   ├── config/            # Configuration management
│   └── models/            # Data models and schemas
├── examples/              # Demo scripts and usage examples
├── configs/              # Configuration files
├── cli_demo.py          # Command-line interface
└── README.md            # This file
```

## 📚 Examples

The `examples/` directory contains:
- **vector_embeddings_pipeline_demo.py** - Complete pipeline showcase
- **jsondoc_embeddingchunk_textconvertor_demo.py** - Document chunking only
- Sample configuration files for different domains

## 🛠️ Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest

# Format code
black src/
isort src/

# Check imports and validate setup
python -c "from dynamic_embeddings import EmbeddingPipeline; print('✓ Import successful')"
```

## 📈 Performance Features

- **Batch Processing** - Efficient OpenAI API usage
- **Connection Pooling** - Optimized database connections
- **Quality Validation** - Ensures meaningful chunks
- **Error Handling** - Robust retry logic and error recovery
- **Usage Tracking** - Monitor API costs and performance
- **Streaming Support** - Handle large documents efficiently
- **Document Replacement** - Automatic handling of duplicate documents across collections
- **Token Management** - Intelligent batching with tiktoken for OpenAI API efficiency
- **Extended Schema** - Support for complex hierarchical paths up to 4096 characters

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for your changes
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🔗 Links

- 🐛 [Issues](https://github.com/adhir-potdar/dynamic-embeddings/issues)
- 💬 [Discussions](https://github.com/adhir-potdar/dynamic-embeddings/discussions)
- 📧 Contact: adhir.potdar@isanasystems.com

---

**Ready to get started?** Run `python cli_demo.py --full-demo` to see the complete pipeline in action! 🚀