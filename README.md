# Wormhole Vectors: Bridging Dense and Sparse Search Spaces

This project demonstrates the concept of "Wormhole Vectors" - a technique that bridges dense vector embeddings with sparse keyword representations using statistical analysis in OpenSearch.

## Concept Overview

Wormhole vectors solve the problem of traversing between different search spaces:

1. **Dense Vector Space**: Semantic embeddings that capture meaning but lack explainability
2. **Sparse Keyword Space**: Traditional keyword search that's explainable but misses semantic relationships

The "wormhole" is created by:
- Running a dense vector query to get a "Foreground Set" of documents
- Using statistical analysis (significant_terms aggregation) to find keywords that are statistically significant in the foreground compared to the background
- Generating a sparse keyword query that "explains" what the dense vector region represents

## Key Features

- **Disambiguation**: Automatically distinguishes between ambiguous terms (e.g., "Java" = programming vs coffee)
- **Explainability**: Shows users why results appeared by mapping vectors back to keywords
- **Zero Result Problem**: Bridges vocabulary gaps without manual synonym lists

## Prerequisites

- Python 3.8+
- An Aiven OpenSearch instance (or any OpenSearch/Elasticsearch cluster)
- OpenSearch with k-NN plugin enabled

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables:
```bash
export OPENSEARCH_URL="https://your-instance.aivencloud.com:12345"
export OPENSEARCH_USERNAME="avnadmin"
export OPENSEARCH_PASSWORD="your-password"
```

Or update the values directly in `wormhole_vectors.py` in the `main()` function.

## Usage

### Basic Usage

Run the full demonstration (creates index and loads sample data):

```bash
python wormhole_vectors.py
```

The script will:
1. Create an OpenSearch index with k-NN vector support
2. Generate and index sample data (including ambiguous "Java" examples)
3. Demonstrate disambiguation using wormhole vectors
4. Show how the same query produces different keyword explanations based on context

### CLI Options

Skip indexing and only run queries (useful after initial setup):

```bash
python wormhole_vectors.py --skip-indexing
```

Run a single query:

```bash
python wormhole_vectors.py --skip-indexing --query "Java programming"
```

Interactive query mode (experiment with multiple queries):

```bash
python wormhole_vectors.py --skip-indexing --interactive
```

Run the full disambiguation demo:

```bash
python wormhole_vectors.py --skip-indexing --demo
```

Customize query parameters:

```bash
python wormhole_vectors.py --skip-indexing --query "server" --k 20 --keyword-size 15
```

Use a custom index name:

```bash
python wormhole_vectors.py --index-name "my_custom_index"
```

### Available CLI Arguments

- `--skip-indexing`: Skip creating index and indexing documents (use existing index)
- `--query`, `-q`: Run a single query and display results
- `--interactive`, `-i`: Enter interactive query mode
- `--demo`: Run the full disambiguation demonstration
- `--k`: Number of results to retrieve for foreground set (default: 10)
- `--keyword-size`: Number of significant keywords to extract (default: 10)
- `--index-name`: Name of the OpenSearch index (default: wormhole_demo)
- `--help`, `-h`: Show help message with all options

## How It Works

### The Math

The wormhole traversal uses statistical significance:

```
Score(term) = P(term | Foreground) / P(term | Background)
```

Terms with high scores appear disproportionately in the foreground set compared to the entire index, making them good "explanatory keywords" for that vector region.

### Example: Java Disambiguation

**Query 1**: "Java programming language"
- **Foreground**: Documents about Java programming
- **Wormhole Keywords**: `hibernate`, `scala`, `jvm`, `backend`, `programming`
- **Sparse Query**: `(hibernate) AND (scala) AND (jvm) OR (backend) OR (programming)`

**Query 2**: "Java coffee beans"
- **Foreground**: Documents about Java coffee
- **Wormhole Keywords**: `sumatra`, `coffee`, `roast`, `island`, `beans`
- **Sparse Query**: `(sumatra) AND (coffee) AND (roast) OR (island) OR (beans)`

The same ambiguous term produces different keyword explanations!

## Code Structure

- `wormhole_vectors.py`: Main implementation with `WormholeVectorSearch` class
- `requirements.txt`: Python dependencies
- `README.md`: This file

## Key Methods

- `create_index()`: Sets up OpenSearch index with k-NN vector field
- `generate_sample_data()`: Creates sample documents with embeddings
- `dense_vector_search()`: Performs k-NN vector search
- `wormhole_traversal()`: Complete wormhole traversal (vector → keywords)
- `demonstrate_disambiguation()`: Shows the Java disambiguation example

## Customization

You can customize:
- **Embedding model**: Change `SentenceTransformer` model in `__init__`
- **Index settings**: Modify `create_index()` for different vector dimensions
- **Sample data**: Add your own documents in `generate_sample_data()`
- **Query parameters**: Adjust `k` (foreground size) and `keyword_size` in `wormhole_traversal()`

## Production Considerations

For production use:
- Use a production-grade embedding model (e.g., `all-mpnet-base-v2`)
- Adjust HNSW parameters based on your data size
- Add error handling and retry logic
- Implement caching for embeddings
- Monitor query performance

## References

- [OpenSearch k-NN Plugin](https://opensearch.org/docs/latest/search-plugins/knn/index/)
- [Significant Terms Aggregation](https://opensearch.org/docs/latest/aggregations/bucket/significant-terms/)
- [Sentence Transformers](https://www.sbert.net/)

## License

This is a demonstration project. Feel free to use and modify as needed.

