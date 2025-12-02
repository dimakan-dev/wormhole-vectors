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
export OPENSEARCH_USERNAME="osd-user-name"
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
- Scale OpenSearch cluster according to your needs (we can help out at Aiven)

## References
- [Vector Podcast episode with Trey Grainger on Wormhole vector traversal idea](https://github.com/dimakan-dev/wormhole-vectors)
- [OpenSearch k-NN Plugin](https://opensearch.org/docs/latest/search-plugins/knn/index/)
- [Significant Terms Aggregation](https://opensearch.org/docs/latest/aggregations/bucket/significant-terms/)
- [Sentence Transformers](https://www.sbert.net/)
- [Blog post](https://aiven.io/blog)

## License

This is a demonstration project. Feel free to use and modify as needed.

## Traversal examples
```

Query: Java coffee


% python wormhole_vectors.py --skip-indexing --query "Java coffee" --k 10
Loading embedding model...
Model loaded successfully!
Using existing index: wormhole_demo

============================================================
WORMHOLE VECTOR TRAVERSAL
============================================================
Query: 'Java coffee'
Step 1: Dense Vector Search → Retrieving top 10 documents (Foreground Set)
Step 2: Statistical Analysis → Finding significant terms (Foreground vs Background)
Step 3: Positional Weighting → Boosting terms from top-ranked documents
Step 4: Generate Sparse Query → Combine statistical + positional scores
Step 5: Execute Sparse Query → Show documents retrieved by keyword search

============================================================
RAW OPENSEARCH QUERY: Dense Vector (k-NN)
============================================================
{
  "size": 10,
  "query": {
    "knn": {
      "embedding_vector": {
        "vector": [
          -0.07056304812431335,
          0.014885498210787773,
          0.039055194705724716,
          0.043486181646585464,
          -0.01573297567665577
        ],
        "k": 10
      }
    }
  },
  "aggs": {
    "wormhole_keywords": {
      "significant_terms": {
        "field": "description_keywords",
        "size": 10,
        "background_filter": {
          "match_all": {}
        }
      }
    }
  },
  "_source": [
    "title",
    "description",
    "category"
  ]
}

Note: Vector truncated for display. Full vector has 384 dimensions.

Found 10 results in foreground set

Top Results:
------------------------------------------------------------
1. [coffee] Coffee Roast Techniques
   Master the art of roasting Java coffee beans from Sumatra...
   Score: 0.8752

2. [coffee] Java Coffee Beans from Sumatra
   Premium coffee beans from the Indonesian island of Java, Sumatra region...
   Score: 0.8530

3. [coffee] Indonesian Coffee Origins
   Explore Java and Sumatra coffee origins, tasting notes and brewing methods...
   Score: 0.8497

4. [coffee] Island Coffee Collection
   Exotic coffee from Java island, featuring unique flavor profiles...
   Score: 0.8478

5. [coffee] Sumatran Coffee Roasting Guide
   Learn to roast Java and Sumatra coffee beans to perfection...
   Score: 0.8356

6. [coffee] Espresso Coffee Guide
   Learn to make perfect espresso, latte art, and coffee brewing techniques...
   Score: 0.7605

7. [coffee] Italian Coffee Culture
   Explore Italian coffee traditions, espresso machines, and café culture...
   Score: 0.7310

8. [programming] Java Programming Guide
   Learn Java programming with Hibernate, Spring Framework, and JVM optimization...
   Score: 0.7192

9. [programming] Scala for Java Developers
   Transition from Java to Scala, covering functional programming and JVM internals...
   Score: 0.7011

10. [programming] JVM Performance Tuning
   Optimize Java Virtual Machine performance for enterprise applications...
   Score: 0.6678


============================================================
POSITIONAL WEIGHTING FROM TOP DOCUMENTS
============================================================
Top terms extracted from top 5 documents (weighted by rank):
Rank weights: #1=1.00, #2=0.71, #3=0.58, #4=0.50, #5=0.45

 1. 'coffee': 6.4633 [from: #1(1.00), #1(1.00), #2(0.71)]
 2. 'java': 3.9388 [from: #1(1.00), #2(0.71), #2(0.71)]
 3. 'sumatra': 3.4388 [from: #1(1.00), #2(0.71), #2(0.71)]
 4. 'beans': 2.8614 [from: #1(1.00), #2(0.71), #2(0.71)]
 5. 'island': 1.7071 [from: #2(0.71), #4(0.50), #4(0.50)]
 6. 'roast': 1.4472 [from: #1(1.00), #5(0.45)]
 7. 'roasting': 1.4472 [from: #1(1.00), #5(0.45)]
 8. 'indonesian': 1.2845 [from: #2(0.71), #3(0.58)]
 9. 'origins': 1.1547 [from: #3(0.58), #3(0.58)]
10. 'techniques': 1.0000 [from: #1(1.00)]
11. 'master': 1.0000 [from: #1(1.00)]
12. 'art': 1.0000 [from: #1(1.00)]
13. 'premium': 0.7071 [from: #2(0.71)]
14. 'region': 0.7071 [from: #2(0.71)]
15. 'explore': 0.5774 [from: #3(0.58)]


============================================================
WORMHOLE KEYWORDS (Combined: Statistical + Top Document Weighting)
============================================================
Terms are weighted by:
  - Statistical significance (40%): Foreground vs Background
  - Positional importance (60%): Terms from top-ranked documents

1. 'coffee'
   Combined Score: 1.0000 (40% stat + 60% positional)
   Statistical: 0.9800 | Positional: 1.0000
   Boosted from top docs: ⭐⭐⭐⭐⭐
   Foreground: 7 docs | Background: 7 docs

2. 'java'
   Combined Score: 0.6337 (40% stat + 60% positional)
   Statistical: 0.7360 | Positional: 0.6094
   Boosted from top docs: ⭐⭐⭐
   Foreground: 8 docs | Background: 10 docs

3. 'sumatra'
   Combined Score: 0.4922 (40% stat + 60% positional)
   Statistical: 0.5600 | Positional: 0.5320
   Boosted from top docs: ⭐⭐
   Foreground: 4 docs | Background: 4 docs

4. 'beans'
   Combined Score: 0.3629 (40% stat + 60% positional)
   Statistical: 0.4200 | Positional: 0.4427
   Boosted from top docs: ⭐⭐
   Foreground: 3 docs | Background: 3 docs

5. 'learn'
   Combined Score: 0.0415 (40% stat + 60% positional)
   Statistical: 0.2400 | Positional: 0.0692
   Boosted from top docs: 
   Foreground: 3 docs | Background: 4 docs

============================================================
GENERATED SPARSE QUERY
============================================================
Query: coffee AND (java OR sumatra OR beans OR learn)

This sparse query 'explains' what the dense vector region represents!
(Lucene boosts left as an exercise for the reader)

============================================================
STEP 5: EXECUTING SPARSE KEYWORD QUERY (BM25 with boosts)
============================================================
Searching for: coffee, java, sumatra, beans, learn

============================================================
RAW OPENSEARCH QUERY: Sparse Keyword (BM25 with boosts)
============================================================
{
  "size": 10,
  "query": {
    "bool": {
      "must": [
        {
          "multi_match": {
            "query": "coffee",
            "fields": [
              "description^2",
              "description_keywords"
            ],
            "type": "best_fields",
            "operator": "and"
          }
        }
      ],
      "should": [
        {
          "multi_match": {
            "query": "java",
            "fields": [
              "description^2",
              "description_keywords"
            ],
            "type": "best_fields",
            "operator": "and"
          }
        },
        {
          "multi_match": {
            "query": "sumatra",
            "fields": [
              "description^2",
              "description_keywords"
            ],
            "type": "best_fields",
            "operator": "and"
          }
        },
        {
          "multi_match": {
            "query": "beans",
            "fields": [
              "description^2",
              "description_keywords"
            ],
            "type": "best_fields",
            "operator": "and"
          }
        },
        {
          "multi_match": {
            "query": "learn",
            "fields": [
              "description^2",
              "description_keywords"
            ],
            "type": "best_fields",
            "operator": "and"
          }
        }
      ],
      "minimum_should_match": 1
    }
  },
  "_source": [
    "title",
    "description",
    "category"
  ]
}

Found 6 results using sparse keyword search

Top Results from Sparse Query:
------------------------------------------------------------
1. [coffee] Sumatran Coffee Roasting Guide
   Learn to roast Java and Sumatra coffee beans to perfection...
   Score: 14.7001

2. [coffee] Coffee Roast Techniques
   Master the art of roasting Java coffee beans from Sumatra...
   Score: 11.3243

3. [coffee] Java Coffee Beans from Sumatra
   Premium coffee beans from the Indonesian island of Java, Sumatra region...
   Score: 10.8695

4. [coffee] Indonesian Coffee Origins
   Explore Java and Sumatra coffee origins, tasting notes and brewing methods...
   Score: 7.1544

5. [coffee] Espresso Coffee Guide
   Learn to make perfect espresso, latte art, and coffee brewing techniques...
   Score: 5.5152

6. [coffee] Island Coffee Collection
   Exotic coffee from Java island, featuring unique flavor profiles...
   Score: 4.2560

============================================================
COMPARISON: Dense Vector vs Sparse Keyword Results
============================================================
Dense Vector Results: 10 documents
Sparse Keyword Results: 6 documents
Overlap: 6 documents (60.0%)

This demonstrates how the wormhole bridges semantic and lexical search!


Query: Java programming

% python wormhole_vectors.py --skip-indexing --query "Java programming" --k 10
Loading embedding model...
Model loaded successfully!
Using existing index: wormhole_demo

============================================================
WORMHOLE VECTOR TRAVERSAL
============================================================
Query: 'Java programming'
Step 1: Dense Vector Search → Retrieving top 10 documents (Foreground Set)
Step 2: Statistical Analysis → Finding significant terms (Foreground vs Background)
Step 3: Positional Weighting → Boosting terms from top-ranked documents
Step 4: Generate Sparse Query → Combine statistical + positional scores
Step 5: Execute Sparse Query → Show documents retrieved by keyword search

============================================================
RAW OPENSEARCH QUERY: Dense Vector (k-NN)
============================================================
{
  "size": 10,
  "query": {
    "knn": {
      "embedding_vector": {
        "vector": [
          -0.030333904549479485,
          0.07656581699848175,
          -0.07458636909723282,
          -0.05459993705153465,
          -0.11299372464418411
        ],
        "k": 10
      }
    }
  },
  "aggs": {
    "wormhole_keywords": {
      "significant_terms": {
        "field": "description_keywords",
        "size": 10,
        "background_filter": {
          "match_all": {}
        }
      }
    }
  },
  "_source": [
    "title",
    "description",
    "category"
  ]
}

Note: Vector truncated for display. Full vector has 384 dimensions.

Found 10 results in foreground set

Top Results:
------------------------------------------------------------
1. [programming] Java Programming Guide
   Learn Java programming with Hibernate, Spring Framework, and JVM optimization...
   Score: 0.8241

2. [programming] Scala for Java Developers
   Transition from Java to Scala, covering functional programming and JVM internals...
   Score: 0.7604

3. [programming] JVM Performance Tuning
   Optimize Java Virtual Machine performance for enterprise applications...
   Score: 0.6991

4. [programming] Backend Development with Java
   Build scalable backend systems using Java, Hibernate ORM, and microservices...
   Score: 0.6970

5. [programming] Java Enterprise Patterns
   Design patterns for Java enterprise applications using Spring and Hibernate...
   Score: 0.6951

6. [coffee] Coffee Roast Techniques
   Master the art of roasting Java coffee beans from Sumatra...
   Score: 0.6600

7. [coffee] Sumatran Coffee Roasting Guide
   Learn to roast Java and Sumatra coffee beans to perfection...
   Score: 0.6399

8. [coffee] Java Coffee Beans from Sumatra
   Premium coffee beans from the Indonesian island of Java, Sumatra region...
   Score: 0.6288

9. [coffee] Indonesian Coffee Origins
   Explore Java and Sumatra coffee origins, tasting notes and brewing methods...
   Score: 0.6270

10. [coffee] Island Coffee Collection
   Exotic coffee from Java island, featuring unique flavor profiles...
   Score: 0.6138


============================================================
POSITIONAL WEIGHTING FROM TOP DOCUMENTS
============================================================
Top terms extracted from top 5 documents (weighted by rank):
Rank weights: #1=1.00, #2=0.71, #3=0.58, #4=0.50, #5=0.45

 1. 'java': 5.8860 [from: #1(1.00), #1(1.00), #2(0.71)]
 2. 'programming': 2.7071 [from: #1(1.00), #1(1.00), #2(0.71)]
 3. 'jvm': 2.2845 [from: #1(1.00), #2(0.71), #3(0.58)]
 4. 'hibernate': 1.9472 [from: #1(1.00), #4(0.50), #5(0.45)]
 5. 'enterprise': 1.4718 [from: #3(0.58), #5(0.45), #5(0.45)]
 6. 'spring': 1.4472 [from: #1(1.00), #5(0.45)]
 7. 'scala': 1.4142 [from: #2(0.71), #2(0.71)]
 8. 'performance': 1.1547 [from: #3(0.58), #3(0.58)]
 9. 'applications': 1.0246 [from: #3(0.58), #5(0.45)]
10. 'guide': 1.0000 [from: #1(1.00)]
11. 'learn': 1.0000 [from: #1(1.00)]
12. 'framework': 1.0000 [from: #1(1.00)]
13. 'optimization': 1.0000 [from: #1(1.00)]
14. 'backend': 1.0000 [from: #4(0.50), #4(0.50)]
15. 'using': 0.9472 [from: #4(0.50), #5(0.45)]


============================================================
WORMHOLE KEYWORDS (Combined: Statistical + Top Document Weighting)
============================================================
Terms are weighted by:
  - Statistical significance (40%): Foreground vs Background
  - Positional importance (60%): Terms from top-ranked documents

1. 'java'
   Combined Score: 1.0000 (40% stat + 60% positional)
   Statistical: 1.4000 | Positional: 1.0000
   Boosted from top docs: ⭐⭐⭐⭐⭐
   Foreground: 10 docs | Background: 10 docs

2. 'programming'
   Combined Score: 0.2760 (40% stat + 60% positional)
   Statistical: 0.0000 | Positional: 0.4599
   Boosted from top docs: ⭐⭐

3. 'jvm'
   Combined Score: 0.2329 (40% stat + 60% positional)
   Statistical: 0.0000 | Positional: 0.3881
   Boosted from top docs: ⭐

4. 'hibernate'
   Combined Score: 0.2226 (40% stat + 60% positional)
   Statistical: 0.4200 | Positional: 0.3308
   Boosted from top docs: ⭐
   Foreground: 3 docs | Background: 3 docs

5. 'sumatra'
   Combined Score: 0.0778 (40% stat + 60% positional)
   Statistical: 0.5600 | Positional: 0.0000
   Foreground: 4 docs | Background: 4 docs

6. 'beans'
   Combined Score: 0.0241 (40% stat + 60% positional)
   Statistical: 0.4200 | Positional: 0.0000
   Foreground: 3 docs | Background: 3 docs

7. 'coffee'
   Combined Score: 0.0000 (40% stat + 60% positional)
   Statistical: 0.3571 | Positional: 0.0000
   Foreground: 5 docs | Background: 7 docs

============================================================
GENERATED SPARSE QUERY
============================================================
Query: java AND (programming OR jvm OR hibernate OR sumatra)

This sparse query 'explains' what the dense vector region represents!
(Lucene boosts left as an exercise for the reader)

============================================================
STEP 5: EXECUTING SPARSE KEYWORD QUERY (BM25 with boosts)
============================================================
Searching for: java, programming, jvm, hibernate, sumatra

============================================================
RAW OPENSEARCH QUERY: Sparse Keyword (BM25 with boosts)
============================================================
{
  "size": 10,
  "query": {
    "bool": {
      "must": [
        {
          "multi_match": {
            "query": "java",
            "fields": [
              "description^2",
              "description_keywords"
            ],
            "type": "best_fields",
            "operator": "and"
          }
        }
      ],
      "should": [
        {
          "multi_match": {
            "query": "programming",
            "fields": [
              "description^2",
              "description_keywords"
            ],
            "type": "best_fields",
            "operator": "and"
          }
        },
        {
          "multi_match": {
            "query": "jvm",
            "fields": [
              "description^2",
              "description_keywords"
            ],
            "type": "best_fields",
            "operator": "and"
          }
        },
        {
          "multi_match": {
            "query": "hibernate",
            "fields": [
              "description^2",
              "description_keywords"
            ],
            "type": "best_fields",
            "operator": "and"
          }
        },
        {
          "multi_match": {
            "query": "sumatra",
            "fields": [
              "description^2",
              "description_keywords"
            ],
            "type": "best_fields",
            "operator": "and"
          }
        }
      ],
      "minimum_should_match": 1
    }
  },
  "_source": [
    "title",
    "description",
    "category"
  ]
}

Found 8 results using sparse keyword search

Top Results from Sparse Query:
------------------------------------------------------------
1. [programming] Java Programming Guide
   Learn Java programming with Hibernate, Spring Framework, and JVM optimization...
   Score: 14.6442

2. [programming] Scala for Java Developers
   Transition from Java to Scala, covering functional programming and JVM internals...
   Score: 10.3410

3. [programming] Backend Development with Java
   Build scalable backend systems using Java, Hibernate ORM, and microservices...
   Score: 5.5783

4. [programming] Java Enterprise Patterns
   Design patterns for Java enterprise applications using Spring and Hibernate...
   Score: 5.5783

5. [coffee] Sumatran Coffee Roasting Guide
   Learn to roast Java and Sumatra coffee beans to perfection...
   Score: 5.0836

6. [coffee] Coffee Roast Techniques
   Master the art of roasting Java coffee beans from Sumatra...
   Score: 5.0836

7. [coffee] Java Coffee Beans from Sumatra
   Premium coffee beans from the Indonesian island of Java, Sumatra region...
   Score: 4.8794

8. [coffee] Indonesian Coffee Origins
   Explore Java and Sumatra coffee origins, tasting notes and brewing methods...
   Score: 4.8794

============================================================
COMPARISON: Dense Vector vs Sparse Keyword Results
============================================================
Dense Vector Results: 10 documents
Sparse Keyword Results: 8 documents
Overlap: 8 documents (80.0%)

This demonstrates how the wormhole bridges semantic and lexical search!


Query: Java

```