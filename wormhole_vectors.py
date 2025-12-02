"""
Wormhole Vectors: Bridging Dense Vector Space to Sparse Keyword Space

This script demonstrates the concept of "wormhole vectors" by:
1. Performing a dense vector search in OpenSearch
2. Using statistical analysis (significant_terms aggregation) to find keywords
   that are statistically significant in the foreground (query results) vs background (entire index)
3. Generating a sparse keyword query that "explains" the dense vector region
"""

import os
import json
import argparse
from typing import List, Dict, Any, Optional
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import bulk
import numpy as np
from sentence_transformers import SentenceTransformer


class WormholeVectorSearch:
    """
    Implements the Wormhole Vector traversal pattern:
    Dense Vector Query → Document Set → Statistical Keyword Analysis → Sparse Query
    """
    
    def __init__(self, opensearch_url: str, username: str, password: str, 
                 index_name: str = "wormhole_demo", use_ssl: bool = True):
        """
        Initialize the OpenSearch client and embedding model.
        
        Args:
            opensearch_url: URL of the OpenSearch instance (e.g., https://your-instance.aivencloud.com:port)
            username: OpenSearch username
            password: OpenSearch password
            index_name: Name of the index to use
            use_ssl: Whether to use SSL (typically True for Aiven)
        """
        self.index_name = index_name
        self.client = OpenSearch(
            hosts=[opensearch_url],
            http_auth=(username, password),
            use_ssl=use_ssl,
            verify_certs=use_ssl,
            connection_class=RequestsHttpConnection
        )
        
        # Load embedding model (using a lightweight model for demo)
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded successfully!")
    
    def create_index(self, dimension: int = 384):
        """
        Create an OpenSearch index with both text and k-NN vector fields.
        
        Args:
            dimension: Dimension of the embedding vectors (384 for all-MiniLM-L6-v2)
        """
        index_body = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100
                },
                "analysis": {
                    "analyzer": {
                        "keyword_analyzer": {
                            "type": "standard",
                            "stopwords": "_english_"
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "title": {
                        "type": "text",
                        "fields": {
                            "keyword": {
                                "type": "keyword"
                            }
                        }
                    },
                    "description": {
                        "type": "text",
                        "fields": {
                            "keyword": {
                                "type": "keyword"
                            }
                        }
                    },
                    "description_keywords": {
                        "type": "text",
                        "analyzer": "keyword_analyzer",
                        "fielddata": True
                    },
                    "embedding_vector": {
                        "type": "knn_vector",
                        "dimension": dimension,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                            "parameters": {
                                "ef_construction": 128,
                                "m": 24
                            }
                        }
                    },
                    "category": {
                        "type": "keyword"
                    }
                }
            }
        }
        
        # Delete index if it exists
        if self.client.indices.exists(index=self.index_name):
            print(f"Deleting existing index: {self.index_name}")
            self.client.indices.delete(index=self.index_name)
        
        # Create new index
        print(f"Creating index: {self.index_name}")
        self.client.indices.create(index=self.index_name, body=index_body)
        print("Index created successfully!")
    
    def generate_sample_data(self) -> List[Dict[str, Any]]:
        """
        Generate sample data that demonstrates the wormhole concept.
        Includes ambiguous terms like "Java" (programming vs coffee) to show disambiguation.
        """
        sample_docs = [
            # Java Programming Cluster
            {
                "title": "Java Programming Guide",
                "description": "Learn Java programming with Hibernate, Spring Framework, and JVM optimization",
                "category": "programming"
            },
            {
                "title": "Scala for Java Developers",
                "description": "Transition from Java to Scala, covering functional programming and JVM internals",
                "category": "programming"
            },
            {
                "title": "Backend Development with Java",
                "description": "Build scalable backend systems using Java, Hibernate ORM, and microservices",
                "category": "programming"
            },
            {
                "title": "JVM Performance Tuning",
                "description": "Optimize Java Virtual Machine performance for enterprise applications",
                "category": "programming"
            },
            {
                "title": "Java Enterprise Patterns",
                "description": "Design patterns for Java enterprise applications using Spring and Hibernate",
                "category": "programming"
            },
            
            # Java Coffee Cluster
            {
                "title": "Java Coffee Beans from Sumatra",
                "description": "Premium coffee beans from the Indonesian island of Java, Sumatra region",
                "category": "coffee"
            },
            {
                "title": "Sumatran Coffee Roasting Guide",
                "description": "Learn to roast Java and Sumatra coffee beans to perfection",
                "category": "coffee"
            },
            {
                "title": "Island Coffee Collection",
                "description": "Exotic coffee from Java island, featuring unique flavor profiles",
                "category": "coffee"
            },
            {
                "title": "Coffee Roast Techniques",
                "description": "Master the art of roasting Java coffee beans from Sumatra",
                "category": "coffee"
            },
            {
                "title": "Indonesian Coffee Origins",
                "description": "Explore Java and Sumatra coffee origins, tasting notes and brewing methods",
                "category": "coffee"
            },
            
            # Server Ambiguity Examples
            {
                "title": "Linux Server Administration",
                "description": "Manage Linux servers, DevOps practices, rack mounting, and server infrastructure",
                "category": "tech"
            },
            {
                "title": "DevOps Server Management",
                "description": "Automate server deployment, container orchestration, and infrastructure as code",
                "category": "tech"
            },
            {
                "title": "Server Hardware Guide",
                "description": "Choose the right server hardware, rack configurations, and networking equipment",
                "category": "tech"
            },
            {
                "title": "Restaurant Server Training",
                "description": "Professional waiter training, food service tips, and customer service excellence",
                "category": "hospitality"
            },
            {
                "title": "Food Service Best Practices",
                "description": "Learn restaurant server skills, tipping etiquette, and food presentation",
                "category": "hospitality"
            },
            {
                "title": "Waiter Service Guide",
                "description": "Master the art of serving food, managing tables, and providing excellent service",
                "category": "hospitality"
            },
            
            # Additional documents for background
            {
                "title": "Python Data Science",
                "description": "Data analysis with Python, pandas, numpy, and machine learning",
                "category": "programming"
            },
            {
                "title": "JavaScript Web Development",
                "description": "Build modern web applications with React, Node.js, and TypeScript",
                "category": "programming"
            },
            {
                "title": "Espresso Coffee Guide",
                "description": "Learn to make perfect espresso, latte art, and coffee brewing techniques",
                "category": "coffee"
            },
            {
                "title": "Italian Coffee Culture",
                "description": "Explore Italian coffee traditions, espresso machines, and café culture",
                "category": "coffee"
            },
            {
                "title": "Cloud Infrastructure",
                "description": "Deploy applications on AWS, Azure, and Google Cloud platforms",
                "category": "tech"
            },
            {
                "title": "Database Management",
                "description": "SQL, NoSQL databases, data modeling, and query optimization",
                "category": "tech"
            },
            {
                "title": "Cooking Techniques",
                "description": "Master culinary skills, food preparation, and kitchen management",
                "category": "hospitality"
            },
            {
                "title": "Hotel Management",
                "description": "Hospitality industry management, guest services, and hotel operations",
                "category": "hospitality"
            }
        ]
        
        # Generate embeddings and prepare documents
        documents = []
        for doc in sample_docs:
            # Create embedding
            text = f"{doc['title']} {doc['description']}"
            embedding = self.embedding_model.encode(text).tolist()
            
            # Extract keywords from description (simple tokenization)
            keywords = doc['description'].lower().split()
            # Remove common stopwords
            stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
            keywords = [k.strip('.,!?;:()[]{}') for k in keywords if k not in stopwords and len(k) > 2]
            
            documents.append({
                "_index": self.index_name,
                "_source": {
                    "title": doc["title"],
                    "description": doc["description"],
                    "description_keywords": " ".join(keywords),
                    "embedding_vector": embedding,
                    "category": doc["category"]
                }
            })
        
        return documents
    
    def index_documents(self, documents: List[Dict[str, Any]]):
        """
        Index documents into OpenSearch using bulk API.
        
        Args:
            documents: List of documents to index
        """
        print(f"Indexing {len(documents)} documents...")
        success, failed = bulk(self.client, documents, chunk_size=100, request_timeout=60)
        print(f"Indexed {success} documents successfully")
        if failed:
            print(f"Failed to index {len(failed)} documents")
        
        # Refresh index to make documents searchable
        self.client.indices.refresh(index=self.index_name)
        print("Index refreshed!")
    
    def dense_vector_search(self, query_text: str, k: int = 50) -> Dict[str, Any]:
        """
        Perform a dense vector search using k-NN.
        
        Args:
            query_text: Text query to convert to embedding
            k: Number of results to retrieve (foreground set size)
        
        Returns:
            Search response from OpenSearch
        """
        # Generate query embedding
        query_vector = self.embedding_model.encode(query_text).tolist()
        
        # Perform k-NN search
        query_body = {
            "size": k,
            "query": {
                "knn": {
                    "embedding_vector": {
                        "vector": query_vector,
                        "k": k
                    }
                }
            },
            "_source": ["title", "description", "category"]
        }
        
        print(f"\n{'='*60}")
        print(f"DENSE VECTOR SEARCH")
        print(f"{'='*60}")
        print(f"Query: '{query_text}'")
        print(f"Retrieving top {k} documents (Foreground Set)...\n")
        
        response = self.client.search(index=self.index_name, body=query_body)
        
        print(f"Found {response['hits']['total']['value']} results\n")
        print("Top Results:")
        print("-" * 60)
        for i, hit in enumerate(response['hits']['hits'][:10], 1):
            print(f"{i}. [{hit['_source']['category']}] {hit['_source']['title']}")
            print(f"   {hit['_source']['description'][:80]}...")
            print(f"   Score: {hit['_score']:.4f}\n")
        
        return response
    
    def extract_and_weight_terms_from_top_docs(self, hits: List[Dict[str, Any]], top_n: int = 5) -> Dict[str, float]:
        """
        Extract terms from top-ranked documents and weight them by position.
        Terms from higher-ranked documents get more weight.
        
        Args:
            hits: List of search result hits
            top_n: Number of top documents to consider
        
        Returns:
            Dictionary mapping terms to their weighted scores
        """
        term_weights = {}
        term_sources = {}  # Track which documents contributed to each term
        
        # Process top N documents with position-based weighting
        for rank, hit in enumerate(hits[:top_n], 1):
            # Weight decreases with rank: top doc gets weight 1.0, 2nd gets ~0.71, etc.
            position_weight = 1.0 / (rank ** 0.5)  # Decay factor
            
            # Extract text from title and description
            title = hit['_source'].get('title', '').lower()
            description = hit['_source'].get('description', '').lower()
            text = f"{title} {description}"
            
            # Simple tokenization (matching the indexing approach)
            stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
            keywords = text.split()
            keywords = [k.strip('.,!?;:()[]{}') for k in keywords if k not in stopwords and len(k) > 2]
            
            # Count term frequency in this document and apply position weight
            for term in keywords:
                if term not in term_weights:
                    term_weights[term] = 0.0
                    term_sources[term] = []
                term_weights[term] += position_weight
                term_sources[term].append((rank, position_weight, hit['_source'].get('title', 'Untitled')))
        
        # Store sources for later display
        self._term_sources = term_sources
        
        return term_weights
    
    def combine_keyword_scores(self, statistical_keywords: List[Dict[str, Any]], 
                               positional_weights: Dict[str, float],
                               statistical_weight: float = 0.4,
                               positional_weight: float = 0.6) -> List[Dict[str, Any]]:
        """
        Combine statistical significance scores with positional weights from top documents.
        
        Args:
            statistical_keywords: Keywords from significant_terms aggregation
            positional_weights: Term weights from top documents
            statistical_weight: Weight for statistical significance (0-1)
            positional_weight: Weight for positional importance (0-1)
        
        Returns:
            Re-ranked list of keywords with combined scores
        """
        # Normalize positional weights to 0-1 range
        normalized_pos_weights = {}
        if positional_weights:
            max_pos_weight = max(positional_weights.values())
            if max_pos_weight > 0:
                normalized_pos_weights = {k: v / max_pos_weight for k, v in positional_weights.items()}
        
        # Normalize statistical scores to 0-1 range
        normalized_stat_scores = {}
        if statistical_keywords:
            stat_scores = [kw['score'] for kw in statistical_keywords]
            max_stat_score = max(stat_scores) if stat_scores else 1.0
            min_stat_score = min(stat_scores) if stat_scores else 0.0
            stat_range = max_stat_score - min_stat_score if max_stat_score > min_stat_score else 1.0
            
            for kw in statistical_keywords:
                # Normalize to 0-1 range
                if stat_range > 0:
                    normalized_stat_scores[kw['term']] = (kw['score'] - min_stat_score) / stat_range
                else:
                    normalized_stat_scores[kw['term']] = 1.0
        
        # Create a combined score for each keyword
        keyword_scores = {}
        
        # Add statistical keywords
        for kw in statistical_keywords:
            term = kw['term']
            stat_score = normalized_stat_scores.get(term, 0.0)
            pos_score = normalized_pos_weights.get(term, 0.0)
            
            # Combine normalized scores
            keyword_scores[term] = {
                'term': term,
                'statistical_score': kw['score'],  # Keep original for display
                'positional_score': pos_score,
                'combined_score': (statistical_weight * stat_score) + (positional_weight * pos_score),
                'foreground_count': kw.get('foreground_count', 0),
                'background_count': kw.get('background_count', 0)
            }
        
        # Add high-positional-weight terms that might not be in statistical results
        for term, pos_score in normalized_pos_weights.items():
            if term not in keyword_scores and pos_score > 0.3:  # Threshold for inclusion
                keyword_scores[term] = {
                    'term': term,
                    'statistical_score': 0.0,
                    'positional_score': pos_score,
                    'combined_score': positional_weight * pos_score,
                    'foreground_count': 0,
                    'background_count': 0
                }
        
        # Sort by combined score
        sorted_keywords = sorted(keyword_scores.values(), key=lambda x: x['combined_score'], reverse=True)
        
        return sorted_keywords
    
    def sparse_keyword_search(self, keywords: List[str], size: int = 10) -> Dict[str, Any]:
        """
        Execute a sparse keyword search (BM25) using the generated wormhole keywords.
        
        Args:
            keywords: List of keywords to search for
            size: Number of results to retrieve
        
        Returns:
            Search response from OpenSearch
        """
        if not keywords:
            return {"hits": {"hits": [], "total": {"value": 0}}}
        
        # Build a bool query with proper boolean logic:
        # Top keyword (most significant) is required (MUST)
        # Remaining keywords are optional (SHOULD with minimum_should_match = 1)
        primary_term = keywords[0]
        optional_terms = keywords[1:] if len(keywords) > 1 else []
        
        # Must clause: primary term is required
        must_clause = {
            "multi_match": {
                "query": primary_term,
                "fields": ["description^2", "description_keywords"],
                "type": "best_fields",
                "operator": "and"
            }
        }
        
        # Should clauses: optional terms (at least one should match)
        should_clauses = []
        for term in optional_terms:
            should_clauses.append({
                "multi_match": {
                    "query": term,
                    "fields": ["description^2", "description_keywords"],
                    "type": "best_fields",
                    "operator": "and"
                }
            })
        
        # Build query: primary_term AND (term1 OR term2 OR ...)
        bool_query = {"must": [must_clause]}
        if should_clauses:
            bool_query["should"] = should_clauses
            bool_query["minimum_should_match"] = 1  # At least one optional term should match
        
        query_body = {
            "size": size,
            "query": {
                "bool": bool_query
            },
            "_source": ["title", "description", "category"]
        }
        
        # Store query body for display
        self._last_sparse_query = query_body
        
        response = self.client.search(index=self.index_name, body=query_body)
        return response
    
    def wormhole_traversal(self, query_text: str, k: int = 50, keyword_size: int = 10) -> Dict[str, Any]:
        """
        Perform the complete wormhole traversal:
        1. Dense vector search (foreground set)
        2. Statistical keyword analysis (significant_terms)
        3. Generate sparse keyword query
        
        Args:
            query_text: Text query to convert to embedding
            k: Number of results to retrieve (foreground set size)
            keyword_size: Number of significant keywords to extract
        
        Returns:
            Dictionary containing search results and wormhole keywords
        """
        # Generate query embedding
        query_vector = self.embedding_model.encode(query_text).tolist()
        
        # Perform k-NN search with significant_terms aggregation
        query_body = {
            "size": k,
            "query": {
                "knn": {
                    "embedding_vector": {
                        "vector": query_vector,
                        "k": k
                    }
                }
            },
            "aggs": {
                "wormhole_keywords": {
                    "significant_terms": {
                        "field": "description_keywords",
                        "size": keyword_size,
                        "background_filter": {
                            "match_all": {}
                        }
                    }
                }
            },
            "_source": ["title", "description", "category"]
        }
        
        print(f"\n{'='*60}")
        print(f"WORMHOLE VECTOR TRAVERSAL")
        print(f"{'='*60}")
        print(f"Query: '{query_text}'")
        print(f"Step 1: Dense Vector Search → Retrieving top {k} documents (Foreground Set)")
        print(f"Step 2: Statistical Analysis → Finding significant terms (Foreground vs Background)")
        print(f"Step 3: Positional Weighting → Boosting terms from top-ranked documents")
        print(f"Step 4: Generate Sparse Query → Combine statistical + positional scores")
        print(f"Step 5: Execute Sparse Query → Show documents retrieved by keyword search\n")
        
        # Print raw OpenSearch query for dense vector search
        print("="*60)
        print("RAW OPENSEARCH QUERY: Dense Vector (k-NN)")
        print("="*60)
        query_display = json.loads(json.dumps(query_body))  # Deep copy
        # Show vector info separately
        vector_length = 0
        if 'query' in query_display and 'knn' in query_display['query']:
            vector = query_display['query']['knn']['embedding_vector']['vector']
            vector_length = len(vector)
            # Show first 5 dimensions as example
            sample_size = min(5, vector_length)
            query_display['query']['knn']['embedding_vector']['vector'] = vector[:sample_size]
        print(json.dumps(query_display, indent=2))
        if vector_length > 5:
            print(f"\nNote: Vector truncated for display. Full vector has {vector_length} dimensions.")
        print()
        
        response = self.client.search(index=self.index_name, body=query_body)
        
        # Extract results
        hits = response['hits']['hits']
        total = response['hits']['total']['value']
        
        # Display results
        print(f"Found {total} results in foreground set\n")
        print("Top Results:")
        print("-" * 60)
        for i, hit in enumerate(hits[:10], 1):
            print(f"{i}. [{hit['_source']['category']}] {hit['_source']['title']}")
            print(f"   {hit['_source']['description'][:80]}...")
            print(f"   Score: {hit['_score']:.4f}\n")
        
        # Extract statistical keywords from aggregation
        statistical_keywords = []
        if 'aggregations' in response and 'wormhole_keywords' in response['aggregations']:
            buckets = response['aggregations']['wormhole_keywords']['buckets']
            statistical_keywords = [
                {
                    "term": bucket['key'],
                    "score": bucket['score'],
                    "foreground_count": bucket['doc_count'],
                    "background_count": bucket['bg_count']
                }
                for bucket in buckets
            ]
        
        # Extract and weight terms from top documents
        positional_weights = self.extract_and_weight_terms_from_top_docs(hits, top_n=5)
        
        # Store raw positional weights for boost calculation
        self._raw_positional_weights = positional_weights
        
        # Show positional weighting breakdown
        if positional_weights and hasattr(self, '_term_sources'):
            print("\n" + "="*60)
            print("POSITIONAL WEIGHTING FROM TOP DOCUMENTS")
            print("="*60)
            print("Top terms extracted from top 5 documents (weighted by rank):")
            print("Rank weights: #1=1.00, #2=0.71, #3=0.58, #4=0.50, #5=0.45\n")
            sorted_pos_weights = sorted(positional_weights.items(), key=lambda x: x[1], reverse=True)
            for i, (term, weight) in enumerate(sorted_pos_weights[:15], 1):
                sources = self._term_sources.get(term, [])
                source_info = ", ".join([f"#{r}({w:.2f})" for r, w, _ in sources[:3]])
                print(f"{i:2}. '{term}': {weight:.4f} [from: {source_info}]")
            print()
        
        # Combine statistical significance with positional weighting
        wormhole_keywords = self.combine_keyword_scores(
            statistical_keywords,
            positional_weights,
            statistical_weight=0.4,  # Less weight on pure statistics
            positional_weight=0.6    # More weight on top document terms
        )
        
        print("\n" + "="*60)
        print("WORMHOLE KEYWORDS (Combined: Statistical + Top Document Weighting)")
        print("="*60)
        print("Terms are weighted by:")
        print("  - Statistical significance (40%): Foreground vs Background")
        print("  - Positional importance (60%): Terms from top-ranked documents\n")
        
        if wormhole_keywords:
            for i, kw in enumerate(wormhole_keywords[:keyword_size], 1):
                print(f"{i}. '{kw['term']}'")
                print(f"   Combined Score: {kw['combined_score']:.4f} (40% stat + 60% positional)")
                print(f"   Statistical: {kw['statistical_score']:.4f} | Positional: {kw['positional_score']:.4f}")
                if kw['positional_score'] > 0:
                    boost_indicator = "⭐" * min(5, int(kw['positional_score'] * 5))
                    print(f"   Boosted from top docs: {boost_indicator}")
                if kw['foreground_count'] > 0:
                    print(f"   Foreground: {kw['foreground_count']} docs | Background: {kw['background_count']} docs")
                print()
            
            # Generate sparse query string with proper boolean logic
            # Format: primary_term AND (term1 OR term2 OR term3 OR ...)
            top_keywords = wormhole_keywords[:5]
            
            if len(top_keywords) > 1:
                primary_kw = top_keywords[0]
                optional_terms = [kw['term'] for kw in top_keywords[1:]]
                sparse_query = f"{primary_kw['term']} AND ({' OR '.join(optional_terms)})"
            else:
                primary_kw = top_keywords[0] if top_keywords else None
                if primary_kw:
                    sparse_query = primary_kw['term']
                else:
                    sparse_query = ""
            
            print("="*60)
            print("GENERATED SPARSE QUERY")
            print("="*60)
            print(f"Query: {sparse_query}")
            print("\nThis sparse query 'explains' what the dense vector region represents!")
            print("(Lucene boosts left as an exercise for the reader)")
            
            # Step 5: Execute the sparse query
            print("\n" + "="*60)
            print("STEP 5: EXECUTING SPARSE KEYWORD QUERY (BM25 with boosts)")
            print("="*60)
            top_terms_list = [kw['term'] for kw in wormhole_keywords[:5]]
            print(f"Searching for: {', '.join(top_terms_list)}\n")
            
            sparse_results = self.sparse_keyword_search(top_terms_list, size=k)
            
            # Print raw OpenSearch query for sparse keyword search
            if hasattr(self, '_last_sparse_query'):
                print("="*60)
                print("RAW OPENSEARCH QUERY: Sparse Keyword (BM25 with boosts)")
                print("="*60)
                print(json.dumps(self._last_sparse_query, indent=2))
                print()
            sparse_hits = sparse_results['hits']['hits']
            sparse_total = sparse_results['hits']['total']['value']
            
            print(f"Found {sparse_total} results using sparse keyword search\n")
            print("Top Results from Sparse Query:")
            print("-" * 60)
            
            if sparse_hits:
                for i, hit in enumerate(sparse_hits[:10], 1):
                    print(f"{i}. [{hit['_source']['category']}] {hit['_source']['title']}")
                    print(f"   {hit['_source']['description'][:80]}...")
                    print(f"   Score: {hit['_score']:.4f}\n")
            else:
                print("No results found with the sparse query.")
            
            # Compare overlap between dense and sparse results
            dense_doc_ids = {hit['_id'] for hit in hits}
            sparse_doc_ids = {hit['_id'] for hit in sparse_hits}
            overlap = len(dense_doc_ids & sparse_doc_ids)
            overlap_percentage = (overlap / len(dense_doc_ids) * 100) if dense_doc_ids else 0
            
            print("="*60)
            print("COMPARISON: Dense Vector vs Sparse Keyword Results")
            print("="*60)
            print(f"Dense Vector Results: {len(dense_doc_ids)} documents")
            print(f"Sparse Keyword Results: {len(sparse_doc_ids)} documents")
            print(f"Overlap: {overlap} documents ({overlap_percentage:.1f}%)")
            print("\nThis demonstrates how the wormhole bridges semantic and lexical search!")
        else:
            print("No significant keywords found.")
            sparse_hits = []
            sparse_total = 0
        
        return {
            "query": query_text,
            "total_results": total,
            "hits": hits,
            "wormhole_keywords": wormhole_keywords,
            "sparse_query_results": sparse_hits,
            "sparse_total": sparse_total
        }
    
    def demonstrate_disambiguation(self):
        """
        Demonstrate the disambiguation power of wormhole vectors using the "Java" example.
        """
        print("\n" + "="*80)
        print("DEMONSTRATION: Disambiguation via Wormhole Vectors")
        print("="*80)
        print("\nThe word 'Java' is ambiguous - it could mean:")
        print("  1. Java programming language (Hibernate, Scala, JVM, Backend)")
        print("  2. Java coffee/region (Sumatra, Coffee, Roast, Island)")
        print("\nLet's see how wormhole vectors help disambiguate...\n")
        
        # Query 1: Should retrieve programming-related Java docs
        print("\n" + "="*80)
        print("QUERY 1: 'Java programming language'")
        print("="*80)
        result1 = self.wormhole_traversal("Java programming language", k=10, keyword_size=10)
        
        # Query 2: Should retrieve coffee-related Java docs
        print("\n" + "="*80)
        print("QUERY 2: 'Java coffee beans'")
        print("="*80)
        result2 = self.wormhole_traversal("Java coffee beans", k=10, keyword_size=10)
        
        # Compare the wormhole keywords
        print("\n" + "="*80)
        print("COMPARISON: How Wormhole Keywords Differ")
        print("="*80)
        print("\nQuery 1 Keywords (Programming):")
        for kw in result1['wormhole_keywords'][:5]:
            print(f"  - {kw['term']} (combined score: {kw['combined_score']:.4f})")
        
        print("\nQuery 2 Keywords (Coffee):")
        for kw in result2['wormhole_keywords'][:5]:
            print(f"  - {kw['term']} (combined score: {kw['combined_score']:.4f})")
        
        print("\n" + "="*80)
        print("CONCLUSION")
        print("="*80)
        print("The same ambiguous term 'Java' produces different wormhole keywords,")
        print("allowing the system to understand context and disambiguate automatically!")
    
    def interactive_query_mode(self):
        """
        Interactive mode for experimenting with queries.
        """
        print("\n" + "="*80)
        print("INTERACTIVE QUERY MODE")
        print("="*80)
        print("Enter queries to explore wormhole vectors. Type 'quit' or 'exit' to stop.\n")
        
        while True:
            try:
                query = input("Enter your query (or 'quit' to exit): ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\nExiting interactive mode. Goodbye!")
                    break
                
                if not query:
                    print("Please enter a valid query.")
                    continue
                
                self.wormhole_traversal(query, k=10, keyword_size=10)
                
            except KeyboardInterrupt:
                print("\n\nExiting interactive mode. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")


def main():
    """
    Main function to run the wormhole vector demonstration.
    """
    parser = argparse.ArgumentParser(
        description="Wormhole Vectors: Bridging Dense Vector Space to Sparse Keyword Space",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full demo with indexing
  python wormhole_vectors.py
  
  # Skip indexing, only run queries
  python wormhole_vectors.py --skip-indexing
  
  # Run a single query
  python wormhole_vectors.py --skip-indexing --query "Java programming"
  
  # Interactive query mode
  python wormhole_vectors.py --skip-indexing --interactive
  
  # Custom parameters
  python wormhole_vectors.py --skip-indexing --query "server" --k 20 --keyword-size 15
        """
    )
    
    parser.add_argument(
        "--skip-indexing",
        action="store_true",
        help="Skip creating index and indexing documents (use existing index)"
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Run a single query and display results"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Enter interactive query mode"
    )
    
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of results to retrieve for foreground set (default: 10)"
    )
    
    parser.add_argument(
        "--keyword-size",
        type=int,
        default=10,
        help="Number of significant keywords to extract (default: 10)"
    )
    
    parser.add_argument(
        "--index-name",
        type=str,
        default="wormhole_demo",
        help="Name of the OpenSearch index (default: wormhole_demo)"
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the full disambiguation demonstration"
    )
    
    args = parser.parse_args()
    
    # Configuration - Update these with your Aiven OpenSearch credentials
    OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "https://os-91a381d-wormhole-vectors-dev-sandbox.d.aivencloud.com:12691")
    OPENSEARCH_USERNAME = os.getenv("OPENSEARCH_USERNAME", "user")
    OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD", "password")
    
    # Ensure URL has protocol
    if not OPENSEARCH_URL.startswith(("http://", "https://")):
        OPENSEARCH_URL = f"https://{OPENSEARCH_URL}"
    
    if "your-instance" in OPENSEARCH_URL:
        print("="*80)
        print("CONFIGURATION REQUIRED")
        print("="*80)
        print("\nPlease set the following environment variables:")
        print("  - OPENSEARCH_URL: Your Aiven OpenSearch URL")
        print("  - OPENSEARCH_USERNAME: Your OpenSearch username")
        print("  - OPENSEARCH_PASSWORD: Your OpenSearch password")
        print("\nOr update the values in the main() function.")
        print("\nExample:")
        print("  export OPENSEARCH_URL='https://your-instance.aivencloud.com:12345'")
        print("  export OPENSEARCH_USERNAME='user'")
        print("  export OPENSEARCH_PASSWORD='your-password'")
        return
    
    # Initialize wormhole search
    wormhole = WormholeVectorSearch(
        opensearch_url=OPENSEARCH_URL,
        username=OPENSEARCH_USERNAME,
        password=OPENSEARCH_PASSWORD,
        index_name=args.index_name
    )
    
    # Check if index exists when skipping indexing
    if args.skip_indexing:
        if not wormhole.client.indices.exists(index=args.index_name):
            print(f"Error: Index '{args.index_name}' does not exist.")
            print("Please run without --skip-indexing first to create the index and load data.")
            return
        print(f"Using existing index: {args.index_name}")
    else:
        # Create index
        wormhole.create_index()
        
        # Generate and index sample data
        documents = wormhole.generate_sample_data()
        wormhole.index_documents(documents)
    
    # Handle different execution modes
    if args.query:
        # Single query mode
        wormhole.wormhole_traversal(args.query, k=args.k, keyword_size=args.keyword_size)
    elif args.interactive:
        # Interactive mode
        wormhole.interactive_query_mode()
    elif args.demo:
        # Full demonstration
        wormhole.demonstrate_disambiguation()
        
        # Additional examples
        print("\n" + "="*80)
        print("ADDITIONAL EXAMPLES")
        print("="*80)
        
        # Example: Server ambiguity
        print("\nExample: 'server' query (ambiguous between tech and hospitality)")
        wormhole.wormhole_traversal("server", k=args.k, keyword_size=args.keyword_size)
        
        print("\n" + "="*80)
        print("Demo complete! You can now experiment with your own queries.")
        print("="*80)
    else:
        # Default: Run full demo
        wormhole.demonstrate_disambiguation()
        
        # Additional examples
        print("\n" + "="*80)
        print("ADDITIONAL EXAMPLES")
        print("="*80)
        
        # Example: Server ambiguity
        print("\nExample: 'server' query (ambiguous between tech and hospitality)")
        wormhole.wormhole_traversal("server", k=args.k, keyword_size=args.keyword_size)
        
        print("\n" + "="*80)
        print("Demo complete! Use --interactive or --query to experiment with your own queries.")
        print("="*80)


if __name__ == "__main__":
    main()

