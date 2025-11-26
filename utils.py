"""
Utility functions for GAT-ETM training and evaluation
"""
import numpy as np
from scipy.sparse import coo_matrix
from collections import Counter
from sklearn.neighbors import NearestNeighbors

def nearest_neighbors(X, k=10):
    """
    Find k nearest neighbors using cosine similarity
    
    Args:
        X: feature matrix (N x D)
        k: number of neighbors
    
    Returns:
        indices: neighbor indices (N x k)
        distances: neighbor distances (N x k)
    """
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(X)
    distances, indices = nbrs.kneighbors(X)
    return indices[:, 1:], distances[:, 1:]

def get_topic_coherence(beta, train_data, top_n=10):
    """
    Calculate Topic Coherence metric
    
    Topic Coherence measures how well the top words in a topic
    co-occur together in the training documents.
    
    Args:
        beta: topic-word distribution (K x V) where K is num_topics, V is vocab_size
        train_data: scipy sparse matrix (N x V) where N is num_docs
        top_n: number of top words to consider for coherence
    
    Returns:
        coherence: average topic coherence score
    """
    K, V = beta.shape
    coherence_scores = []
    
    for k in range(K):
        # Get top_n words with highest probability
        top_words = np.argsort(beta[k])[-top_n:][::-1]
        
        # Calculate coherence for topic k
        coherence_k = 0
        count = 0
        
        for i, word_i in enumerate(top_words):
            for j, word_j in enumerate(top_words[i+1:], start=i+1):
                # Count documents containing both word_i and word_j
                docs_with_i = train_data[:, word_i].toarray().flatten() > 0
                docs_with_j = train_data[:, word_j].toarray().flatten() > 0
                docs_with_both = docs_with_i & docs_with_j
                
                if docs_with_j.sum() > 0:
                    # PMI-like measure: log(P(word_i, word_j) / P(word_j))
                    coherence_k += np.log((docs_with_both.sum() + 1) / docs_with_j.sum())
                    count += 1
        
        if count > 0:
            coherence_scores.append(coherence_k / count)
        else:
            coherence_scores.append(0)
    
    return np.mean(coherence_scores)

def get_topic_diversity(beta, top_n=10):
    """
    Calculate Topic Diversity metric
    
    Topic Diversity measures how diverse the topics are by counting
    the unique words across all topics.
    
    Args:
        beta: topic-word distribution (K x V)
        top_n: number of top words to consider for each topic
    
    Returns:
        diversity: diversity score (0-1), higher is better
    """
    K, V = beta.shape
    unique_words = set()
    
    for k in range(K):
        # Get top_n words for topic k
        top_words = np.argsort(beta[k])[-top_n:][::-1]
        unique_words.update(top_words)
    
    # Diversity = unique words / (num_topics * top_n)
    diversity = len(unique_words) / (K * top_n)
    return diversity

