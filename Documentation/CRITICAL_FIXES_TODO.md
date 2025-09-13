# Critical Fixes TODO List
**Date**: 2025-09-06  
**Priority**: HIGH - Production Blockers
**Status**: Ready for GPT-5 Implementation

---

## 🔥 CRITICAL PRIORITY FIXES (Production Blockers)

### 1. Fix Guardrails Logic Flaws
**File**: `app/core/guardrails.py`
**Issue**: IndexError potential + hardcoded thresholds

#### 1.1 Fix Array Index Bug (HIGH)
```python
# CURRENT (BROKEN):
if np.mean(rerank_scores[:3]) < self.min_mean_top3:

# FIX TO:
safe_top_scores = rerank_scores[:min(3, len(rerank_scores))]
if len(safe_top_scores) > 0 and np.mean(safe_top_scores) < self.min_mean_top3:
```

#### 1.2 Make Thresholds Configurable (MEDIUM)
```python
# ADD to __init__:
@dataclass
class GuardrailsConfig:
    min_rel_threshold: float = 0.22
    min_mean_top3: float = 0.18
    domain_type: str = "cx_consulting"  # Allow per-domain tuning

# UPDATE constructor:
def __init__(self, config: GuardrailsConfig = None):
    self.config = config or GuardrailsConfig()
```

#### 1.3 Add Confidence Gradients (MEDIUM)
```python
# ADD confidence levels instead of binary yes/no:
@dataclass
class GuardrailsDecision:
    status: str  # "high_confidence", "medium_confidence", "low_evidence", "no_answer"
    confidence_score: float  # 0.0-1.0
    evidence_quality: str  # "strong", "weak", "insufficient"
    recommended_caveats: List[str]  # ["Limited evidence", "Partial coverage"]
```

### 2. Fix RetrievalPipeline Critical Flaws
**File**: `app/services/retrieval_pipeline.py`

#### 2.1 Fix Double Encoding Performance Issue (CRITICAL)
```python
# CURRENT (INEFFICIENT):
"embedding": self.embedder.encode([h.text])[0],  # Re-encodes every doc

# FIX TO:
def _add_embeddings(self, hits: List[RetrievalHit]) -> List[RetrievalHit]:
    """Add embeddings only if missing"""
    texts_to_encode = []
    indices_needing_embedding = []
    
    for i, hit in enumerate(hits):
        if not hasattr(hit, 'embedding') or hit.embedding is None:
            texts_to_encode.append(hit.text)
            indices_needing_embedding.append(i)
    
    if texts_to_encode:
        new_embeddings = self.embedder.encode(texts_to_encode)
        for i, emb in zip(indices_needing_embedding, new_embeddings):
            hits[i].embedding = emb
    
    return hits
```

#### 2.2 Fix Broken RRF Implementation (CRITICAL)
```python
# CURRENT (BROKEN):
def _rrf_fuse(self, a: List[RetrievalHit], b: List[RetrievalHit], k: int, k_rrf: int = 60):
    combined = a + b  # Just concatenates!

# FIX TO:
def _rrf_fuse(self, dense_hits: List[RetrievalHit], sparse_hits: List[RetrievalHit], 
              k: int, k_rrf: int = 60) -> List[RetrievalHit]:
    """Proper Reciprocal Rank Fusion implementation"""
    scores = {}
    
    # Score dense results
    for rank, hit in enumerate(dense_hits):
        doc_id = hit.doc_id
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k_rrf + rank + 1)
    
    # Score sparse results  
    for rank, hit in enumerate(sparse_hits):
        doc_id = hit.doc_id
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k_rrf + rank + 1)
    
    # Merge and deduplicate
    all_hits = {hit.doc_id: hit for hit in dense_hits + sparse_hits}
    
    # Sort by RRF score
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    result = []
    for doc_id, rrf_score in sorted_docs[:k]:
        hit = all_hits[doc_id]
        hit.score = rrf_score  # Update with RRF score
        result.append(hit)
    
    return result
```

#### 2.3 Implement Provider Hooks (CRITICAL)
```python
# CURRENT (USELESS):
def _dense_search(self, query: str, k: int) -> List[RetrievalHit]:
    return []

# FIX TO:
def _dense_search(self, query: str, k: int) -> List[RetrievalHit]:
    """Connect to existing vector search"""
    # Wire to existing DocumentService or similar
    from app.services.document_service import DocumentService
    doc_service = DocumentService()
    
    results = doc_service.vector_search(query, limit=k)
    
    hits = []
    for result in results:
        hits.append(RetrievalHit(
            doc_id=result.get("doc_id"),
            text=result.get("text", ""),
            score=result.get("score", 0.0),
            metadata=result.get("metadata", {})
        ))
    
    return hits

def _sparse_search(self, query: str, k: int) -> List[RetrievalHit]:
    """Connect to existing BM25 search"""
    # Wire to existing BM25 implementation
    from app.services.retrieval import HybridRetriever
    retriever = HybridRetriever()
    
    bm25_results = retriever.bm25_search(query, k=k)
    
    hits = []
    for result in bm25_results:
        hits.append(RetrievalHit(
            doc_id=result.get("doc_id"),
            text=result.get("content", ""),
            score=result.get("bm25_score", 0.0),
            metadata=result.get("metadata", {})
        ))
    
    return hits
```

### 3. Fix Integration Score Mismatch
**File**: `app/agents/agent_runner.py`
**Issue**: Using wrong score types for guardrails

#### 3.1 Use Cross-Encoder Scores (CRITICAL)
```python
# CURRENT (WRONG):
rerank_scores = [float(x.get("score", 0.0)) for x in sorted_ctx[:6]]

# FIX TO:
# Method 1: Use RetrievalPipeline cross-encoder scores
from app.services.retrieval_pipeline import RetrievalPipeline

pipeline = RetrievalPipeline(embedder, bm25, reranker)
hits, metadata = pipeline.search(query, final_k=6)
rerank_scores = [hit.score for hit in hits]  # These are cross-encoder scores

# Method 2: If keeping existing system, adjust thresholds
guardrails_config = GuardrailsConfig(
    min_rel_threshold=0.5,  # Adjusted for BM25/dense scores
    min_mean_top3=0.3      # Adjusted for BM25/dense scores
)
```

#### 3.2 Fix Citation System Integration (MEDIUM)
```python
# CURRENT (HACK):
guard.require_citations = False
answer_json_str = json.dumps({"answer": str(final_answer), "citations": []})

# FIX TO:
# Option 1: Support plaintext mode properly
guardrails_config = GuardrailsConfig(
    require_citations=False,  # Explicit configuration
    output_format="plaintext"  # vs "json"
)

decision = guard.gate(final_answer, rerank_scores, output_format="plaintext")

# Option 2: Generate proper citations
citations = []
for i, context in enumerate(sorted_ctx[:3]):  # Top 3 sources
    citations.append({
        "doc_id": context.get("doc_id", f"source_{i}"),
        "title": context.get("title", "Unknown Source"),
        "relevance_score": context.get("score", 0.0),
        "excerpt": context.get("text", "")[:100]
    })

decision = guard.gate(final_answer, rerank_scores, citations=citations)
```

---

## 🚀 HIGH PRIORITY ENHANCEMENTS

### 4. Add Caching System
**File**: NEW - `app/core/embedding_cache.py`

```python
from functools import lru_cache
import hashlib
import numpy as np

class EmbeddingCache:
    """Cache embeddings to avoid recomputation"""
    
    def __init__(self, max_size: int = 10000):
        self.cache = {}
        self.max_size = max_size
    
    def get_embedding(self, text: str, embedder) -> np.ndarray:
        """Get cached embedding or compute new one"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash in self.cache:
            return self.cache[text_hash]
        
        # Compute new embedding
        embedding = embedder.encode([text])[0]
        
        # Cache management
        if len(self.cache) >= self.max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[text_hash] = embedding
        return embedding
```

### 5. Add Error Recovery
**File**: `app/core/error_recovery.py`

```python
from typing import Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)

class ErrorRecovery:
    """Graceful degradation for pipeline failures"""
    
    @staticmethod
    def with_fallback(primary_func: Callable, fallback_func: Callable, 
                     error_msg: str = "Primary function failed") -> Any:
        """Execute primary function with fallback on failure"""
        try:
            return primary_func()
        except Exception as e:
            logger.warning(f"{error_msg}: {e}, using fallback")
            try:
                return fallback_func()
            except Exception as fallback_e:
                logger.error(f"Fallback also failed: {fallback_e}")
                raise fallback_e from e
    
    @staticmethod
    def safe_mmr(query_emb, candidates, **kwargs):
        """MMR with fallback to simple ranking"""
        try:
            from app.services.mmr_diversifier import mmr_select
            return mmr_select(query_emb, candidates, **kwargs)
        except Exception as e:
            logger.warning(f"MMR failed: {e}, falling back to score ranking")
            # Simple fallback: sort by score
            return sorted(candidates, key=lambda x: x.score, reverse=True)[:kwargs.get('top_k', 10)]
```

### 6. Add Configuration Management
**File**: NEW - `app/core/pipeline_config.py`

```python
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class RetrievalConfig:
    """Centralized configuration for retrieval pipeline"""
    
    # Pipeline settings
    enable_mmr: bool = True
    enable_cross_encoder: bool = True
    enable_caching: bool = True
    
    # Retrieval parameters
    dense_k: int = 20
    sparse_k: int = 20
    final_k: int = 6
    rrf_k: int = 60
    
    # MMR parameters
    mmr_lambda: float = 0.5
    mmr_fetch_k: int = 40
    
    # Guardrails thresholds (per domain)
    guardrails_config: Dict[str, float] = field(default_factory=lambda: {
        "cx_consulting": {"min_rel": 0.22, "min_mean_top3": 0.18},
        "technical_docs": {"min_rel": 0.35, "min_mean_top3": 0.25},
        "general": {"min_rel": 0.20, "min_mean_top3": 0.15}
    })
    
    # Performance settings
    cache_size: int = 10000
    embedding_batch_size: int = 32
    max_text_length: int = 8192
    
    def get_guardrails_config(self, domain: str = "cx_consulting"):
        """Get domain-specific guardrails configuration"""
        return self.guardrails_config.get(domain, self.guardrails_config["general"])
```

---

## 🧪 TESTING REQUIREMENTS

### 7. Add Comprehensive Tests
**Files**: Create test files for each component

#### 7.1 Unit Tests
- `tests/test_mmr_diversifier.py` - MMR algorithm correctness
- `tests/test_guardrails.py` - Decision logic and edge cases  
- `tests/test_retrieval_pipeline.py` - RRF fusion and integration
- `tests/test_error_recovery.py` - Fallback mechanisms

#### 7.2 Integration Tests  
- `tests/integration/test_end_to_end_pipeline.py` - Full pipeline
- `tests/integration/test_score_alignment.py` - Score type compatibility
- `tests/integration/test_performance.py` - Performance benchmarks

#### 7.3 Performance Tests
```python
def test_embedding_cache_performance():
    """Verify caching improves performance by >5x"""
    pass

def test_mmr_performance_large_dataset():
    """Verify MMR handles 1000+ candidates in <2 seconds"""
    pass

def test_pipeline_throughput():
    """Verify pipeline handles 100 concurrent requests"""  
    pass
```

---

## 📋 IMPLEMENTATION PHASES

### Phase 1: Critical Fixes (Week 1)
- [ ] Fix Guardrails array indexing bug
- [ ] Fix RRF implementation  
- [ ] Wire provider hooks to existing services
- [ ] Fix score type mismatch in integration

### Phase 2: Performance & Reliability (Week 2)
- [ ] Add embedding caching system
- [ ] Add error recovery mechanisms
- [ ] Implement configuration management
- [ ] Add comprehensive logging

### Phase 3: Testing & Optimization (Week 3)
- [ ] Unit test coverage >90%
- [ ] Integration test suite
- [ ] Performance benchmarking
- [ ] Load testing with realistic data

### Phase 4: Production Deployment (Week 4)
- [ ] A/B testing framework
- [ ] Monitoring and alerting
- [ ] Documentation updates
- [ ] Rollout plan

---

## ✅ Success Criteria

**Production Readiness Checklist**:
- [ ] All critical bugs fixed (no crashes)
- [ ] Performance meets SLA (95th percentile <2s response time)
- [ ] Error rate <0.1% 
- [ ] Test coverage >90%
- [ ] Monitoring and alerting configured
- [ ] A/B testing shows improvement over current system

**Quality Gates**:
- [ ] Code review approved
- [ ] Security review passed  
- [ ] Performance benchmarks met
- [ ] Integration tests passing
- [ ] Staging deployment successful

This TODO list provides specific, actionable fixes for every critical issue identified in the code quality critique.