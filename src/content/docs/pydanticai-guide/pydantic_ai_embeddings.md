---
title: "PydanticAI: Embeddings"
description: "Embedder, EmbeddingModel, EmbeddingResult, and EmbeddingSettings — generate text embeddings with every major provider using a single API."
framework: pydanticai
language: python
---

# Embeddings

Verified against **pydantic-ai==1.102.0** — source module: `pydantic_ai.embeddings`.

The `Embedder` class provides a high-level, provider-agnostic API for generating vector embeddings. Specify a model name with `provider:model-name` syntax and call `embed_query()` or `embed_documents()` — the same interface works across OpenAI, Cohere, Google, Bedrock, and SentenceTransformers.

## Minimal runnable example

```python
import asyncio
from pydantic_ai import Embedder

embedder = Embedder('openai:text-embedding-3-small')

async def main():
    result = await embedder.embed_query('What is machine learning?')
    print(result.embeddings[0][:5])    # first 5 dimensions
    print(result.model)                # 'text-embedding-3-small'
    print(result.usage.total_tokens)   # token count

asyncio.run(main())
```

Synchronous variant:

```python
result = embedder.embed_query_sync('What is machine learning?')
print(result.embeddings[0][:5])
```

## Provider prefixes

```python
# OpenAI
Embedder('openai:text-embedding-3-small')
Embedder('openai:text-embedding-3-large')
Embedder('openai:text-embedding-ada-002')

# Cohere
Embedder('cohere:embed-v4.0')
Embedder('cohere:embed-english-v3.0')
Embedder('cohere:embed-multilingual-v3.0')

# Google
Embedder('google:gemini-embedding-001')
Embedder('google:gemini-embedding-2-preview')

# Amazon Bedrock
Embedder('bedrock:amazon.titan-embed-text-v2:0')
Embedder('bedrock:cohere.embed-english-v3')

# SentenceTransformers (local / no API key needed)
# pip install "pydantic-ai[sentence-transformers]"
Embedder('sentence-transformers:all-MiniLM-L6-v2')
```

## `embed_query` vs `embed_documents`

The `input_type` distinction tells the model whether the text is a *search query* or a *document* to index. Providers that distinguish these (Cohere, Voyage) optimise differently:

```python
import asyncio
from pydantic_ai import Embedder

embedder = Embedder('cohere:embed-v4.0')

async def main():
    # Embed a user search query
    q_result = await embedder.embed_query('best Italian restaurants')

    # Embed documents for indexing
    docs = ['Trattoria Roma: authentic pasta', 'Osteria del Sole: seafood specialist']
    d_result = await embedder.embed_documents(docs)

    # Compute cosine similarity
    import numpy as np
    q_vec = np.array(q_result.embeddings[0])
    for i, doc_vec in enumerate(d_result.embeddings):
        sim = float(np.dot(q_vec, doc_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(doc_vec)))
        print(f'doc[{i}] similarity: {sim:.4f}')

asyncio.run(main())
```

## Embedding multiple texts at once

Pass a list to batch embed in a single API call:

```python
async def batch_embed(texts: list[str]) -> list[list[float]]:
    result = await embedder.embed_documents(texts)
    return result.embeddings   # list of float vectors, one per input
```

## `EmbeddingSettings` — customise model parameters

```python
from pydantic_ai import Embedder
from pydantic_ai.embeddings import EmbeddingSettings

settings = EmbeddingSettings(
    dimensions=256,       # truncate output dimension (OpenAI text-embedding-3-* only)
    encoding_format='float',  # 'float' | 'base64' (provider-dependent)
)

embedder = Embedder('openai:text-embedding-3-large', settings=settings)

# Override per-call
result = await embedder.embed_query(
    'hello world',
    settings=EmbeddingSettings(dimensions=512),
)
```

## Token counting

```python
embedder = Embedder('openai:text-embedding-3-small')

# Async
token_count = await embedder.count_tokens('How many tokens is this?')
print(token_count)

# Sync
token_count = embedder.count_tokens_sync('How many tokens is this?')

# Model input limit
max_tokens = await embedder.max_input_tokens()
print(max_tokens)   # e.g. 8191 for text-embedding-3-small
```

## `EmbeddingResult` — what you get back

```python
result = await embedder.embed_query('test')

result.embeddings        # list[list[float]] — one vector per input text
result.model             # model name string
result.usage             # EmbeddingUsage(total_tokens=N, prompt_tokens=N)
result.provider_details  # raw provider response dict, if available
```

## Testing with `TestEmbeddingModel`

```python
from pydantic_ai import Embedder
from pydantic_ai.embeddings import TestEmbeddingModel

test_model = TestEmbeddingModel(embedding_size=128)  # returns deterministic unit vectors

embedder = Embedder(test_model)
result = embedder.embed_query_sync('anything')
print(len(result.embeddings[0]))  # 128
```

Use `embedder.override()` in tests to avoid hitting the API:

```python
from pydantic_ai import Embedder
from pydantic_ai.embeddings import TestEmbeddingModel

embedder = Embedder('openai:text-embedding-3-small')

def test_my_pipeline():
    with embedder.override(model=TestEmbeddingModel(embedding_size=1536)):
        result = embedder.embed_query_sync('test query')
        assert len(result.embeddings[0]) == 1536
```

## Custom `EmbeddingModel` — implementing your own

```python
from collections.abc import Sequence
from pydantic_ai.embeddings.base import EmbeddingModel
from pydantic_ai.embeddings.result import EmbeddingResult, EmbedInputType
from pydantic_ai.embeddings.settings import EmbeddingSettings

class LocalFaissModel(EmbeddingModel):
    """Embedding model backed by a local FAISS index."""

    def __init__(self, index_path: str, dim: int = 768):
        super().__init__()
        import faiss
        self._index = faiss.read_index(index_path)
        self._dim = dim

    @property
    def model_name(self) -> str:
        return 'local-faiss'

    @property
    def system(self) -> str:
        return 'faiss'

    async def embed(
        self,
        inputs: str | Sequence[str],
        *,
        input_type: EmbedInputType,
        settings: EmbeddingSettings | None = None,
    ) -> EmbeddingResult:
        texts, _ = self.prepare_embed(inputs, settings)
        # Replace with actual embedding call
        import numpy as np
        vecs = [np.random.rand(self._dim).tolist() for _ in texts]
        return EmbeddingResult(embeddings=vecs, model=self.model_name)

embedder = Embedder(LocalFaissModel('/path/to/index'))
```

## OpenTelemetry instrumentation

```python
from pydantic_ai import Embedder

# Enable for a single embedder
embedder = Embedder('openai:text-embedding-3-small', instrument=True)

# Enable globally for all embedders created after this call
Embedder.instrument_all(True)

# Customise the settings
from pydantic_ai.models.instrumented import InstrumentationSettings
Embedder.instrument_all(InstrumentationSettings(event_mode='body'))
```

## Similarity search pipeline

A complete example: embed a corpus, embed queries, return top-k by cosine similarity.

```python
import asyncio
import math
from pydantic_ai import Embedder

embedder = Embedder('openai:text-embedding-3-small')

def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b + 1e-9)

async def build_and_search(corpus: list[str], query: str, top_k: int = 3):
    corpus_result = await embedder.embed_documents(corpus)
    query_result  = await embedder.embed_query(query)

    q_vec = query_result.embeddings[0]
    scored = [
        (cosine_similarity(q_vec, doc_vec), doc)
        for doc_vec, doc in zip(corpus_result.embeddings, corpus)
    ]
    scored.sort(reverse=True)
    return scored[:top_k]

docs = [
    'Python is a high-level programming language.',
    'Machine learning uses statistical methods.',
    'The Eiffel Tower is in Paris.',
    'Deep learning is a subset of machine learning.',
]

async def main():
    results = await build_and_search(docs, 'neural networks and deep learning')
    for score, doc in results:
        print(f'{score:.3f}  {doc}')

asyncio.run(main())
```

## Hybrid: embeddings + agent output

Generate embeddings alongside an agent run to persist semantic representations of agent outputs:

```python
import asyncio
from pydantic_ai import Agent, Embedder

agent = Agent('openai:gpt-4o', output_type=str)
embedder = Embedder('openai:text-embedding-3-small')

async def run_and_embed(prompt: str):
    result = await agent.run(prompt)
    embedding = await embedder.embed_documents([result.output])
    return result.output, embedding.embeddings[0]

asyncio.run(run_and_embed('Summarise the Python programming language in one sentence.'))
```

## VoyageAI embeddings — domain-specific models

VoyageAI offers specialised embedding models for code, legal text, and finance:

```python
# pip install voyageai
import asyncio
from pydantic_ai import Embedder

async def main():
    # General-purpose, high quality
    gen_embedder = Embedder('voyageai:voyage-4')

    # Optimised for code snippets
    code_embedder = Embedder('voyageai:voyage-code-3')

    # Optimised for legal documents
    legal_embedder = Embedder('voyageai:voyage-law-2')

    code_snippet = 'def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)'
    result = await code_embedder.embed_query(code_snippet)
    print(f'Code embedding dimensions: {len(result.embeddings[0])}')

asyncio.run(main())
```

## Cohere multi-lingual embeddings

Cohere's models embed multiple languages into the same vector space, enabling cross-lingual retrieval:

```python
import asyncio
from pydantic_ai import Embedder

embedder = Embedder('cohere:embed-multilingual-v3.0')

async def cross_lingual_search():
    # Documents in different languages
    docs = [
        'Python is a high-level programming language.',       # English
        'Python est un langage de programmation de haut niveau.',  # French
        'Python ist eine höhere Programmiersprache.',              # German
        'Pythonは高レベルのプログラミング言語です。',                   # Japanese
    ]

    # Query in English — finds semantically similar docs in all languages
    doc_result = await embedder.embed_documents(docs)
    query_result = await embedder.embed_query('high level scripting language')

    import math
    def cosine_sim(a, b):
        dot = sum(x*y for x, y in zip(a, b))
        return dot / (math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(x*x for x in b)) + 1e-9)

    q_vec = query_result.embeddings[0]
    scored = sorted(
        [(cosine_sim(q_vec, dvec), doc) for dvec, doc in zip(doc_result.embeddings, docs)],
        reverse=True,
    )
    for score, doc in scored:
        print(f'{score:.4f}  {doc[:60]}')

asyncio.run(cross_lingual_search())
```

## `EmbeddingResult` — full property reference

```python
import asyncio
from pydantic_ai import Embedder

embedder = Embedder('openai:text-embedding-3-small')

async def main():
    result = await embedder.embed_query('What is machine learning?')

    # Core embedding data
    print(result.embeddings)          # list[list[float]] — one vector per input
    print(result.inputs)              # list[str] — the original input texts
    print(result.input_type)          # 'query' or 'document'

    # Model metadata
    print(result.model_name)          # e.g. 'text-embedding-3-small'
    print(result.provider_name)       # e.g. 'openai'
    print(result.timestamp)           # datetime of the request

    # Usage statistics
    print(result.usage.total_tokens)  # total tokens consumed
    print(result.usage.prompt_tokens) # synonym for total_tokens on embedding models

    # Provider-specific metadata
    print(result.provider_details)    # raw provider response dict, or None
    print(result.provider_response_id)  # provider's response ID, or None

    # Access by index or by input text
    print(result[0])                  # first embedding vector
    print(result['What is machine learning?'])  # embedding for this exact input

asyncio.run(main())
```

## Embedding multiple input types in one call

```python
import asyncio
from pydantic_ai import Embedder
from pydantic_ai.embeddings.result import EmbedInputType

embedder = Embedder('openai:text-embedding-3-large')

async def embed_mixed_corpus():
    """Embed a corpus with explicit input_type control."""

    # Documents for indexing — use 'document' type for best retrieval performance
    documents = [
        'Introduction to machine learning with Python.',
        'Advanced neural network architectures.',
        'Practical data preprocessing techniques.',
    ]

    doc_result = await embedder.embed(
        documents,
        input_type='document',
    )
    print(f'Embedded {len(doc_result.embeddings)} documents')
    print(f'Input type: {doc_result.input_type}')  # 'document'

    # Single query — use 'query' type
    query_result = await embedder.embed(
        'deep learning best practices',
        input_type='query',
    )
    print(f'Query embedding dims: {len(query_result.embeddings[0])}')

asyncio.run(embed_mixed_corpus())
```

## Dimensionality reduction with OpenAI

OpenAI's `text-embedding-3-*` models support truncating the output dimension. Lower dimensions reduce storage and similarity computation cost with minimal quality loss:

```python
import asyncio
from pydantic_ai import Embedder
from pydantic_ai.embeddings import EmbeddingSettings

async def main():
    # Full dimensions: 3072 for text-embedding-3-large
    full_embedder = Embedder('openai:text-embedding-3-large')
    full_result = await full_embedder.embed_query('hello world')
    print(f'Full dims: {len(full_result.embeddings[0])}')  # 3072

    # Reduced to 256 dims — 12x smaller, still high quality
    compact_embedder = Embedder(
        'openai:text-embedding-3-large',
        settings=EmbeddingSettings(dimensions=256),
    )
    compact_result = await compact_embedder.embed_query('hello world')
    print(f'Compact dims: {len(compact_result.embeddings[0])}')  # 256

asyncio.run(main())
```

## Production batch processing

```python
import asyncio
from pydantic_ai import Embedder

embedder = Embedder('openai:text-embedding-3-small')

async def embed_corpus_in_batches(
    texts: list[str],
    batch_size: int = 100,
) -> list[list[float]]:
    """Embed a large corpus in batches to avoid rate limits."""
    all_vectors: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        result = await embedder.embed_documents(batch)
        all_vectors.extend(result.embeddings)
        print(f'Processed {min(i + batch_size, len(texts))}/{len(texts)} texts')

    return all_vectors

async def main():
    corpus = [f'Document about topic {i}' for i in range(350)]
    vectors = await embed_corpus_in_batches(corpus, batch_size=100)
    print(f'Total vectors: {len(vectors)}, each dim: {len(vectors[0])}')

asyncio.run(main())
```

## Complete RAG pipeline

End-to-end example: embed a knowledge base, then use an agent to answer questions via semantic search:

```python
import asyncio
import math
from pydantic_ai import Agent, Embedder, RunContext
from dataclasses import dataclass, field

@dataclass
class KnowledgeBase:
    documents: list[str]
    vectors: list[list[float]] = field(default_factory=list)

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb + 1e-9)

    def search(self, query_vec: list[float], top_k: int = 3) -> list[str]:
        scored = [
            (self.cosine_similarity(query_vec, doc_vec), doc)
            for doc_vec, doc in zip(self.vectors, self.documents)
        ]
        scored.sort(reverse=True)
        return [doc for _, doc in scored[:top_k]]

embedder = Embedder('openai:text-embedding-3-small')
agent = Agent('openai:gpt-4o', deps_type=KnowledgeBase)

@agent.system_prompt
def knowledge_base_prompt(ctx: RunContext[KnowledgeBase]) -> str:
    return f'You answer questions using a knowledge base of {len(ctx.deps.documents)} documents.'

@agent.tool
async def retrieve_docs(ctx: RunContext[KnowledgeBase], query: str) -> list[str]:
    """Search the knowledge base for relevant documents."""
    query_result = await embedder.embed_query(query)
    return ctx.deps.search(query_result.embeddings[0], top_k=3)

async def main():
    # Build knowledge base
    docs = [
        'Python was created by Guido van Rossum and first released in 1991.',
        'Python 3.0 was released in 2008 and broke backward compatibility with Python 2.',
        'PEP 8 is the official Python style guide, promoting code readability.',
        'Python uses indentation to define code blocks instead of braces.',
        'The Python Package Index (PyPI) hosts over 500,000 packages.',
    ]

    kb = KnowledgeBase(documents=docs)
    kb_result = await embedder.embed_documents(docs)
    kb.vectors = list(kb_result.embeddings)

    result = await agent.run('When was Python first released?', deps=kb)
    print(result.output)

asyncio.run(main())
```

## Reference

| Symbol | Module | Notes |
|---|---|---|
| `Embedder` | `pydantic_ai.embeddings` | High-level embedding interface |
| `EmbeddingModel` | `pydantic_ai.embeddings` | Abstract base — subclass for custom models |
| `EmbeddingResult` | `pydantic_ai.embeddings` | Holds `embeddings`, `model`, `usage`, `inputs`, `input_type` |
| `EmbeddingSettings` | `pydantic_ai.embeddings` | `dimensions`, `encoding_format` |
| `TestEmbeddingModel` | `pydantic_ai.embeddings` | Deterministic unit vectors for tests |
| `KnownEmbeddingModelName` | `pydantic_ai.embeddings` | Literal type of all known model strings |
| `WrapperEmbeddingModel` | `pydantic_ai.embeddings` | Base class for custom model wrappers |
| `InstrumentedEmbeddingModel` | `pydantic_ai.embeddings` | OTel-instrumented model wrapper |
