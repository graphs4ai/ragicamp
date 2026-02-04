"""Constants and enums for RAGiCamp.

Centralizes all magic strings and constants used throughout the codebase.
"""

from enum import Enum


class AgentType(str, Enum):
    """Agent types supported by RAGiCamp."""

    DIRECT_LLM = "direct_llm"
    FIXED_RAG = "fixed_rag"
    BANDIT_RAG = "bandit_rag"
    MDP_RAG = "mdp_rag"

    @classmethod
    def rag_types(cls) -> list["AgentType"]:
        """Get agent types that use retrieval."""
        return [cls.FIXED_RAG, cls.BANDIT_RAG, cls.MDP_RAG]

    @classmethod
    def requires_retriever(cls, agent_type: str) -> bool:
        """Check if agent type requires a retriever."""
        return agent_type in [t.value for t in cls.rag_types()]


class ModelType(str, Enum):
    """Model types supported by RAGiCamp."""

    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"  # Future
    VLLM = "vllm"  # Future


class RetrieverType(str, Enum):
    """Retriever types supported by RAGiCamp."""

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"  # Future


class MetricType(str, Enum):
    """Metric types supported by RAGiCamp."""

    # Standard metrics
    EXACT_MATCH = "exact_match"
    F1 = "f1"

    # Semantic metrics
    BERTSCORE = "bertscore"
    BLEURT = "bleurt"

    # RAG-specific (Ragas)
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCY = "answer_relevancy"
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_RECALL = "context_recall"
    ANSWER_SIMILARITY = "answer_similarity"
    ANSWER_CORRECTNESS = "answer_correctness"

    # LLM-based
    LLM_JUDGE = "llm_judge"
    LLM_JUDGE_QA = "llm_judge_qa"

    # Legacy (can be deprecated)
    HALLUCINATION = "hallucination"

    @classmethod
    def standard_metrics(cls) -> list["MetricType"]:
        """Fast, deterministic metrics."""
        return [cls.EXACT_MATCH, cls.F1]

    @classmethod
    def semantic_metrics(cls) -> list["MetricType"]:
        """Neural-based semantic metrics."""
        return [cls.BERTSCORE, cls.BLEURT]

    @classmethod
    def ragas_metrics(cls) -> list["MetricType"]:
        """Ragas-powered RAG metrics."""
        return [
            cls.FAITHFULNESS,
            cls.ANSWER_RELEVANCY,
            cls.CONTEXT_PRECISION,
            cls.CONTEXT_RECALL,
            cls.ANSWER_SIMILARITY,
            cls.ANSWER_CORRECTNESS,
        ]

    @classmethod
    def llm_metrics(cls) -> list["MetricType"]:
        """LLM-based metrics."""
        return [cls.LLM_JUDGE, cls.LLM_JUDGE_QA]

    @classmethod
    def requires_context(cls, metric_type: str) -> bool:
        """Check if metric requires retrieved context."""
        context_metrics = {m.value for m in cls.ragas_metrics()}
        context_metrics.add(cls.FAITHFULNESS.value)
        context_metrics.add(cls.HALLUCINATION.value)
        return metric_type in context_metrics


class DatasetType(str, Enum):
    """Dataset types supported by RAGiCamp."""

    NATURAL_QUESTIONS = "natural_questions"
    TRIVIAQA = "triviaqa"
    HOTPOTQA = "hotpotqa"
    SQUAD = "squad"  # Future


class EvaluationMode(str, Enum):
    """Evaluation modes."""

    GENERATE = "generate"  # Only generate predictions
    EVALUATE = "evaluate"  # Only compute metrics
    BOTH = "both"  # Generate and evaluate


class PhaseStatus(str, Enum):
    """Status of an experiment phase."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# === File Extensions ===


class FileExtension:
    """Common file extensions."""

    JSON = ".json"
    YAML = ".yaml"
    YML = ".yml"
    CHECKPOINT = "_checkpoint.json"
    STATE = "_state.json"
    LOG = ".log"


# === Default Values ===


class Defaults:
    """Default configuration values."""

    # Model
    MAX_TOKENS = 256
    TEMPERATURE = 0.7
    TOP_P = 1.0

    # Retrieval
    TOP_K = 5
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    # Evaluation & Execution
    BATCH_SIZE = 8
    MIN_BATCH_SIZE = 1  # Minimum batch size for auto-reduction
    CHECKPOINT_INTERVAL = 50  # Save checkpoint every N items
    EXECUTOR_BATCH_SIZE = 32  # Default for ResilientExecutor

    # Paths
    DATA_DIR = "data"
    OUTPUT_DIR = "outputs"
    ARTIFACTS_DIR = "artifacts"
    CACHE_DIR = "data/datasets"

    # GPU Memory Partitioning
    # With FAISS on CPU, vLLM can use most of the GPU
    VLLM_GPU_MEMORY_FRACTION = 0.70  # 70% for vLLM - reduced to leave room for other processes
    FAISS_GPU_MEMORY_FRACTION = 0.0  # FAISS runs on CPU (B200 not yet supported)
    # Remaining 5% for PyTorch overhead, embedding model, etc.

    # FAISS GPU Configuration
    # NOTE: GPU FAISS disabled by default - B200 (Blackwell) not yet supported by faiss-gpu-cu12.
    # Enable once faiss adds Blackwell kernels, or build from source.
    FAISS_USE_GPU = False  # Disable GPU FAISS until Blackwell support added
    FAISS_GPU_TEMP_MEMORY_MB = (
        65536  # Temporary memory for FAISS operations (64GB for large index transfers)
    )
    FAISS_INDEX_TYPE = (
        "hnsw"  # Default index type: flat, ivf, ivfpq, hnsw (hnsw is fastest for CPU)
    )
    FAISS_IVF_NLIST = 4096  # Number of clusters for IVF indexes
    FAISS_IVF_NPROBE = 128  # Number of clusters to search (higher = better recall)
    FAISS_CPU_THREADS = 0  # 0 = auto-detect (use all available cores)

    # Embedding Model Optimization
    # These settings optimize sentence-transformers inference similar to how vLLM
    # optimizes causal LM inference.
    EMBEDDING_BATCH_SIZE = 256  # Batch size for encoding (tune based on GPU memory)
    EMBEDDING_USE_FP16 = True  # Use FP16 for faster encoding (minimal quality loss)
    EMBEDDING_NORMALIZE = True  # L2 normalize embeddings (required for cosine similarity)
    EMBEDDING_SHOW_PROGRESS = True  # Show progress bar during encoding
    EMBEDDING_USE_FLASH_ATTENTION = True  # Use Flash Attention 2 if available
    EMBEDDING_USE_TORCH_COMPILE = True  # Apply torch.compile() for speedup
    # For production, consider using infinity-emb or HuggingFace TEI server
    # which provide vLLM-style continuous batching for embeddings


# === Prompt Templates ===


class PromptTemplates:
    """Default prompt templates."""

    QA_SYSTEM = "You are a helpful assistant. Answer the question based on the provided context."

    QA_CONTEXT = """Context:
{context}

Question: {query}

Answer:"""

    QA_NO_CONTEXT = """Question: {query}

Answer:"""

    LLM_JUDGE = """You are an expert judge. Evaluate if the answer is correct.

Question: {question}
Reference Answer: {reference}
Candidate Answer: {candidate}

Is the candidate answer correct? Respond with only "yes" or "no"."""
