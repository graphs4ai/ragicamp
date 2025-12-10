"""Constants and enums for RAGiCamp.

Centralizes all magic strings and constants used throughout the codebase.
"""

from enum import Enum
from typing import List


class AgentType(str, Enum):
    """Agent types supported by RAGiCamp."""
    
    DIRECT_LLM = "direct_llm"
    FIXED_RAG = "fixed_rag"
    BANDIT_RAG = "bandit_rag"
    MDP_RAG = "mdp_rag"
    
    @classmethod
    def rag_types(cls) -> List["AgentType"]:
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
    def standard_metrics(cls) -> List["MetricType"]:
        """Fast, deterministic metrics."""
        return [cls.EXACT_MATCH, cls.F1]
    
    @classmethod
    def semantic_metrics(cls) -> List["MetricType"]:
        """Neural-based semantic metrics."""
        return [cls.BERTSCORE, cls.BLEURT]
    
    @classmethod
    def ragas_metrics(cls) -> List["MetricType"]:
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
    def llm_metrics(cls) -> List["MetricType"]:
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
    
    GENERATE = "generate"   # Only generate predictions
    EVALUATE = "evaluate"   # Only compute metrics
    BOTH = "both"           # Generate and evaluate


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
    
    # Evaluation
    BATCH_SIZE = 8
    CHECKPOINT_EVERY = 20
    
    # Paths
    DATA_DIR = "data"
    OUTPUT_DIR = "outputs"
    ARTIFACTS_DIR = "artifacts"
    CACHE_DIR = "data/datasets"


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
