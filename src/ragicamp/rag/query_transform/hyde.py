"""HyDE (Hypothetical Document Embeddings) query transformer.

HyDE generates a hypothetical answer to the query using an LLM,
then uses that hypothetical answer as the search query instead
of the original question. This often improves retrieval because
the hypothetical answer is more similar to actual documents.

Reference: https://arxiv.org/abs/2212.10496
"""

from typing import TYPE_CHECKING, List, Optional

from ragicamp.rag.query_transform.base import QueryTransformer

if TYPE_CHECKING:
    from ragicamp.models.base import LanguageModel


class HyDETransformer(QueryTransformer):
    """Generate hypothetical answers and search with those.

    HyDE (Hypothetical Document Embeddings) asks an LLM to generate
    a hypothetical answer to the user's question, then uses that
    answer for retrieval instead of the original query.

    This works because:
    - The hypothetical answer is written in "document style"
    - It contains domain vocabulary that matches real documents
    - Embedding similarity works better for document-document matching
    """

    DEFAULT_PROMPT = """Please write a short paragraph that directly answers the following question.
Write as if you are explaining the answer in a textbook or encyclopedia.
Be factual and informative.

Question: {query}

Answer:"""

    def __init__(
        self,
        llm: "LanguageModel",
        prompt_template: Optional[str] = None,
        include_original: bool = True,
        num_hypothetical: int = 1,
        max_tokens: int = 150,
    ):
        """Initialize HyDE transformer.

        Args:
            llm: Language model to generate hypothetical answers
            prompt_template: Custom prompt template with {query} placeholder
            include_original: Whether to also search with the original query
            num_hypothetical: Number of hypothetical answers to generate
            max_tokens: Maximum tokens for hypothetical answer generation
        """
        self.llm = llm
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT
        self.include_original = include_original
        self.num_hypothetical = num_hypothetical
        self.max_tokens = max_tokens

    def transform(self, query: str) -> List[str]:
        """Generate hypothetical answer(s) and return as search queries.

        Args:
            query: The original user query

        Returns:
            List containing hypothetical answer(s), optionally with original query
        """
        queries = []

        # Optionally include original query
        if self.include_original:
            queries.append(query)

        # Generate hypothetical answer(s)
        prompt = self.prompt_template.format(query=query)

        for _ in range(self.num_hypothetical):
            hypothetical = self.llm.generate(
                prompt,
                max_tokens=self.max_tokens,
                temperature=0.7,  # Some variation if generating multiple
            )
            # Clean up the response
            hypothetical = hypothetical.strip()
            if hypothetical:
                queries.append(hypothetical)

        return queries

    def batch_transform(self, queries: List[str]) -> List[List[str]]:
        """Generate hypothetical answers for multiple queries in one batch call.

        This is much faster than calling transform() sequentially because
        it uses a single batched LLM call for all queries.

        Args:
            queries: List of original user queries

        Returns:
            List of query lists, one per input query
        """
        if not queries:
            return []

        # Build prompts for all queries
        prompts = [self.prompt_template.format(query=q) for q in queries]

        # Batch generate hypothetical answers (single batched LLM call!)
        hypotheticals = self.llm.generate(
            prompts,
            max_tokens=self.max_tokens,
            temperature=0.7,
        )

        # Build result: for each query, return [original, hypothetical]
        results = []
        for query, hyp in zip(queries, hypotheticals):
            query_list = []
            if self.include_original:
                query_list.append(query)
            if hyp and hyp.strip():
                query_list.append(hyp.strip())
            results.append(query_list)

        return results

    def __repr__(self) -> str:
        return (
            f"HyDETransformer(include_original={self.include_original}, "
            f"num_hypothetical={self.num_hypothetical})"
        )
