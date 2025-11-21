"""MDP-based RAG agent - Iterative action selection with reinforcement learning."""

from typing import Any, Dict, List

from ragicamp.agents.base import RAGAgent, RAGContext, RAGResponse
from ragicamp.models.base import LanguageModel
from ragicamp.policies.base import Policy
from ragicamp.retrievers.base import Document, Retriever
from ragicamp.utils.formatting import ContextFormatter
from ragicamp.utils.prompts import PromptBuilder


class MDPRAGAgent(RAGAgent):
    """RAG agent that uses MDP formulation for iterative decision-making.

    This agent treats RAG as a sequential decision process:
    - State: current query, retrieved docs, intermediate results
    - Actions: retrieve more, reformulate query, generate answer, etc.
    - Policy: learned strategy for selecting actions

    The agent can take multiple steps before generating the final answer,
    allowing for iterative refinement and exploration.
    """

    def __init__(
        self,
        name: str,
        model: LanguageModel,
        retriever: Retriever,
        policy: Policy,
        max_steps: int = 5,
        system_prompt: str = "You are a helpful assistant. Use the provided context to answer questions accurately.",
        **kwargs: Any,
    ):
        """Initialize the MDP-based RAG agent.

        Args:
            name: Agent identifier
            model: The language model to use
            retriever: The retriever for finding relevant documents
            policy: MDP policy for action selection
            max_steps: Maximum number of steps before forcing answer generation
            system_prompt: System prompt for the LLM
            **kwargs: Additional configuration
        """
        super().__init__(name, **kwargs)
        self.model = model
        self.retriever = retriever
        self.policy = policy
        self.max_steps = max_steps
        self.prompt_builder = PromptBuilder(system_prompt=system_prompt)

    def answer(self, query: str, **kwargs: Any) -> RAGResponse:
        """Generate an answer using iterative MDP-based approach.

        Args:
            query: The input question
            **kwargs: Additional generation parameters

        Returns:
            RAGResponse with the answer and action history
        """
        # Initialize state
        state = self._initialize_state(query)
        context = RAGContext(query=query)

        # Iterative action loop
        for step in range(self.max_steps):
            # Select action based on current state
            action = self.policy.select_action(state=state)

            # Log the step
            step_info = {
                "step": step,
                "action_type": action["type"],
                "action_params": action.get("params", {}),
            }

            # Execute action and update state
            if action["type"] == "retrieve":
                docs = self._execute_retrieve(query, action, context)
                state = self._update_state(state, action, docs)
                step_info["num_docs_retrieved"] = len(docs)

            elif action["type"] == "reformulate":
                new_query = self._execute_reformulate(query, state, action)
                state = self._update_state(state, action, {"reformulated_query": new_query})
                step_info["reformulated_query"] = new_query

            elif action["type"] == "generate":
                # Final action: generate answer
                answer = self._execute_generate(query, context, state, **kwargs)
                step_info["answer_generated"] = True
                context.intermediate_steps.append(step_info)
                break

            else:
                # Unknown action - default to generate
                answer = self._execute_generate(query, context, state, **kwargs)
                step_info["action_type"] = "generate"
                step_info["note"] = f"Unknown action '{action['type']}', defaulting to generate"
                context.intermediate_steps.append(step_info)
                break

            context.intermediate_steps.append(step_info)
        else:
            # Max steps reached, force generation
            answer = self._execute_generate(query, context, state, **kwargs)
            context.intermediate_steps.append(
                {
                    "step": self.max_steps,
                    "action_type": "generate",
                    "note": "Max steps reached, forcing generation",
                }
            )

        # Return response with full trajectory
        return RAGResponse(
            answer=answer,
            context=context,
            metadata={
                "agent_type": "mdp_rag",
                "num_steps": len(context.intermediate_steps),
                "trajectory": context.intermediate_steps,
            },
        )

    def update_policy(self, trajectory: List[Dict[str, Any]], reward: float) -> None:
        """Update the MDP policy based on a trajectory and final reward.

        Args:
            trajectory: List of (state, action) pairs
            reward: The final reward/score achieved
        """
        self.policy.update(trajectory=trajectory, reward=reward)

    def _initialize_state(self, query: str) -> Dict[str, Any]:
        """Initialize the MDP state."""
        return {"query": query, "retrieved_docs": [], "reformulations": [], "step": 0}

    def _update_state(
        self, state: Dict[str, Any], action: Dict[str, Any], result: Any
    ) -> Dict[str, Any]:
        """Update state based on action and result."""
        new_state = state.copy()
        new_state["step"] += 1

        if action["type"] == "retrieve":
            new_state["retrieved_docs"].extend(result)
        elif action["type"] == "reformulate":
            new_state["reformulations"].append(result["reformulated_query"])

        return new_state

    def _execute_retrieve(
        self, query: str, action: Dict[str, Any], context: RAGContext
    ) -> List[Document]:
        """Execute a retrieval action."""
        params = action.get("params", {})
        top_k = params.get("top_k", 5)
        docs = self.retriever.retrieve(query, top_k=top_k, **params)
        context.retrieved_docs.extend(docs)
        return docs

    def _execute_reformulate(
        self, query: str, state: Dict[str, Any], action: Dict[str, Any]
    ) -> str:
        """Execute a query reformulation action."""
        # Use LLM to reformulate the query based on current state
        prompt = f"Reformulate this question to be more specific: {query}"
        reformulated = self.model.generate(prompt, max_tokens=100)
        return reformulated

    def _execute_generate(
        self, query: str, context: RAGContext, state: Dict[str, Any], **kwargs: Any
    ) -> str:
        """Execute answer generation action."""
        # Format all retrieved documents using utility
        context_text = ContextFormatter.format_numbered(context.retrieved_docs)

        # Build prompt using utility
        prompt = self.prompt_builder.build_prompt(query=query, context=context_text)
        return self.model.generate(prompt, **kwargs)
