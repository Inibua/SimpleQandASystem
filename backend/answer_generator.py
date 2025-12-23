"""
answer_generator.py

Grounded answer generation with citation enforcement using Ollama and LangGraph.
"""

import logging
from typing import List, Dict, Any
import requests
from langgraph.graph.state import StateGraph, CompiledStateGraph, START, END
from pydantic import BaseModel, Field

from backend.backend_config import BackendConfig


class GenerationState(BaseModel):
    """State for answer generation workflow."""
    query: str = Field(description="User's original question")
    conversation_context: List[Dict] = Field(default_factory=list, description="Previous conversation turns")
    retrieved_chunks: List[Dict] = Field(default_factory=list, description="Retrieved document chunks")
    answer: str = Field(default="", description="Generated answer")
    citations: List[Dict] = Field(default_factory=list, description="Citations mapping to chunks")
    should_answer: bool = Field(default=True, description="Whether to generate an answer")
    clarification_question: str = Field(default="", description="Clarification question if no context")


class AnswerGenerator:
    """
    Generates grounded answers with strict citation enforcement using LangGraph.
    """

    def __init__(self, config: BackendConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> CompiledStateGraph:
        """Build the LangGraph workflow for answer generation."""
        workflow = StateGraph(GenerationState)

        # Define nodes
        # workflow.add_node("validate_context", self._validate_context)
        workflow.add_node("generate_answer", self._generate_answer)
        workflow.add_node("extract_citations", self._extract_citations)
        workflow.add_node("create_clarification", self._create_clarification)

        # Define edges
        # workflow.set_entry_point("validate_context")
        # workflow.set_entry_point(START)

        # workflow.add_edge("validate_context", "generate_answer", "has_context")
        # workflow.add_edge("validate_context", "create_clarification", "no_context")
        workflow.add_conditional_edges(START, self._validate_context,
                                       {"no_context": "create_clarification", "has_context": "generate_answer"})

        workflow.add_edge("generate_answer", "extract_citations")
        workflow.add_edge("extract_citations", END)
        workflow.add_edge("create_clarification", END)

        return workflow.compile()

    async def generate_answer(self, query: str, conversation_context: List[Dict],
                              retrieved_chunks: List[Dict]) -> Dict[str, Any]:
        """
        Generate a grounded answer with citations.

        Args:
            query: User's question
            conversation_context: Previous conversation turns
            retrieved_chunks: Retrieved document chunks from retrieval engine

        Returns:
            Dictionary containing answer, citations, and metadata
        """
        self.logger.info(f"Generating answer for query: '{query}'")

        try:
            # Initialize state
            initial_state = GenerationState(
                query=query,
                conversation_context=conversation_context,
                retrieved_chunks=retrieved_chunks
            )

            # Execute workflow
            final_state = self.workflow.invoke(initial_state)

            return {
                "answer": final_state["answer"],
                "citations": final_state["citations"],
                "clarification_question": final_state["clarification_question"],
                "should_answer": final_state["should_answer"],
                "retrieved_chunks_count": len(retrieved_chunks)
            }

        except Exception as e:
            self.logger.error(f"Answer generation failed: {e}")
            return {
                "answer": "I encountered an error while processing your question. Please try again.",
                "citations": [],
                "clarification_question": "",
                "should_answer": False,
                "error": str(e)
            }

    def _validate_context(self, state: GenerationState) -> GenerationState:
        """Validate if we have sufficient context to answer."""
        if not state.retrieved_chunks:
            state.should_answer = False
            return "no_context"

        # Check if chunks are relevant (simple threshold - can be enhanced)
        # relevant_chunks = [chunk for chunk in state.retrieved_chunks
        #                    if chunk.get('scores', {}).get('final_score', 0) > 0.3]
        #
        # if not relevant_chunks:
        #     state.should_answer = False
        #     return "no_context"

        return "has_context"

    def _generate_answer(self, state: GenerationState) -> GenerationState:
        """Generate answer using Ollama LLM with retrieved context."""
        try:
            # Prepare context from retrieved chunks
            context_parts = []
            for i, chunk in enumerate(state.retrieved_chunks):
                context_parts.append(
                    f"[Source {i + 1}] Document: {chunk['document_name']}, "
                    f"Page: {chunk['page_number']}, "
                    f"Heading: {chunk.get('section_heading', 'N/A')}\n"
                    f"Content: {chunk['chunk_text']}\n"
                )

            context = "\n".join(context_parts)

            # Prepare conversation history
            history = ""
            if state.conversation_context:
                history = "Previous conversation:\n"
                for turn in state.conversation_context[-2:]:  # Last 2 turns
                    history += f"Q: {turn.get('question', '')}\nA: {turn.get('answer', '')}\n\n"

            # Generate prompt
            prompt = self._build_prompt(state.query, context, history)

            # Call Ollama
            response = self._call_ollama(prompt)

            state.answer = response.strip()
            return state

        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            state.answer = "I couldn't generate an answer due to an error. Please try again."
            return state

    def _extract_citations(self, state: GenerationState) -> GenerationState:
        """Extract and validate citations from the generated answer."""
        if not state.answer or not state.retrieved_chunks:
            state.citations = []
            return state

        try:
            # Simple citation extraction - look for [Source X] patterns
            citations = []
            used_chunk_indices = set()

            # Extract citations from answer text
            import re
            citation_pattern = r'\[Source\s+(\d+)\]'
            matches = re.findall(citation_pattern, state.answer)

            for match in matches:
                try:
                    chunk_index = int(match) - 1  # Convert to 0-based index
                    if 0 <= chunk_index < len(state.retrieved_chunks):
                        chunk = state.retrieved_chunks[chunk_index]
                        citations.append({
                            "document_name": chunk["document_name"],
                            "page_number": chunk["page_number"],
                            "section_heading": chunk.get("section_heading", ""),
                            "chunk_id": chunk["chunk_id"],
                            "source_index": chunk_index + 1
                        })
                        used_chunk_indices.add(chunk_index)
                except (ValueError, IndexError):
                    continue

            # Validate that citations are used
            if not citations:
                self.logger.warning("No valid citations found in answer")
                state.answer = "I cannot provide an answer without proper citations to the source documents."
                state.citations = []
            else:
                state.citations = citations

            return state

        except Exception as e:
            self.logger.error(f"Citation extraction failed: {e}")
            state.answer = "I cannot provide an answer due to citation validation issues."
            state.citations = []
            return state

    def _create_clarification(self, state: GenerationState) -> GenerationState:
        """Create a clarification question when no relevant context is found."""
        # Prepare conversation context for clarification
        context_text = ""
        if state.conversation_context:
            recent_questions = [turn.get('question', '') for turn in state.conversation_context[-2:]]
            context_text = " ".join(recent_questions)

        full_context = f"{context_text} {state.query}".strip()

        # Generate clarification question
        prompt = f"""Based on the following conversation context, create a single clarifying question to help find relevant information. The user asked: "{full_context}"

Create one clear, specific question that would help narrow down what they're looking for. Return only the question."""

        try:
            clarification = self._call_ollama(prompt)
            state.clarification_question = clarification.strip()
            state.answer = ""  # No answer when asking clarification
        except Exception as e:
            self.logger.error(f"Clarification generation failed: {e}")
            state.clarification_question = "Could you provide more specific details about what you're looking for?"

        return state

    def _build_prompt(self, query: str, context: str, history: str) -> str:
        """Build the prompt for answer generation with citation enforcement."""
        return f"""You are a helpful assistant that answers questions based strictly on the provided context. You must cite your sources using [Source X] notation.

GROUND RULES:
1. Answer ONLY using information from the provided context
2. Cite every factual statement with [Source X] where X is the source number
3. If information isn't in the context, say you cannot answer
4. Be concise and accurate
5. Use multiple citations when information comes from multiple sources

{history}

CONTEXT SOURCES:
{context}

USER QUESTION: {query}

Your answer with citations:"""

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama LLM API."""
        try:
            response = requests.post(
                f"http://{self.config.llm_host}:{self.config.llm_port}/api/generate",
                json={
                    "model": self.config.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0,  # Low temperature for factual answers
                        "top_p": 0.9,
                        "num_predict": self.config.max_tokens
                    }
                },
                timeout=60
            )

            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")

        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to Ollama. Please ensure Ollama is running.")
        except Exception as e:
            raise Exception(f"Ollama call failed: {e}")

    def health_check(self) -> bool:
        """Check if Ollama is available."""
        try:
            response = requests.get(
                f"http://{self.config.llm_host}:{self.config.llm_port}/api/tags",
                timeout=10
            )
            return response.status_code == 200
        except:
            return False
