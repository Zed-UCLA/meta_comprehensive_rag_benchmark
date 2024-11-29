import os
from typing import Any, Dict, List
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from bs4 import BeautifulSoup  # For cleaning HTML content
from transformer_inference import TransformerModel
import vllm
from openai import OpenAI


class ImprovedRAGModel:
    def __init__(self, llm_name="meta-llama/Llama-3.2-3B-Instruct",
                 retriever_model="sentence-transformers/all-mpnet-base-v2",
                 reranker_model="cross-encoder/ms-marco-MiniLM-L-12-v2",
                 is_server=False,
                 vllm_server=None,
                 use_transformers=True):
        """
        Initialize the retriever, reranker, and generator with flexible backends (vLLM or transformers).
        """
        # Configure device for GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize retriever (offloaded to CPU for memory efficiency)
        self.retriever = SentenceTransformer(retriever_model, device="cpu")
        self.retriever_batch_size = 16  # Reduced batch size to optimize memory usage

        # Initialize reranker (offloaded to CPU)
        self.reranker = CrossEncoder(reranker_model, device="cpu")

        # Model configuration
        self.llm_name = llm_name
        self.is_server = is_server
        self.vllm_server = vllm_server
        self.use_transformers = use_transformers

        # Initialize generator
        if self.use_transformers:
            print("Initializing Transformers model...")
            self.generator = TransformerModel(llm_name)
            self.generator.model.gradient_checkpointing_enable()  # Enable gradient checkpointing for memory efficiency
        elif self.is_server:
            # Initialize the model with vLLM server
            openai_api_key = "EMPTY"
            openai_api_base = self.vllm_server
            self.generator = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
        else:
            # Initialize vLLM offline inference
            self.generator = vllm.LLM(
                model=self.llm_name,
                worker_use_ray=True,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.8,  # Adjusted GPU utilization for memory efficiency
                trust_remote_code=True,
                dtype="half",  # Use half precision for memory savings
                enforce_eager=True
            )
            self.tokenizer = self.generator.get_tokenizer()

    def retrieve_contexts(self, queries: List[str], search_results: List[List[Dict]]) -> List[List[str]]:
        """
        Retrieve top contexts using dense retrieval from search results.
        """
        all_contexts = []
        for query, results in zip(queries, search_results):
            passages = [result["page_result"] for result in results]
            query_embedding = self.retriever.encode(query, convert_to_tensor=True, batch_size=self.retriever_batch_size)
            passage_embeddings = self.retriever.encode(passages, convert_to_tensor=True, batch_size=self.retriever_batch_size)
            scores = torch.cosine_similarity(query_embedding, passage_embeddings, dim=-1).cpu().numpy()
            top_indices = scores.argsort()[-5:][::-1]  # Top 5
            # Limit passage length to 512 characters and clean HTML
            limited_passages = [
                BeautifulSoup(passages[i][:512], "lxml").get_text() for i in top_indices
            ]
            all_contexts.append(limited_passages)
        return all_contexts

    def rerank_contexts(self, query: str, contexts: List[str]) -> List[str]:
        """
        Rerank retrieved contexts based on relevance to the query.
        """
        if not contexts:
            return []
        scores = self.reranker.predict([(query, context) for context in contexts])
        ranked_contexts = [context for _, context in sorted(zip(scores, contexts), reverse=True)]

        # Filter out low-scoring contexts
        threshold = max(scores) * 0.6  # Discard contexts with scores < 60% of the highest
        ranked_contexts = [context for context, score in zip(ranked_contexts, scores) if score >= threshold]

        return ranked_contexts[:5]  # Limit to top 5

    def generate_answer(self, query: str, contexts: List[str]) -> str:
        """
        Generate an answer using the selected backend.
        """
        # Format input as a structured prompt with limited context length
        max_context_length = 1024  # Define maximum context length
        input_text = "\n".join([f"Context: {context}" for context in contexts[:3]])
        input_text = f"{input_text[:max_context_length]}\n\nQuestion: {query}\n\nAnswer:"

        if self.use_transformers:
            # Generate with transformers
            raw_output = self.generator.generate_response([input_text], max_new_tokens=50, temperature=0.5, top_p=0.8)[0]

            # Post-process the output to clean unnecessary text
            if "Answer:" in raw_output:
                return raw_output.split("Answer:")[-1].strip()
            return raw_output.strip()

        elif self.is_server:
            # Generate with vLLM server
            response = self.generator.chat.completions.create(
                model=self.llm_name,
                messages=[
                    {"role": "system", "content": "Provide a concise and accurate response."},
                    {"role": "user", "content": input_text}
                ],
                max_tokens=50,
                temperature=0.7,
                top_p=0.9
            )
            return response.choices[0].message.content.strip()

        else:
            # Generate with vLLM offline
            formatted_input = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": input_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
            responses = self.generator.generate(
                formatted_input,
                vllm.SamplingParams(max_tokens=50, temperature=0.7, top_p=0.9)
            )
            return responses[0].outputs[0].text.strip()

    def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
        """
        Generate answers for a batch of queries.
        """
        queries = batch["query"]
        search_results = batch["search_results"]
        answers = []

        for query, results in zip(queries, search_results):
            # Step 1: Retrieve relevant contexts
            contexts = self.retrieve_contexts([query], [results])[0]

            # Step 2: Rerank the contexts
            contexts = self.rerank_contexts(query, contexts)

            # Step 3: Generate the answer
            answer = self.generate_answer(query, contexts)
            answers.append(answer)

        return answers

    def get_batch_size(self) -> int:
        """
        Return the batch size to be used for processing queries.
        """
        return 1
