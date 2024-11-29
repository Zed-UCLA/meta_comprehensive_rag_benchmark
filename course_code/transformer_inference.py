from transformers import AutoModelForCausalLM, AutoTokenizer 
import torch
from torch.cuda.amp import autocast  # For mixed precision

class TransformerModel:
    def __init__(self, model_name="meta-llama/Llama-3.2-3B-Instruct"):
        """
        Initialize the model and tokenizer from transformers.
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # Use eos_token as pad_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,  # Use half precision to save memory
            device_map="auto",         # Automatically map layers to GPUs/CPUs
            low_cpu_mem_usage=True     # Optimize CPU memory usage
        )

        # Explicitly set pad_token_id in the model config
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Enable gradient checkpointing for memory efficiency (optional)
        self.model.gradient_checkpointing_enable()

    def generate_response(self, prompts, max_new_tokens=50, temperature=0.5, top_p=0.8):
        """
        Generate responses for a list of prompts using the model.

        Args:
            prompts (list of str): The input prompts.
            max_new_tokens (int): Maximum new tokens to generate.
            temperature (float): Sampling temperature.
            top_p (float): Nucleus sampling probability.

        Returns:
            list of str: Generated responses.
        """
        batch_size = 1  # Adjust this based on available GPU memory
        responses = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
            with torch.amp.autocast('cuda'):  # Use mixed precision
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,  # Use max_new_tokens instead of max_length
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id  # Explicitly set pad_token_id
                )
            responses.extend([self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs])

        return responses

