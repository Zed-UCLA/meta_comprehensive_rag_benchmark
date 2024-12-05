
# RAG Model Generation and Evaluation Framework

## Environment Check
To start the framework, ensure that the environment settings meet the following requirements. Note that some configurations not listed below might still work, but the listed ones are well-tested.

### Recommended Environment
- **Machine**: Local machine
- **CPU**: AMD Ryzen 9 4900HS with Radeon Graphics
- **RAM**: 40GB
- **GPU**: NVIDIA GeForce RTX 2060
- **GPU RAM**: 6GB
- **OS**: Linux (WSL2)
- **Python**: 3.10.15

---

## Environment Setup
To set up the environment, follow these steps:

### Python Environment Setup
```bash
# Navigate to the course code directory
cd course_code

# Create and activate a Conda virtual environment
conda create -n crag python=3.10
conda activate crag

# Install the required Python packages
pip install -r requirements.txt
pip install --upgrade openai
```

### Hugging Face Setup
1. Go to the [Hugging Face website](https://huggingface.co/) and create an account.
2. Obtain your Hugging Face Access Token:
   - After logging in, click on your profile icon in the top-right corner and go to **Settings**.
   - Navigate to the **Access Tokens** section.
   - Click **Create New Token**, name it (e.g., `student-api-token`), and copy the token.

3. Run the following commands to log in:
   ```bash
   huggingface-cli login --token "your_access_token"
   export CUDA_VISIBLE_DEVICES=0
   ```

For more details or alternative setup options, please refer to the [TA's page](https://ccye.notion.site/F24-CS245-Course-Project-Description-12781f4313e2800cb7ebfd85870bff40#12981f4313e2809fbf13e69b60f4059f).

---

## Model Development
You can develop your own RAG model. Once developed, update `generate.py` and `evaluate.py` to include your model.

### Example Modification in `generate.py`
```python
# In generate.py
parser.add_argument("--model_name", type=str, default="vanilla_baseline",
    choices=["vanilla_baseline",
             "rag_baseline",
             "improved_rag",
             # Add your model here
             ],
)

...

if model_name == "vanilla_baseline":
    from vanilla_baseline import InstructModel
    model = InstructModel(llm_name=llm_name, is_server=args.is_server, vllm_server=args.vllm_server, 
        use_transformers=args.use_transformers)
elif model_name == "rag_baseline":
    from rag_baseline import RAGModel
    model = RAGModel(llm_name=llm_name, is_server=args.is_server, vllm_server=args.vllm_server, use_transformers=args.use_transformers)
elif model_name == "improved_rag":
    from improved_rag import ImprovedRAGModel
    model = ImprovedRAGModel(llm_name=llm_name, is_server=args.is_server, vllm_server=args.vllm_server, use_transformers=args.use_transformers, batch_size=1)
# elif model_name == "your_model":
#     Add your model here
else:
    raise ValueError("Model name not recognized.")
```

### Example Modification in `evaluate.py`
```python
# In evaluate.py
parser.add_argument("--model_name", type=str, default="vanilla_baseline",
    choices=["vanilla_baseline",
             "rag_baseline",
             "improved_rag",
             # Add your model here
             ],
)
```

---

## Generate Predictions
Use `generate.py` to generate predictions based on either a vLLM server or a local transformer model. **Note:** The prediction results may vary significantly depending on the method used. For optimal results, we recommend using the vLLM server.

### Using vLLM Server (Preferred)
1. Open a new terminal and start the vLLM server:
   ```bash
   vllm serve meta-llama/Llama-3.2-3B-Instruct \\
   --tensor_parallel_size=1 \\
   --dtype="half" \\
   --port=8088 \\
   --enforce_eager \\
   --gpu_memory_utilization=1.0 \\ # Maximize the GPU memory utilization  
   --max_len 3000 \\ # Adjust based on your GPU memory size
   ```

   You should see the following output if the server starts correctly:
   ```bash
   INFO:     Started server process [17972]
   INFO:     Waiting for application startup.
   INFO:     Application startup complete.
   INFO:     Uvicorn running on http://0.0.0.0:8088 (Press CTRL+C to quit)
   ```

2. In your original terminal, run the prediction script:
   ```bash
   python generate.py    \\
   --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2"    \\
   --split 1    \\
   --model_name "YOUR_MODEL_NAME"    \\
   --llm_name "meta-llama/Llama-3.2-1B-Instruct"    \\
   --is_server    \\
   --vllm_server "http://localhost:8088/v1"\\
   ```

3. Results will be saved to:
   ```plaintext
   /output/data/improved_rag/Llama-3.2-3B-Instruct/predictions.json
   ```

### Using Local Transformer Model
If vLLM is unavailable, use the local transformer model:
```bash
python generate.py \\
--dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \\
--split 1 \\
--model_name "YOUR_MODEL_NAME" \\
--llm_name "meta-llama/Llama-3.2-1B-Instruct" \\
--is_transformer\\
```

---

## Evaluation
Use `evaluate.py` to evaluate your prediction results. Ensure the predictions file exists at:
```plaintext
/output/data/improved_rag/Llama-3.2-3B-Instruct/predictions.json
```

### Using vLLM Server (Preferred)
1. Start the vLLM server (refer to the previous section).
2. Run the evaluation script:
   ```bash
   python evaluate.py    \\
   --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2"    \\
   --model_name "YOUR_MODEL_NAME"    \\
   --llm_name "meta-llama/Llama-3.2-3B-Instruct"    \\
   --is_server    \\
   --vllm_server "http://localhost:8088/v1"    \\
   --max_retries 10\\
   ```

3. Results will be saved in:
   ```plaintext
   /output/data/improved_rag/Llama-3.2-3B-Instruct/
   ├── llm_evaluation_logs.json
   └── evaluation_results.json
   ```

---
