import asyncio
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from datasets import load_dataset, Dataset
from tqdm.asyncio import tqdm_asyncio

# Initialize the local API client
client = AsyncOpenAI(api_key="dummy", base_url="http://localhost:8001/v1")

# Define the prompt template
THOUGHT_PROMPT_TEMPLATE = """You are an expert algorithm thinker. Given a problem, please provide a detailed high-level strategy without writing any code. 
Ensure your response is comprehensive and well-structured with clear steps and considerations.
Steps:
1. **Define the input-output structure**: Clearly specify the types and meanings of the input and output variables.
2. **Describe the solving logic** using:
   - **Sequence**: Step-by-step operations in the correct order.
   - **Branch**: Conditions (if / if-else) that lead to different solution paths.
   - **Loop**: Repetitive operations (for / while) needed to process the input.
3. **Key Considerations**: Identify any constraints, limitations, or potential pitfalls to keep in mind while solving the problem (e.g., time complexity, memory usag
e, edge cases). Provide solutions or optimizations where applicable.
Your response should provide a thorough strategy, avoid being overly concise, and include at least a few sentences for each step.
Example:
- **Input**:
  - **N** (an integer): The number of vertices in the graph.
  - **p** (a list of integers): A list of size N, where the i-th element, `p_i`, denotes the vertex that vertex i points to.
  
- **Output**:
  - Print "POSSIBLE" if a valid assignment exists, otherwise print "IMPOSSIBLE".
1. **Graph Construction**:
   1. Treat the given list `p` as a directed graph, where each vertex i points to `p_i`.
   2. The graph is essentially a **directed cycle structure** because each vertex has exactly one outgoing edge, forming a set of cycles.
2. **Identify Cycles**:
   1. Perform a Depth First Search (DFS) or Kosaraju’s algorithm to find the **strongly connected components (SCCs)** in the graph.

3. **Assigning Values**:
   1. For each SCC:
      1. Assign a **unique value** to each vertex in the SCC.
      2. Ensure that for each cycle, the vertices get distinct values. This is necessary for the condition that for each `x < a_i`, there exists a vertex `j` such th
at `a_j = x`.
4. **Check Feasibility**:
   1. After assigning values, verify that there are **disjoint cycles**.
   2. Ensure that there are no overlapping assignments between vertices belonging to different cycles.
   3. If any cycle is connected improperly or the structure is too intertwined, mark the assignment as **IMPOSSIBLE**.
5. **Return Result**:
   1. If all cycles are successfully assigned values, print **"POSSIBLE"**.
   2. If it is impossible to assign values (due to overlapping cycles or conflicting connections), print **"IMPOSSIBLE"**.
### Step 3: **Key Considerations**
1. **Time Complexity**:
   - The **SCC detection** using algorithms like Tarjan's or Kosaraju's runs in **O(N)** time.
   - Thus, the entire algorithm can be expected to run in **O(N)** time, which is optimal for the given problem constraints.
2. **Memory Usage**:
   - The space complexity primarily depends on the storage for the graph and the SCCs, leading to an **O(N)** space requirement.
3. **Edge Cases**:
   - The graph could consist of a single cycle (where every vertex points to exactly one other in a circular manner), which will be possible to assign distinct value
s.
   - If the graph has disconnected parts (not possible here since the graph is weakly connected), the solution might fail. But we are guaranteed weak connectivity, w
hich simplifies the situation.
4. **Optimizations**:
   - Ensure that you can check cycles efficiently using union-find structures or DFS for SCCs.
   - Use efficient memory handling as the graph size can go up to 200,000 vertices.
Problem:
{question}
"""

# Function to call the LLM API
async def call_llm(prompt: str) -> str:
    response = await client.chat.completions.create(
        model="/scratch/pioneer/jobs/job.2664465.hpc/models/Qwen2.5-Coder-14B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1024,
        temperature=1.0,
    )
    return response.choices[0].message.content.strip()

# Function to process each entry and retry if the thought is too short
async def process_entry(example, min_length=100, max_retries=5):
    prompt = THOUGHT_PROMPT_TEMPLATE.format(question=example["question"])
    
    # Initial call to the LLM
    thinking = await call_llm(prompt)
    
    retries = 0
    while len(thinking) < min_length and retries < max_retries:
        print(f"[WARNING] Thought generation too short: {thinking[:50]}... Retrying {retries + 1}/{max_retries}...")
        thinking = await call_llm(prompt)
        retries += 1
    
    # If the response is still too short after the retries, log it
    if len(thinking) < min_length:
        print(f"[ERROR] Thought generation failed to meet the minimum length after {max_retries} retries.")
    
    # Add the generated thinking as a high-level strategy in the question
    example["question"] += f"\n\n<High-Level-Strategy>\n{thinking}</High-Level-Strategy>"
    return example
async def async_map(dataset, min_length=100, max_retries=5):
    all_results = []
    for i in range(0, len(dataset), 500):
        batch = dataset[i:i+500]
        batch_dict = batch
        keys = batch_dict.keys()
        values = zip(*batch_dict.values())
        batch_list = [dict(zip(keys, v)) for v in values]
        # Process entries asynchronously
        results = await tqdm_asyncio.gather(*[process_entry(ex, min_length, max_retries) for ex in batch_list])
        all_results.extend(results)
    return Dataset.from_list(all_results)

# Main function
async def main():
    # Load the dataset from Huggingface
    ds = load_dataset("Gen-Verse/CodeContests_train", split="train")
    
    # Process the dataset in parallel
    processed_ds = await async_map(ds, min_length=50, max_retries=5)
    # Save the processed dataset as a Huggingface dataset
    processed_ds.save_to_disk("agent2_train_hf")
    # Optionally, save as JSONL
    processed_ds.to_json("agent2_train.jsonl", orient="records", lines=True, force_ascii=False)
if __name__ == "__main__":
    asyncio.run(main())