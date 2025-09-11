# Getting Started
## Setting up Environment
1. Use python=3.10
2. pip install packages from requirements.txt

# Running the framework
## Pre-Run Instructions
1. Set / export the parent folder of `src` to `PYTHONPATH`.
2. Update the `src_dir` and `data_dir` of `src.utils.filepaths` accordingly (should point to your `src` folder and your `data` folder accordingly.
3. Set / export the OPENAI_API_KEY variable

## Run on ConFinQA Dataset
1. Run `src.runme`
   
   ```python
   python path/to/src/runme.py
   ```
2. To score the dataset (after running the `runme.py` file to generate the output files required for scoring), run `src.utils.scoring`
   
   ```python
   python path/to/src/utils/scoring.py
   ```

## Run chatbot
1. Run `src.app.cli`
   
   ```python
   python path/to/src/app/cli.py record_file_name
   ```
   where `record_file_name` is the filename of the record (e.g. `Single_UNP/2014/page_35.pdf-3`)

# How it works
## Agentic Pipeline
![Agentic Framework diagram](https://raw.githubusercontent.com/lemousehunter/CRDFAgent/main/diagrams/agentic_framework.png)

## Agents
### 1️⃣ Clarify
Detects ambiguity and reformulates vague user questions into precise, self-contained forms using conversation history.

### 2️⃣ Route
Implements a “short-circuit” mechanism to directly answer questions when sufficient context is already available, reducing latency.

### 3️⃣ Decompose & Retrieve
Breaks complex queries into subproblems and runs parallel retrieval agents to gather supporting facts.

### 4️⃣ Aggregate
Generates a deterministic Python function to compute the final answer, ensuring reproducibility and auditability.

### 5️⃣ Planner
A planner agent monitors outputs and injects immediate feedback for retries if needed, ending the pipeline when a direct or aggregated answer is produced.
 


 # Results
 ## Metrics Used
 1. The first metric used is the 'correctness' of answer - be it via direct matching or tolerance-based evaluation.
 2. The second metric used modulates the 'Quality of Reasoning' with 'Efficiency of Reasoning'. 'Quality of Reasoning' is defined to be the number of agents used in the pipeline that had successfully completed their tasks (therefore it is a binary number of either 0 or 1). 'Efficiency of Reasoning' is defined to be the number of attempts each agent takes to complete its task. Therefore, the afore-mentioned can be formalized as such:
    ```math
      \text{SQ\_METRIC} = \frac{\sum_{i=1}^{N} q_i}{\sum_{i=1}^{N} a_i}
    ```

where:  
- $q_i \in \{0, 1\}$ is the quality indicator of agent $i$ (1 if successful, 0 otherwise).  
- $a_i \in \mathbb{N}^+$ is the number of attempts made by agent $i$.  
- $N$ is the total number of agents in the pipeline.

## Results
The pipeline was run on the top 8 records of the `ConvFinQA` dataset to get the agent outputs, before being scored by the scorer. The scores are as follows:

 | Agent/Metric              | Score    |
 |---------------------------|----------|
 | ClarifierAgent_score      | 0.862069 |
 | DirectQAAgent_score       | 0.890805 |
 | DecomposerAgent_score     | 0.947368 |
 | FreeAgent_score           | 0.645833 |
 | AggregatorAgent_score     | 0.312500 |
 | `SQ_METRIC`               | 0.583908 |
 | Correctness               | 0.689655 |
 


# Some notes
1. This system was developed on mac using pycharm, compatibility with Windows systems cannot be guaranteed
2. If I had missed out any packages in `requirements.txt`, please let me know :)
3. The system might get stuck in "reflection" loops, especially at the aggregation stage. If that happens, just re-run the system. An `agent_retry_limit` has already been set (defaults to 5).
4. Each 'agent' file can be run independently, I developed it as such to test out the pipeline incrementally
5. Currently, 'gpt-5-mini' is being used as the LLM for all agents. It works fairly well

