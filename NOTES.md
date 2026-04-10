## Submission

**Flow:** 1. Analysis → 2. Research → 3. Improvements → 4. Next steps

---

## 1. Analysis

### Current architecture and data flow (agents, retrieval, persistence)

Current architecture creates a fixed stack of agents that is led by the Planner Agent. The Document QA is initialized by first embedding the table and text data into a local vectorDB. Each question would then trigger a sequential chain (Clarifier -> DirectQA -> Decomposer -> Free Agents -> Aggregator). Planner will trigger a rerun with feedback when `self_consistency` and `further_action_required` is true. Each agent has their own memory thread and a fixed retry attempts to improve their output given feedback. Each retry mean pushing the agent back in the stack pipeline.

Essentially, each agent makes an LLM call with their customized prompts and input, and output things required as inputs for subsequent agents. These inputs and outputs are persisted in states and shared amongst agents.

### Pros of this approach

1. The pipeline contains a short circuit path from the DirectQA agent. If direct answer is found in text, it will return directly and skip the rest of the stack.
2. The pipeline breaks down roles for each specialised agent to execute with their customized prompts.
3. Agents can execute with a customized feedback loop, and regenerate for better output when self-consistency flag is activated.
4. Runs Free Agent on parallel as the subproblems are not dependent on each other.
5. Every Agent updates step history which makes AI response explainable and transparent.

### Cons, risks, and failure modes

1. React Agent might be an overkill in this case as all of the agents are not equipped with tools -> no reasoning-tooluse loop
2. Planner agent always runs, and the feedback could go unused.
3. It is very slow as many sequential LLM calls has to be made. (min 10 and max 58 + 7N calls if the run completes the stack with aggregator agent! min 4 calls if DirectQA exits early)
4. Vector RAG could be overkill given the size of the record. Also, default chunking (240 words with 8-word sliding window) + top K is being used. This could introduce noise in the retrieved context.
5. No caching attempts for retrieval (many related questions could use previous RAG results)
6. The sequential LLM calls could lead to weaker data preservation and propagate effects of hallucination. (ie. if Clarifier Agent rewrites question and misses out on a point, all agents in the chain will suffer from the loss of information).
7. Manual Python Exec is not entirely safe with restricted builtins. Running on the same process can be dangerous and a sandbox to execute code is safer. (ie. SystemExit can bypass except Exception)

### Data and context (text, tables, splits); implications for chunking, retrieval, and prompting

Texts and tables are well parsed in the `convfinqa_dataset.json`. After scanning through using a script, I realized that the ~3500 records includes a median of < 4000 words and max < 15,000 words. This is considered small in the context window size of current general LLM models.

Units of the numerical values are often included in the texts, texts near tables, or table headers. It is important that we parse that information out as it is essential for accurate data retrieval and calculation, especially since the tasks expects raw numerical outputs. It is also important that we save context of previous questions as the QnA is asked sequentially and contextualized. Some questions might be financial-specific, thus we could few-shot prompt to have the LLM understand the question requirements better.

I also realized that while we are using executed_answers as goldens, they are not always TRUE. (ie. Double_HII/2017/page_104.pdf q1 and q2). That is possibly because in the original dataset, our conv_questions were dialogue_break, and thus the intermediate questions are not fully validated. ("exe_ans" is golden but "exe_ans_list" is not)

---

## 2. Research

### ConvFinQA (paper / task) and relation to this pipeline

The dataset and task is most probably inspired by the [ConvFinQA Research Paper](https://arxiv.org/pdf/2210.03849). It is written in late 2022 during early GPT3 era. It is targetted to do multi-step, conversational and document-backed reasoning, to answer various contextualised financial questions. The outputs are usually numerical values, deterministic and unambiguous. (ie. there exist golden values)

The paper experimented with **FinQANet with a RoBERTa‑large encoder** and **GPT‑3 (175B param)**, and the former performed better.
The former has 2 core strengths: Neural network to identify and extract useful numerical information and operations needed, and a symbolic executor to run the sequenced operations. GPT3 back then, however, was only used with few-shot prompting to generate the direct answer. Thus, it lacks reliability in reasoning and mathemical execution.

### ReAct-style agents here — when justified vs overkill

This perfectly aligns with ReAct style agents. We want an unambiguous outcome, reasoning (potentially loops of reasoning to check through with additional information), and tool use to execute arithmetic operations without hallucinations.

This is inspired by the current repo, where all agents initialises as a langgraph react agent that has built-in ReAct-like loop. This allows the agent to maintain context and execute loops of reasoning-action-reasoning.

This framework is preferred over Multi-agent systems for a few reasons. Our task looks into a single record, expects multi-turn QA, expects unambiguous numerical output. The process of getting these output is often formulaic, sequential and relatively simple. (ie. Query rewrite to contextualise question -> Retrieve relevant context and numbers -> Generate operations to answer question -> call calculation tool with sequence of operations -> Optional Check)

Multi Agent System is often advantageous when we expect open-ended and complex reasoning (ie. Create a deep research report about Financial market today). Many more specialisation might be required here (ie. Market research to gather data, Domain expert to interpret trends, Aggregator to compile report as per user preferences, etc), thus it justifies to split the expertise to different agents and have them work sometimes independently and on parallel.

### Evaluation (correctness, pipeline metrics, LLM-as-judge tradeoffs)

We first need to identify our goals for the task before designing for evaluation. The research paper identifies its goals as:

- To Handle Multi-Turn QA (later questions depend on earlier ones)
- To Perform Numerical reasoning over tables + text
- To Support Multi-step Arithmetic operations (ratios, differences, aggregations)
- To Ensure interpretability via sequenced operation steps

As for the repo, it is not specifically documented. So, I came up with 3 main goals that references the aforementioned and seem reasonable if this were to be a consumer product: (in order of priority)

1. Accuracy
2. Explainability
3. Latency

The current scoring pipeline is good as it targets raw accuracy by comparing raw output and golden output. This is best and most important indicator of performance. This is also done with exact match, then LLM-as-a-judge. We can improve this with a deterministic linient matching to reduce the usage of the fallback LLM judge. Example of this is standardising the decimal significance, typing, percentage formats, etc. We could also have separate grading for questions in different turns (ie. Find average of accuracy for Q1s or Q4s). This highlights the significance of multi-turn QA. Theoretically, questions that depend on earlier context should be a more complex question, thus harder to answer and have lower accuracy. Optionally, we can also grade accuracy of retrieval and execution separately to optimize this system. For retrieval, there will be ground truths for correct sources/clauses, thus we can evaluate its recall to ensure high retrieval rate of all necessary information. For execution, we can evaluate number of attempts in running python execution and the code accuracy based on operations. This helps us determine the operations to code quality during tool-use stage.

Explainability is also well established in the step-history. This is a good trait for reasoning agents and helps reproduce the answer that the Agent got. To be more transparent, we could also include sources of the retrieved context. LLM Judges are also used to evaluate how well each agent executed their tasks and its faithfulness. Using binary scoring (0 or 1) for LLM Judging also eliminates some ambiguity and subjectivity compared to 1-5 scales.

## Latency is mostly related to the user experience. This goal is included as I faced long waiting time when executing the multi-agent pipeline. Rationally, as an end-user, the response time should be faster than if I were to reference the document and answer the question manually. This also assumes that this tool is not used as a background agent where latency is less of a concern. This is a reasonable assumption since its a QA system which suggests an end user chatting with the agent. To evaluate, we can easily track latency at all stages of the ReAct loop and use it to compare cross systems or optimize current system. We can use LangSmith for this.

## 3. Improvements

### Changes, builds, or documentation delivered

As I have to experiment with the different approaches, I decided to store them as different versions of source files while using a universal grader for simplicity. We will also just work on the first 10 datasets due to constraint of not having the true golden + save cost of LLM calls during experimentation phase. True evaluation will require running through entire dataset.

Key changes:

- Build a true golden dataset (first 10 due to time constraint)
- Evaluate the current SRC code
- build and eval SRC_V1 -- Single LLM call with question history (Vanilla)
- build and eval SRC_V2 -- Single LLM call with question rewrite
- build and eval SRC_V3 -- ReAct Agent with question rewrite + Python Execution Tool
- build and eval SRC_V4 -- ReAct Agent with question rewrite + Python Execution Tool + KB building

### Measurement, profiling, validation

### Backlog

---

## 4. Next steps

### Model adaptation (LoRA / similar): specialization, data, signals

### Product / platform (e.g. Manus): intent routing and adapter switching for task-specific stacks

---

# Braindump

Initial Code Deep Dive

- many pydantic warnings (can clean up but not serious)
- env file + configs need clean up
- can store and display logs better
- we need better testing loop rather than manual inspection of answer

[first attempt] - Tried cli to test flow on a single doc

- there is no caching for DirectQA actions (ie. if we've queried before, we should be able to retrieve directQA answers since its duplicate action and document stays the same)
- each question take very long. maybe need better parallelism if possible. can do latency check on each stage to improve speed as this user experience isnt the best
- got 'assistant: I'm sorry, I couldn't find an answer to that question.' after waiting for very long, i suspect some agent has reached the limit. it seems like it managed to get some kind of answer but replied with a fallback instead of giving 'inconfident' answer or 'ran out of steps to think?'
- what are the params / flags? self consistency seem interesting (self_consistency here only mean ability to rerun a step)
  [Self-consistency = generating multiple independent answers and selecting the most consistent one, normalize, then pick majority / semantic clustering / confidence weight (freq. x confidence) voting]
- clarifier does self clarify? can i ask the users questions to clarify in case the prompt given is not good enough or confusing
- tool seems to target financial analysts, auditors who need grounded, traceable answers from long PDFs
- seems like direct qa type questions work quite well
- it would be nice if there are references (which line or which table cell) to back the data found for credibility if this is productionize
- using a stack for the pipeline and there are retry limits per agent, so stack is finite and answer will eventually be provided.
- neural encoder loaded twice
- reAct agent per agent seems unnecessary since there are no tools thus no actions to be taken. it seems to be created only for structured formatting and checkpointing which could be done with normal llm calls
- cli runs per record id which is at page level now.
- the retrieved chunks observed in data/results is huge, might need to shrink chunk size to observe something helpful; issue faced as late agents take everything into context and this may either include duplicated context
- i exceeded the context window for one of the queries (inspect what context is being stacked / duplicated)
- what does feedback for each agent return us?
- grading agent uses langgraph react agent again + uses LLM as a judge most of the time. maybe for golden numerical answer, we can expect LLM judge as fallback but use actual text matching? (do some leniency, like numerically equal but string not exact match, for text matching before doing llm judging)
- improve llm as a judge;
- could invest time into preprocessing + embedding+chunking the data
- what are some types of questions + how can we scale from there
- 'max_workers < 1:' should be changed to 'max_workers >= 1:'.
- inspect dataset for text length and table size
- common problems come from scale and unit issue (mishandling units in table header, can be done by preprocessing the data + gathering implied information), unevaluated values, unsure of the output format based on the question, agent tasks are sequential (so if clarifier dropped details / weakened intent) other tasks will fail accordingly,
- step history and error logging is wrong for direct qa agent
- direct qa agent is tasked to do both routing and parsing out direct ans from context
- sometimes python execution can fail and result in error at aggregator stage

## CRDFAgent/data/convfinqa_dataset.json

### Record Counts by Split

| Split     | Number of Records |
| --------- | ----------------- |
| Train     | 3,037             |
| Dev       | 421               |
| **Total** | **3,458**         |

### TRAIN (n = 3,037 records)

| Statistic                             | Min | Median  | Max    |
| ------------------------------------- | --- | ------- | ------ |
| pre_text length (characters)          | 1   | 1,417.0 | 7,153  |
| post_text length (characters)         | 1   | 1,681.0 | 8,769  |
| combined pre+post length (characters) | 103 | 3,619.0 | 14,166 |
| table cell count (sum of row widths)  | 1   | 12.0    | 114    |

### DEV (n = 421 records)

| Statistic                             | Min | Median  | Max   |
| ------------------------------------- | --- | ------- | ----- |
| pre_text length (characters)          | 17  | 1,072.0 | 5,639 |
| post_text length (characters)         | 1   | 1,750.0 | 5,850 |
| combined pre+post length (characters) | 210 | 3,479.0 | 8,730 |
| table cell count (sum of row widths)  | 2   | 12.0    | 48    |

[PLAN]

- spend first half of time understanding the objective, data, codebase. This allows me to pick up pros and cons of current implementation to brainstorm on improvements in architecture

Phase 1 (MVP — do this)
1 LLM planner
DSL output
Python executor
Trace output
Phase 2 (improve accuracy)
Add schema constraints
Improve table grounding
Add simple verifier
Phase 3 (research-level)
Train/fine-tune planner on ConvFinQA
Add program-of-thought supervision
Beam search over programs

---

## **Phase 1 — Document Preprocessing & Execution Prototype**

**Goal:** Get a working QA pipeline on a single document.
**Steps:**

1. **Preprocess the document**
   - Normalize units, numeric formats, and dates.
   - Extract tables and key sections into structured formats (JSON, tables).

2. **Contextualized query handling**
   - Optionally rewrite user queries to explicit forms.
   - Feed to Planner LLM to generate execution steps (DSL or single-step).

3. **Programmatic execution**
   - Compute answers from preprocessed tables or structured data.

**Testing/Grading:**

- Compare output with golden/reference answers for **accuracy**.
- Log **latency** for end-to-end QA.

**Deliverable:** Working single-document QA prototype.

---

## **Phase 2 — Explicit KB Introduction**

**Goal:** Capture all raw facts from the document systematically.
**Steps:**

1. Extract **explicit knowledge items** from tables and numeric statements.
2. Store in **KB structure**:
   - Fields: `id`, `info`, `type: explicit`, `value`, `unit`, `tags`.

3. Optional: include reasoning placeholders for future implicit facts.

**Testing/Grading:**

- Verify KB contains all expected explicit facts.
- Compare execution results with and without KB to ensure consistency.

**Deliverable:** Structured KB for explicit information.

---

## **Phase 3 — Implicit Knowledge & Reasoning**

**Goal:** Enable multi-step calculations and derived facts.
**Steps:**

1. Define **reasoning operations** (`subtract`, `percentage_change`, `multiply`, etc.).
2. Generate **implicit knowledge items** with reasoning chains.
3. Update execution engine to **read from KB nodes** instead of raw tables where applicable.

**Testing/Grading:**

- Check correctness of **derived facts** against manual calculations.
- Optional: log **step-level correctness**.

**Deliverable:** KB enriched with both explicit and implicit facts; reasoning chains stored.

---

## **Phase 4 — Graph-based RAG & Multi-document Scaling**

**Goal:** Make the system scalable and explainable for large documents.
**Steps:**

1. Convert KB to **graph structure**:
   - Nodes = KB items
   - Edges = `derived_from` relationships

2. Add **vector embeddings** for semantic retrieval (supporting RAG).
3. QA execution:
   - User query → optional rewrite → retrieve relevant KB nodes + raw text chunks → Planner LLM executes reasoning.

4. Append **new knowledge** back to KB after QA.

**Testing/Grading:**

- Track **final QA accuracy** against golden answers.
- Log **latency** for end-to-end retrieval + reasoning.
- Optionally visualize reasoning graph for **explainability**.

**Deliverable:** Scalable QA system with KB + graph reasoning + vector RAG for large documents.

---

### **Additional Notes on Grading & Explainability**

- **Grading:** focus primarily on **accuracy**; latency is optional.
- **Explainability:** store and return **reasoning chains** and links to KB nodes; step-level grading is optional.
- **Testing Strategy:** start small with single documents, then gradually scale to multi-page or multi-document scenarios.

---

- Redesign the agent pipeline to achieve

1. high accuracy in FinQA result (float) -- able to calculate with metrics.
2. clear Explainable step (bool)
3. Fast response (float) -- able to get Five-Number Summary of latency.

- Optimization

1. Caching Queries (need to refetch doc qa queries) + calculations (most steps seem to contain duplicates -- might be unnecessary if calculation latency is negligible) (Conversational + Multi-turn Dependency)
2. Do actual self-consistency (by cross checking answers)

---
