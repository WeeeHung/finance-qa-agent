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
- invest time into preprocessing + embedding+chunking the data
- what are some types of questions + how can we scale from there

todo:

- try fixing 'max_workers < 1:' and rerun cli for results.
- inspect dataset for text length and table size

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

[CurrentGoal]

- List out pros and cons of

- Redesign the agent pipeline to achieve

1. high accuracy in FinQA result (float) -- able to calculate with metrics.
2. clear Explainable step (bool)
3. Fast response (float) -- able to get Five-Number Summary of latency.

- Optimization

1. Caching Queries (need to refetch doc qa queries) + calculations (most steps seem to contain duplicates -- might be unnecessary if calculation latency is negligible) (Conversational + Multi-turn Dependency)
2. Do actual self-consistency (by cross checking answers)
