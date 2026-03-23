# Learning Large Language Models (LLMs)

This is a repository to house the journey to understanding, developing, training, and deploying LLMs from the ground up.

## Weeks 1–2

Objective

Stop treating transformers as magic. Build one line-by-line.

What you do (hands-on)
	•	Implement:
	•	Token embedding
	•	Positional encoding
	•	Self-attention
	•	Multi-head attention
	•	Transformer block
	•	Train a small GPT-like model on:
	•	Tiny Shakespeare
	•	Wikipedia subset
	•	Code (Python files)

Constraints (important)
	•	No HuggingFace Trainer at first
	•	Pure PyTorch
	•	You must understand:
	•	Why attention is O(n²)
	•	Why layer norm placement matters
	•	What residuals actually do

Deliverables
	•	A model that:
	•	Trains
	•	Generates text
	•	A README explaining:
	•	Why it diverges
	•	Why it overfits
	•	What hyperparameters matter most

📌 If you do nothing else in this plan, do this phase well.
This is where “chef mode” begins.

⸻

Phase 2: “I can fine-tune and evaluate LLMs properly”

Weeks 3–4

Objective

Move from training to using models intelligently.

What you do
	•	Take a pretrained model (LLaMA-like / Mistral-like)
	•	Do:
	•	Full fine-tuning
	•	LoRA / QLoRA
	•	Tasks:
	•	Instruction tuning
	•	Domain adaptation
	•	Classification via prompting vs fine-tuning

You must explicitly learn:
	•	Loss curves ≠ model quality
	•	Overfitting in LLMs
	•	Evaluation pitfalls
	•	Prompt sensitivity

Evaluation (critical)
	•	Implement:
	•	Perplexity
	•	Task-based metrics
	•	Human eval heuristics
	•	Build a small eval harness

Deliverables
	•	A comparison doc:
	•	Prompting vs fine-tuning
	•	LoRA vs full fine-tune
	•	One bad experiment and why it failed

This is where you start sounding like a real ML researcher in meetings.

⸻

Phase 3: “I experiment like an R&D engineer”

Weeks 5–6

Objective

Become fluent in experimental thinking, not just execution.

What you do

Pick one axis and explore deeply:

Examples:
	•	Context length scaling
	•	Dataset quality vs size
	•	Tokenization choices
	•	Attention variants (Flash, grouped query)
	•	RLHF vs supervised tuning
	•	Tool use / function calling

Rules
	•	One variable at a time
	•	Everything logged
	•	Write why you expected something to happen

Papers to re-implement (light versions)
	•	GPT-style scaling laws
	•	LoRA paper (core idea)
	•	One RLHF-adjacent paper (even partial)

Deliverables
	•	2–3 experiment reports
	•	One surprising result

📌 This phase converts you from “engineer who runs code” to engineer who generates insight.

⸻

Phase 4: “I can speak LLM fluently to leadership”

Weeks 7–8

Objective

Translate ML R&D into business language.

What you do
	•	Pick a SynthBee-relevant problem
	•	Propose:
	•	Model choice
	•	Data strategy
	•	Evaluation
	•	Deployment plan
	•	Build:
	•	A prototype
	•	A short decision memo

You must practice:
	•	Explaining why not to do something
	•	Cost vs performance tradeoffs
	•	When to fine-tune vs prompt
	•	Risk and failure modes

Deliverables
	•	A 2–3 page internal proposal
	•	A demo notebook or API

This directly maps to points 1, 3, and 4 in your job description.

