Hiver AI Support Agent — AmazonHelp

A small AI customer-support agent built from real Customer Support on Twitter (TWCS) conversations.

The system:

Classifies a customer message into a support intent.

Retrieves historical AmazonHelp support evidence.

Generates a grounded draft reply.

Checks the reply for obvious data leakage.

Decides whether the interaction should be automatically handled or escalated.

------------------------------------------------------------------------------------------------------------------------------------------

1. Complete Pipeline

+-------------------------+
|    Customer Message    |
+-----------+-------------+
            |
            v
+-------------------------+
|  Intent Classification  |
|       Qwen 2.5 7B       |
+-----------+-------------+
            |
            v
+-------------------------+
|    Predicted Intent     |
|     21-intent taxonomy  |
+-----------+-------------+
            |
            v
+-------------------------+
|   Historical Retrieval  |
|     nomic-embed-text    |
+-----------+-------------+
            |
            v
+-------------------------+
|   AmazonHelp Historical |
|     Support Evidence    |
+-----------+-------------+
            |
            v
+-------------------------+
|    Reply Generation     |
|       Qwen 2.5 3B       |
+-----------+-------------+
            |
            v
+-------------------------+
|  Safety / Leakage Check |
|   Deterministic Rules   |
+-----------+-------------+
            |
            v
+-------------------------+
|   Escalation Decision   |
|    Rule-based Engine    |
+-----------+-------------+
            |
       +----+----+
       |         |
       v         v
 AUTO_HANDLE  ESCALATE

The overall architecture is hybrid:

Qwen 2.5 7B Intent Classification
              +
Semantic Retrieval
              +
Qwen 2.5 3B Reply Generation
              +
Deterministic Safety Rules
              +
Rule-based Escalation

------------------------------------------------------------------------------------------------------------------------------------------

2. Problem Framing

What this project builds

The project builds an experimental customer-support agent that can:

identify the customer's support intent,

retrieve similar historical AmazonHelp conversations,

draft a concise support response,

check the draft for obvious leakage,

and determine whether the interaction should be auto-handled or escalated.

What this project does not build

This is not a production customer-service system.

It does not:

access real customer accounts,

perform refunds or cancellations,

change orders,

access private customer information,

contact customers,

guarantee delivery outcomes,

replace human support agents,

or claim production-level reliability.

The generated response is a draft only.

------------------------------------------------------------------------------------------------------------------------------------------

3. Dataset

Source

Customer Support on Twitter (TWCS) dataset.

Kaggle source:

https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

Target brand

AmazonHelp

AmazonHelp extraction

169,840 AmazonHelp tweets

82,556 reconstructed conversations

70,956 direct customer → AmazonHelp pairs

66,347 unique customer messages

The historical data is used for retrieval and response-grounding experiments.

Main generated data files are stored under:

results/

The raw dataset is stored under:

data/raw/twcs.csv

------------------------------------------------------------------------------------------------------------------------------------------

4. Golden Evaluation Set

A manually labelled golden set of 200 unique customer messages was created.

The golden examples were removed from the historical reference pool to reduce evaluation leakage.

Intent taxonomy

The final taxonomy contains 21 intents:

ACCOUNT
CASUAL_ENGAGEMENT
CUSTOMER_SERVICE
DELIVERY_DELAY
DELIVERY_INSTRUCTIONS
DELIVERY_PROBLEM
DELIVERY_TRACKING
DEVICE
FEEDBACK
INSUFFICIENT_CONTEXT
ORDER_CANCELLATION
ORDER_DISPATCH
OTHER
PACKAGE_NOT_RECEIVED
PAYMENT
PRODUCT_INFORMATION
REFUND
RESOLUTION_CONFIRMATION
RETURN
SELLER
WEBSITE_OR_APP

Golden set:

data/golden/golden_set.csv
results/amazonhelp_golden_labeled.csv

The evaluation set contains 200 examples and is frozen for comparison between experiments.

------------------------------------------------------------------------------------------------------------------------------------------

5. Models

Intent Classification

The current classifier uses:

Qwen 2.5 7B
Ollama

The classifier receives:

the customer message,

the 21 intent definitions,

explicit intent boundaries,

classification instructions,

and deterministic handling for several high-confidence cases.

The final current evaluation result is:

111 / 200 = 55.50%

Semantic Retrieval

Historical conversations are embedded using:

nomic-embed-text

A 3,000-example historical reference subset is used for the semantic retrieval experiment.

The embedding cache is stored at:

results/amazonhelp_embeddings_3k.pkl

Reply Generation

Reply generation uses:

Qwen 2.5 3B

The generator receives the new customer message and cleaned historical customer issues as grounding evidence.

The prompt explicitly prevents the model from claiming access to private data or inventing operational outcomes.

Safety

Deterministic checks detect obvious leakage such as:

URLs

Twitter usernames

placeholders

agent identifiers

signatures

order/tracking identifiers

other unsafe response patterns

Escalation

A rule-based engine produces:

AUTO_HANDLE

or:

ESCALATE

The escalation result is treated as a policy outcome, not an accuracy metric, because the current evaluation data does not contain human-labelled escalation ground truth.

------------------------------------------------------------------------------------------------------------------------------------------

6. Intent Classification Results

The classifier was evaluated on the same frozen 200-example golden set.

Baseline 1 — Majority Class

The trivial majority-class baseline achieved:

37 / 200 = 18.50%

This provides a lower-bound reference for a classifier that always predicts the most frequent intent.

Baseline 2 — TF-IDF Similarity

The simple TF-IDF nearest-neighbour baseline achieved:

30 / 200 = 15.00%

Original LLM Classifier

The original Qwen 2.5 3B classifier achieved:

77 / 200 = 38.50%

Few-Shot Experiment

The few-shot v2 classifier achieved:

35.00%

Current Classifier

The current improved classifier uses Qwen 2.5 7B with stronger intent boundaries and deterministic handling for high-confidence patterns.

Result:

111 / 200 = 55.50%

Comparison

Approach

Accuracy

TF-IDF nearest neighbour

15.00%

Majority-class baseline

18.50%

Few-shot v2

35.00%

Original Qwen 2.5 3B

38.50%

Current Qwen 2.5 7B classifier

55.50%

The current result is an experimental result on the frozen 200-example golden set. It should not be interpreted as production accuracy.

------------------------------------------------------------------------------------------------------------------------------------------

7. Retrieval Results

Two retrieval approaches were explored.

TF-IDF

The TF-IDF approach relied mainly on lexical overlap.

Result:

15.00% top-1 intent match

Semantic Retrieval

Semantic retrieval uses:

nomic-embed-text

The embedding retriever produced more semantically meaningful neighbours during qualitative inspection.

However, semantic similarity alone was not treated as a replacement for intent classification.

The earlier intent-aware retrieval experiment achieved:

38.50% top-1 intent match

This showed that retrieval can provide useful historical context without necessarily solving the intent-classification problem.

------------------------------------------------------------------------------------------------------------------------------------------

8. Reply Generation

Historical AmazonHelp support conversations are used as grounding evidence.

The reply-generation prompt instructs the model not to:

claim private account access,

claim investigations it cannot perform,

invent refunds,

invent replacements,

invent cancellations,

invent delivery status,

invent policies,

expose historical IDs,

expose URLs,

expose usernames,

expose agent information.

Historical evidence is cleaned before being included in the generation prompt.

Generated replies are passed through deterministic safety checks.

The generated text is intended to be a support-reply draft, not an executed customer-service action.

------------------------------------------------------------------------------------------------------------------------------------------

9. Reply Safety Results

20 generated replies were evaluated.

Deterministic leakage evaluation

20 / 20 replies passed

The leakage evaluator reported:

Total replies:       20
Leakage-free:        20
Leakage detected:     0
Leakage rate:       0.0%

The checks covered:

URLs

Twitter URLs

placeholders

usernames

agent initials

signatures

No obvious leakage was detected in the evaluated 20 replies.

Result file:

results/reply_leakage_results_20.csv

This is a deterministic safety signal and does not establish that every generated reply is factually correct.

------------------------------------------------------------------------------------------------------------------------------------------

10. LLM Reply Quality Judge

20 generated replies were evaluated using a local LLM judge.

The judge evaluates six dimensions from 0 to 2:

Metric

Score

Relevance

1.85 / 2

Usefulness

1.30 / 2

Grounding

0.30 / 2

Factual safety

0.30 / 2

Leakage safety

0.00 / 2

Style

1.95 / 2

Total

5.70 / 12

All 20 judgments were successfully parsed.

The evaluation suggests that the generated replies were generally relevant and stylistically acceptable, while grounding and factual-safety scores were weak in this particular judge run.

The deterministic leakage evaluation provides a separate signal: all 20 evaluated replies passed the implemented leakage checks.

Important evaluation caveat

The LLM judge output contains inconsistencies between some numerical scores and its textual reasons. Therefore, the judge should be treated as a supporting evaluation signal rather than authoritative ground truth.

Result file:

results/reply_quality_judge_20.csv

------------------------------------------------------------------------------------------------------------------------------------------

11. Judge-Human Calibration

A small independent human calibration sample of 10 replies was compared with the LLM judge.

Results:

Exact agreement: 20%
Quadratic weighted kappa: -0.129

The calibration sample is very small.

These results indicate weak agreement in this sample, so the LLM judge is not treated as a replacement for human evaluation.

This is especially important because an LLM judge can produce plausible explanations while still applying an inconsistent scoring rubric.

------------------------------------------------------------------------------------------------------------------------------------------

12. Escalation Decision

The escalation layer uses the predicted intent and generated reply to determine whether the request can be auto-handled or should be escalated.

20-example reply evaluation

Decision

Count

AUTO_HANDLE

6

ESCALATE

14

Total

20

Escalation rate:

70%

200-example policy coverage

A separate run over all 200 golden examples produced:

AUTO_HANDLE: 63 / 200 = 31.5%
ESCALATE:    137 / 200 = 68.5%

The escalation result is a policy outcome, not an accuracy metric, because there is no human-labelled escalation ground truth in the current evaluation set.

Result files:

results/escalation_results_reply_20.csv
results/escalation_results_200.csv

------------------------------------------------------------------------------------------------------------------------------------------

13. Top Five Failure Modes

The current classifier errors show several recurring boundaries.

1. RESOLUTION_CONFIRMATION → CASUAL_ENGAGEMENT

Short messages such as:

Thanks
Merci
Thank you

can be difficult to distinguish between a genuine resolution confirmation and casual engagement.

The largest observed confusion was:

RESOLUTION_CONFIRMATION → CASUAL_ENGAGEMENT

Hypothesis:

The two intents contain overlapping short conversational language. More state-aware examples are needed.

2. CUSTOMER_SERVICE → ACCOUNT / INSUFFICIENT_CONTEXT / RESOLUTION_CONFIRMATION

Some customer-service requests contain account-related or conversational language without explicitly describing an account problem.

Examples include requests that effectively mean:

I already contacted support.
I need someone to help me.
I already replied.

Hypothesis:

The broader CUSTOMER_SERVICE category competes with several specific categories, especially when the message contains little operational detail.

3. DELIVERY_PROBLEM → DELIVERY_TRACKING

Examples involving packages, couriers, and delivery status can be ambiguous.

The distinction is:

DELIVERY_TRACKING
= asking where the package is or asking for its status

DELIVERY_PROBLEM
= reporting an unacceptable or failed delivery event

Hypothesis:

Both intents share strong delivery vocabulary, so the classifier must reason about the customer's requested outcome rather than only package-related words.

4. OTHER → PACKAGE_NOT_RECEIVED / PRODUCT_INFORMATION / CUSTOMER_SERVICE

Short or unusual messages were sometimes forced into a more specific category.

Examples include messages with limited context, complaints, or mixed issues.

Hypothesis:

OTHER acts as a broad residual class, while the model tends to prefer recognizable specific intents when a message contains a strong keyword.

5. DELIVERY_DELAY → DELIVERY_TRACKING / ORDER_DISPATCH

Delivery delay, tracking, and dispatch are closely related operational states.

The distinction is:

DELIVERY_DELAY
= shipment is late or past the expected date

DELIVERY_TRACKING
= customer wants current location/status

ORDER_DISPATCH
= order has not yet been dispatched/shipped

Hypothesis:

Correct classification requires understanding the fulfilment timeline and state rather than matching isolated words such as "shipping", "delivery", or "order".

------------------------------------------------------------------------------------------------------------------------------------------

14. What Is Misleading About the Headline Number?

The current headline number is:

55.50%

More precisely:

111 / 200 intent classifications correct

This number should not be interpreted as the success rate of the complete support agent.

It measures only intent classification on the frozen, manually labelled 200-example golden set.

The complete system also contains:

semantic retrieval,

reply generation,

grounding,

safety validation,

leakage prevention,

escalation.

Each component has different evaluation signals and limitations.

Therefore, the headline should be written as:

55.50% intent classification accuracy on the frozen 200-example manually labelled AmazonHelp golden set.

It should not be presented as:

55.50% overall support-agent accuracy

The reply-generation evaluation also demonstrates why a single classifier number is insufficient: a response can be fluent and stylistically acceptable while still having grounding or factual-safety weaknesses.

------------------------------------------------------------------------------------------------------------------------------------------

15. Key Findings

Finding 1 — Model size and prompt design affected classification

The tested classifier results were:

Experiment

Accuracy

Majority-class baseline

18.50%

TF-IDF baseline

15.00%

Few-shot v2

35.00%

Original Qwen 2.5 3B

38.50%

Current Qwen 2.5 7B

55.50%

The current approach improved substantially over the original LLM baseline on the frozen evaluation set.

Finding 2 — Semantic retrieval is useful but insufficient

Embedding retrieval produced semantically relevant historical examples.

However, semantic similarity alone did not solve the intent-classification problem.

This supports using retrieval primarily as contextual evidence for response generation rather than assuming that nearest-neighbour similarity is sufficient for intent classification.

Finding 3 — Leakage safety and response quality are different dimensions

The deterministic leakage evaluation achieved:

20 / 20 clean

At the same time, the LLM judge produced:

5.70 / 12 overall

with particularly low grounding and factual-safety scores.

Therefore:

A response can pass deterministic leakage checks while still requiring improvement in grounding and factual reliability.

Finding 4 — The LLM judge requires human calibration

The 10-example calibration produced:

20% exact agreement
-0.129 quadratic weighted kappa

The sample is too small for a production reliability estimate, but it demonstrates why automated judging should be calibrated against human evaluation.

Finding 5 — Short and ambiguous support messages are a major challenge

Many classification errors occur around:

resolution confirmation,

casual engagement,

customer service,

insufficient context,

other,

delivery tracking,

delivery problems,

dispatch,

and delay.

These categories require interpreting conversational state rather than simply identifying keywords.

------------------------------------------------------------------------------------------------------------------------------------------

16. Limitations

The golden evaluation set contains only 200 manually labelled examples.

Intent frequencies are uneven, with some intents having very few examples.

The semantic reference set contains 3,000 examples.

The reply-quality evaluation contains only 20 examples.

The escalation evaluation does not contain human-labelled escalation ground truth.

The escalation percentages therefore represent policy behaviour rather than escalation accuracy.

The LLM reply-quality judge showed weak agreement with the small human calibration sample.

The judge also showed inconsistencies between some scores and explanations.

Deterministic leakage checks cover only the implemented patterns and cannot guarantee complete safety.

The generated reply is a draft and does not execute real customer-service operations.

Local model behaviour can vary with model versions, prompts, and runtime configuration.

Therefore, the reported metrics should be treated as experimental evaluation results rather than production-level benchmarks.

------------------------------------------------------------------------------------------------------------------------------------------

17. Reproduction

Install dependencies

pip install -r requirements.txt

Pull the required models

For the current intent classifier:

ollama pull qwen2.5:7b

For reply generation:

ollama pull qwen2.5:3b

For semantic retrieval:

ollama pull nomic-embed-text

Run the live demo

With Ollama running:

python run.py "My package was meant to arrive yesterday and has not arrived."

The demo displays:

predicted intent,

delivery route,

retrieved historical evidence,

generated reply,

safety-check result,

escalation decision.

Run intent evaluation

python test/evaluate_classifier.py

Results:

results/classification_results_200.csv

Run reply generation evaluation

python test/evaluate_reply_generation.py

Results:

results/reply_generation_eval_20.csv

Run reply-quality judge

python test/evaluate_reply_quality.py

Results:

results/reply_quality_judge_20.csv

Run deterministic leakage evaluation

python test/evaluate_leakage.py

Results:

results/reply_leakage_results_20.csv

Run escalation evaluation

python test/evaluate_escalation.py

Results:

results/escalation_results_reply_20.csv

Run classifier error analysis

python test/error_analysis.py

------------------------------------------------------------------------------------------------------------------------------------------

18. Important Result Files

results/
├── amazonhelp_golden_labeled.csv
├── classification_results_200.csv
├── baseline2_similarity_results.csv
├── semantic_retrieval_results_200.csv
├── amazonhelp_embeddings_3k.pkl
├── reply_generation_eval_20.csv
├── reply_leakage_results_20.csv
├── reply_quality_judge_20.csv
├── escalation_results_200.csv
└── escalation_results_reply_20.csv

Additional dataset files include:

results/amazonhelp_conversations.csv
results/amazonhelp_customer_support_pairs.csv
results/amazonhelp_pairs_analyzed.csv
results/amazonhelp_reference_pool.csv
results/reference_labeled_100.csv

------------------------------------------------------------------------------------------------------------------------------------------

19. Project Structure

The core implementation is organized as:

.
├── data/
│   ├── raw/
│   └── golden/
├── results/
├── src/
│   ├── agent/
│   ├── data/
│   ├── escalation/
│   ├── generation/
│   ├── intent/
│   ├── reply/
│   ├── retrieval/
│   └── pipeline.py
├── test/
├── requirements.txt
├── run.py
├── decision_log.md
└── README.md

The main runtime components are:

src/agent/
    End-to-end support-agent orchestration

src/intent/
    Intent taxonomy and classification

src/retrieval/
    Historical semantic retrieval

src/generation/
    Safety-constrained reply generation

src/escalation/
    Auto-handle vs escalation logic

src/data/
    Dataset loading and conversation processing

test/
    Evaluation and analysis scripts

------------------------------------------------------------------------------------------------------------------------------------------

20. Decision Log

The project maintains a decision log containing non-obvious implementation and evaluation decisions.

See:

decision_log.md

The decisions cover areas including:

dataset selection,

conversation reconstruction,

golden-set construction,

intent taxonomy,

evaluation/reference separation,

retrieval,

weak supervision,

reply grounding,

safety,

escalation,

and evaluation methodology.

------------------------------------------------------------------------------------------------------------------------------------------

21. One-Week Next Steps

If development continued for one week, the priority areas would be:

Increase manually labelled examples for the most confused intent pairs.

Improve state-aware classification for:

delay,

tracking,

dispatch,

delivery problems,

package-not-received.

Add explicit handling for short conversational messages such as "thanks", "done", and "okay".

Improve grounded-response validation so generated claims are checked against available evidence.

Expand human evaluation beyond the current 10-example judge calibration sample.

Create a manually labelled escalation ground-truth set.

Improve retrieval using support-state and intent information.

Reduce unsupported claims about account access, investigations, policies, and operational actions.

------------------------------------------------------------------------------------------------------------------------------------------

22. Summary

This project demonstrates an experimental AI customer-support pipeline using real AmazonHelp conversations from the Customer Support on Twitter dataset.

The current intent classifier achieved:

111 / 200 = 55.50%

on the frozen 200-example manually labelled golden set.

For comparison:

Majority baseline:       18.50%
TF-IDF baseline:         15.00%
Original Qwen 2.5 3B:    38.50%
Few-shot v2:             35.00%
Current Qwen 2.5 7B:     55.50%

The reply-generation pipeline achieved:

20 / 20 deterministic leakage-free replies

The LLM reply-quality judge produced:

5.70 / 12

on 20 examples, with grounding and factual safety identified as weak dimensions in that judge run.

The 10-example judge-human calibration showed:

20% exact agreement
-0.129 quadratic weighted kappa

so the LLM judge is treated as a supporting signal rather than ground truth.

The main conclusion is that a useful support agent requires more than a capable language model. Intent boundaries, historical retrieval, grounded response generation, deterministic safety checks, and escalation policy all need to work together, and each component needs its own evaluation.
