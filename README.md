
## 1. `README.md`

# Hiver AI Support Agent — AmazonHelp

A small AI customer-support agent built from real Customer Support on Twitter (TWCS) conversations.

The system:

1. Classifies a customer message into a support intent.
2. Retrieves historical AmazonHelp support evidence.
3. Generates a grounded draft reply.
4. Checks the reply for obvious data leakage.
5. Decides whether the interaction should be automatically handled or escalated.

---

## 1. Complete Pipeline

Customer Message
       |
       v
+-------------------------+
| Intent Classification   |
| Qwen 2.5 3B             |
+-----------+-------------+
            |
            v
+-------------------------+
| Predicted Intent        |
| 21-intent taxonomy      |
+-----------+-------------+
            |
            v
+-------------------------+
| Historical Retrieval    |
| nomic-embed-text        |
+-----------+-------------+
            |
            v
+-------------------------+
| AmazonHelp Historical   |
| Support Evidence        |
+-----------+-------------+
            |
            v
+-------------------------+
| Reply Generation        |
| Qwen 2.5 3B             |
+-----------+-------------+
            |
            v
+-------------------------+
| Safety / Leakage Check  |
| Deterministic Rules     |
+-----------+-------------+
            |
            v
+-------------------------+
| Escalation Decision     |
| Rule-based Engine       |
+-----------+-------------+
            |
       +----+----+
       |         |
       v         v
 AUTO_HANDLE  ESCALATE

The overall architecture is hybrid:

Qwen 2.5 3B
      +
Semantic Retrieval
      +
Deterministic Safety Rules
      +
Rule-based Escalation

---

## 2. Dataset

Source:

Customer Support on Twitter (TWCS) dataset.
Kaggle source: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

Target brand:

**AmazonHelp**

AmazonHelp data extracted:

* 169,840 AmazonHelp tweets
* 82,556 reconstructed conversations
* 70,956 direct customer → AmazonHelp pairs
* 66,347 unique customer messages

Main data files are stored under:

results/

---

## 3. Golden Evaluation Set

A manually labelled golden set of 200 unique customer messages was created.

The taxonomy contains 21 intents:

```text
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

Golden examples were removed from the historical reference pool to reduce evaluation leakage.

Golden set:

results/amazonhelp_golden_labeled.csv

---

## 4. Models

### Intent Classification

Qwen 2.5 3B

The model receives:

* customer message
* 21 intent definitions
* classification instructions

It must return one intent.

### Semantic Retrieval

nomic-embed-text

A 3,000-example historical reference subset was embedded for the semantic retrieval experiment.

### Reply Generation

Qwen 2.5 3B

The model receives historical AmazonHelp support evidence and generates a draft reply.

### Safety

Deterministic checks detect obvious leakage such as:

* URLs
* Twitter usernames
* placeholders
* agent identifiers
* signatures
* order/tracking identifiers

### Escalation

A rule-based engine decides:

AUTO_HANDLE

or:

ESCALATE

---

## 5. Intent Classification Results

### Baseline 1 — LLM Classifier

Qwen 2.5 3B evaluated on the 200-example golden set:

77 / 200 = 38.50%

### Baseline 2 — TF-IDF Similarity

TF-IDF nearest-neighbour baseline:


30 / 200 = 15.00%

### Prompt Experiment

Improved prompt experiment:

29.00%

### Few-Shot v2

Few-shot classifier:

35.00%

The original LLM baseline remained the best tested classifier.

---

## 6. Retrieval Results

Two retrieval approaches were tested.

### TF-IDF

The TF-IDF approach relied mainly on lexical overlap.

Result:

15.00% top-1 intent match

### Semantic Retrieval

Semantic retrieval used:

nomic-embed-text

The embedding retriever produced more semantically meaningful neighbours during qualitative inspection.

However, intent-aware retrieval did not improve the classification result.

Result:

38.50% top-1 intent match

Therefore semantic similarity alone was not treated as a solution to intent classification.

---

## 7. Reply Generation

Historical AmazonHelp support responses are used as grounding evidence.

The generation prompt instructs the model not to:

* claim private account access
* claim investigations it cannot perform
* invent refunds
* invent replacements
* invent cancellations
* invent delivery status
* invent policies
* expose historical IDs
* expose URLs
* expose usernames
* expose agent information

Generated replies are passed through a deterministic safety check.

---

## 8. Reply Safety Results

20 generated replies were evaluated.

Deterministic leakage check:

20 / 20 PASSED

The checks detected:

* URLs
* usernames
* placeholders
* agent identifiers
* signatures
* order/tracking identifiers

No obvious leakage was detected in the evaluated replies.

---

## 9. LLM Reply Quality Judge

20 generated replies were evaluated using a local LLM judge.

The judge scores six dimensions from 0 to 2:

| Metric | Score |
|---|---:|
| Relevance | 1.65 / 2 |
| Usefulness | 1.55 / 2 |
| Grounding | 0.65 / 2 |
| Factual safety | 1.90 / 2 |
| Leakage safety | 1.90 / 2 |
| Style | 1.40 / 2 |
| **Total** | **9.05 / 12** |

All 20 judgments were successfully parsed.

The results show that the generated replies were generally relevant and reasonably useful, while grounding remained the weakest dimension.

The deterministic leakage checks provide a separate safety signal: all 20 evaluated replies passed the implemented leakage checks.

### Judge-human calibration

A small independent human calibration sample of 10 replies was scored using the same 1–5 overall-quality scale used by the LLM judge.

Results:

- Exact agreement: **20%**
- Quadratic weighted kappa: **−0.129**

This indicates weak agreement on the calibration sample. Therefore, the LLM judge is treated as a supporting evaluation signal rather than a replacement for human evaluation.

The calibration sample is small, so the agreement statistic should not be interpreted as a production-level estimate of judge reliability.

---

## 10. Escalation Decision

The escalation layer uses the predicted intent and generated reply to decide whether the request can be auto-handled or should be escalated to a human agent.

The current evaluation harness runs on the 20 examples used for reply-generation evaluation.

Results:

| Decision | Count |
|---|---:|
| AUTO_HANDLE | 6 |
| ESCALATE | 14 |
| **Total** | **20** |

**Escalation rate: 70%**

The policy is intentionally conservative for intents involving account access, payments, refunds, cancellations, missing packages, delivery problems, devices, sellers, website/app problems, customer-service requests, insufficient context, and order-dispatch issues.

The escalation result is a **policy outcome, not an accuracy metric**, because the current evaluation set does not contain human-labelled escalation ground truth.

A separate 200-example policy-coverage run is available in:

`results/escalation_results_200.csv`

The current reproducible evaluation result is the 20-example run:

`results/escalation_results_reply_20.csv`

---

## 11. Escalation Results

Evaluation on all 200 golden examples:


AUTO_HANDLE: 63 / 200 = 31.5%

ESCALATE: 137 / 200 = 68.5%


The escalation policy is intentionally conservative.

---

## 12. Top Five Failure Modes

### 1. CUSTOMER_SERVICE → RESOLUTION_CONFIRMATION

The classifier sometimes interprets support-related language as confirmation that a problem has already been resolved.

Hypothesis:

The two intents contain overlapping conversational language. More explicit state-based examples may help.

---

### 2. OTHER → ORDER_CANCELLATION

Generic or unusual support requests were sometimes forced into transactional intents.

Hypothesis:

The classifier appears biased toward recognizable transactional categories when the message does not clearly fit another intent.

---

### 3. CUSTOMER_SERVICE → ORDER_DISPATCH

Messages mentioning orders or shipments were sometimes classified as dispatch problems even when the customer was primarily asking for support.

Hypothesis:

Strong order-related vocabulary can dominate the broader customer-service intent.

---

### 4. DELIVERY_DELAY → ORDER_DISPATCH

The classifier sometimes confused late delivery with an order that has not yet been dispatched.

Hypothesis:

These intents require reasoning about the order's fulfilment state and timeline rather than keyword matching.

---

### 5. DELIVERY_PROBLEM → DELIVERY_TRACKING

Delivery problems and tracking requests contain similar vocabulary.

Hypothesis:

The distinction depends on delivery state:

* Tracking = where/status of package
* Delivery problem = an unacceptable delivery event

---

## 13. What Is Misleading About the Headline Number?

The main headline number is:


38.50% intent classification accuracy


This number should not be interpreted as the success rate of the complete support agent.

It measures only intent classification on the 200-example golden set.

The complete system also contains:

* retrieval
* reply generation
* grounding
* safety
* leakage prevention
* escalation

The reply-quality evaluation demonstrates why this distinction matters.

The generated replies achieved strong relevance and style but weak grounding and factual safety.

Therefore:

38.50%

should be described as:

> Intent classification accuracy on the manually labelled 200-example golden evaluation set.

It should not be presented as overall customer-support-agent accuracy.

---

## 14. Key Findings

### Finding 1

Adding complexity did not automatically improve classification.

Results:

Baseline LLM       38.50%
Prompt experiment  29.00%
Few-shot v2        35.00%

The original baseline was better.

---

### Finding 2

Semantic retrieval is useful but insufficient.

Embedding retrieval produced semantically relevant historical examples, but intent-aware retrieval did not improve the classification result.

The intent-aware retrieval experiment achieved **38.50% top-1 intent match**, tying the baseline classifier rather than improving it.

---

### Finding 3

Fluent replies can still be insufficiently grounded.

The deterministic leakage checks passed all 20 evaluated replies.

The LLM judge gave the following results:

| Metric | Score |
|---|---:|
| Relevance | 1.65 / 2 |
| Usefulness | 1.55 / 2 |
| Grounding | 0.65 / 2 |
| Factual safety | 1.90 / 2 |
| Leakage safety | 1.90 / 2 |
| Style | 1.40 / 2 |
| **Total** | **9.05 / 12** |

Grounding was the weakest dimension.

Therefore:

> A professional-sounding support reply is not necessarily a well-grounded support reply.

---

### Finding 4

The LLM judge should not be treated as a replacement for human evaluation.

On a small 10-example human calibration sample:

- Exact agreement: **20%**
- Quadratic weighted kappa: **−0.129**

This indicates weak agreement between the LLM judge and human scores. The judge is therefore treated as a supporting evaluation signal rather than definitive evidence of reply quality.

---

### Finding 5

The main classification failures came from ambiguous support categories.

The most important observed failure modes were:

1. Customer-service requests confused with resolution confirmations.
2. Vague/other requests pulled toward transactional intents.
3. Customer-service requests confused with order-dispatch issues.
4. Delivery-delay requests confused with order-dispatch issues.
5. Delivery problems confused with delivery-tracking requests.

These failures suggest that the main challenge is not simply model size, but distinguishing closely related support situations from short and ambiguous customer messages.
---

## 15. Limitations

The golden evaluation set contains only 200 manually labelled examples.

Intent frequencies are uneven, with some intents having very few examples.

The semantic reference set contains 3,000 examples and uses weakly transferred labels for the intent-aware retrieval experiment rather than full manual annotation.

The reply-quality evaluation currently contains 20 examples.

The escalation evaluation does not contain human-labelled escalation ground truth, so its 70% escalation rate is a policy outcome rather than an accuracy measure.

The LLM judge also showed weak agreement with the small human calibration sample: 20% exact agreement and a quadratic weighted kappa of -0.129. Therefore, the LLM judge should be treated as a supporting evaluation signal rather than a replacement for human evaluation.

Therefore, the reported metrics should be treated as experimental evaluation results rather than production-level benchmarks.

---

## 16. Reproduction

### Install dependencies

pip install -r requirements.txt

### Pull local models

ollama pull qwen2.5:3b
ollama pull nomic-embed-text

### Run intent evaluation

python evaluation/evaluate_intent.py

### Run trivial majority-class baseline

python baselines/trivial_baseline.py

### Run TF-IDF baseline

python baselines/simple_baseline.py

### Run reply generation evaluation

python test/evaluate_reply_generation.py

### Run reply quality evaluation

python evaluation/evaluate_replies.py

### Run escalation evaluation

python evaluation/evaluate_escalation.py

### Run human-vs-LLM judge agreement

python evaluation/judge_agreement.py

### Results are written to:

results/

---

## 17. Important Result Files

results/amazonhelp_golden_labeled.csv
results/classification_results_200.csv
results/baseline2_similarity_results.csv
results/semantic_retrieval_results_200.csv
results/reply_generation_eval_20.csv
results/reply_leakage_results_20.csv
results/reply_quality_judge_20.csv
results/escalation_results_200.csv
results/escalation_results_reply_20.csv


---

## 18. Project Structure

.
├── baselines/
├── data/
├── data_pack/
├── evaluation/
├── notebooks/
├── reports/
│   └── final_report.md
├── results/
├── src/
├── step/
├── test/
├── decision_log.md
├── requirements.txt
├── run.py
└── README.md


---

## 19. Decision Log

The project contains 15 non-obvious implementation and evaluation decisions.

See:

decision_log.md

The decision log covers:

* dataset selection
* conversation reconstruction
* golden-set construction
* intent taxonomy
* evaluation/reference separation
* retrieval
* weak supervision
* reply grounding
* safety
* escalation
* evaluation methodology

---

## 20. Next Improvements

The next iteration would focus on:

1. Increasing manually labelled examples for confused intent pairs.
2. Improving delivery-state classification.
3. Adding stronger grounded-response validation.
4. Creating a manually labelled escalation ground truth.
5. Expanding reply-quality evaluation.
6. Improving retrieval using support-state information.
7. Reducing unsupported claims about account access and support actions.

---

## 21. Summary

This project demonstrates an experimental AI customer-support pipeline using real AmazonHelp conversations.

The strongest tested intent-classification baseline achieved:

38.50% accuracy

The TF-IDF baseline achieved:

15.00%

The reply generator achieved:

20/20 deterministic leakage checks passed

while the LLM judge highlighted grounding and factual-safety weaknesses.

The main conclusion is that building a useful support agent requires more than a capable language model. Intent boundaries, historical retrieval, grounding, safety validation, and escalation decisions all need to work together.


---