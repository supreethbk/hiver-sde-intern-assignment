# Final report
# Hiver Support-Agent Final Report

## 1. Problem Framing

The goal is to build a small AI customer-support agent using real historical customer-support conversations.

For each incoming customer message, the system performs three tasks:

1. Classifies the message into an intent taxonomy derived from the historical data.
2. Drafts a support reply grounded in historical AmazonHelp resolution patterns.
3. Decides whether the reply can be automatically handled or should be escalated.

The target brand selected from the Customer Support on Twitter (TWCS) dataset was AmazonHelp.

The main challenge is not simply generating fluent replies. The system must distinguish similar support situations, use historical evidence without leaking private information, avoid unsupported claims, and recognize when a human/support workflow is required.

---

## 2. Dataset and Evaluation Set

The TWCS dataset contains approximately 3 million tweets.

For AmazonHelp:

- AmazonHelp tweets: 169,840
- Reconstructed conversations: 82,556
- Direct customer → AmazonHelp pairs: 70,956
- Unique customer messages in direct pairs: 66,347

A manually labelled golden set of 200 unique customer messages was created using random sampling (`random_state=42`).

The golden set contains 21 intents:

- ACCOUNT
- CASUAL_ENGAGEMENT
- CUSTOMER_SERVICE
- DELIVERY_DELAY
- DELIVERY_INSTRUCTIONS
- DELIVERY_PROBLEM
- DELIVERY_TRACKING
- DEVICE
- FEEDBACK
- INSUFFICIENT_CONTEXT
- ORDER_CANCELLATION
- ORDER_DISPATCH
- OTHER
- PACKAGE_NOT_RECEIVED
- PAYMENT
- PRODUCT_INFORMATION
- REFUND
- RESOLUTION_CONFIRMATION
- RETURN
- SELLER
- WEBSITE_OR_APP

The golden examples were excluded from the historical retrieval pool to reduce evaluation leakage.

---

## 3. System Approach

The system consists of four main stages:

### Stage 1 — Intent Classification

A local Qwen 2.5 3B model is given the customer message and the 21 intent definitions.

The model must return exactly one intent.

Several prompt and few-shot variations were tested. The original baseline performed better than the attempted prompt and few-shot variants.

### Stage 2 — Historical Retrieval

Historical AmazonHelp customer/support pairs are used as potential evidence for response generation.

Two retrieval approaches were evaluated:

- TF-IDF nearest-neighbour retrieval
- Semantic embedding retrieval using `nomic-embed-text`

A 3,000-example historical reference subset was embedded locally for the semantic retrieval experiment.

### Stage 3 — Reply Generation

The retrieved historical support response is supplied as evidence to the local language model.

The generation prompt instructs the model to:

- stay grounded in the historical evidence,
- avoid inventing policies or account information,
- avoid claiming private account access,
- avoid unsupported refunds, replacements, cancellations, investigations, or status checks,
- avoid leaking historical URLs, IDs, usernames, dates, carriers, sellers, or agent identifiers.

A deterministic safety check is applied to generated replies.

### Stage 4 — Escalation

A rule-based escalation layer determines whether an interaction should be automatically handled or escalated.

Sensitive intents such as account, payment, refund, cancellation, package-not-received, delivery problems, seller issues, and website/app problems are routed toward escalation.

The purpose is to prevent the system from automatically handling cases that may require private account access or a support action.

---

## 4. Baselines and Results

### Baseline 1 — LLM Intent Classification

The local Qwen 2.5 3B classifier was evaluated on all 200 golden examples.

**Result: 77/200 = 38.50% accuracy**

This provides the primary intent-classification baseline.

The largest observed confusions included:

- CUSTOMER_SERVICE → RESOLUTION_CONFIRMATION: 8
- OTHER → ORDER_CANCELLATION: 6
- CUSTOMER_SERVICE → ORDER_DISPATCH: 5
- DELIVERY_DELAY → ORDER_DISPATCH: 4
- DELIVERY_PROBLEM → DELIVERY_TRACKING: 4

These errors show that the difficult cases are often semantically close support situations rather than completely unrelated intents.

### Baseline 2 — TF-IDF Similarity

A TF-IDF nearest-neighbour baseline was evaluated using the manually labelled reference examples.

**Result: 30/200 = 15.00%**

This was substantially worse than the LLM classifier.

The experiment also showed that surface word similarity can be misleading. Messages with highly similar wording can represent different support intents.

### Prompt/Few-Shot Experiments

A stricter delivery-focused prompt was tested.

**Result: 29.00%**

A few-shot classifier using up to three examples per intent was also tested.

**Result: 35.00%**

Neither improved on the 38.50% baseline.

This was retained as an important negative result rather than selecting a worse approach simply because it was more complex.

### Semantic Retrieval

Semantic retrieval using `nomic-embed-text` was tested on the 3,000-example reference subset.

The semantic retriever produced stronger semantic matches than the TF-IDF approach in qualitative inspection, but intent-aware retrieval did not improve the final classification result.

The intent-aware retrieval experiment achieved:

**38.50% top-1 intent match**

Therefore, semantic similarity alone was not treated as evidence that the retrieval approach solved the intent problem.

---

## 5. Reply Generation Evaluation

Reply generation was evaluated on 20 examples.

### Deterministic leakage safety

**20/20 generated replies passed the implemented leakage checks.**

The checks looked for obvious leakage such as:

- URLs
- Twitter usernames
- placeholders
- agent identifiers
- signatures
- order/tracking identifiers

This demonstrates that the generation pipeline can remove obvious historical-data artifacts.

---

### LLM-as-Judge

The 20 replies were evaluated using an LLM judge across six dimensions.

| Dimension | Score |
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

This highlights an important difference between fluent response generation and trustworthy support behaviour: a reply can sound professional while still being insufficiently grounded in historical resolution patterns.

The LLM judge is therefore treated as a supporting evaluation signal rather than definitive evidence of reply quality.

---

## 6. Escalation Results

The current escalation evaluation was run on the 20 examples used for reply-generation evaluation.

| Decision | Count | Percentage |
|---|---:|---:|
| AUTO_HANDLE | 6 | 30% |
| ESCALATE | 14 | 70% |
| **Total** | **20** | **100%** |

The policy consistently escalated sensitive categories such as:

- ACCOUNT
- PAYMENT
- REFUND
- ORDER_CANCELLATION
- PACKAGE_NOT_RECEIVED
- DELIVERY_PROBLEM
- DEVICE
- SELLER
- WEBSITE_OR_APP
- CUSTOMER_SERVICE
- OTHER
- INSUFFICIENT_CONTEXT
- ORDER_DISPATCH

It automatically handled lower-risk categories such as:

- CASUAL_ENGAGEMENT
- DELIVERY_INSTRUCTIONS
- DELIVERY_TRACKING
- FEEDBACK
- PRODUCT_INFORMATION

Some conditional categories require further refinement because a safe decision can depend on the content of the generated reply, not only the predicted intent.

The **70% escalation rate is a policy outcome, not an accuracy metric**, because the current evaluation set does not contain human-labelled escalation ground truth.

A separate 200-example policy-coverage run is retained in `results/escalation_results_200.csv`.

---

## 7. Top Five Failure Modes

### Failure Mode 1 — Customer Service vs Resolution Confirmation

**Observed confusion:** CUSTOMER_SERVICE → RESOLUTION_CONFIRMATION

The model often interprets resolution-related language as confirmation that a support issue has already been resolved.

**Hypothesis:** The two intents contain overlapping support language and conversational context. The model needs stronger distinction between a customer requesting help and a customer confirming an already completed resolution.

---

### Failure Mode 2 — Other vs Transactional Intents

**Observed confusion:** OTHER → ORDER_CANCELLATION

Generic or unusual support requests were sometimes forced into a specific transactional intent.

**Hypothesis:** The classifier appears biased toward recognizable transactional categories when the message does not clearly fit one of the defined intents.

---

### Failure Mode 3 — Customer Service vs Order Dispatch

**Observed confusion:** CUSTOMER_SERVICE → ORDER_DISPATCH

Messages mentioning an order or shipment were sometimes classified as dispatch problems even when the main purpose was requesting general assistance.

**Hypothesis:** Strong order-related keywords can dominate the broader customer-service intent.

---

### Failure Mode 4 — Delivery Delay vs Order Dispatch

**Observed confusion:** DELIVERY_DELAY → ORDER_DISPATCH

The classifier sometimes failed to distinguish a package that has already entered the delivery process but is late from an order that has not yet been dispatched.

**Hypothesis:** Both intents describe order fulfilment delays and require temporal/state reasoning rather than simple keyword matching.

---

### Failure Mode 5 — Delivery Problem vs Delivery Tracking

**Observed confusion:** DELIVERY_PROBLEM → DELIVERY_TRACKING

Messages about where a package is and messages about an incorrectly delivered package can contain similar delivery vocabulary.

**Hypothesis:** The distinction depends on the state of delivery: tracking asks for current location/status, while delivery-problem cases describe an unacceptable delivery event.

---

## 8. What Is Misleading About My Headline Number?

The most tempting headline number is the **38.50% intent-classification accuracy**.

However, this number should not be interpreted as meaning that the complete support agent is 38.5% effective.

First, intent accuracy is only one component of the system. Reply generation, grounding, leakage prevention, and escalation are separate problems.

Second, the 200-example golden set is relatively small and contains highly uneven intent frequencies. Several intents have very few examples.

Third, classification accuracy does not measure whether a generated response is useful or safe.

The reply-quality evaluation demonstrates this clearly. Across 20 examples, the generated replies achieved:

- Relevance: **1.65 / 2**
- Usefulness: **1.55 / 2**
- Grounding: **0.65 / 2**
- Factual safety: **1.90 / 2**
- Leakage safety: **1.90 / 2**
- Style: **1.40 / 2**
- Total: **9.05 / 12**

Grounding was the weakest dimension, showing that a fluent response is not necessarily well grounded in historical support patterns.

The LLM judge also had weak agreement with the small human calibration sample: **20% exact agreement** and **−0.129 quadratic weighted kappa**. Therefore, the judge scores should be treated as supporting evidence rather than definitive measurements of reply quality.

The escalation evaluation also does not provide an accuracy measure because it has no human-labelled escalation ground truth.

Therefore, the **38.50% number should be presented as a baseline classification result on the manually labelled golden set, not as an overall support-agent success rate**.

---

## 9. Main Findings

The experiments produced five important findings.

### 1. More complexity did not automatically improve classification

The baseline LLM classifier achieved **38.50%**, while the tested prompt and few-shot variants achieved **29.00%** and **35.00%**.

This shows that additional instructions and examples did not automatically improve classification.

### 2. Semantic retrieval is useful but insufficient

Embedding-based retrieval produced semantically relevant historical examples, but intent-aware retrieval achieved **38.50% top-1 intent match**, tying rather than improving on the baseline classifier.

Retrieval quality therefore cannot be assumed to solve the classification problem.

### 3. Fluent replies can still be insufficiently grounded

The generated replies passed the deterministic leakage checks on **20/20 examples**.

However, the LLM judge gave grounding the lowest score at **0.65/2**.

This is an important distinction: a reply can sound professional and relevant while still being insufficiently grounded in historical support patterns.

### 4. LLM-as-judge evaluation has limitations

The LLM judge achieved an average score of **9.05/12** across the 20 evaluated replies.

However, comparison with the small human calibration sample produced only **20% exact agreement** and a **−0.129 quadratic weighted kappa**.

Therefore, the LLM judge should be treated as a supporting evaluation signal rather than a replacement for human evaluation.

### 5. The main challenge is distinguishing ambiguous support states

The largest classification errors involved closely related intents such as:

- CUSTOMER_SERVICE vs RESOLUTION_CONFIRMATION
- CUSTOMER_SERVICE vs ORDER_DISPATCH
- DELIVERY_DELAY vs ORDER_DISPATCH
- DELIVERY_PROBLEM vs DELIVERY_TRACKING
- OTHER vs ORDER_CANCELLATION

These failures suggest that the main difficulty is distinguishing similar support situations and conversational states rather than simply recognizing keywords.

---

## 10. Next Week

The next iteration would focus on:

1. Improving the intent taxonomy boundaries using more manually labelled examples for the most confused intents.

2. Increasing the golden-set size while preserving a clear sampling and labelling strategy.

3. Improving retrieval with intent-aware and support-state-aware representations rather than relying only on semantic similarity.

4. Adding stronger grounding checks that detect unsupported claims about actions, investigations, account access, or order status.

5. Creating a manually labelled escalation ground truth and measuring escalation precision and recall.

6. Expanding reply-quality evaluation beyond the initial 20 examples.

7. Recalibrating the LLM judge against a larger human-labelled sample before relying on it as a stronger evaluation signal.

8. Testing whether structured support-state information can better distinguish delivery tracking, delivery delay, dispatch, delivery problems, and package-not-received cases.