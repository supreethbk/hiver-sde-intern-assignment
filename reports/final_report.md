Hiver Support-Agent Final Report


1. Problem Framing

The goal is to build a small AI customer-support agent using real historical customer-support conversations from the Customer Support on Twitter (TWCS) dataset.

For each incoming customer message, the system:

Classifies the message into a support-intent taxonomy.

Retrieves historical AmazonHelp support evidence.

Drafts a concise response grounded in historical support patterns.

Checks the generated response for obvious leakage.

Decides whether the interaction should be automatically handled or escalated.

The target brand is AmazonHelp.

The project deliberately does not attempt to access real customer accounts, perform refunds/cancellations, modify orders, contact customers, or replace human support workflows. The generated response is a draft.

The main challenge is therefore not just fluent text generation. The system must distinguish closely related support situations, use historical evidence without leaking artifacts, avoid unsupported operational claims, and recognize cases that should be escalated.


2. Dataset and Evaluation Set

The TWCS dataset contains approximately 3 million tweets.

For AmazonHelp, the extracted data contains:

AmazonHelp tweets: 169,840

Reconstructed conversations: 82,556

Direct customer → AmazonHelp pairs: 70,956

Unique customer messages in direct pairs: 66,347

A manually labelled golden set of 200 unique customer messages was created using random sampling with random_state=42.

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

The 200 golden examples were excluded from the historical reference pool to reduce direct evaluation leakage.


3. System Approach

Stage 1 — Intent Classification

The current classifier uses a local Qwen 2.5 7B model through Ollama.

The model receives the customer message and the 21 intent definitions with explicit intent boundaries. Deterministic rules handle several high-confidence patterns before/around the model classification.

The current frozen evaluation result is:

111 / 200 = 55.50% accuracy

Stage 2 — Historical Retrieval

Historical AmazonHelp customer/support pairs are used as evidence for response generation.

Two retrieval approaches were explored:

TF-IDF nearest-neighbour retrieval

Semantic embedding retrieval using nomic-embed-text

A reproducible 3,000-example historical reference subset was embedded for the semantic retrieval experiment.

Stage 3 — Reply Generation

Reply generation uses a local Qwen 2.5 3B model.

The generation prompt uses cleaned historical customer issues as contextual evidence and instructs the model not to:

claim private account access,

claim investigations it cannot perform,

invent refunds, replacements, or cancellations,

invent delivery status or policies,

expose historical URLs, usernames, IDs, or agent identifiers.

A deterministic safety check is applied to generated replies.

Stage 4 — Escalation

A rule-based escalation layer determines whether the interaction is:

AUTO_HANDLE

ESCALATE

Sensitive or operational cases are generally routed toward escalation.

The escalation result is treated as a policy outcome, not an accuracy metric, because the current evaluation data has no human-labelled escalation ground truth.


4. Baselines and Intent Results

The same frozen 200-example golden set was used for classifier comparisons.

Approach

Correct

Accuracy

TF-IDF nearest neighbour

30 / 200

15.00%

Majority-class baseline

37 / 200

18.50%

Few-shot v2

70 / 200

35.00%

Original Qwen 2.5 3B

77 / 200

38.50%

Current Qwen 2.5 7B

111 / 200

55.50%

The current classifier therefore improves over the original Qwen 2.5 3B baseline on the frozen evaluation set.

Interpretation

The result should be treated as an experimental benchmark on 200 manually labelled examples. It is not a production accuracy estimate.

The largest observed errors are concentrated around semantically similar support states rather than completely unrelated topics.


5. Retrieval Results

TF-IDF

The TF-IDF nearest-neighbour approach achieved:

30 / 200 = 15.00% top-1 intent match

This provides a simple lexical-similarity baseline.

Semantic Retrieval

Semantic retrieval uses nomic-embed-text and a 3,000-example historical reference subset.

Qualitative inspection showed semantically meaningful neighbours. However, the intent-aware retrieval experiment achieved:

38.50% top-1 intent match

This did not improve on the original LLM classifier.

The result supports using retrieval as historical context for response generation rather than assuming semantic similarity alone can solve intent classification.


6. Reply Generation Evaluation

Reply generation was evaluated on 20 examples.

Deterministic leakage safety

The leakage evaluation reported:

20 / 20 replies clean

That is:

Leakage-free: 20

Leakage detected: 0

Leakage rate: 0.0%

The implemented checks cover obvious artifacts including:

URLs

Twitter usernames

placeholders

agent identifiers

signatures

order/tracking identifiers

This demonstrates that the evaluated replies did not contain the tested leakage patterns. It does not prove complete factual or security safety.

LLM-as-Judge

The 20 replies were also evaluated with a local LLM judge.

Dimension

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

The judge output suggests that response relevance and style were stronger than grounding and factual-safety dimensions in this run.

However, some numerical scores were inconsistent with the accompanying judge explanations. Therefore, these scores are treated as a supporting evaluation signal, not ground truth.


7. Human Calibration of the LLM Judge

A small independent human calibration sample of 10 replies was compared with the LLM judge.

Results:

Exact agreement: 20%

Quadratic weighted kappa: -0.129

The sample is small, so this is not a production reliability estimate.

The result nevertheless provides evidence that the current judge should not replace human evaluation. A larger human-labelled calibration set would be needed before relying on the judge as a strong quality metric.


8. Escalation Results

The reply-generation evaluation set contains 20 examples.

Decision

Count

Percentage

AUTO_HANDLE

6

30%

ESCALATE

14

70%

Total

20

100%

A separate policy-coverage run over all 200 golden examples produced:

AUTO_HANDLE: 63 / 200 = 31.5%

ESCALATE: 137 / 200 = 68.5%

These numbers describe the behaviour of the implemented escalation policy. They are not escalation accuracy, because no human-labelled escalation ground truth is available.


9. Top Five Failure Modes

1. RESOLUTION_CONFIRMATION vs CASUAL_ENGAGEMENT

Short messages such as "Thanks", "Merci", or similar acknowledgements can be ambiguous.

Observed error pattern:

RESOLUTION_CONFIRMATION → CASUAL_ENGAGEMENT

Hypothesis: Both categories contain short conversational language. The classifier needs stronger state-aware examples distinguishing confirmation of a resolved problem from simple social acknowledgement.

2. CUSTOMER_SERVICE vs ACCOUNT / INSUFFICIENT_CONTEXT / RESOLUTION_CONFIRMATION

General requests for assistance can contain account-related or conversational wording without clearly identifying an account problem.

Hypothesis: CUSTOMER_SERVICE is a broad category that competes with several more specific intents when the customer provides little operational detail.

3. DELIVERY_PROBLEM vs DELIVERY_TRACKING

Both categories frequently contain package, courier, delivery, and shipment vocabulary.

The intended distinction is:

DELIVERY_TRACKING: asking where/status of the package.

DELIVERY_PROBLEM: reporting an unacceptable or failed delivery event.

Hypothesis: The distinction requires reasoning about the delivery state and requested outcome rather than matching delivery keywords.

4. OTHER vs Specific Transactional Intents

Vague or unusual requests can be forced into recognizable categories such as PACKAGE_NOT_RECEIVED, PRODUCT_INFORMATION, or CUSTOMER_SERVICE.

Hypothesis: The model tends to prefer a recognizable specific intent when the message contains a strong keyword, even when the available context is insufficient.

5. DELIVERY_DELAY vs DELIVERY_TRACKING / ORDER_DISPATCH

These categories depend on the shipment timeline:

DELIVERY_DELAY: an expected delivery is late.

DELIVERY_TRACKING: the customer wants current location/status.

ORDER_DISPATCH: the order has not yet been dispatched/shipped.

Hypothesis: These cases require temporal and fulfilment-state reasoning instead of keyword matching.


10. What Is Misleading About My Headline Number?

The current headline classifier result is:

55.50% intent classification accuracy (111/200).

This number should not be interpreted as the success rate of the complete support agent.

It measures one component: intent classification on a frozen 200-example manually labelled golden set.

The complete system also includes:

retrieval,

reply generation,

grounding,

safety validation,

leakage prevention,

escalation.

Each component has separate evaluation signals and limitations.

The reply evaluation illustrates this distinction. All 20 replies passed the deterministic leakage checks, while the LLM judge produced a total score of 5.70/12 and particularly low grounding and factual-safety scores in this run.

The LLM judge also showed only 20% exact agreement with the small human calibration sample.

Therefore the headline should be written precisely as:

55.50% intent classification accuracy on the frozen 200-example manually labelled AmazonHelp golden evaluation set.

It should not be described as overall support-agent accuracy.


11. Main Findings

Finding 1 — The current classifier improved over the original baseline

The original Qwen 2.5 3B classifier achieved 38.50%.

The current Qwen 2.5 7B classifier achieved 55.50% on the same frozen golden set.

This is an experimental improvement on this evaluation set.

Finding 2 — Simple baselines remain useful for context

The TF-IDF baseline achieved 15.00%, while the majority-class baseline achieved 18.50%.

These baselines establish reference points for interpreting the LLM results.

Finding 3 — Semantic retrieval is useful but insufficient

Semantic retrieval produced meaningful historical neighbours, but the intent-aware retrieval experiment reached 38.50% top-1 intent match and did not improve on the original LLM baseline.

Retrieval is therefore better treated as contextual evidence than as a standalone solution to classification.

Finding 4 — Leakage safety and response quality are different dimensions

The deterministic leakage evaluator found 20/20 clean replies.

The LLM judge nevertheless identified weaknesses in grounding and factual safety.

This shows that passing leakage checks does not guarantee a high-quality or fully grounded support response.

Finding 5 — Automated judging needs human calibration

The current judge produced 5.70/12, while the 10-example human calibration showed 20% exact agreement and -0.129 quadratic weighted kappa.

The current judge should therefore remain a supporting metric.

Finding 6 — Ambiguous support states are the main classification challenge

Recurring confusion involves:

CUSTOMER_SERVICE vs RESOLUTION_CONFIRMATION

DELIVERY_PROBLEM vs DELIVERY_TRACKING

DELIVERY_DELAY vs ORDER_DISPATCH

OTHER vs specific transactional intents

CUSTOMER_SERVICE vs ACCOUNT / INSUFFICIENT_CONTEXT

These errors indicate that conversational and operational state is more important than isolated keywords for several categories.


12. Limitations

The golden evaluation set contains only 200 manually labelled examples.

Intent frequencies are uneven, with some intents having very few examples.

The semantic reference subset contains 3,000 examples.

The reply-generation evaluation contains only 20 examples.

The escalation evaluation has no human-labelled ground truth.

Escalation percentages therefore represent policy behaviour rather than accuracy.

The deterministic leakage checker only covers the implemented patterns.

The LLM judge showed weak agreement with the small human calibration sample.

Some judge scores were inconsistent with their textual explanations.

Local model behaviour can vary with model versions and runtime configuration.

Generated replies are drafts and do not execute real customer-support actions.

The results should therefore be treated as experimental evidence rather than production-level benchmarks.


13. One-Week Next Steps

Increase manual labels for the most confused intent pairs.

Improve state-aware classification for delay, tracking, dispatch, delivery problems, and package-not-received cases.

Add stronger handling for short acknowledgements such as "thanks", "done", and "okay".

Add stronger grounding validation for unsupported claims about actions, investigations, account access, policies, and order status.

Create a manually labelled escalation ground-truth set.

Expand reply-quality evaluation beyond 20 examples.

Recalibrate the LLM judge using a larger human-labelled sample.

Improve retrieval using intent-aware and support-state-aware representations.


14. Conclusion

The project demonstrates an experimental AI customer-support pipeline built from real AmazonHelp support conversations.

The current classifier achieved:

111 / 200 = 55.50%

on the frozen manually labelled golden set.

The system also demonstrates:

historical semantic retrieval,

grounded response generation,

deterministic leakage detection,

rule-based escalation,

automated reply-quality evaluation,

and human calibration of the LLM judge.

The main lesson is that building a useful support agent requires more than generating fluent text. Intent boundaries, historical evidence, grounding, safety validation, and escalation policy each introduce separate failure modes and therefore need separate evaluation.
