# Decision log

## 1. Selected AmazonHelp as the target brand
The TWCS dataset contains conversations from many brands. AmazonHelp was selected because it has a large number of customer-support interactions, providing enough data for intent discovery, retrieval, reply generation, and evaluation.

## 2. Reconstructed conversations instead of treating tweets independently
Tweets were linked using the available response relationships so that customer messages could be paired with the corresponding AmazonHelp support responses. This preserves the customer-to-support context needed for the task.

## 3. Used direct customer → AmazonHelp response pairs for the main analysis
Only direct customer-to-support pairs were used for the main supervised analysis. This reduces ambiguity compared with using unrelated tweets from the same conversation.

## 4. Created a manually labelled 200-example golden set
A random sample of 200 unique customer messages was manually reviewed and labelled. The golden set is used for evaluation rather than training, so the reported results measure performance on examples that were not used as model references.

## 5. Kept a relatively fine-grained intent taxonomy
The taxonomy contains 21 intents instead of collapsing everything into broad categories such as "delivery" or "order". This preserves meaningful differences such as delivery delay, tracking, delivery problems, package not received, and dispatch.

## 6. Kept low-frequency intents in the golden taxonomy
Some intents contain fewer than five examples in the 200-example golden set. They were not removed solely because of low frequency because the golden set is an evaluation set, not a training dataset.

## 7. Removed golden examples from the historical reference pool
The 200 golden customer messages were excluded from the historical retrieval pool. This prevents direct evaluation examples from being retrieved as historical evidence.

## 8. Used historical support responses as grounding evidence
Reply generation uses previous AmazonHelp support responses as evidence rather than generating replies entirely from general language-model knowledge.

## 9. Used a 3,000-example semantic reference subset
The full historical pool was too large for the initial local embedding experiment. A 3,000-example subset was therefore embedded to make semantic retrieval practical while keeping the experiment reproducible.

## 10. Treated transferred reference labels as weak supervision
Labels transferred from the 100 manually labelled reference examples to the 3,000-example reference set were treated as weak labels, not ground truth. This distinction prevents the retrieval experiment from being presented as fully manually labelled data.

## 11. Compared semantic retrieval against TF-IDF similarity
Two retrieval approaches were evaluated: traditional TF-IDF nearest-neighbour similarity and embedding-based semantic retrieval. This provides a baseline comparison instead of assuming that embeddings automatically improve retrieval.

## 12. Added output-safety checks to reply generation
Generated replies are checked for URLs, Twitter usernames, placeholders, agent identifiers, signatures, and order/tracking identifiers. The purpose is to prevent historical customer-support artifacts from leaking into generated responses.

## 13. Removed private/action claims from generated replies
The reply-generation prompt was constrained against claiming private account access, investigations, refunds, replacements, cancellations, or other actions that the system cannot actually perform.

## 14. Separated automatic handling from escalation
A separate escalation layer was introduced after intent classification and reply generation. Sensitive intents such as account, payment, refund, cancellation, and package-not-received issues are escalated rather than automatically handled.

## 15. Evaluated both deterministic safety and LLM-judged reply quality
Reply quality was evaluated using an LLM judge across relevance, usefulness, grounding, factual safety, leakage safety, and style, while deterministic checks were retained for concrete leakage detection. This separates subjective response quality from objective safety checks.