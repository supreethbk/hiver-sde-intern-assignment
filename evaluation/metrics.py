"""Evaluation metrics."""
"""Shared evaluation metrics."""

from sklearn.metrics import accuracy_score, cohen_kappa_score


def classification_accuracy(y_true, y_pred):
    """Return classification accuracy."""
    return accuracy_score(y_true, y_pred)


def exact_agreement(y_true, y_pred):
    """Return the proportion of exact matches."""
    if len(y_true) == 0:
        return 0.0

    matches = sum(
        true == pred
        for true, pred in zip(y_true, y_pred)
    )

    return matches / len(y_true)


def quadratic_weighted_kappa(y_true, y_pred):
    """Return quadratic weighted Cohen's kappa."""
    return cohen_kappa_score(
        y_true,
        y_pred,
        weights="quadratic",
    )


def escalation_rate(decisions):
    """Return the proportion of examples marked for escalation."""
    if len(decisions) == 0:
        return 0.0

    return sum(
        decision == "ESCALATE"
        for decision in decisions
    ) / len(decisions)