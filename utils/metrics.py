import torch


class ConfusionMeter:
    """Accumulate global pixel counts before computing the five paper metrics."""

    def __init__(self, threshold=0.5):
        self.threshold = float(threshold)
        self.tn = self.fp = self.fn = self.tp = 0

    @torch.no_grad()
    def update(self, probabilities, targets):
        prediction = probabilities >= self.threshold
        truth = targets >= 0.5
        self.tn += int(((~prediction) & (~truth)).sum().item())
        self.fp += int((prediction & (~truth)).sum().item())
        self.fn += int(((~prediction) & truth).sum().item())
        self.tp += int((prediction & truth).sum().item())

    @staticmethod
    def _ratio(numerator, denominator):
        return float(numerator) / float(denominator) if denominator else 0.0

    def compute(self):
        tn, fp, fn, tp = self.tn, self.fp, self.fn, self.tp
        return {
            "DSC": self._ratio(2 * tp, 2 * tp + fp + fn),
            "mIoU": self._ratio(tp, tp + fp + fn),
            "Sensitivity": self._ratio(tp, tp + fn),
            "Specificity": self._ratio(tn, tn + fp),
            "Accuracy": self._ratio(tn + tp, tn + fp + fn + tp),
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
        }
