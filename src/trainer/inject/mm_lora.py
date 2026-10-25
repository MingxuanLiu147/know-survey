"""Image/text LoRA injection using positive answer cross-entropy."""

import torch
from torch.utils.data import DataLoader

from trainer.unlearn.mm_base import MMUnlearnBase


class MMLoRA(MMUnlearnBase):
    """Reuse the multimodal optimizer loop with training and retention probes.

    ``forget_loader`` is the existing loop's internal primary-loader argument;
    for injection it contains only the public ``train`` split.
    """

    def __init__(self, *args, eval_max_batches: int = 8, **kwargs):
        if int(eval_max_batches) < 1:
            raise ValueError("eval_max_batches must be positive")
        self.eval_max_batches = int(eval_max_batches)
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, batch):
        """Forward all processor fields, including image grids and pixel values."""
        return model(**batch).loss

    @torch.no_grad()
    def _probe(self, source):
        if source is None:
            return None, []
        # Stable order is necessary to compare each prediction before and after
        # training, even though the training loader itself is shuffled.
        loader = DataLoader(
            source.dataset,
            batch_size=source.batch_size or 1,
            shuffle=False,
            collate_fn=source.collate_fn,
        )
        self.model.eval()
        total_loss = 0.0
        correct = tokens = 0
        rows, predictions = [], []
        for batch_index, batch in enumerate(loader):
            if batch_index >= self.eval_max_batches:
                break
            batch = {
                key: value.to(self.accelerator.device)
                if torch.is_tensor(value)
                else value
                for key, value in batch.items()
            }
            outputs = self.model(**batch)
            labels = batch["labels"][:, 1:]
            predicted = outputs.logits[:, :-1].argmax(dim=-1)
            mask = labels.ne(-100)
            count = int(mask.sum())
            if count == 0:
                raise ValueError("Injection probe has no supervised answer tokens")
            total_loss += float(outputs.loss) * count
            tokens += count
            correct += int((predicted.eq(labels) & mask).sum())
            for row in range(labels.shape[0]):
                row_mask = mask[row]
                row_count = int(row_mask.sum())
                if not row_count:
                    raise ValueError("Injection probe contains an empty answer")
                row_correct = int(
                    predicted[row, row_mask].eq(labels[row, row_mask]).sum()
                )
                rows.append(
                    {
                        "index": len(rows),
                        "tokens": row_count,
                        "token_accuracy": row_correct / row_count,
                    }
                )
                predictions.append(predicted[row, row_mask].cpu().tolist())
        if not tokens:
            raise ValueError("Injection probe loader is empty")
        return {
            "token_accuracy": correct / tokens,
            "loss": total_loss / tokens,
            "samples": len(rows),
            "tokens": tokens,
            "per_example": rows,
        }, predictions

    def train(self):
        before_train, _ = self._probe(self.forget_loader)
        before_retain, retain_predictions = self._probe(self.retain_loader)
        summary = super().train()
        after_train, _ = self._probe(self.forget_loader)
        after_retain, updated_predictions = self._probe(self.retain_loader)
        retention = None
        if before_retain is not None:
            if [len(row) for row in retain_predictions] != [
                len(row) for row in updated_predictions
            ]:
                raise ValueError("Retention probe token counts changed between passes")
            agreement = sum(
                old == new
                for old_row, new_row in zip(retain_predictions, updated_predictions)
                for old, new in zip(old_row, new_row)
            )
            retention = {
                "before": before_retain,
                "after": after_retain,
                "locality_agreement": agreement / before_retain["tokens"],
            }
        summary["evaluation"] = {
            "protocol": "decoder_only_teacher_forcing",
            "scope": "first_eval_max_batches",
            "eval_max_batches": self.eval_max_batches,
            "train": {"before": before_train, "after": after_train},
            "retain": retention,
        }
        return summary
