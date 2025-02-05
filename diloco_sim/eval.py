import torch
from .config import DilocoSimulatorConfig
from .setup import DilocoSetup
import math
from dataclasses import dataclass
import wandb


@dataclass
class EvalStats:
    loss: float
    perplexity: float
    glob: bool


class Evaluator(DilocoSetup):

    def __init__(self, config: DilocoSimulatorConfig) -> None:
        super().__init__(config)

    def _evaluate(self):
        original_state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.model.eval()

        for param in self.model.parameters():
            torch.distributed.all_reduce(param.data, op=torch.distributed.ReduceOp.SUM)
            param.data /= self.config.num_nodes

        if self.rank == 0:
            ## Rank 0 calculates Local loss
            self.model.load_state_dict(original_state_dict)

            losses = []
            num_batches = math.ceil(self.config.eval_iters / self.config.batch_size)
            with torch.no_grad():
                for _ in range(num_batches):
                    x, y = self._get_batch(eval=True)
                    output = self.model(x)
                    loss = self.config.loss_fn(output, y)
                    losses.append(loss.item())

            avg_loss = sum(losses) / len(losses)

            print(f"LOCAL: Eval Loss: {avg_loss:.4f}, Eval Perplexity: {math.exp(avg_loss):.4f}")
            self._log_eval(EvalStats(loss=avg_loss, perplexity=math.exp(avg_loss), glob=False))

        elif self.rank == 1:
            ## Rank 1 calculates Global loss
            losses = []
            num_batches = math.ceil(self.config.eval_iters / self.config.batch_size)
            with torch.no_grad():
                for _ in range(num_batches):
                    x, y = self._get_batch(eval=True)
                    output = self.model(x)
                    loss = self.config.loss_fn(output, y)
                    losses.append(loss.item())

            avg_loss = sum(losses) / len(losses)

            print(f"GLOBAL: Eval Loss: {avg_loss:.4f}, Eval Perplexity: {math.exp(avg_loss):.4f}")
            self._log_eval(EvalStats(loss=avg_loss, perplexity=math.exp(avg_loss), glob=True))

            self.model.load_state_dict(original_state_dict)

        else:
            self.model.load_state_dict(original_state_dict)

        self.model.load_state_dict(original_state_dict)

        self.model.train()

    def _log_eval(self, eval_stats: EvalStats):
        if self.config.wandb_project is None:
            return

        if eval_stats.glob:
            name = 'global'
        else:
            name = 'local'

        wandb.log(
            {
                f"val_{name}_loss": eval_stats.loss,
                f"val_{name}_perplexity": eval_stats.perplexity,
            }, step=self.local_step
        )
