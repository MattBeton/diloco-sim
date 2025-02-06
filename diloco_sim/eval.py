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
            # Compute local loss
            self.model.load_state_dict(original_state_dict)
            losses_local = []
            num_batches = math.ceil(self.config.eval_iters / self.config.batch_size)
            with torch.no_grad():
                for _ in range(num_batches):
                    x, y = self._get_batch(eval=True)
                    output = self.model(x)
                    loss = self.config.loss_fn(output, y)
                    losses_local.append(loss.item())
            local_loss = sum(losses_local) / len(losses_local)
            print(f"LOCAL: Eval Loss: {local_loss:.4f}, Eval Perplexity: {math.exp(local_loss):.4f}")
            self._log_eval(EvalStats(loss=local_loss, perplexity=math.exp(local_loss), glob=False))
        elif self.rank == 1:
            # Compute global loss
            losses_global = []
            num_batches = math.ceil(self.config.eval_iters / self.config.batch_size)
            with torch.no_grad():
                for _ in range(num_batches):
                    x, y = self._get_batch(eval=True)
                    output = self.model(x)
                    loss = self.config.loss_fn(output, y)
                    losses_global.append(loss.item())
            global_loss = sum(losses_global) / len(losses_global)
        else:
            # Other ranks do not perform evaluation but must participate in broadcast.
            pass

        # Broadcast the global loss from rank 1 to all ranks.
        # All ranks create a dummy tensor to participate.
        global_loss_tensor = torch.empty(1, device=next(self.model.parameters()).device)
        if self.rank == 1:
            global_loss_tensor[0] = global_loss
        torch.distributed.broadcast(global_loss_tensor, src=1)

        # Only rank 0 logs the global evaluation.
        if self.rank == 0:
            global_loss = global_loss_tensor.item()
            print(f"GLOBAL: Eval Loss: {global_loss:.4f}, Eval Perplexity: {math.exp(global_loss):.4f}")
            self._log_eval(EvalStats(loss=global_loss, perplexity=math.exp(global_loss), glob=True))

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
