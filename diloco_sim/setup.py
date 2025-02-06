import torch
import os
from typing import Optional, Iterator, Tuple
from torch.distributed import init_process_group as init_process_group, destroy_process_group
import torch.distributed as dist
from copy import deepcopy
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import LambdaLR
from .config import DilocoSimulatorConfig
import wandb
from tqdm import tqdm
import math


class DilocoSetup:
    rank: int
    device: torch.device
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Optional[torch.optim.lr_scheduler.CosineAnnealingLR] = None
    master_model: torch.nn.Module
    master_optimizer: torch.optim.Optimizer
    train_dataloader: DataLoader
    eval_dataloader: Optional[DataLoader] = None
    train_data_iter: Iterator[Tuple[torch.Tensor, torch.Tensor]]
    eval_data_iter: Optional[Iterator[Tuple[torch.Tensor, torch.Tensor]]] = None
    max_local_step: int
    local_step: int = 0
    epoch: int = 0
    pbar: Optional[tqdm] = None

    def __init__(self, config: DilocoSimulatorConfig) -> None:
        self.config = config
        self.max_local_step = (
            self.config.num_epochs * len(self.config.train_dataset) // (self.config.batch_size * self.config.num_nodes)
        )
        if self.config.max_local_step:
            self.max_local_step = min(self.max_local_step, self.config.max_local_step)

        if self.config.wandb_project:
            wandb.login()

    def _initialize_logging(self) -> None:
        print(f"DilocoSimulator initialized with config: {self.config}")
        self.pbar = tqdm(total=self.max_local_step)

        if self.config.wandb_project:
            # Handle wandb initialization across ranks
            if self.rank == 0:
                wandb.init(project=self.config.wandb_project, 
                           config=self.config.__dict__,
                           name=self.config.wandb_name if self.config.wandb_name else None)
                run_id = wandb.run.id
                run_name = wandb.run.name
                # Broadcast run_id and run_name to other ranks
                self._broadcast_run_info(run_id, run_name)
            else:
                run_id, run_name = self._receive_run_info()
                # print(run_id, run_name)
                wandb.init(project=self.config.wandb_project, config=self.config.__dict__,
                        id=run_id.strip(), name=run_name.strip(), resume="allow")

    def _broadcast_run_info(self, run_id: str, run_name: str):
        """Broadcast run ID and name from rank 0 to others."""
        max_length = 256
        run_id_encoded = run_id.ljust(max_length).encode('utf-8')
        run_name_encoded = run_name.ljust(max_length).encode('utf-8')
        
        run_id_tensor = torch.tensor(list(run_id_encoded), dtype=torch.uint8).to(self.device)
        run_name_tensor = torch.tensor(list(run_name_encoded), dtype=torch.uint8).to(self.device)
        
        torch.distributed.broadcast(run_id_tensor, src=0)
        torch.distributed.broadcast(run_name_tensor, src=0)

    def _receive_run_info(self) -> tuple:
        """Receive run ID and name from rank 0."""
        max_length = 256
        run_id_tensor = torch.zeros(max_length, dtype=torch.uint8).to(self.device)
        run_name_tensor = torch.zeros(max_length, dtype=torch.uint8).to(self.device)
        
        torch.distributed.broadcast(run_id_tensor, src=0)
        torch.distributed.broadcast(run_name_tensor, src=0)
        
        run_id = run_id_tensor.cpu().numpy().tobytes().decode('utf-8').strip('\x00')
        run_name = run_name_tensor.cpu().numpy().tobytes().decode('utf-8').strip('\x00')
        return run_id, run_name

    def _initialize_distributed(self, rank: int):
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(self.config.port)
        self.rank = rank
        init_process_group(
            backend=(
                "nccl" if torch.cuda.is_available() and len(self.config.devices) == self.config.num_nodes else "gloo"
            ),
            # init_method="env://",
            rank=rank,
            world_size=self.config.num_nodes,
        )
        self.device = torch.device(
            f"cuda:{self.config.devices[rank % len(self.config.devices)]}"
            if torch.cuda.is_available() and self.config.devices
            else "cpu"
        )
        torch.cuda.set_device(self.device) if self.device.type == "cuda" else None
        print(f"Initialized process group with rank {rank} on device {self.device}")

    def _cleanup(self):
        if self.rank == 0:
            wandb.finish()
        if self.pbar:
            self.pbar.close()
        if dist.is_initialized():
            destroy_process_group()

    def _setup_master_model(self):
        print("Setting up master model")
        self.master_model = deepcopy(self.model).to("cpu")
        for param in self.master_model.parameters():
            param.requires_grad = True

    def _setup_master_optimizer(self):
        print("Setting up master optimizer")
        self.master_optimizer = self.config.outer_optimizer_cls(
            self.master_model.parameters(), **self.config.outer_optimizer_kwargs
        )

    def _setup_model(self):
        if self.rank == 0:
            print("Setting up model")
        self.model = self.config.model_cls(**self.config.model_kwargs).to(self.device)
        for name, param in self.model.named_parameters():
            dist.broadcast(param.data, src=0)

        self.model.train()

    def _setup_optimizer(self):
        if self.rank == 0:
            print("Setting up optimizer")
        self.optimizer = self.config.optimizer_cls(self.model.parameters(), **self.config.optimizer_kwargs)

    def _setup_scheduler(self):
        if self.rank == 0:
            print("Setting up scheduler")

        def lr_lambda(current_step):
            if current_step < self.config.warmup_steps:
                return float(current_step) / float(max(self.config.warmup_steps, 1))
            elif self.config.cosine_anneal:
                min_lr_factor = 0.1
                progress = (current_step - self.config.warmup_steps) / float(
                    max(1, self.config.max_local_step - self.config.warmup_steps)
                )
                cosine_term = 0.5 * (1.0 + math.cos(math.pi * progress))
                return (1 - min_lr_factor) * cosine_term + min_lr_factor
            else:
                return 1.0

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

    def _setup_train_dataloader(self):
        if self.rank == 0:
            print("Setting up train dataloader")
        sampler = DistributedSampler(
            self.config.train_dataset, num_replicas=self.config.num_nodes, rank=self.rank, shuffle=True, drop_last=True
        )  # May want to do different data split between workers when looping over epochs
        self.train_dataloader = DataLoader(
            self.config.train_dataset, batch_size=self.config.batch_size, sampler=sampler, pin_memory=True
        )
        self.train_data_iter = iter(self.train_dataloader)

    def _setup_eval_dataloader(self):
        if self.rank == 0:
            print("Setting up eval dataloader")
        self.eval_dataloader = DataLoader(
            self.config.eval_dataset, batch_size=self.config.batch_size, pin_memory=True, shuffle=True
        )
        self.eval_data_iter = iter(self.eval_dataloader)
    
    def _save_checkpoint(self):
        wandb_run_id = wandb.run.name
        checkpoint_dir = os.path.join(self.config.save_dir, self.config.wandb_project, wandb_run_id, str(self.rank))

        # Ensure directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)

        filename = f"{self.local_step}.pt"
        save_path = os.path.join(checkpoint_dir, filename)

        try:
            torch.save(self.model.state_dict(), save_path)
        except OSError as e:
            if "No space left on device" in str(e):
                print("Warning: Not enough space. Attempting to free up space by deleting old checkpoints.")
                try:
                    # Remove old checkpoint files
                    for f in os.listdir(checkpoint_dir):
                        file_path = os.path.join(checkpoint_dir, f)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            print(f"Deleted: {file_path}")

                    # Retry saving after cleanup
                    torch.save(self.model.state_dict(), save_path)
                    print("Model saved successfully after cleanup.")
                except Exception as cleanup_error:
                    print(f"Failed to free up space: {cleanup_error}")
            else:
                print(f"Failed to save model due to unexpected error: {e}")

    def _get_batch(self, eval=False):
        if not eval or self.eval_data_iter is None:
            try:
                x, y = next(self.train_data_iter)
            except StopIteration:
                self.epoch += 1
                self.train_data_iter = iter(self.train_dataloader)
                x, y = next(self.train_data_iter)
        else:
            try:
                x, y = next(self.eval_data_iter)
            except StopIteration:
                self.eval_data_iter = iter(self.eval_dataloader)
                x, y = next(self.eval_data_iter)

        x, y = x.to(self.device), y.to(self.device)

        return x, y

    def _setup(self, rank: int):
        self._initialize_distributed(rank)
        self._setup_model()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_train_dataloader()
        self._initialize_logging()
        if self.rank == 0:
            self._setup_master_model()
            self._setup_master_optimizer()
            if self.config.eval_dataset:
                self._setup_eval_dataloader()

    def load_model(self, path):
        self.master_model.load_state_dict(torch.load(path))
        for model in self.models:
            model.load_state_dict(self.master_model.state_dict())
