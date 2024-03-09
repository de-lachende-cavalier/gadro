from typing import Any, Dict, List, Optional, Type

import numpy as np
import torch
import torch.nn.functional as F

from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch
from tianshou.policy import PPOPolicy


class GAILPrimePolicy(PPOPolicy):
    """A customized GAIL policy, made to work with our custom environments."""

    def __init__(
        self,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        optim: torch.optim.Optimizer,
        dist_fn: Type[torch.distributions.Distribution],
        expert_buffer: ReplayBuffer,
        disc_net: torch.nn.Module,
        disc_optim: torch.optim.Optimizer,
        disc_update_num: int = 4,
        eps_clip: float = 0.2,
        dual_clip: Optional[float] = None,
        value_clip: bool = False,
        advantage_normalization: bool = True,
        recompute_advantage: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            actor,
            critic,
            optim,
            dist_fn,
            eps_clip,
            dual_clip,
            value_clip,
            advantage_normalization,
            recompute_advantage,
            **kwargs,
        )

        self.disc_net = disc_net
        self.disc_optim = disc_optim
        self.disc_update_num = disc_update_num
        self.expert_buffer = expert_buffer
        self.action_dim = actor.output_dim

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        # update reward
        with torch.no_grad():
            batch.rew = to_numpy(-F.logsigmoid(-self.disc(batch)).flatten())
        return super().process_fn(batch, buffer, indices)

    def disc(self, batch: Batch) -> torch.Tensor:
        obs = batch.obs

        act = to_torch(batch.act, device=self.disc_net.device).long()
        n_classes = len(obs.speaker_info[0])  # how many patches there are
        act = F.one_hot(act, num_classes=n_classes).float()

        return self.disc_net(obs, act)

    def learn(
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        # update discriminator
        losses = []
        acc_pis = []
        acc_exps = []
        bsz = len(batch) // self.disc_update_num

        for b in batch.split(bsz, merge_last=True):
            logits_pi = self.disc(b)
            exp_b = self.expert_buffer.sample(bsz)[0]
            logits_exp = self.disc(exp_b)
            loss_pi = -F.logsigmoid(-logits_pi).mean()
            loss_exp = -F.logsigmoid(logits_exp).mean()
            loss_disc = loss_pi + loss_exp
            self.disc_optim.zero_grad()
            loss_disc.backward()
            self.disc_optim.step()
            losses.append(loss_disc.item())
            acc_pis.append((logits_pi < 0).float().mean().item())
            acc_exps.append((logits_exp > 0).float().mean().item())

        # update policy
        res = super().learn(batch, batch_size, repeat, **kwargs)
        res["loss/disc"] = losses
        res["stats/acc_pi"] = acc_pis
        res["stats/acc_exp"] = acc_exps

        return res
