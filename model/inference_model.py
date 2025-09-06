from dataclasses import dataclass
import dataclasses as dc
from enum import Enum

import numpy as np

import torch
import torch.nn as nn

from .mamba2 import Mamba2
from .retnet import RetNet
from .hawk import Hawk
from .xlstm import XLSTM
from .modules import SimpleMLP


class LayerType(Enum):
    MAMBA2 = "mamba2"
    RETNET = "retnet"
    HAWK = "hawk"
    XLSTM = "xlstm"


@dataclass(frozen=True, kw_only=True)
class ModelConfig:
    hidden_size: int
    proj_size: int
    tower_depth: int
    num_heads: int
    num_features: int
    good_towers: tuple[int,...] = dc.field(default_factory=lambda: (1, 2))


def create_layer(layer_type: LayerType, hidden_size: int, num_heads: int = 8):
    """Factory function to create baseline models

    Args:
        layer_type: Type of layer to create
        hidden_size: Hidden dimension
        num_heads: Number of attention heads (for models that use it)
    """
    match layer_type:
        case LayerType.MAMBA2:
            return Mamba2(hidden_size=hidden_size)
        case LayerType.RETNET:
            return RetNet(hidden_size=hidden_size, num_heads=num_heads)
        case LayerType.HAWK:
            return Hawk(hidden_size=hidden_size)
        case LayerType.XLSTM:
            return XLSTM(
                hidden_size=hidden_size,
                mlstm_num_heads=num_heads,
                slstm_num_heads=num_heads // 2,
            )
        case _:
            raise ValueError(f"Unknown layer type: {layer_type}.")


class Block(nn.Module):
    def __init__(self, layer_type: LayerType, config: ModelConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size)
        self.layer = create_layer(layer_type, config.hidden_size, config.num_heads)
        self.norm2 = nn.RMSNorm(config.hidden_size)
        self.mlp = SimpleMLP(config.hidden_size, config.proj_size)

    def forward(self, x: torch.Tensor, state):
        skip = x
        x, new_state = self.layer(self.norm1(x), state)
        x = x + skip
        x = x + self.mlp(self.norm2(x))
        return x, new_state

    def init_state(self, batch_size: int, device: torch.device):
        return self.layer.init_state(batch_size, device)


class Tower(nn.Module):
    def __init__(self, config: ModelConfig, layer_type):
        super().__init__()

        self.input_up = nn.Linear(config.num_features, config.proj_size)
        self.input_down = nn.Linear(config.proj_size, config.hidden_size)

        self.blocks = nn.ModuleList(
            [Block(layer_type, config) for _ in range(config.tower_depth)]
        )

    def forward(self, x: torch.Tensor, state):
        new_state = []

        x = self.input_up(x)
        x = nn.functional.relu(x)
        x = self.input_down(x)

        for block, block_state in zip(self.blocks, state, strict=True):
            x, new_block_state = block(x, block_state)
            new_state.append(new_block_state)
        return x, new_state

    def init_state(self, batch_size: int, device: torch.device):
        return [
            block.init_state(batch_size, device) for block in self.blocks
        ]  # pyright: ignore[reportCallIssue]


class MultiTowerModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        layer_types = [
            LayerType.XLSTM,
            LayerType.MAMBA2,
            LayerType.RETNET,
            LayerType.HAWK,
        ]

        self.towers = nn.ModuleList(
            [Tower(config, layer_type) for layer_type in layer_types]
        )
        self.output_proj = nn.Linear(config.hidden_size, 1)

    @property
    def good_towers(self):
        return [self.towers[i] for i in self.config.good_towers]


    def forward(self, x: torch.Tensor, state):
        results = [
            tower(x, tower_state)
            for tower_state, tower in zip(state, self.good_towers, strict=True)
        ]
        xs, new_state = zip(*results, strict=True)

        return list(xs), list(new_state)

    def init_state(self, batch_size, device):
        return [tower.init_state(batch_size, device) for tower in self.good_towers]

class EdgeLordModel(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        p = np.load('reg.npz')
        w = p['weight']
        b = p['bias']
        lin = nn.Linear(in_features=w.shape[-1], out_features=w.shape[0])
        lin.load_state_dict({
            'weight': torch.tensor(w, dtype=torch.float32),
            'bias': torch.tensor(b, dtype=torch.float32)
        })
        self.lin = lin

    def forward(self, x: torch.Tensor, state):
        x, state = self.inner(x, state)
        x = torch.cat(x, axis=-1)
        o = self.inner.output_proj(x.reshape((x.size(0), len(self.inner.config.good_towers), -1)))
        a = self.lin(x)
        a[:, self.inner.config.good_towers] = o[..., 0]
        return a, state

    def init_state(self, batch_size, device):
        return self.inner.init_state(batch_size, device)
