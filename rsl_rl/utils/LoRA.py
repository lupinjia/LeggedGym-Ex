
#sources https://github.com/priyammaz/PyTorch-Adventures/tree/main/PyTorch%20Tools/LoRA
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

import math
from typing import Optional, List, Iterator
from collections import OrderedDict
from itertools import repeat
from abc import abstractmethod


class LoRALayer():
    def __init__(
        self, 
        rank: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.merged = False
        self.merge_weights = merge_weights
        self.scaling = self.lora_alpha / self.rank
        self.lora_dropout_func = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...


#maybe replace nn.Linear with FrozenLinear for consistent implementations?
class FrozenLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        nn.Linear.__init__(self, *args, **kwargs)
        self.weight.requires_grad = False
        self.bias.requires_grad = False

    @staticmethod
    def _from_linear(module: nn.Linear, **kwargs):
        if isinstance(module, nn.Linear):
            out_size, in_size = module.weight.shape
            device = module.weight.device
            dtype = module.weight.dtype
            lora_module = FrozenLinear(
                in_size, 
                out_size,
                **kwargs
            )
            lora_module.weight.data = module.weight.data
            lora_module.bias.data = module.bias.data
            return lora_module
        else:
            raise ValueError(
                "_from_linear classmethod supports only objects "
                f"of torch.nn.Linear class, but {str(type(module))} is given."
            )
    

class LoRALinear(nn.Linear, LoRALayer):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        rank: int,
        bias: bool = True,
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        merge_weights: bool = True,
        device=None,
        dtype=None,
    ):
        assert rank > 0, "LoRA rank must be greater than 0"
        factory_kwargs = {"device": device, "dtype": dtype}
        nn.Linear.__init__(self, in_features=in_features, out_features=out_features, bias=bias, **factory_kwargs)
        # Freezing the pre-trained weight matrix and bias (following peft example)
        self.weight.requires_grad = False
        self.bias.requires_grad = False

        LoRALayer.__init__(self, rank=rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        # Actual trainable parameters
        self.lora_A = nn.Parameter(
            self.weight.new_zeros((rank, in_features), **factory_kwargs),
            requires_grad=True,
        )
        self.lora_B = nn.Parameter(
            self.weight.new_zeros((out_features, rank), **factory_kwargs),
            requires_grad=True,
        )
        
        # Reset
        self.reset_parameters()
        self.merged = False
    
    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, "lora_A"):
            nn.init.uniform_(self.lora_A)
            nn.init.zeros_(self.lora_B)

    @torch.no_grad()
    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                self.weight.data -= self.lora_B @ self.lora_A * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                self.weight.data += self.lora_B @ self.lora_A * self.scaling
                self.merged = True
    
    @torch.jit.export
    def merge(self, merge: bool = True):
        with torch.no_grad():
            if merge and not self.merged:
                self.weight.add_(self.lora_B @ self.lora_A * self.scaling)
                self.merged = True
            if not merge and self.merged:
                self.weight.sub_(self.lora_B @ self.lora_A * self.scaling)
                self.merged = False     

    def forward(self, x: torch.Tensor):
        if not self.merged:
            result = F.linear(x, self.weight, bias=self.bias)            
            result += (self.lora_dropout_func(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, self.weight, bias=self.bias)

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, rank={self.rank}"
    
    def state_dict(self, *args, keep_weights=False, keep_bias=False, **kwargs):
        dest = super().state_dict(*args, **kwargs)
        if not keep_weights or not keep_bias:
            for k in list(dest.keys()):
                if not keep_weights and "weight" in k:
                    del dest[k]
                elif not keep_bias and "bias" in k:
                    del dest[k]
        return dest

    @staticmethod
    def _from_linear(
        module: nn.Linear,
        rank: int,
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        merge_weights: bool = True,
        share_mem: bool=False,
        device=None,
        dtype=None,
    ):
        if isinstance(module, nn.Linear):
            if rank == 0:
                return FrozenLinear._from_linear(module)
            out_size, in_size = module.weight.shape
            device = module.weight.device
            dtype = module.weight.dtype
            lora_module = LoRALinear(
                in_size, 
                out_size, 
                rank=rank, 
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                merge_weights=merge_weights,
                device=device,
                dtype=dtype,
            )
            if share_mem:
                lora_module.weight = module.weight
                lora_module.bias = module.bias
            else:
                lora_module.weight.data = module.weight.data
                lora_module.bias.data = module.bias.data
            return lora_module
        else:
            raise ValueError(
                "_from_linear method supports only objects "
                f"of torch.nn.Linear class, but {str(type(module))} is given."
            )
    
def _from_sequential(model: nn.Sequential, ranks: Iterator[int] = None, targets: tuple[int] = None):
    modules = []
    check = targets is None
    if ranks is None:
        ranks = iter([])
    elif isinstance(ranks, int):
        ranks = repeat(ranks)
    else:
        ranks = iter(ranks)
    for i, layer in enumerate(model):
        if (check or i in targets) and isinstance(layer, nn.Linear):
            rank = next(ranks, 1)
            modules.append(LoRALinear._from_linear(layer, rank))
        elif isinstance(layer, nn.Linear):
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False
            modules.append(FrozenLinear._from_linear(layer))
        else:
            modules.append(layer)
    lora_model = nn.Sequential(*modules)
    return lora_model


class Constructed_LoRAs(nn.Module):
    def __init__(self, base_model: nn.Sequential, loras_models: List[nn.Sequential]):
        super().__init__()

        base_params = dict(base_model.named_parameters())
        base_buffers = dict(base_model.named_buffers())
        self.loras = nn.ModuleList()
        
        for lora_model in loras_models:
            for name, param in lora_model.named_parameters():
                if name in base_params:
                    parent_name, param_name = self._get_parent_and_name(lora_model, name)
                    setattr(parent_name, param_name, base_params[name])            
            for name, buf in lora_model.named_buffers():
                if name in base_buffers:
                    parent_name, buf_name = self._get_parent_and_name(lora_model, name)
                    setattr(parent_name, buf_name, base_buffers[name])
            
            self.loras.append(lora_model)
        #if all(list(x.parameters())[0].data_ptr() == list(y.parameters())[0].data_ptr() for x in self.loras for y in self.loras for z in range(2)):
        #    print("✅ Sharing memory!")
        #else:
        #    print("❌ Independent memory allocations.")
        #    assert 1 == 0
        self.index: int = 0
        self.current = self.loras[0]
        for layer in self.current:
            if isinstance(layer, LoRALinear):
                layer.merge()
    
    def _get_parent_and_name(self, model, target_name):
        """Helper to find the parent module of a nested parameter name."""
        names = target_name.split(".")
        curr = model
        for i in range(len(names) - 1):
            curr = getattr(curr, names[i])
        return curr, names[-1]
    
    @torch.jit.export
    def swap(self, inp: int):
        for layer in self.current:
            if isinstance(layer, LoRALinear):
                layer.merge(False)
        self.index = inp
        for i, model in enumerate(self.loras):
            if i == self.index:
                self.current = model
        for layer in self.current:
            if isinstance(layer, LoRALinear):
                layer.merge(True)

    def forward(self, inp):
        return self.current.forward(inp)

if __name__ == "__main__":
    base_model = nn.Sequential(
        nn.Linear(100, 20), nn.ReLU(), nn.Linear(20, 70), nn.ReLU()
    )

    base_model_2 = nn.Sequential(
        nn.Linear(100, 20), nn.ReLU(), nn.Linear(20, 70), nn.ReLU()
    )

    lora_model = _from_sequential(base_model, [10, 3])
    lora_model_2 = _from_sequential(base_model_2, [10, 3])
    lora_model_3 = _from_sequential(base_model_2, [10, 3])  

    lora_model_2.load_state_dict(base_model.state_dict(), strict=False)
    lora_model_2.load_state_dict(lora_model.state_dict(), strict=False)

    d = base_model.state_dict()
    d.update(lora_model.state_dict())

    lora_model_3.load_state_dict(d)

    lora_model_4 = _from_sequential(base_model_2, [10, 3])

    factory = Constructed_LoRAs(base_model, [lora_model, lora_model_4, lora_model_2])

    print("Saving TorchScript model...")
    scripted_model = torch.jit.script(factory)
    torch.jit.save(scripted_model, "lora_factory.pt")

    print("Loading TorchScript model...")
    loaded_model = torch.jit.load("lora_factory.pt")

    x = torch.randn(1, 100)
    
    loaded_model.swap(0)
    out_a = loaded_model(x)
    print(out_a)

    loaded_model.swap(1)
    out_b = loaded_model(x)
    print(out_b)

    loaded_model.swap(0)
    out_c = loaded_model(x)
    print(out_c)
    
    # Verification
    is_same = torch.allclose(out_a, out_c)
    print(f"Consistency Check (Swap 0 vs Repeat Swap 0): {'PASSED' if is_same else 'FAILED'}")