import copy
import torch.nn as nn
import torch.nn.utils.prune as prune

def prune_model_global_l1(model: nn.Module, sparsity: float) -> nn.Module:
    # Create a deep copy to avoid modifying the original model
    pruned_model = copy.deepcopy(model)

    # Gather parameters to prune: all weights in Conv2d and Linear layers
    parameters_to_prune = []
    for module in pruned_model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))

    # Apply global unstructured L1 pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=sparsity
    )

    # Remove pruning reparameterization, making the pruning permanent
    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')

    return pruned_model