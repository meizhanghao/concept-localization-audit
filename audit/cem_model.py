"""Frozen CEM inference and concept-logit Grad-CAM.

The no-intervention forward was checked against the checkpoint authors' source.
Raw reference generation uses CUDA RNG; numerical audits need only NumPy.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

class CEMInference(nn.Module):
    """No-intervention evaluation path; modules retain checkpoint names."""
    def __init__(self, state):
        super().__init__()
        self.pre_concept_model = models.resnet34(weights=None)
        self.concept_context_generators = nn.ModuleList([
            nn.Sequential(nn.Linear(1000, 32), nn.LeakyReLU()) for _ in range(112)])
        self.concept_prob_generators = nn.ModuleList([nn.Linear(32, 1)])
        self.c2y_model = nn.Sequential(nn.Linear(112 * 16, 200))
        self.loss_concept = nn.BCELoss(weight=state['loss_concept.weight'])
        self.load_state_dict(state, strict=True)

    def spatial_forward(self, x):
        m = self.pre_concept_model
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        x = m.layer4(m.layer3(m.layer2(m.layer1(x))))
        pre = m.fc(m.avgpool(x).flatten(1))
        return x, pre

    def contexts(self, pre):
        return torch.stack([module(pre) for module in self.concept_context_generators], 1)

    def forward(self, x):
        spatial, pre = self.spatial_forward(x)
        context = self.contexts(pre)
        logits = self.concept_prob_generators[0](context).squeeze(-1)
        probability = logits.sigmoid()
        mixed = context[:, :, :16] * probability[..., None] + context[:, :, 16:] * (1 - probability[..., None])
        species = self.c2y_model(mixed.flatten(1))
        return probability, species, logits, spatial, pre

def parameter_bank(model):
    context_weight = torch.stack([v[0].weight for v in model.concept_context_generators])
    context_bias = torch.stack([v[0].bias for v in model.concept_context_generators])
    probability_weight = model.concept_prob_generators[0].weight.flatten()
    probability_bias = model.concept_prob_generators[0].bias.flatten()
    bank = [(context_weight, context_bias, probability_weight, probability_bias)]
    for seed in range(2026112301, 2026112306):
        generator = torch.Generator(device='cuda').manual_seed(seed)
        context = torch.randn(context_weight.shape, device='cuda', generator=generator)
        context *= context_weight.norm(dim=-1, keepdim=True) / context.norm(dim=-1, keepdim=True)
        probability = torch.randn(probability_weight.shape, device='cuda', generator=generator)
        probability *= probability_weight.norm() / probability.norm()
        bank.append((context, torch.zeros_like(context_bias), probability, torch.zeros_like(probability_bias)))
    return bank

def cam_maps(spatial, pre, params, fc_weight, mapping):
    context_weight, context_bias, probability_weight, _ = params
    weight = context_weight[mapping]
    activation = torch.einsum('bi,aki->bak', pre, weight) + context_bias[mapping]
    slopes = torch.where(activation >= 0, 1., .01)
    gradient_pre = torch.einsum('bak,aki->bai', slopes * probability_weight, weight)
    alpha = (gradient_pre @ fc_weight) / (spatial.shape[-1] * spatial.shape[-2])
    return torch.einsum('bak,bkhw->bahw', alpha, spatial).relu()

def crop_origin(length):
    if length >= 299:
        return int(round((length - 299) / 2))
    return -((299 - length) // 2)

def scored_peaks(maps, grid):
    native = F.interpolate(maps, (299, 299), mode='bilinear', align_corners=False)
    projected = F.grid_sample(native, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    full = F.interpolate(projected, (224, 224), mode='bicubic', align_corners=False)
    index = full.flatten(-2).argmax(-1)
    return torch.stack([index % 224, index // 224], -1), maps.flatten(-2).amax(-1) <= 0
