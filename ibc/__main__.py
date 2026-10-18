import argparse
import copy
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ViT_B_16_Weights, vit_b_16


def images(x):
    x = x.float() / 255
    x = F.interpolate(x, (224, 224), mode='bilinear', align_corners=False)
    mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
    std = x.new_tensor([.229, .224, .225])[None, :, None, None]
    return (x - mean) / std


class Learner(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.head = nn.Linear(768, classes)
        for p in self.backbone.encoder.layers[-1].self_attention.out_proj.parameters():
            p.requires_grad_(True)

    def forward(self, x):
        z = self.backbone.conv_proj(images(x)).flatten(2).transpose(1, 2)
        z = torch.cat([self.backbone.class_token.expand(len(x), -1, -1), z], 1)
        feature = self.backbone.encoder(z)[:, 0]
        return self.head(feature), feature

    def block(self):
        return list(self.backbone.encoder.layers[-1].self_attention.out_proj.parameters())


def vector(parameters):
    return torch.cat([p.detach().flatten() for p in parameters])


def assign(parameters, value):
    offset = 0
    with torch.no_grad():
        for p in parameters:
            p.copy_(value[offset:offset + p.numel()].reshape_as(p))
            offset += p.numel()


def features(model, anchors):
    return F.normalize(model(anchors)[1], dim=1).flatten() / math.sqrt(len(anchors))


def sketch(model, anchors, c):
    if not len(anchors):
        return torch.empty(sum(p.numel() for p in model.block()), 0)
    model.eval()
    output = features(model, anchors)
    generator = torch.Generator().manual_seed(c['probe_seed'])
    rows = []
    for _ in range(c['probes']):
        signs = (torch.randint(2, output.shape, generator=generator) * 2 - 1).to(output)
        grad = torch.autograd.grad((output * signs).sum(), model.block(), retain_graph=True)
        rows.append(torch.cat([p.flatten() for p in grad]).detach())
    _, _, vh = torch.linalg.svd(torch.stack(rows), full_matrices=False)
    return vh[:min(c['rank'], len(vh))].T.contiguous()


def project(proposal, basis, bound):
    component = basis.T @ proposal
    norm = component.norm()
    return proposal + basis @ (component * (min(1., bound / float(norm)) - 1)) if norm > 0 else proposal


def prepare(c, out, source):
    d = dict(np.load(source, allow_pickle=False))
    if not {'x', 'y', 'task', 'split', 'session'} <= set(d):
        raise ValueError('NPZ requires uint8 x(N,3,H,W), y, task, split, session')
    if d['x'].dtype != np.uint8:
        raise ValueError('Retain input images as uint8 for anchor storage accounting')
    for s in np.unique(d['session']):
        if len(np.unique(d['split'][d['session'] == s])) > 1:
            raise ValueError('Acquisition session crosses data partitions')
    np.savez_compressed(out / 'data.npz', **d)


def train(c, out):
    torch.manual_seed(c['seed'])
    rng = np.random.default_rng(c['seed'])
    d = np.load(out / 'data.npz')
    x, y = torch.tensor(d['x']), torch.tensor(d['y'], dtype=torch.long)
    model = Learner(c['classes'])
    parameters = model.block()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=c['learning_rate'], momentum=.9, weight_decay=c['weight_decay'])
    tasks = sorted(np.unique(d['task']))
    anchors = torch.empty_like(x[:0])
    anchor_ids = np.array([], dtype=int)
    observed = []
    rows, accuracy = [], []
    for stage, task in enumerate(tasks):
        for param_group in optimizer.param_groups:
            param_group['lr'] = c['learning_rate']
        ids = np.flatnonzero((d['task'] == task) & (d['split'] == 'train'))
        current_classes = np.unique(d['y'][ids])
        observed.extend(current_classes.tolist())
        allowed = torch.tensor(sorted(set(observed)), dtype=torch.long)
        monitor_ids = ids[:min(c['batch_size'], len(ids))]
        work_ids = ids[len(monitor_ids):]
        if not len(work_ids):
            raise ValueError('Each stage needs separate monitor and training examples')
        basis = sketch(model, anchors, c)
        if c['variant'] == 'random' and basis.shape[1]:
            basis = torch.linalg.qr(torch.randn_like(basis), mode='reduced').Q
        initial_model, initial_optimizer = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
        scales = []
        for batch in np.array_split(work_ids[:c['calibration_steps'] * c['batch_size']], c['calibration_steps']):
            if not len(batch):
                continue
            before = vector(parameters)
            logits = model(x[batch])[0]
            loss = F.cross_entropy(logits[:, allowed], torch.searchsorted(allowed, y[batch]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scales.append(float((vector(parameters) - before).norm()))
        model.load_state_dict(initial_model)
        optimizer.load_state_dict(initial_optimizer)
        nonzero = [v for v in scales if v > 0]
        scale = float(np.median(nonzero)) if nonzero else 1.
        remaining = c['stage_budget']
        steps = c['epochs'] * math.ceil(len(work_ids) / c['batch_size'])
        desired = c['stage_budget'] / steps
        progress, disturbance, step = 0., 0., 0
        with torch.no_grad():
            start_features = features(model, anchors).clone() if len(anchors) else None
        for epoch in range(c['epochs']):
            if epoch and epoch % c['refresh_epochs'] == 0 and c['variant'] != 'frozen-sketch':
                basis = sketch(model, anchors, c)
                if c['variant'] == 'random' and basis.shape[1]:
                    basis = torch.linalg.qr(torch.randn_like(basis), mode='reduced').Q
            for group in np.array_split(rng.permutation(work_ids), math.ceil(len(work_ids) / c['batch_size'])):
                model.eval()
                with torch.no_grad():
                    monitor_before = F.cross_entropy(model(x[monitor_ids])[0][:, allowed], torch.searchsorted(allowed, y[monitor_ids]))
                    protected_before = features(model, anchors).clone() if len(anchors) else None
                before = vector(parameters)
                logits = model(x[group])[0]
                loss = F.cross_entropy(logits[:, allowed], torch.searchsorted(allowed, y[group]))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                proposal = vector(parameters) - before
                request = c['stage_budget'] / steps if c['variant'] == 'uniform' else desired
                bound = scale * min(request, remaining)
                if c['variant'] == 'strict':
                    bound = 0
                elif c['variant'] == 'no-ledger':
                    bound = scale * request
                accepted = proposal if c['variant'] in ['finetune', 'global-cap'] else project(proposal, basis, bound)
                if c['variant'] != 'finetune':
                    accepted *= min(1., c['trust_cap'] * scale / max(float(accepted.norm()), 1e-12))
                assign(parameters, before + accepted)
                expenditure = float((basis.T @ accepted).norm()) / scale
                if c['variant'] != 'no-ledger':
                    remaining = max(0., remaining - expenditure)
                with torch.no_grad():
                    after = F.cross_entropy(model(x[monitor_ids])[0][:, allowed], torch.searchsorted(allowed, y[monitor_ids]))
                    local_progress = float((monitor_before - after) / (monitor_before + 1e-8))
                    q = float((features(model, anchors) - protected_before).norm()) if len(anchors) else 0.
                if step % c['monitor_steps'] == 0:
                    progress = .9 * progress + .1 * local_progress
                    disturbance = .9 * disturbance + .1 * q
                    feedback = max(0, 1 - progress / c['progress_target']) - max(0, disturbance / c['drift_target'] - 1)
                    desired = float(np.clip(desired * np.exp(c['controller_gain'] * feedback), 0, 2 * c['stage_budget'] / steps))
                rows.append(dict(stage=stage, step=step, expenditure=expenditure, remaining=remaining, request=request,
                                 progress=local_progress, disturbance=q, accepted_norm=float(accepted.norm())))
                step += 1
            for param_group in optimizer.param_groups:
                param_group['lr'] = c['learning_rate'] * .5 * (1 + math.cos(math.pi * (epoch + 1) / c['epochs']))
        with torch.no_grad():
            retained_drift = float((features(model, anchors) - start_features).norm()) if len(anchors) else 0.
            for prior in tasks[:stage + 1]:
                test = np.flatnonzero((d['task'] == prior) & (d['split'] == 'test'))
                predicted = []
                for batch in np.array_split(test, max(1, math.ceil(len(test) / c['batch_size']))):
                    predicted.extend(allowed[model(x[batch])[0][:, allowed].argmax(1)].numpy())
                accuracy.append(dict(stage=stage, task=int(prior), accuracy=float(np.mean(np.array(predicted) == d['y'][test])), feature_drift=retained_drift))
        pool = np.unique(np.concatenate([anchor_ids, ids]))
        selected = []
        for label in sorted(set(observed)):
            available = pool[d['y'][pool] == label]
            selected.extend(rng.permutation(available)[:max(1, c['anchors'] // len(set(observed)))])
        anchor_ids = np.array(selected[:c['anchors']], dtype=int)
        anchors = x[anchor_ids]
        state_bytes = sum(p.numel() * p.element_size() for p in model.parameters()) + basis.numel() * basis.element_size() + anchors.numel() * anchors.element_size()
        state_bytes += sum(v.numel() * v.element_size() for state in optimizer.state.values() for v in state.values() if isinstance(v, torch.Tensor))
        if state_bytes > c['memory_mib'] * 2 ** 20:
            raise MemoryError('Retained-state ceiling exceeded')
        np.save(out / f'anchors_stage{stage}.npy', anchor_ids)
    torch.save(model.state_dict(), out / 'model.pt')
    pd.DataFrame(rows).to_csv(out / 'updates.csv', index=False)
    pd.DataFrame(accuracy).to_csv(out / 'accuracy.csv', index=False)


def evaluate(c, out):
    d = pd.read_csv(out / 'accuracy.csv')
    a = d.pivot(index='stage', columns='task', values='accuracy')
    final = a.iloc[-1]
    forgetting = (a.iloc[:-1].max() - final).iloc[:-1].mean()
    updates = pd.read_csv(out / 'updates.csv')
    pd.DataFrame([dict(method=c['variant'], final_accuracy=final.mean(), forgetting=forgetting,
                       new_task_accuracy=np.nanmean(np.diag(a)), feature_drift=d.feature_drift.mean(),
                       expenditure=updates.expenditure.sum())]).to_csv(out / 'metrics.csv', index=False)


def plot(out):
    d = pd.read_csv(out / 'updates.csv')
    fig, ax = plt.subplots(figsize=(7, 4))
    for stage, g in d.groupby('stage'):
        ax.plot(g.step, g.expenditure.cumsum(), label=str(stage))
    ax.set(xlabel='Optimizer step', ylabel='Cumulative normalized sensitive motion')
    ax.legend(title='Stage')
    fig.tight_layout()
    fig.savefig(out / 'expenditure.png', dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'train', 'evaluate', 'plot'])
    p.add_argument('--input')
    p.add_argument('--config', default='config.json')
    p.add_argument('--output', default='runs/default')
    p.add_argument('--variant')
    args = p.parse_args()
    c = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.variant:
        c['variant'] = args.variant
    if c['variant'] not in ['full', 'uniform', 'strict', 'no-ledger', 'random', 'frozen-sketch', 'finetune', 'global-cap']:
        p.error('Unknown variant')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    settings = out / 'settings.json'
    if args.variant is None and settings.exists():
        c['variant'] = json.loads(settings.read_text(encoding='utf-8')).get('variant', c.get('variant', 'full'))
    settings.write_text(json.dumps(c, indent=2))
    if args.command == 'prepare':
        if not args.input:
            p.error('--input required')
        prepare(c, out, args.input)
    elif args.command == 'plot':
        plot(out)
    else:
        globals()[args.command](c, out)


if __name__ == '__main__':
    main()
