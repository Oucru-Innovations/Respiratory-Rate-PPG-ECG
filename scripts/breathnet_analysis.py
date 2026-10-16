"""Interpretation tools for :class:`dl.model.BreathNet`.

``explain_forward`` re-runs the model's forward pass step by step, returning
every intermediate (resonator energy maps, harmonic-summed evidence, frame
attention, per-pathway evidence, reliability gate) and optionally switching a
component off -- an inference-time ablation of the *trained* model, no
retraining. ``integrated_gradients`` attributes the chosen rate's logit to the
input samples of each pathway (Sundararajan et al. 2017).

Used by ``notebooks/04_breathnet_explained.ipynb``.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dl.model import LabelDistribution, build_model

PATHWAYS = ("BW", "AM", "FM")
ABLATIONS = {
    "full model": set(),
    "no harmonic summation": {"harmonics"},
    "no frequency refinement": {"refine"},
    "uniform attention over time": {"attention"},
    "uniform pathway gate": {"gate"},
    "no learned rate prior": {"prior"},
    "resonators at initial width": {"bandwidth_init"},
    "BW pathway only": {"only_0"},
    "AM pathway only": {"only_1"},
    "FM pathway only": {"only_2"},
}


def load_run(run_dir):
    run_dir = Path(run_dir)
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model = build_model(**ckpt["model"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    ldl = LabelDistribution(sigma=ckpt["ldl"]["sigma"])
    return model, ldl, ckpt


def load_features(features_dir, patients=None):
    """Feature rows (memmap) and their index rows for the given patients."""
    features_dir = Path(features_dir)
    idx = pd.read_parquet(features_dir / "index.parquet")
    idx["feature_row"] = np.arange(len(idx))
    idx = idx[idx["feature_ok"]]
    if patients is not None:
        idx = idx[idx["patient_id"].isin(patients)]
    x = np.load(features_dir / "features.npy", mmap_mode="r")
    return x, idx.reset_index(drop=True)


@torch.no_grad()
def explain_forward(model, x, ablate=frozenset(), keep_maps=False):
    """BreathNet forward pass with intermediates; mirrors ``BreathNet.forward``.

    ``ablate`` may contain: harmonics, refine, attention, gate, prior,
    bandwidth_init, only_<pathway index>.
    """
    m = model
    if "bandwidth_init" in ablate:
        saved = m.bank.log_sigma.data.clone()
        m.bank.log_sigma.data = torch.log(torch.clamp(1.5 / m.bank.freq_hz, max=8.0))
    try:
        raw = m.bank(x)                                               # (B, P, K, T) log energy
    finally:
        if "bandwidth_init" in ablate:
            m.bank.log_sigma.data = saved
    e = (raw - raw.mean(dim=(2, 3), keepdim=True)) / (raw.std(dim=(2, 3), keepdim=True) + 1e-5)
    normed = e
    if "harmonics" not in ablate:
        e = m._harmonics(e)
    b, p, k, t = e.shape
    frames = e.permute(0, 3, 1, 2).reshape(b * t, p, k)
    if "refine" not in ablate:
        frames = frames + m.refine(frames)
    e = frames.reshape(b, t, p, k)
    evid, attn = [], []
    for i in range(p):
        if "attention" in ablate:
            w = torch.full((b, t), 1.0 / t)
        else:
            w = torch.softmax(m.frame_score[i](e[:, :, i]).squeeze(-1), dim=1)
        evid.append((w.unsqueeze(-1) * e[:, :, i]).sum(1))
        attn.append(w)
    evid, attn = torch.stack(evid, 1), torch.stack(attn, 1)
    prob = torch.softmax(evid, dim=-1)
    ent = -(prob * torch.log(prob + 1e-9)).sum(-1)
    att_ent = -(attn * torch.log(attn + 1e-9)).sum(-1)
    feats = torch.stack([evid.max(-1).values - evid.mean(-1), evid.std(-1), ent, att_ent], -1)
    gate = torch.softmax(m.gate(feats).squeeze(-1), dim=1)
    only = [a for a in ablate if a.startswith("only_")]
    if "gate" in ablate:
        gate = torch.full_like(gate, 1.0 / p)
    if only:
        gate = torch.zeros_like(gate)
        gate[:, int(only[0].split("_")[1])] = 1.0
    prior = torch.zeros_like(m.rate_bias) if "prior" in ablate else m.rate_bias
    logits = m.temperature * (gate.unsqueeze(-1) * evid).sum(1) * p + prior
    out = {"logits": logits, "evidence": evid, "gate": gate, "attention": attn, "gate_features": feats}
    if keep_maps:
        out.update(raw_energy=raw, normalised_energy=normed, refined_energy=e.permute(0, 2, 3, 1))
    return out


def predict(model, ldl, x, rows, ablate=frozenset(), batch=1024):
    """Decoded RR (bpm) plus per-window gate and per-pathway argmax rates."""
    rr, gates, path_rr, att_ent = [], [], [], []
    rates = ldl.bins
    for i in range(0, len(rows), batch):
        xb = torch.from_numpy(np.asarray(x[rows[i:i + batch]], dtype=np.float32))
        o = explain_forward(model, xb, ablate)
        rr.append(ldl.decode(o["logits"]).numpy())
        gates.append(o["gate"].numpy())
        path_rr.append(rates[o["evidence"].argmax(-1)].numpy())
        att_ent.append(o["gate_features"][..., 3].numpy())
    return {"rr": np.concatenate(rr), "gate": np.concatenate(gates),
            "pathway_rr": np.concatenate(path_rr), "attention_entropy": np.concatenate(att_ent)}


def integrated_gradients(model, x, target_bin, steps=64):
    """Attribution of logit ``target_bin`` to each input sample, (P, N), using a
    zero baseline (z-scored inputs, so zero = no respiratory modulation)."""
    x = torch.as_tensor(x, dtype=torch.float32)[None]
    alphas = torch.linspace(0, 1, steps)[:, None, None]
    path = (alphas * x).requires_grad_(True)
    logits = model(path)
    grad, = torch.autograd.grad(logits[:, target_bin].sum(), path)
    return (x[0] * grad.mean(0)).numpy()


def split_patients(run_dir, split="test"):
    return json.loads((Path(run_dir) / "split.json").read_text())[split]


# ---------------------------------------------------------------- export for the notebook
def _component_params(model):
    out = {}
    for n, p in model.named_parameters():
        out[n.split(".")[0]] = out.get(n.split(".")[0], 0) + p.numel()
    return out


def _frequency_response(model, f_bpm):
    """|H(f)| of each resonator (rows) on a frequency grid (bpm), max-normalised per filter."""
    k = model.bank.kernels().detach()[:, 0]                       # (2K, L)
    nk = k.shape[0] // 2
    t = model.bank.t.numpy()
    ph = np.exp(-2j * np.pi * (f_bpm[:, None] / 60.0) * t[None])  # (F, L)
    h = np.abs((k[:nk].numpy() + 1j * k[nk:].numpy()) @ ph.T)     # (K, F)
    return h / h.max(1, keepdims=True)


def _pick_examples(idx, rr, pathway_rr):
    """Three illustrative held-out windows (deterministic)."""
    d = idx.assign(pred=rr, err=np.abs(rr - idx["ref_rr"].to_numpy()),
                   bw_err=np.abs(pathway_rr[:, 0] - idx["ref_rr"].to_numpy()),
                   fm_err=np.abs(pathway_rr[:, 2] - idx["ref_rr"].to_numpy()))
    ex = {}
    typical = d[(d["err"] < 0.7) & d["ref_rr"].between(20, 28) & (d["ref_interval_cv"] < 0.12)]
    ex["Typical window"] = typical.sort_values("ref_interval_cv").iloc[len(typical) // 3]
    rescue = d[(d["err"] < 1.0) & (d[["bw_err", "fm_err"]].max(axis=1) > 8)]
    ex["One pathway misleads, fusion recovers"] = rescue.sort_values("ref_interval_cv").iloc[0]
    burst = d[d["err"] < 1.5]
    ex["Movement burst inside the window"] = burst.sort_values("ref_burst_ratio").iloc[-1]
    return ex


def export(signal, fold=0, out_root="runs", n_sample=4000, seed=0):
    """Compute everything notebook 04 plots for one signal; writes
    ``runs/breathnet_f<fold>/<signal>.npz`` and ``<signal>_ablation.csv``."""
    from common.metrics import rr_metrics
    from data_prep.build_dataset import TARGET_FS
    from estimate_rr.resp_signals import RESP_FS, respiratory_waveform
    from estimate_rr.time_domain import count_adv

    torch.set_num_threads(8)
    run = Path(out_root) / f"{signal}_modwave_w60_breathnet_f{fold}"
    out = Path(out_root) / f"breathnet_f{fold}"
    out.mkdir(parents=True, exist_ok=True)
    model, ldl, ckpt = load_run(run)
    x, idx = load_features(f"dataset/01NVa_monitor/features/{signal}_modwave_w60", split_patients(run))
    rates = ldl.bins.numpy()
    rows = idx["feature_row"].to_numpy()

    # ablations on every held-out window
    abl, full = [], None
    for name, ab in ABLATIONS.items():
        p = predict(model, ldl, x, rows, ab)
        m = rr_metrics(idx["ref_rr"], p["rr"], idx["patient_id"])
        abl.append({"variant": name, "mae": m["mae"], "patient_mae": m["patient_macro_mae"], "within_2bpm": m["within_2bpm"]})
        if name == "full model":
            full = p
    abl = pd.DataFrame(abl)
    abl["delta_patient_mae"] = abl["patient_mae"] - abl.loc[abl["variant"] == "full model", "patient_mae"].iloc[0]
    abl.to_csv(out / f"{signal}_ablation.csv", index=False)

    # internals on a random sample of held-out windows
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(idx), size=min(n_sample, len(idx)), replace=False))
    xb = torch.from_numpy(np.asarray(x[rows[pick]], dtype=np.float32))
    with torch.no_grad():
        o = explain_forward(model, xb, keep_maps=True)
        harm = model._harmonics(o["normalised_energy"])
        frames = harm.permute(0, 3, 1, 2).reshape(-1, 3, len(rates))
        delta = model.refine(frames).reshape(len(pick), -1, 3, len(rates)).mean(1)    # (B, P, K), time-averaged
    ref = idx["ref_rr"].to_numpy()[pick]
    ev = o["evidence"].numpy()
    fused = (o["gate"].unsqueeze(-1) * o["evidence"]).sum(1).numpy() * 3 * float(model.temperature)
    # evidence profiles aligned on the true rate: offset (bpm) -> mean / quartiles
    offsets = np.arange(-20, 21)
    k_true = np.clip(np.round(ref - rates[0]).astype(int), 0, len(rates) - 1)
    def aligned(e):
        z = (e - e.mean(-1, keepdims=True)) / (e.std(-1, keepdims=True) + 1e-9)
        cols = k_true[:, None] + offsets[None]
        ok = (cols >= 0) & (cols < len(rates))
        vals = np.where(ok, np.take_along_axis(z, np.clip(cols, 0, len(rates) - 1), 1), np.nan)
        return np.vstack([np.nanmean(vals, 0), np.nanpercentile(vals, 25, 0), np.nanpercentile(vals, 75, 0)])
    aligned_profiles = np.stack([aligned(ev[:, i]) for i in range(3)] + [aligned(fused)])   # (4, 3, O)

    # example windows: full anatomy + integrated gradients + the real impedance breaths
    examples = _pick_examples(idx, full["rr"], full["pathway_rr"])
    cnn = pd.read_csv(Path(out_root) / f"{signal}_mod_w60_f{fold}" / "test_predictions.csv").set_index("window_row")
    ex_arrays = {}
    for j, (label, r) in enumerate(examples.items()):
        xi = torch.from_numpy(np.asarray(x[int(r["feature_row"])], dtype=np.float32))[None]
        with torch.no_grad():
            oi = explain_forward(model, xi, keep_maps=True)
        prob = torch.softmax(oi["logits"], -1)[0].numpy()
        target = int(np.argmax(prob))
        ig = integrated_gradients(model, xi[0], target)
        sig = np.load(Path("dataset/01NVa_monitor/signals") / f"{r['segment_id']}.npz")
        a = int(round(r["start_s"] * TARGET_FS["resp"]))
        resp = respiratory_waveform(sig["resp"][a:a + int(60 * TARGET_FS["resp"])].astype(float), TARGET_FS["resp"])
        _, breaths = count_adv(resp, RESP_FS)
        ex_arrays[f"ex{j}_input"] = xi[0].numpy()
        ex_arrays[f"ex{j}_raw_energy"] = oi["raw_energy"][0].numpy()
        ex_arrays[f"ex{j}_refined_energy"] = oi["refined_energy"][0].numpy()
        ex_arrays[f"ex{j}_attention"] = oi["attention"][0].numpy()
        ex_arrays[f"ex{j}_evidence"] = oi["evidence"][0].numpy()
        ex_arrays[f"ex{j}_gate"] = oi["gate"][0].numpy()
        ex_arrays[f"ex{j}_prob"] = prob
        ex_arrays[f"ex{j}_ig"] = ig
        ex_arrays[f"ex{j}_resp"] = resp
        ex_arrays[f"ex{j}_breaths"] = breaths
        ex_arrays[f"ex{j}_meta"] = np.array(json.dumps({
            "label": label, "segment_id": r["segment_id"], "patient_id": r["patient_id"], "start_s": float(r["start_s"]),
            "ref_rr": float(r["ref_rr"]), "pred": float(r["pred"]),
            "cnn": float(cnn.loc[int(r["window_row"]), "pred_rr"]) if int(r["window_row"]) in cnn.index else None,
            "burst_ratio": float(r["ref_burst_ratio"])}))

    f_grid = np.linspace(0, 70, 281)
    np.savez_compressed(
        out / f"{signal}.npz",
        rates=rates, sigma=model.bank.log_sigma.exp().detach().numpy(),
        sigma_init=np.minimum(1.5 / model.bank.freq_hz.numpy(), 8.0),
        kernel_t=model.bank.t.numpy(), kernels=model.bank.kernels().detach()[:, 0].numpy(),
        f_grid=f_grid, freq_response=_frequency_response(model, f_grid),
        harm=model.harm.detach().numpy(), rate_bias=model.rate_bias.detach().numpy(),
        temperature=float(model.temperature),
        params=np.array(json.dumps(_component_params(model))),
        sample_ref=ref, sample_pathway_rr=rates[ev.argmax(-1)], sample_fused_rr=rates[fused.argmax(-1)],
        sample_gate=o["gate"].numpy(), sample_attention=o["attention"].numpy(),
        sample_burst=idx["ref_burst_ratio"].to_numpy()[pick],
        refine_delta_mean=delta.mean(0).numpy(), refine_in_mean=frames.reshape(len(pick), -1, 3, len(rates)).mean(1).mean(0).numpy(),
        refine_corr=np.array([np.corrcoef(delta[:, i].numpy().ravel(),
                                          frames.reshape(len(pick), -1, 3, len(rates)).mean(1)[:, i].numpy().ravel())[0, 1]
                              for i in range(3)]),
        aligned_offsets=offsets, aligned_profiles=aligned_profiles,
        n_examples=len(examples), **ex_arrays)
    print(f"{signal}: wrote {out / (signal + '.npz')} and {out / (signal + '_ablation.csv')}")


if __name__ == "__main__":
    import sys
    for s in (sys.argv[1:] or ["ppg", "ecg"]):
        export(s)
