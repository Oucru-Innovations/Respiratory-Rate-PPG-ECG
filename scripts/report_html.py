"""Render the offline benchmark report as ONE self-contained HTML file.

Figures are embedded as base64 PNG, styles are inline, nothing is fetched, so
the file can be copied, e-mailed or opened without network access.
Called from :mod:`scripts.make_report`.
"""
import base64
import html
from pathlib import Path

import numpy as np

CSS = """
:root{--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--accent:#2a78d6;--accent2:#eb6834;--ring:rgba(11,11,11,.10);
--good:#006300;--warn-bg:#fff6e0}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--page:#0d0d0d;--surface:#1a1a19;
--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--accent:#3987e5;
--accent2:#d95926;--ring:rgba(255,255,255,.10);--good:#0ca30c;--warn-bg:#2b2416}}
:root[data-theme="dark"]{--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--accent:#3987e5;--accent2:#d95926;--ring:rgba(255,255,255,.10);
--good:#0ca30c;--warn-bg:#2b2416}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);
font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:980px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:28px;line-height:1.2;margin:0 0 6px}
h2{font-size:20px;margin:44px 0 10px;padding-top:12px;border-top:1px solid var(--grid)}
h3{font-size:16px;margin:26px 0 8px}
p,li{max-width:72ch}
.byline{color:var(--muted);font-size:13px;margin-bottom:20px}
.lead{font-size:17px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:22px 0}
.tile{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:14px 16px}
.tile .label{font-size:13px;color:var(--ink2)}
.tile .value{font-size:30px;font-weight:600;margin-top:2px}
.tile .note{font-size:12px;color:var(--muted)}
.tile.accent{border-left:4px solid var(--accent)} .tile.accent2{border-left:4px solid var(--accent2)}
figure{margin:18px 0;background:#fcfcfb;border:1px solid var(--ring);border-radius:10px;padding:10px}
figure img{width:100%;height:auto;display:block}
figcaption{color:#52514e;font-size:13px;padding:6px 4px 0}
.tablewrap{overflow-x:auto;margin:12px 0;border:1px solid var(--ring);border-radius:10px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{padding:7px 10px;text-align:left;border-bottom:1px solid var(--grid);white-space:nowrap}
th{color:var(--ink2);font-weight:600;background:var(--surface);position:sticky;top:0}
td.num{text-align:right;font-variant-numeric:tabular-nums}
tr.dl td{font-weight:600}
.ci{color:var(--muted);font-weight:400;font-size:12px}
.note{color:var(--muted);font-size:13.5px}
td.wrap{white-space:normal;min-width:150px}
table.arch th{white-space:normal;vertical-align:bottom}
table.prose td{white-space:normal;vertical-align:top}
table.prose td:first-child{white-space:nowrap}
tr:last-child td{border-bottom:none}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;
background:var(--surface);border:1px solid var(--ring);border-radius:4px;padding:0 4px}
.pending{background:var(--warn-bg);border:1px solid var(--ring);border-radius:10px;padding:12px 16px}
details{margin:10px 0} summary{cursor:pointer;color:var(--accent);font-weight:600}
nav.toc{font-size:14px;margin:10px 0 0} nav.toc a{color:var(--accent);margin-right:14px;text-decoration:none}
a{color:var(--accent)}
.theme{position:fixed;top:12px;right:12px;font-size:12px;background:var(--surface);color:var(--ink2);
border:1px solid var(--ring);border-radius:8px;padding:4px 8px;cursor:pointer}
"""

THEME_JS = """
const b=document.querySelector('.theme');b.addEventListener('click',()=>{const r=document.documentElement;
const dark=r.dataset.theme?r.dataset.theme==='dark':matchMedia('(prefers-color-scheme: dark)').matches;
r.dataset.theme=dark?'light':'dark';});
"""


def esc(x):
    return html.escape(str(x))


def img(path, caption):
    p = Path(path)
    if not p.exists():
        return ""
    data = base64.b64encode(p.read_bytes()).decode()
    return (f'<figure><img alt="{esc(caption)}" src="data:image/png;base64,{data}">'
            f"<figcaption>{esc(caption)}</figcaption></figure>")


def fmt_params(n):
    """Exact below 10 k (a 585-parameter model should not read as '1 k'), else thousands."""
    return f"{int(n):,}" if n < 10_000 else f"{n / 1e3:,.0f} k"


def breathnet2_headline(ctx):
    """One sentence for the lead when a BreathNet-2 variant beats the CNN on the same
    (fully cross-validated) windows; empty otherwise."""
    rows = ctx.get("arch") or []
    wins = {}
    for sig in ("ppg", "ecg"):
        cnn = next((r for r in rows if r["signal"] == sig and r.get("is_baseline") and r["run"].endswith("_mod_w60")), None)
        if not cnn:
            continue
        cand = [r for r in rows if r["signal"] == sig and "breathnet2" in r["run"] and not r.get("pending")
                and r.get("n_folds") == cnn.get("n_folds") and r.get("p", 1) < 0.05 and r.get("mean_diff", 0) < 0]
        if cand:
            wins[sig] = min(cand, key=lambda r: r["val_patient_mae"])
    if not wins:
        return ""
    bits = []
    for sig, r in wins.items():
        cnn = next(x for x in rows if x["signal"] == sig and x.get("is_baseline") and x["run"].endswith("_mod_w60"))
        bits.append(f"{sig.upper()} {r['patient_macro_mae']:.2f} vs {cnn['patient_macro_mae']:.2f} per patient, "
                    f"better for {r['better_than_cnn']} of {r['n_patients_common']} patients, p {'&lt; 0.001' if r['p'] < 0.001 else f'= {r[chr(112)]:.2g}'}")
    same = len({r["run"].split("_w60_")[-1] for r in wins.values()}) == 1
    name = next(iter(wins.values()))["model"]
    return (f" A physiology-built model with {next(iter(wins.values()))['params'] / 1e3:.1f} k parameters "
            f"(<b>{esc(name)}</b>) does better than the {cnn['params'] / 1e3:.0f} k-parameter CNN"
            + (" on both signals" if same and len(wins) == 2 else "") + ": " + "; ".join(bits) + ".")


def tile(label, value, note="", cls=""):
    return (f'<div class="tile {cls}"><div class="label">{esc(label)}</div>'
            f'<div class="value">{esc(value)}</div><div class="note">{esc(note)}</div></div>')


def metrics_table(rows):
    head = ("<tr><th>Estimator</th><th>Coverage</th><th>MAE (bpm)</th><th>Per-patient MAE<br><span class=ci>95 % CI</span></th>"
            "<th>Bias</th><th>95 % limits<br>of agreement</th><th>Within ±2 bpm</th></tr>")
    body = []
    for r in rows:
        cls = ' class="dl"' if r["estimator"].startswith("DL") else ""
        if r.get("pending") is True or r.get("mae") != r.get("mae"):
            done = f" ({int(r['folds_done'])} of 5 folds trained)" if r.get("folds_done") is not None else ""
            body.append(f"<tr{cls}><td>{esc(r['estimator'])}</td><td colspan=6>still training{done}</td></tr>")
            continue
        ci = (f"<br><span class=ci>{r['ci_lo']:.2f}–{r['ci_hi']:.2f}</span>"
              if r.get("ci_lo") == r.get("ci_lo") and r.get("ci_lo") is not None else "")
        body.append(
            f"<tr{cls}><td>{esc(r['estimator'])}</td><td class=num>{r['coverage']:.0%}</td>"
            f"<td class=num>{r['mae']:.2f}</td><td class=num>{r['patient_macro_mae']:.2f}{ci}</td>"
            f"<td class=num>{r['bias']:+.2f}</td><td class=num>{r['loa_low']:.1f} to {r['loa_high']:.1f}</td>"
            f"<td class=num>{r['within_2bpm']:.0%}</td></tr>")
    return f'<div class="tablewrap"><table>{head}{"".join(body)}</table></div>'


def simple_table(header, rows):
    h = "".join(f"<th>{esc(c)}</th>" for c in header)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tablewrap"><table class="prose"><tr>{h}</tr>{b}</table></div>'


KEY_ROWS = ("DL", "CtO · smart fusion", "ARM · smart fusion", "WCH · smart fusion", "PKS · smart fusion",
            "ARM · median fusion", "CtO · median fusion", "ARM · smart fusion, as published")

DEFECTS = [
    ("Labels", "Philips <code>MPDataExport.csv</code> rows carry one field more than the header, so "
     "<code>NOM_RESP_RATE</code> read by name is the heart rate (median 137).", "Targets were heart rates"),
    ("Labels", "Label computed over 60 s of reference, input = last 3,000 samples assuming 100 Hz.",
     "Input and label covered different time spans"),
    ("Split", "Patients 001 and 004 used the identical recording for train and validation; other "
     "“validation” files were other days of the same patients; windows drawn with replacement.",
     "Validation scores were leaked"),
    ("DL input", "0.5–3.5 Hz band-pass plus 2nd derivative before the spectrogram.",
     "99.9 % of respiratory-band power removed"),
    ("DL input", "2.56-s STFT window, 1.28-s hop, 0–50 Hz axis resized to 224 px; dB ÷ max then "
     "<code>ToPILImage</code>.", "Rates > 23 bpm aliased; respiration in 2 % of rows; pixel values wrapped mod 256"),
    ("DL loss", "<code>ClosenessLoss</code> applied index logic to one-hot targets.",
     "Plain cross-entropy; no ordinal penalty"),
    ("DL eval", "AUC from shuffled predictions vs unshuffled targets; evaluation script loaded a "
     "different model and crashed.", "No valid metric existed"),
    ("Signal processing", "All methods ran on raw PPG/ECG band-passed to 0.1–0.5 Hz; WCH/ARM returned "
     "peak <em>counts</em>, ARP a pole magnitude, AR_RR used coefficients as roots.",
     "Only CtA returned correct rates on clean synthetic signals"),
    ("Filters", "Causal <code>lfilter</code> low-pass beside zero-phase high-pass; transfer-function "
     "form at 0.1 Hz.", "Phase lag and numerical instability"),
]

STAGES = [
    "<b>Read</b> the GE monitor export (ECG, Pleth, thoracic impedance on one clock, 300 Hz); split at gaps; "
    "resample to ECG 250 Hz, PPG 100 Hz, impedance 25 Hz (<code>data_prep/</code>).",
    "<b>Window</b> into 60-s windows with a 30-s stride. <b>Reference</b> = breath-by-breath rate "
    "(count-adv) on the impedance over exactly the window; kept only when time- and frequency-domain "
    "estimates agree within 3 bpm and the channel is not clipped, flat or bursting.",
    "<b>Quality-check</b> the PPG/ECG window: no clipping or flat-line, plausible HR, ≤ 20 % rejected beats.",
    "<b>Extract</b> baseline wander, amplitude and frequency modulation from detected beats "
    "(Charlton 2016; Karlen 2013), resample to 4 Hz, band-pass 4–60 breaths/min (<code>estimate_rr/</code>).",
    "<b>Signal processing</b>: nine estimators (CtO, CtA, PKS, ZeX, FTS, WCH, ARS, ARM, ARP) per "
    "respiratory signal, fused by smart fusion or median, with a correction for the very-low-frequency "
    "background (see <a href=\"#fix\">the fix</a>); the published rules remain available.",
    "<b>Deep learning</b>: log-power spectrograms of the three respiratory signals (16-s STFT, 1-s hop, "
    "0.94-bpm bins) → small CNN (0.4 M parameters) → distribution over 4–60 breaths/min, trained with "
    "Gaussian soft labels; estimate = its mean (<code>dl/</code>). Transformer, LSTM, GRU, U-Net, MLP and a "
    "physiology-inspired BreathNet (<code>--arch</code>) are compared in "
    "<a href=\"#arch\">Network architectures</a>.",
    "<b>Evaluate</b> on patients never used for training or model selection: MAE, per-patient MAE, "
    "Bland–Altman, coverage, errors by RR range and RR/HR ratio (<code>common/</code>, <code>scripts/</code>).",
]


def render(ctx):
    d = ctx["dataset"]
    parts = [f"<!doctype html><html lang=en><head><meta charset=utf-8>"
             f"<meta name=viewport content='width=device-width,initial-scale=1'>"
             f"<title>RR benchmark report</title><style>{CSS}</style></head><body>"
             f"<button class=theme>Toggle theme</button><main>"]
    parts.append("<h1>Respiratory-rate estimation: audit and benchmark</h1>")
    if ctx.get("cv"):
        nf = max((s["dl_full"]["n_folds"] for s in ctx["signals"].values()), default=0)
        scope = (f"5-fold cross-validation, every patient tested once" if nf == 5 else
                 f"cross-validation in progress: {nf} of 5 folds finished")
    else:
        scope = f"fold {ctx['fold']} of 5"
    parts.append(f'<div class="byline">{scope} · generated {esc(ctx["date"])} · '
                 f"01NVa_Dengue GE monitor data · <code>bash scripts/build_report.sh</code></div>")

    ppg = ctx["signals"].get("ppg")
    if ppg and ppg.get("dl"):
        f = ppg["dl_full"]
        who = (f"all {f['n_patients']} patients, each scored by a model that never saw them"
               if ctx.get("cv") else f"{f['n_patients']} held-out patients")
        parts.append(
            f'<p class="lead">On {who}, the deep-learning model estimates respiratory rate from monitor PPG with a '
            f"mean absolute error of {f['mae']:.2f} breaths/min (per patient {f['patient_macro_mae']:.2f}, 95 % CI "
            f"{f['ci_lo']:.2f}–{f['ci_hi']:.2f}), against {ppg['best']['mae']:.2f} for the best signal-processing "
            f"method ({esc(ppg['best']['estimator'])}, {ppg['best']['coverage']:.0%} of windows). The 2023 pipeline "
            f"could not produce trustworthy numbers; its code is in <code>archive/</code>."
            + breathnet2_headline(ctx) + "</p>")
        parts.append('<div class="tiles">'
                     + tile("DL error, PPG", f'{ppg["dl_full"]["mae"]:.2f} bpm', "mean absolute error", "accent")
                     + tile("Best signal-processing", f'{ppg["best"]["mae"]:.2f} bpm',
                            f'{ppg["best"]["estimator"]}, {ppg["best"]["coverage"]:.0%} coverage', "accent2")
                     + tile("DL within ±2 bpm", f'{ppg["dl_full"]["within_2bpm"]:.0%}', "of test windows")
                     + tile("Test patients", f'{ppg["dl_full"]["n_patients"]}',
                            f'{ppg["dl_full"]["n"]:,} windows, never seen in training'
                            + (f' · {ppg["dl_full"]["n_folds"]} fold{"s" if ppg["dl_full"]["n_folds"] != 1 else ""}' if ctx.get("cv") else ""))
                     + "</div>")
    parts.append('<nav class="toc"><a href="#results">Results</a><a href="#errors">Where errors are</a><a href="#arch">Architectures</a><a href="#breathnet">BreathNet</a>'
                 '<a href="#fix">Fixing the methods</a><a href="#cases">Sample windows</a><a href="#trends">Trends</a>'
                 '<a href="#wearable">Wearables</a>'
                 '<a href="#data">Data</a><a href="#old">Old pipeline</a><a href="#new">New pipeline</a>'
                 '<a href="#caveats">Caveats</a><a href="#next">Next steps</a><a href="#files">Files</a></nav>')

    # ---- results per signal
    parts.append('<h2 id="results">Results on held-out patients</h2>')
    which = ("every test window of the folds in which all of them have finished, pooled across folds"
             if ctx.get("cv") else "a random sample of the test patients' windows")
    parts.append(f"<p>Every estimator in a table is scored on the same windows: {which}. Per-patient MAE carries a "
                 "95 % interval from resampling patients. <b>Coverage</b> is the share of windows that get an estimate; smart "
                 "fusion drops windows where the three respiratory signals disagree. <b>Per-patient MAE</b> "
                 "counts each patient once, so long recordings cannot dominate.</p>")
    for sig in ("ppg", "ecg"):
        s = ctx["signals"].get(sig)
        parts.append(f"<h3>{sig.upper()}</h3>")
        if not s:
            parts.append('<div class="pending">Not finished yet: the runs for this signal are still in progress. '
                         "Re-run <code>bash scripts/build_report.sh</code> when it finishes.</div>")
            continue
        parts.append(f'<p>{s["n_windows"]:,} windows from {s["n_patients"]} test patients. The DL model has the '
                     f'lower per-patient error for {s["dl_better_patients"]:.0%} of them.</p>')
        key = [r for r in s["rows"] if r["estimator"].startswith("DL") or r["estimator"] in KEY_ROWS]
        parts.append(metrics_table(sorted(key, key=lambda r: (r.get("mae") != r.get("mae"), r.get("mae", 0)))))
        parts.append("<details><summary>All estimators</summary>"
                     + metrics_table(sorted(s["rows"], key=lambda r: (r.get("mae") != r.get("mae"), r.get("mae", 0))))
                     + "</details>")
        if s.get("per_fold"):
            folds = sorted({k for r in s["per_fold"] for k in r["folds"]})
            head = "<tr><th>Per-patient MAE</th>" + "".join(f"<th>fold {k}</th>" for k in folds) + "<th>mean ± SD</th></tr>"
            body = "".join(f"<tr><td>{esc(r['estimator'])}</td>" + "".join(f"<td class=num>{r['folds'].get(k, float('nan')):.2f}</td>" for k in folds)
                           + f"<td class=num><b>{r['mean']:.2f}</b> ± {r['sd']:.2f}</td></tr>" for r in s["per_fold"])
            parts.append(f"<p>Fold by fold (each fold tests a different fifth of the patients):</p><div class=tablewrap><table>{head}{body}</table></div>")
        parts.append(img(s["figs"]["methods"], f"{sig.upper()}: MAE of every estimator on the same windows. "
                         "Blue = deep learning, orange = best signal-processing method."))
        parts.append(img(s["figs"]["ba"], f"{sig.upper()}: Bland–Altman. Solid line = bias, dashed = 95 % "
                         "limits of agreement."))

    # ---- where errors are
    parts.append('<h2 id="errors">Where the errors are</h2>')
    if ppg and ppg.get("dl"):
        low = ppg["range_dl"].get("0–12")
        parts.append(
            "<p>Both approaches are most accurate at 12–30 breaths/min, where most windows lie. "
            + (f'Below 12 breaths/min the DL model fails (MAE {low["mae"]:.1f}, {low["n"]} windows) and '
               "over-estimates, pulled towards the common rates it was trained on. " if low else "")
            + "The signal-processing methods degrade as RR approaches half the heart rate, because "
              "respiratory signals sampled once per beat cannot represent faster breathing; the DL "
              "model degrades less.</p>")
    for sig in ("ppg", "ecg"):
        s = ctx["signals"].get(sig)
        if s:
            parts.append(img(s["figs"]["strat"], f"{sig.upper()}: MAE by reference RR range (left) and by "
                             "RR / HR ratio (right); n = windows per bin."))

    parts.extend(render_arch(ctx))
    parts.extend(render_breathnet_cv(ctx))
    parts.extend(render_fix(ctx))
    parts.extend(render_cases(ctx))
    parts.extend(render_trends(ctx))
    parts.extend(render_wearable(ctx))

    # ---- data
    parts.append('<h2 id="data">Data and reference</h2>')
    parts.append(
        f'<p>{d["n_windows"]:,} sixty-second windows from {d["n_patients"]} patients '
        f'({d["hours"]:,.0f} h of monitor recording). The impedance reference passes quality checks in '
        f'{d["ref_ok"]:.0%} of windows; {d["ppg_usable"]:.0%} are usable with PPG and {d["ecg_usable"]:.0%} '
        f'with ECG. Reference RR has median {d["rr_median"]:.1f} breaths/min (5–95 %: '
        f'{d["rr_p5"]:.0f}–{d["rr_p95"]:.0f}); RR/HR has median {d["ratio_median"]:.2f}, and '
        f'{d["ratio_gt04"]:.0%} of windows exceed 0.4.</p>')
    parts.append(img(ctx["figs"]["dataset"], "Distribution of the reference RR, heart rate and their ratio "
                     "across usable PPG windows."))
    if ppg:
        parts.append(img(ppg["figs"]["example"], "One held-out window: the impedance reference, the monitor "
                         "PPG, the three respiratory signals derived from it, and the spectrograms the DL "
                         "model sees (orange line = reference rate)."))

    # ---- old pipeline
    parts.append('<h2 id="old">What was wrong in the 2023 pipeline</h2>')
    parts.append("<p>Every stage had at least one defect that invalidates its results; the code and its "
                 "outputs are kept in <code>archive/</code> with a per-file list in "
                 "<code>archive/README.md</code>. Results produced before October 2026 should not be cited.</p>")
    parts.append(simple_table(["Stage", "Defect", "Effect"], DEFECTS))

    # ---- new pipeline
    parts.append('<h2 id="new">What replaced it</h2><ol>'
                 + "".join(f"<li>{s}</li>" for s in STAGES) + "</ol>")
    parts.append("<p>Unit tests on synthetic ECG, PPG and impedance signals cover every estimator, beat "
                 "detection, reference quality rules, splits and the DL features "
                 "(<code>python -m unittest discover -s tests -t .</code>).</p>")

    # ---- caveats
    parts.append('<h2 id="caveats">Caveats and open questions</h2><ul>')
    if not ctx.get("cv"):
        parts.append("<li><b>One fold.</b> These numbers come from fold 0 of 5; the spread across folds is not known yet.</li>")
    else:
        parts.append("<li><b>Choices made on fold 0.</b> The low-frequency corrections, the architectures and their "
                     "learning rates were chosen while looking at fold 0 (its training and validation patients), who are "
                     "test patients in folds 1–4. The choices are few and coarse, so the optimism should be small, but "
                     "fold 0's test patients are the only ones untouched by them; its numbers are shown separately where "
                     "it matters (the correction table).</li>")
    if (ctx.get("wearable") or {}).get("signals"):
        parts.append("<li><b>Wearable windows are a verified subset.</b> Only stretches whose time alignment was "
                     "confirmed beat by beat are scored, and only where the wearable signal passes the same quality "
                     "checks; windows lost to motion on the wrist or a loose chest patch are not in the numbers "
                     "(see <a href=\"#wearable\">Wearables</a> for how many).</li>")
    else:
        parts.append("<li><b>Monitor PPG, not the wearable.</b> The input is the bedside monitor's Pleth; the "
                     "wearables have not been scored yet (<code>python -m data_prep.build_wearable</code>, then "
                     "<code>python -m scripts.wearable_eval all</code>).</li>")
    parts.append("<li><b>Selected windows.</b> Only windows with a clean, regular impedance trace are scored "
                 f"({d['ref_ok']:.0%}). Irregular breathing, apnoea and motion are under-represented, so accuracy "
                 "on all windows will be lower.</li>")
    locked = d.get("locked") or []
    if locked:
        who = "; ".join(f"{r['rate']:.1f} breaths/min for {r['frac']:.0%} of {r['hours']:.0f} h" for r in locked)
        n = len(locked)
        parts.append(f"<li><b>Possible ventilated patients.</b> {(['One patient', 'Two patients', 'Three patients', 'Four patients', 'Five patients'][n - 1] if n <= 5 else f'{n} patients') + '’'} "
                     f"reference stays within ±0.15 breaths/min of one value ({who}; the median patient: "
                     f"{d['locked_median_frac']:.0%} of windows), as a ventilator set rate would. Ventilation status should be taken from the records "
                     "and those patients analysed separately.</li>")
    parts.append("<li><b>Slow breathing.</b> Fewer than 1 % of windows are below 12 breaths/min. The DL model "
                 "fails there, and so does the low-frequency correction: when breathing itself is below "
                 "about 8 breaths/min it sits inside the background the fix discounts, and the published "
                 "spectral rules do better (see the slowest-breathing sample window).</li>")
    parts.append("<li><b>Ambiguous labels.</b> Irregular breathing can pass the reference rules: the "
                 "largest-DL-error sample window has breath intervals of 2–8.5 s and a CV of 0.33, under the 0.5 "
                 "limit. Tightening the limit to 0.3 would drop 6.6 % of accepted windows.</li>")
    parts.append("</ul>")

    # ---- next steps
    worn = bool((ctx.get("wearable") or {}).get("signals"))
    parts.append('<h2 id="next">Next steps</h2><ol>' +
                 "<li>Validate on an external dataset (CapnoBase, BIDMC).</li>"
                 + ("" if ctx.get("cv") else "<li>Run folds 1–4 for mean ± SD across folds.</li>") +
                 "<li>Mark ventilated periods from clinical records; report them separately.</li>"
                 + ("<li>Fine-tune on wearable windows (patient-wise folds) and compare with the transferred "
                    "monitor models; extend the alignment to the stretches it could not verify.</li>" if worn else
                    "<li>Align the wearable PPG/ECG to the monitor clock (HR cross-correlation) and test the "
                    "monitor-trained model on wearable signals.</li>") +
                 "<li>Add low-rate training examples (re-weighting or augmentation) and evaluate 30-s windows.</li>"
                 "<li>Re-download the 24EIb Philips data and add a reader for it.</li></ol>")

    # ---- files
    parts.append('<h2 id="files">Files in this folder</h2><ul>')
    for title, href in ctx["notebooks"]:
        parts.append(f'<li><a href="{esc(href)}">{esc(title)}</a></li>')
    parts.append("<li><code>summary.json</code>: every number in this report.</li>"
                 "<li><code>fig*.png</code>: the figures, for slides.</li></ul>")
    parts.append(f"</main><script>{THEME_JS}</script></body></html>")
    return "\n".join(parts)


# ---------------------------------------------------------------- new sections
def render_arch(ctx):
    rows = ctx.get("arch") or []
    out = ['<h2 id="arch">Network architectures</h2>']
    if not rows:
        return out + ['<div class="pending">No architecture runs yet: '
                      "<code>bash scripts/run_arch_sweep_fold0.sh</code>.</div>"]
    out.append(
        "<p>Same features, patients, labels and loss; only the network differs. Each model is compared with "
        "the CNN on the same input and the same test windows (pooled over the folds in which both have finished): "
        "<b>folds better</b> counts folds where its per-patient MAE is lower, <b>patients better</b> the test patients whose MAE is lower, "
        "and <b>p</b> is a paired Wilcoxon test on per-patient MAE. <b>CPU ms</b> is the median time for one "
        "60-s window on one CPU thread: network + computing its input (beat detection, respiratory signals, "
        "spectrogram), measured on this machine.</p>")
    out.append("<ul>"
               "<li><b>CNN</b>: 2-D convolutions, pooled over time only, so frequency keeps full resolution.</li>"
               "<li><b>Transformer</b>: attention across spectrogram frames (or 2-D patches), attention-pooled.</li>"
               "<li><b>LSTM / GRU</b>: small bidirectional recurrent net over frames, attention-pooled over time.</li>"
               "<li><b>U-Net</b>: encoder–decoder that draws a “breathing ridge” map over frequency and time, "
               "averaged over time.</li>"
               "<li><b>MLP</b>: the time-averaged spectrum only; what the others gain over it comes from modelling time.</li>"
               "<li><b>BreathNet</b> (physiology-inspired, on the BW/AM/FM waveforms; 6 k parameters): breathing reaches the ECG/PPG "
               "through three pathways (intrathoracic pressure → venous return, stroke volume, vagal sinus "
               "arrhythmia), each kept as its own evidence stream; a bank of damped oscillators tuned to every "
               "rate from 4 to 60 breaths/min stands in for the respiratory rhythm generator, so each output bin "
               "<em>is</em> a breathing rate; harmonic summation folds back the 2× and 3× energy produced by "
               "inspiration being shorter than expiration; attention over time favours the steady rhythm over "
               "brief coughs or movement; and a learned reliability gate fuses the three pathways (a trainable "
               "smart fusion).</li></ul>")
    nb4 = next((href for _, href in ctx.get("notebooks", []) if href.startswith("04_")), None)
    bn = ctx.get("breathnet_cv") or {}
    if nb4 and bn.get("ablation"):
        def rng(variant, nd=1):
            v = [r["delta_patient_mae"] for r in bn["ablation"] if r["variant"] == variant]
            return f"{min(v):.{nd}f}–{max(v):.{nd}f}" if v else "?"
        inv = []
        for sig in ("PPG", "ECG"):
            got = [r["inverted_pathway"] for r in bn["learned"] if r["signal"] == sig]
            if got:
                top = max(set(got), key=got.count)
                inv.append(f"{sig}: {top}" + (f" in {got.count(top)} of {len(got)} folds" if len(got) > 1 else ""))
        scope = f"over both signals and {bn['n_folds']} folds" if bn["n_folds"] > 1 else "PPG and ECG"
        out.append("<p><b>Why BreathNet needs so few parameters</b> (details and interactive figures in "
                   f'<a href="{esc(nb4)}">notebook 04</a>). Frequency analysis is built into its structure: 57 '
                   "resonators at fixed rates whose index is the answer, so only their widths are learned. Its decisive "
                   f"learned step is tiny: a 267-weight layer turns one pathway ({'; '.join(inv)}) into an inverted "
                   "estimate of the slow 1/f background, which cancels when the pathways' evidence is summed, the same "
                   "correction we built by hand for the signal-processing methods. Switching that layer off costs "
                   f"{rng('no frequency refinement')} bpm; switching off the 9 harmonic weights costs "
                   f"{rng('no harmonic summation')} bpm; frame attention, 91 % of the parameters, is worth only "
                   f"{rng('uniform attention over time', 2)} bpm (ranges {scope}).</p>")
    b2 = [r for r in rows if "breathnet2" in r["run"] and not r.get("pending") and r.get("p_bn") == r.get("p_bn")]
    if b2:
        sentences = []
        for sig in ("ppg", "ecg"):
            cand = [r for r in b2 if r["signal"] == sig]
            if not cand:
                continue
            singles = [r for r in cand if r["model"].startswith("BreathNet + ") and " + " not in r["model"][12:]]
            helped = [r["model"][len("BreathNet + "):] for r in singles if r["p_bn"] < 0.05 and r["mean_diff_bn"] < 0]
            best = min(cand, key=lambda r: r["val_patient_mae"])
            vs_cnn = ("indistinguishable from the CNN" if best["p"] >= 0.05
                      else ("better than the CNN" if best["mean_diff"] < 0 else "still behind the CNN"))
            sentences.append(
                f"On {sig.upper()}, {len(helped)} of the {len(singles)} mechanisms beat BreathNet on their own "
                f"({', '.join(helped) if helped else 'none'}); of all {len(cand)} variants and combinations, the best-validated, "
                f"<b>{esc(best['model'])}</b> ({best['params']:,} parameters), reaches a per-patient MAE of "
                f"{best['patient_macro_mae']:.2f} against BreathNet's "
                f"{next(r['patient_macro_mae'] for r in rows if r['signal'] == sig and r['run'].endswith('_breathnet')):.2f} "
                f"({best['better_than_bn']}/{best['n_patients_bn']} patients, p = {best['p_bn']:.2g}) and is {vs_cnn} "
                f"(p = {best['p']:.2g}).")
        out.append("<p><b>BreathNet-2: building the improvisations in.</b> Each thing BreathNet had to learn by "
                   "itself is offered as an explicit mechanism, switchable one at a time: a fitted 1/f background "
                   "removed per frame (<i>whitening</i>), a phase-aligned <i>coherent</i> sum of the three pathways, "
                   "rhythm persistence as a hidden Markov model over rate and time (<i>tracking</i>, which replaces the "
                   "attention stage and most of the parameters), a <i>heart-rate prior</i> from the beat intervals, "
                   "and a small <i>conv head</i> in place of attention and gate. Paired per-patient Wilcoxon tests "
                   "against BreathNet on identical windows: " + " ".join(sentences) + " The mechanisms are "
                   "complementary per signal rather than universally additive; see notebook 04.</p>")
    head = ("<tr><th>Model</th><th>Params</th><th>CPU ms</th><th>Validation<br>per-patient MAE</th>"
            "<th>Test per-patient MAE</th><th>SD across folds</th><th>Folds better than CNN</th>"
            "<th>Patients better than CNN</th><th>p</th></tr>")
    for sig in ("ppg", "ecg"):
        for inp in dict.fromkeys(r["input"] for r in rows if r["signal"] == sig):
            group = [r for r in rows if r["signal"] == sig and r["input"] == inp]
            out.append(f"<h3>{sig.upper()} · {esc(inp)} input</h3>")
            body = []
            for r in sorted(group, key=lambda r: (bool(r.get("pending")), r.get("val_patient_mae", 0))):
                name = esc(r["model"]) + (" <span style='color:var(--muted)'>(baseline)</span>" if r["is_baseline"] else "")
                if r.get("pending"):
                    body.append(f"<tr><td>{name}</td><td colspan=8>still training</td></tr>")
                    continue
                ms = (f"{r['net_ms'] + r['feature_ms']:.0f} ({r['net_ms']:.1f} + {r['feature_ms']:.0f})"
                      if "net_ms" in r else "–")
                vs = ("–" if r["is_baseline"] else f"{r['better_than_cnn']}/{r['n_patients_common']}")
                pv = "–" if r["is_baseline"] else ("&lt;0.001" if r["p"] < 0.001 else f"{r['p']:.2g}")
                cls = ' class="dl"' if r["is_baseline"] else ""
                nf = r.get("n_folds", 1)
                sd = f"{r['fold_sd']:.2f}" if nf > 1 and r.get("fold_sd") == r.get("fold_sd") else "–"
                fb = "–" if r["is_baseline"] else f"{r.get('folds_better_than_cnn', 0)}/{nf}"
                nfold = "" if nf == 5 or not ctx.get("cv") else f" <span class=ci>({nf} fold{'s' if nf != 1 else ''})</span>"
                body.append(f"<tr{cls}><td class=wrap>{name}{nfold}</td><td class=num>{fmt_params(r['params'])}</td><td class=num>{ms}</td>"
                            f"<td class=num>{r['val_patient_mae']:.2f}</td><td class=num>{r['patient_macro_mae']:.2f}</td>"
                            f"<td class=num>{sd}</td><td class=num>{fb}</td><td class=num>{vs}</td><td class=num>{pv}</td></tr>")
            out.append(f'<div class="tablewrap"><table class="arch">{head}{"".join(body)}</table></div>')
    fig = ctx["figs"].get("arch_size")
    if fig:
        out.append(img(fig, "Per-patient MAE against parameter count (log scale). Diamonds = CNN baseline; "
                       "orange = BreathNet; vertical bars = SD across folds."))
    done = [r for r in rows if not r.get("pending") and not r["is_baseline"]]
    if done:
        better = [r for r in done if r["p"] < 0.05 and r["better_than_cnn"] > r["n_patients_common"] / 2]
        worse = [r for r in done if r["p"] < 0.05 and r["better_than_cnn"] < r["n_patients_common"] / 2]
        tie = len(done) - len(better) - len(worse)
        lst = lambda rs: ", ".join(f"{r['model'].split(' (')[0]} ({r['signal'].upper()}, {r['input']})" for r in rs)
        out.append(f"<p>Of {len(done)} comparisons with the CNN (p &lt; 0.05 on per-patient MAE): "
                   f"{len(better)} better{(' — ' + esc(lst(better))) if better else ''}; {len(worse)} worse; "
                   f"{tie} not distinguishable.</p>")
        for sig in ("ppg", "ecg"):
            sr = [r for r in rows if r["signal"] == sig and not r.get("pending")]
            if not sr:
                continue
            # model choice uses the validation patients only; test numbers are reported, never used to choose
            best = min(sr, key=lambda r: r["val_patient_mae"])
            small = min((r for r in sr if r["val_patient_mae"] <= best["val_patient_mae"] + 0.1),
                        key=lambda r: r["params"])
            out.append(f"<p><b>{sig.upper()}</b>: chosen on validation patients: {esc(best['model'].split(' (')[0])} on "
                       f"{esc(best['input'])} (validation {best['val_patient_mae']:.2f}, test {best['patient_macro_mae']:.2f} "
                       f"per-patient MAE, {fmt_params(best['params'])} parameters). Smallest model within 0.1 bpm of it "
                       f"on validation: {esc(small['model'].split(' (')[0])} on {esc(small['input'])}, "
                       f"{fmt_params(small['params'])} parameters (validation {small['val_patient_mae']:.2f}, test "
                       f"{small['patient_macro_mae']:.2f}).</p>")
            test_best = min(sr, key=lambda r: r["patient_macro_mae"])
            if test_best is not best and best["patient_macro_mae"] - test_best["patient_macro_mae"] > 0.05:
                out.append(f"<p><b>Caution:</b> the validation ranking does not carry over to the test patients "
                           f"({esc(best['model'].split(' (')[0])} is best on validation but {best['patient_macro_mae']:.2f} "
                           f"on test, against {test_best['patient_macro_mae']:.2f} for "
                           f"{esc(test_best['model'].split(' (')[0])}). Differences between architectures are smaller "
                           "than the variation between patient splits"
                           + ("." if ctx.get("cv") else ", so folds 1–4 are needed before choosing one.") + "</p>")
    return out


def render_fix(ctx):
    rows = [(sig, s["fix"]) for sig, s in ctx["signals"].items() if s.get("fix")]
    if not rows:
        return []
    out = ['<h2 id="fix">Fixing the signal-processing methods</h2>',
           "<p>ECG- and PPG-derived respiratory signals carry a 1/f-shaped very-low-frequency background "
           "(vasomotion, Mayer waves, posture). On training patients about 20 % of the 4–60 bpm power lies below "
           "8 bpm, against 15–28 % within ±2 bpm of the true rate. The published rules take the <em>largest</em> "
           "spectral peak, which is usually that background, so the spectral methods under-estimated by ~13 bpm; "
           "the background also inflates the thresholds of CtA and the zero crossings of ZeX.</p>",
           "<p>Corrections, chosen on fold-0 <b>training</b> patients only and identical for PPG and ECG: spectral "
           "methods (FTS, WCH, ARS, ARM) pick the peak standing highest above a fitted power-law background "
           "(as in aperiodic-component removal for neural spectra, Donoghue et al. 2020); ARP takes the pole with "
           "the strongest whitened power instead of the largest modulus; CtA, PKS and ZeX run on a half "
           "pre-whitened signal; CtO is unchanged. The band keeps its 4 bpm floor. Below, both rules on the "
           "held-out patients (smart fusion; coverage = windows where BW, AM and FM agree).</p>"]
    for sig, fix in rows:
        f0 = {r["method"]: r for r in (ctx["signals"][sig].get("fix_fold0") or [])}
        extra_h = "<th>MAE corrected, fold 0 only</th>" if f0 else ""
        body = "".join(
            f"<tr><td>{r['method']}</td><td class=num>{r['pub_mae']:.2f}</td><td class=num>{r['pub_bias']:+.2f}</td>"
            f"<td class=num>{r['pub_cov']:.0%}</td><td class=num><b>{r['cor_mae']:.2f}</b></td>"
            f"<td class=num>{r['cor_bias']:+.2f}</td><td class=num>{r['cor_cov']:.0%}</td>"
            f"<td class=num>{r['cor_w2']:.0%}</td>"
            + (f"<td class=num>{f0[r['method']]['cor_mae']:.2f}</td>" if r["method"] in f0 else "") + "</tr>" for r in fix)
        out.append(f"<h3>{sig.upper()}</h3><div class=tablewrap><table><tr><th>Method</th><th>MAE, published</th>"
                   "<th>Bias</th><th>Coverage</th><th>MAE, corrected</th><th>Bias</th><th>Coverage</th>"
                   f"<th>Within ±2 bpm</th>{extra_h}</tr>{body}</table></div>")
        if f0:
            out.append("<p>The last column uses fold 0's test patients only: they played no part in choosing the "
                       "corrections, so it is the clean check that they generalise.</p>")
        out.append(img(ctx["signals"][sig]["figs"]["fix"], f"{sig.upper()}: MAE per method before and after the "
                       "correction (CtO is unchanged, so it is not shown)."))
    return out


CASE_NOTES = {
    "Typical window": "Beats and breaths line up across ECG, PPG and impedance. Compare the orange (published) "
                      "and blue (corrected) lines in the spectra: the published rule takes the hump below 8 bpm.",
    "Fast breathing near HR/2": "About two beats per breath: the derived signals are sampled near their Nyquist "
                                "limit. Spectral methods still find the peak just below HR/2; breath counters miss "
                                "breaths because each cycle is only a couple of samples long.",
    "Slowest breathing": "Genuinely slow breathing (check the impedance breaths). Here the breathing peak lies "
                         "inside the low-frequency band, so the correction works against it: the trade-off of the fix.",
    "Largest DL error": "Check the impedance first: irregular breath intervals mean the label itself is uncertain. "
                        "Windows like this mark where the reference quality rules could be tightened.",
}


def _median(vals):
    vals = [v for v in vals if v is not None]
    return float(np.median(vals)) if vals else float("nan")


def render_cases(ctx):
    if not ctx.get("cases"):
        return []
    out = ['<h2 id="cases">Sample windows</h2>',
           "<p>Four held-out windows, chosen automatically: a typical one, the fastest breathing relative to heart "
           "rate, the slowest breathing and the largest DL error. Each figure shows, on one time axis, what the "
           "methods look at: R peaks (▲) and PPG pulse peaks/onsets (▲▼) from which the respiratory signals are "
           "built, the impedance breaths (●) that define the reference, the derived BW/AM/FM signals with the "
           "breaths CtO accepts, and the PPG spectra with the 1/f background (dashed), the published pick "
           "(orange, dotted), the corrected pick (blue) and the reference (grey). Interactive versions with ECG "
           "spectra are in notebook 03.</p>"]
    spectral = ("FTS", "WCH", "ARS", "ARM", "ARP")
    for c in ctx["cases"]:
        r = {row["method"]: row for row in c["rates"]}
        pub = _median([r[m]["PPG published"] for m in spectral])
        cor = _median([r[m]["PPG corrected"] for m in spectral])
        fmt = lambda v: f"{v:.1f} bpm" if np.isfinite(v) else "all rejected (BW, AM and FM disagree)"
        facts = (f"Reference {c['ref_rr']:.1f} bpm, DL {c['dl']:.1f} bpm, HR {c['hr']:.0f} (RR/HR {c['ratio']:.2f}). "
                 f"PPG spectral methods, median: {fmt(pub)} as published; {fmt(cor)} corrected.")
        out.append(f"<h3>{esc(c['case'])}</h3><p>{esc(CASE_NOTES.get(c['case'], ''))} {facts}</p>")
        out.append(img(c["fig"], f"{c['case']}: {c['patient_id']} at {c['start_s']:.0f} s."))
        cell = lambda v: "<td class=num>–</td>" if v is None else f"<td class=num>{v:.1f}</td>"
        body = "".join(f"<tr><td>{m}</td>" + "".join(cell(row[k]) for k in
                       ("PPG published", "PPG corrected", "ECG published", "ECG corrected")) + "</tr>"
                       for m, row in r.items())
        out.append("<details><summary>Every method's estimate for this window (smart fusion; – = rejected)</summary>"
                   "<div class=tablewrap><table><tr><th>Method</th><th>PPG, published</th><th>PPG, corrected</th>"
                   f"<th>ECG, published</th><th>ECG, corrected</th></tr>{body}</table></div></details>")
    return out


def render_trends(ctx):
    trends = ctx.get("trend") or {}
    if not trends:
        return ['<h2 id="trends">Trends</h2><div class="pending">Trend scoring has not run yet '
                "(<code>python -m scripts.predict_trend</code>).</div>"]
    out = ['<h2 id="trends">Trends over whole recordings</h2>',
           "<p>Every 60-s window of the test patients' recordings (30-s steps), including windows whose "
           "reference failed its quality checks. Clinically, following a patient's <em>changes</em> matters as "
           "much as each value. Trending ability is scored with four-quadrant analysis (Critchley et al. 2010): "
           "for windows 5 minutes apart, does the estimate move in the same direction as the reference? Pairs "
           "with a reference change under 2 bpm are excluded; 90 % concordance is the usual bar.</p>"]
    for sig, tr in trends.items():
        body = "".join(f"<tr><td>{esc(r['estimator'])}</td><td class=num>{r['concordance']:.0%}</td>"
                       f"<td class=num>{r['n_pairs']:,}</td><td class=num>{r['r_change']:.2f}</td>"
                       f"<td class=num>{r.get('mae', float('nan')):.2f}</td><td class=num>{r.get('coverage', float('nan')):.0%}</td></tr>"
                       for r in tr["concordance"])
        out += [f"<h3>{sig.upper()} · {tr['n_patients']} patients, {tr['n_folds']} fold(s)</h3>",
                "<div class=tablewrap><table><tr><th>Estimator</th><th>Concordance</th><th>Pairs</th>"
                "<th>r of changes</th><th>MAE (bpm)</th><th>Coverage</th></tr>" + body + "</table></div>",
                img(tr["fig"], f"{sig.upper()}: RR over the full recordings of the three test patients with the most "
                    "windows. Lines are 5-minute rolling medians; dark dots = reference, grey dots = reference that "
                    "failed quality.")]
    return out


def _wear_row(r, bold=False):
    ci = f"<br><span class=ci>{r['ci_lo']:.2f}–{r['ci_hi']:.2f}</span>" if r.get("ci_lo") == r.get("ci_lo") else ""
    cls = ' class="dl"' if bold else ""
    return (f"<tr{cls}><td>{esc(r['estimator'])}</td><td class=num>{r['coverage']:.0%}</td><td class=num>{r['mae']:.2f}</td>"
            f"<td class=num>{r['patient_macro_mae']:.2f}{ci}</td><td class=num>{r['bias']:+.2f}</td>"
            f"<td class=num>{r['loa_low']:.1f} to {r['loa_high']:.1f}</td><td class=num>{r['within_2bpm']:.0%}</td></tr>")


def render_wearable(ctx):
    wr = ctx.get("wearable")
    if not wr or not wr.get("signals"):
        return []
    al = wr["alignment"]
    out = ['<h2 id="wearable">Wearables: the same methods on the SmartCare PPG and the chest ECG</h2>',
           "<p>Everything above uses the bedside monitor's own PPG and ECG. The patients also wore a SmartCare wrist "
           "PPG and a Faros/ePatch chest ECG. Here every method is scored on the <em>wearable</em> signal against the "
           "same impedance reference, over the same 60-s windows. Nothing is retrained: each patient's wearable windows "
           "are scored by the cross-validation model of the fold in which that patient was a test patient, so no model "
           "has seen the patient.</p>",
           "<p><b>Alignment.</b> The device clocks cannot be trusted: the SmartCare file names and about half the Faros "
           "headers carry a start time, the rest of the Faros files say 1 January 2000, and the monitor's own clock is off "
           "by hours. Each wearable file was therefore aligned to the monitor from the heart itself: heart-rate series "
           "cross-correlated over the whole recording, jointly over offset and clock drift, then checked in 5-minute "
           "blocks on beat-to-beat heart-rate variability (mostly respiratory sinus arrhythmia), which is specific to "
           "each moment. A stretch is kept only if at least three blocks agree on the same time map to within 1 s; a "
           "device restart starts a new stretch. "
           + " ".join(f"{DEVICE_NAMES[k]}: {v['files_aligned']} of {v['files']} files ({v['patients_aligned']} patients, "
                      f"{v['hours_aligned']:,.0f} h) aligned, median block correlation {v['median_block_r']:.2f}, "
                      f"timing residual {v['median_resid_s']:.2f} s."
                      for k, v in al.items() if v["files"]) + "</p>"]
    if wr["figs"].get("alignment"):
        out.append(img(wr["figs"]["alignment"], "Left: ten seconds on both devices after alignment. Right: the monitor's "
                       "clock minus each device's own clock, from the verified alignments: no single correction would "
                       "have worked."))
    head = ("<tr><th>Estimator (wearable input)</th><th>Coverage</th><th>MAE (bpm)</th><th>Per-patient MAE<br>"
            "<span class=ci>95 % CI</span></th><th>Bias</th><th>95 % limits<br>of agreement</th><th>Within ±2 bpm</th></tr>")
    for sig, s in wr["signals"].items():
        out.append(f"<h3>{sig.upper()} · {DEVICE_NAMES[sig]}</h3>")
        out.append(f"<p>{s['n_windows_worn']:,} aligned windows with a usable reference from {s['n_patients']} patients. "
                   f"The wearable signal passes the quality checks in {s['wearable_ok']:.0%} of them, the monitor's "
                   f"{sig.upper()} in {s['monitor_ok']:.0%}. Scored: the {s['all_n']:,} windows where the wearable passes.</p>")
        rows = sorted(s["all"], key=lambda r: r["patient_macro_mae"])
        out.append(f'<div class="tablewrap"><table>{head}'
                   + "".join(_wear_row(r, not r["estimator"].startswith(("ARM", "WCH", "CtO"))) for r in rows) + "</table></div>")
        if s.get("paired"):
            body = "".join(
                f"<tr><td>{esc(r['estimator'])}</td><td class=num>{r['monitor']['patient_macro_mae']:.2f}</td>"
                f"<td class=num>{r['wearable']['patient_macro_mae']:.2f}</td>"
                f"<td class=num>{r['wearable']['patient_macro_mae'] - r['monitor']['patient_macro_mae']:+.2f}</td>"
                f"<td class=num>{r['monitor']['coverage']:.0%} → {r['wearable']['coverage']:.0%}</td>"
                f"<td class=num>{r['wearable_worse_patients']}/{r['n_patients']}</td>"
                f"<td class=num>{'&lt;0.001' if r['p'] < 0.001 else f'{r[chr(112)]:.2g}'}</td></tr>"
                for r in sorted(s["paired"], key=lambda r: r["wearable"]["patient_macro_mae"]))
            out.append(f"<p>Like for like: the {s['paired_n']:,} windows where the monitor's {sig.upper()} is usable "
                       "too, scored with each input. <b>Δ</b> is the change in per-patient MAE from monitor to wearable; "
                       "<b>p</b> is a paired Wilcoxon test over patients.</p>"
                       "<div class=tablewrap><table><tr><th>Estimator</th><th>Monitor input</th><th>Wearable input</th>"
                       "<th>Δ</th><th>Coverage</th><th>Patients worse on wearable</th><th>p</th></tr>" + body + "</table></div>")
    if wr["figs"].get("paired"):
        out.append(img(wr["figs"]["paired"], "Per-patient MAE on identical windows, monitor input against wearable "
                       "input. Points above the diagonal are patients the wearable serves worse."))
    return out


DEVICE_NAMES = {"ppg": "SmartCare wrist PPG", "ecg": "Faros/ePatch chest ECG"}


def render_breathnet_cv(ctx):
    cv = ctx.get("breathnet_cv")
    if not cv:
        return []
    out = ['<h2 id="breathnet">BreathNet: does the same mechanism appear in every fold?</h2>',
           f"<p>BreathNet was retrained from scratch in {cv['n_folds']} fold(s). If its accuracy came from a lucky "
           "initialisation, the parts it relies on would change from fold to fold. Below, each component is switched "
           "off in each fold's trained model, and the learned structure is read out per fold.</p>"]
    if cv.get("fig"):
        out.append(img(cv["fig"], "Extra per-patient MAE when a component is switched off; one dot per fold, bar = mean."))
    if cv.get("learned"):
        rows = [[f"{r['signal']}", f"{r['fold']}", f"{r['median_sigma_s']:.1f} s", f"{r['inverted_pathway']} (r = {r['inversion_r']:.2f})",
                 ", ".join(f"{w:+.2f}" for w in r["half_rate_weights"]), r["strongest_harmonic_pathway"]]
                for r in sorted(cv["learned"], key=lambda r: (r["signal"], r["fold"]))]
        out.append(simple_table(["Signal", "Fold", "Resonator width", "Inverted by refine",
                                 "Half-rate weights (BW, AM, FM)", "Strongest 2×/3×"], rows))
        out.append("<p class=note>Resonator width: median learned σ of the filter bank. Inverted by refine: the pathway "
                   "whose refine correction anti-correlates with its own evidence (it becomes the 1/f background estimate). "
                   "Half-rate weights: negative = octave veto. Strongest 2×/3×: the pathway with the largest harmonic "
                   "weights. The fold-0 model is taken apart in <a href=\"04_breathnet_explained.html\">notebook 04</a>.</p>")
    return out
