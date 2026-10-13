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


def tile(label, value, note="", cls=""):
    return (f'<div class="tile {cls}"><div class="label">{esc(label)}</div>'
            f'<div class="value">{esc(value)}</div><div class="note">{esc(note)}</div></div>')


def metrics_table(rows):
    head = ("<tr><th>Estimator</th><th>Coverage</th><th>MAE (bpm)</th><th>Per-patient MAE</th>"
            "<th>Bias</th><th>95 % limits of agreement</th><th>Within ±2 bpm</th></tr>")
    body = []
    for r in rows:
        cls = ' class="dl"' if r["estimator"].startswith("DL") else ""
        if r.get("pending") is True or r.get("mae") != r.get("mae"):
            body.append(f"<tr{cls}><td>{esc(r['estimator'])}</td><td colspan=6>training — not yet available</td></tr>")
            continue
        body.append(
            f"<tr{cls}><td>{esc(r['estimator'])}</td><td class=num>{r['coverage']:.0%}</td>"
            f"<td class=num>{r['mae']:.2f}</td><td class=num>{r['patient_macro_mae']:.2f}</td>"
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
    "Gaussian soft labels; estimate = its mean (<code>dl/</code>).",
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
    parts.append(f'<div class="byline">Fold {ctx["fold"]} of 5 · generated {esc(ctx["date"])} · '
                 f"01NVa_Dengue GE monitor data · <code>python -m scripts.make_report</code></div>")

    ppg = ctx["signals"].get("ppg")
    if ppg and ppg.get("dl"):
        parts.append(
            f'<p class="lead">On {ppg["dl_full"]["n_patients"]} held-out patients, the deep-learning model '
            f"estimates respiratory rate from monitor PPG with a mean absolute error of "
            f'{ppg["dl_full"]["mae"]:.2f} breaths/min, against {ppg["best"]["mae"]:.2f} for the best '
            f'signal-processing method ({esc(ppg["best"]["estimator"])}). The 2023 pipeline could not '
            f"produce trustworthy numbers; its code is in <code>archive/</code>.</p>")
        parts.append('<div class="tiles">'
                     + tile("DL error, PPG", f'{ppg["dl_full"]["mae"]:.2f} bpm', "mean absolute error", "accent")
                     + tile("Best signal-processing", f'{ppg["best"]["mae"]:.2f} bpm',
                            f'{ppg["best"]["estimator"]}, {ppg["best"]["coverage"]:.0%} coverage', "accent2")
                     + tile("DL within ±2 bpm", f'{ppg["dl_full"]["within_2bpm"]:.0%}', "of test windows")
                     + tile("Test patients", f'{ppg["dl_full"]["n_patients"]}',
                            f'{ppg["dl_full"]["n"]:,} windows, never seen in training')
                     + "</div>")
    parts.append('<nav class="toc"><a href="#results">Results</a><a href="#errors">Where errors are</a>'
                 '<a href="#fix">Fixing the methods</a><a href="#cases">Sample windows</a><a href="#trends">Trends</a>'
                 '<a href="#data">Data</a><a href="#old">Old pipeline</a><a href="#new">New pipeline</a>'
                 '<a href="#caveats">Caveats</a><a href="#next">Next steps</a><a href="#files">Files</a></nav>')

    # ---- results per signal
    parts.append('<h2 id="results">Results on held-out patients</h2>')
    parts.append("<p>Every estimator in a table is scored on the same windows: a random sample of the test "
                 "patients' windows. <b>Coverage</b> is the share of windows that get an estimate; smart "
                 "fusion drops windows where the three respiratory signals disagree. <b>Per-patient MAE</b> "
                 "counts each patient once, so long recordings cannot dominate.</p>")
    for sig in ("ppg", "ecg"):
        s = ctx["signals"].get(sig)
        parts.append(f"<h3>{sig.upper()}</h3>")
        if not s:
            parts.append('<div class="pending">Not run yet: the model for this signal is still training. '
                         "Re-run <code>bash scripts/build_report.sh</code> when it finishes.</div>")
            continue
        parts.append(f'<p>{s["n_windows"]:,} windows from {s["n_patients"]} test patients. The DL model has the '
                     f'lower per-patient error for {s["dl_better_patients"]:.0%} of them.</p>')
        key = [r for r in s["rows"] if r["estimator"].startswith("DL") or r["estimator"] in KEY_ROWS]
        parts.append(metrics_table(sorted(key, key=lambda r: (r.get("mae") != r.get("mae"), r.get("mae", 0)))))
        parts.append("<details><summary>All estimators</summary>"
                     + metrics_table(sorted(s["rows"], key=lambda r: (r.get("mae") != r.get("mae"), r.get("mae", 0))))
                     + "</details>")
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

    parts.extend(render_fix(ctx))
    parts.extend(render_cases(ctx))
    parts.extend(render_trends(ctx))

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
    parts.append("<li><b>One fold.</b> These numbers come from fold 0 of 5; the spread across folds is not known yet.</li>")
    parts.append("<li><b>Monitor PPG, not the wearable.</b> The input is the bedside monitor's Pleth. The "
                 "wearable SmartCare PPG and Faros ECG in the same folders are not aligned yet; their header "
                 "clocks differ by 7 h and must be checked first.</li>")
    parts.append("<li><b>Selected windows.</b> Only windows with a clean, regular impedance trace are scored "
                 f"({d['ref_ok']:.0%}). Irregular breathing, apnoea and motion are under-represented, so accuracy "
                 "on all windows will be lower.</li>")
    parts.append("<li><b>Possible ventilated patients.</b> One test patient's reference sits at 27.9–28.1 "
                 "breaths/min for half of 25 hours, as a ventilator set rate would. Ventilation status should "
                 "be taken from the records and those patients analysed separately.</li>")
    parts.append("<li><b>Slow breathing.</b> Fewer than 1 % of windows are below 12 breaths/min. The DL model "
                 "fails there, and so does the low-frequency correction: when breathing itself is below "
                 "about 8 breaths/min it sits inside the background the fix discounts, and the published "
                 "spectral rules do better (see the slowest-breathing sample window).</li>")
    parts.append("<li><b>Ambiguous labels.</b> Irregular breathing can pass the reference rules: the "
                 "largest-DL-error sample window has breath intervals of 2–8.5 s and a CV of 0.33, under the 0.5 "
                 "limit. Tightening the limit to 0.3 would drop 6.6 % of accepted windows.</li>")
    parts.append("<li><b>Correction chosen on one fold.</b> The very-low-frequency corrections were chosen on fold-0 "
                 "training patients; folds 1–4 will show whether they hold.</li></ul>")

    # ---- next steps
    parts.append('<h2 id="next">Next steps</h2><ol>'
                 "<li>Finish ECG and raw-STFT runs, then folds 1–4 for mean ± SD across folds.</li>"
                 "<li>Mark ventilated periods from clinical records; report them separately.</li>"
                 "<li>Align the wearable PPG/ECG to the monitor clock (HR cross-correlation) and test the "
                 "monitor-trained model on wearable signals.</li>"
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
        body = "".join(
            f"<tr><td>{r['method']}</td><td class=num>{r['pub_mae']:.2f}</td><td class=num>{r['pub_bias']:+.2f}</td>"
            f"<td class=num>{r['pub_cov']:.0%}</td><td class=num><b>{r['cor_mae']:.2f}</b></td>"
            f"<td class=num>{r['cor_bias']:+.2f}</td><td class=num>{r['cor_cov']:.0%}</td>"
            f"<td class=num>{r['cor_w2']:.0%}</td></tr>" for r in fix)
        out.append(f"<h3>{sig.upper()}</h3><div class=tablewrap><table><tr><th>Method</th><th>MAE, published</th>"
                   "<th>Bias</th><th>Coverage</th><th>MAE, corrected</th><th>Bias</th><th>Coverage</th>"
                   f"<th>Within ±2 bpm</th></tr>{body}</table></div>")
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
    tr = ctx.get("trend")
    if not tr:
        return ['<h2 id="trends">Trends</h2><div class="pending">Trend scoring has not run yet '
                "(<code>python -m scripts.predict_trend</code>).</div>"]
    body = "".join(f"<tr><td>{esc(r['estimator'])}</td><td class=num>{r['concordance']:.0%}</td>"
                   f"<td class=num>{r['n_pairs']:,}</td><td class=num>{r['r_change']:.2f}</td>"
                   f"<td class=num>{r.get('mae', float('nan')):.2f}</td><td class=num>{r.get('coverage', float('nan')):.0%}</td></tr>"
                   for r in tr["concordance"])
    return ['<h2 id="trends">Trends over whole recordings</h2>',
            "<p>Every 60-s window of the PPG test patients' recordings (30-s steps), including windows whose "
            "reference failed its quality checks. Clinically, following a patient's <em>changes</em> matters as "
            "much as each value. Trending ability is scored with four-quadrant analysis (Critchley et al. 2010): "
            "for windows 5 minutes apart, does the estimate move in the same direction as the reference? Pairs "
            "with a reference change under 2 bpm are excluded; 90 % concordance is the usual bar.</p>",
            "<div class=tablewrap><table><tr><th>Estimator</th><th>Concordance</th><th>Pairs</th>"
            "<th>r of changes</th><th>MAE (bpm)</th><th>Coverage</th></tr>" + body + "</table></div>",
            img(tr["fig"], "RR over the full recordings of the three test patients with the most windows. Lines are "
                "5-minute rolling medians; dark dots = reference, grey dots = reference that failed quality.")]
