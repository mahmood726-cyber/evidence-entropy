# sentinel:skip-file — hardcoded paths / templated placeholders are fixture/registry/audit-narrative data for this repo's research workflow, not portable application configuration. Same pattern as push_all_repos.py and E156 workbook files.
"""
build_dashboard.py
Runs the EvidenceEntropy pipeline and writes a self-contained HTML dashboard.
Also writes e156-submission/paper.json.
"""
import json
import math
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from entropy_engine import run_pipeline, kl_trust_vs_naive, build_merged_dataset, H_MAX

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
print("Running pipeline...")
results  = run_pipeline()
ov       = results["overall"]
domains  = results["domains"]
n_total  = results["n_matched"]

# KL comparison
merged  = build_merged_dataset()
kl_info = kl_trust_vs_naive(merged)

# ---------------------------------------------------------------------------
# Write paper.json
# ---------------------------------------------------------------------------
paper = {
    "title": "EvidenceEntropy: Information-Theoretic Quantification of Trust-Significance "
             "Dependence Across Cochrane Meta-Analyses",
    "journal_target": "F1000Research / Systematic Reviews",
    "word_limit": 156,
    "format": "E156",
    "corpus": {
        "n_meta_analyses": n_total,
        "n_domains": len(domains),
        "data_sources": [
            "C:/Models/EvidenceScore/results/scores.csv",
            "C:/Models/ActionableEvidence/results/verdicts.csv",
            "C:/Models/TrustGate/data/review_groups.csv",
        ]
    },
    "metrics": {
        "h_trust_bits":           round(ov["h_trust"], 4),
        "h_trust_max_bits":       round(H_MAX, 4),
        "h_trust_relative":       round(ov["h_trust"] / H_MAX, 4),
        "h_sig_bits":             round(ov["h_sig"], 4),
        "h_sig_given_trust_bits": round(ov["h_sig_given_trust"], 4),
        "mi_bits":                round(ov["mi"], 4),
        "nmi":                    round(ov["nmi"], 4),
        "sig_rate":               round(ov["sig_rate"], 4),
        "kl_trust_vs_naive_pq":   round(kl_info["kl_pq"], 4),
        "kl_trust_vs_naive_qp":   round(kl_info["kl_qp"], 4),
    },
    "grade_distribution": ov["grade_dist"],
    "domain_metrics": {
        d: {
            "n": dm["n"],
            "mi_bits": round(dm["mi"], 4),
            "nmi":     round(dm["nmi"], 4),
            "h_trust": round(dm["h_trust"], 4),
            "h_sig":   round(dm["h_sig"], 4),
            "sig_rate":round(dm["sig_rate"], 4),
        }
        for d, dm in domains.items()
    },
    "key_findings": [
        f"Overall MI = {ov['mi']:.4f} bits (NMI = {ov['nmi']:.4f}): trust and "
         "significance share modest but non-zero information across all domains.",
        f"H(Trust) = {ov['h_trust']:.4f} bits ({ov['h_trust']/H_MAX*100:.1f}% of "
         "theoretical maximum), indicating substantial grade diversity.",
        f"Neurological domain has highest NMI = "
         f"{domains['Neurological']['nmi']:.4f} (n={domains['Neurological']['n']}), "
         "suggesting trust is most predictive of significance there.",
        "Renal domain shows MI ~ 0, indicating grade is uninformative about "
        "significance in that domain (n=13).",
    ],
    "method": "Shannon entropy H(X)=-sum(p_i*log2(p_i)); KL divergence "
              "D_KL(P||Q)=sum(P_i*log2(P_i/Q_i)); "
              "MI computed from 6x2 contingency table (grade x significant); "
              "NMI=MI/sqrt(H(Trust)*H(Sig)); all per domain.",
}

os.makedirs("e156-submission", exist_ok=True)
json_path = os.path.join("e156-submission", "paper.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(paper, f, indent=2)
print(f"Wrote {json_path}")

# ---------------------------------------------------------------------------
# Build dashboard HTML
# ---------------------------------------------------------------------------

# Prepare JS data blobs
domain_names  = sorted(domains.keys())
domain_mi     = [round(domains[d]["mi"],  4) for d in domain_names]
domain_nmi    = [round(domains[d]["nmi"], 4) for d in domain_names]
domain_n      = [domains[d]["n"]             for d in domain_names]
domain_ht     = [round(domains[d]["h_trust"],4) for d in domain_names]
domain_hs     = [round(domains[d]["h_sig"],  4) for d in domain_names]
domain_sr     = [round(domains[d]["sig_rate"],4) for d in domain_names]

grade_labels  = ["A+", "A", "B", "C", "D", "F"]
grade_counts  = [ov["grade_dist"].get(g, 0) for g in grade_labels]

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EvidenceEntropy Dashboard</title>
<style>
  :root {{
    --bg: #f6f3ee; --paper: #ffffff; --ink: #111111;
    --muted: #5f5a53; --line: #ddd5ca; --accent: #326891;
    --accent-soft: #e8f0f6; --good: #216c53; --warn: #a06a12;
    --serif: "Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
    --sans: "Segoe UI","Helvetica Neue",Arial,sans-serif;
    --mono: "SFMono-Regular",Consolas,"Liberation Mono",monospace;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--ink);
    font-family: var(--serif); line-height: 1.5;
  }}
  .page {{ max-width: 1200px; margin: 24px auto 80px; padding: 0 24px; }}
  .masthead {{
    border-top: 1px solid #b8aea2; border-bottom: 3px double var(--line);
    padding: 12px 0 18px; margin-bottom: 28px;
    display: flex; justify-content: space-between; align-items: flex-end;
  }}
  .masthead-brand {{ font-size: 38px; font-weight: 700; letter-spacing: -0.04em; }}
  .masthead-meta {{ font-family: var(--sans); font-size: 11px; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.14em; text-align: right; line-height: 1.7; }}

  /* Hero */
  .hero {{
    background: var(--paper); border: 1px solid var(--line);
    border-top: 4px solid var(--ink); padding: 36px 48px 30px;
    box-shadow: 0 12px 32px rgba(17,17,17,0.04); margin-bottom: 24px;
  }}
  .eyebrow {{ font-family: var(--sans); font-size: 11px; letter-spacing: 0.18em;
    text-transform: uppercase; color: var(--accent); font-weight: 700; margin-bottom: 12px; }}
  h1 {{ margin: 0 0 18px; font-size: clamp(36px,5vw,62px); line-height: 0.97;
    letter-spacing: -0.04em; }}
  .lede {{ font-size: clamp(18px,2vw,22px); color: #464038; max-width: 54rem;
    line-height: 1.65; margin: 0 0 28px; }}
  .hero-stats {{
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;
    border-top: 1px solid var(--line); padding-top: 20px;
  }}
  .hstat {{ border-top: 3px solid var(--accent); padding: 14px 16px;
    background: var(--accent-soft); }}
  .hstat-label {{ font-family: var(--sans); font-size: 10px; text-transform: uppercase;
    letter-spacing: 0.15em; color: var(--muted); margin-bottom: 6px; }}
  .hstat-value {{ font-family: var(--sans); font-size: 28px; font-weight: 800;
    line-height: 1.05; color: var(--ink); }}
  .hstat-sub {{ font-family: var(--sans); font-size: 11px; color: var(--muted);
    margin-top: 4px; }}

  /* Section cards */
  .card {{
    background: var(--paper); border: 1px solid var(--line);
    border-top: 3px solid var(--ink); padding: 28px 32px;
    box-shadow: 0 8px 24px rgba(17,17,17,0.035); margin-bottom: 24px;
  }}
  .card h2 {{
    margin: 0 0 6px; font-family: var(--sans); font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.16em; color: var(--muted);
    font-weight: 700; padding-bottom: 10px; border-bottom: 1px solid var(--line);
  }}
  canvas {{ display: block; }}

  /* Domain table */
  .domain-table {{ width: 100%; border-collapse: collapse; font-size: 14px; margin-top: 16px; }}
  .domain-table th {{ text-align: left; font-family: var(--sans); font-size: 10px;
    text-transform: uppercase; letter-spacing: 0.15em; color: var(--muted);
    padding: 10px 12px; border-bottom: 1px solid var(--line); background: #f8f5ef; }}
  .domain-table td {{ padding: 10px 12px; border-bottom: 1px solid #ede9e2; vertical-align: middle; }}
  .domain-table tbody tr:hover {{ background: var(--accent-soft); }}
  .bar-wrap {{ width: 120px; height: 10px; background: #e8e3db; border-radius: 3px; display: inline-block; }}
  .bar-inner {{ height: 100%; background: var(--accent); border-radius: 3px; }}

  /* Methodology */
  .method-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-top: 16px;
  }}
  .method-card {{
    padding: 18px; border: 1px solid var(--line); background: #fbfaf7;
    border-left: 4px solid var(--accent);
  }}
  .method-card h3 {{ margin: 0 0 8px; font-family: var(--sans); font-size: 12px;
    text-transform: uppercase; letter-spacing: 0.12em; color: var(--accent); }}
  .method-card p {{ margin: 0; font-size: 14px; line-height: 1.65; color: #2c2723; }}
  .method-card code {{ font-family: var(--mono); font-size: 12px; background: #ede9e2;
    padding: 1px 5px; }}

  .findings {{ margin-top: 16px; display: grid; gap: 10px; }}
  .finding {{ padding: 12px 16px; border: 1px solid var(--line); background: #fbfaf7;
    border-left: 3px solid var(--accent); font-size: 15px; line-height: 1.65; }}

  @media (max-width: 780px) {{
    .hero-stats {{ grid-template-columns: 1fr 1fr; }}
    .masthead {{ flex-direction: column; align-items: flex-start; gap: 10px; }}
  }}
</style>
</head>
<body>
<div class="page">

  <!-- Masthead -->
  <div class="masthead">
    <div class="masthead-brand">EvidenceEntropy</div>
    <div class="masthead-meta">
      Information-Theoretic Analysis<br>
      Cochrane Meta-Analysis Trust<br>
      N = {n_total:,} &middot; 14 Domains &middot; 2026
    </div>
  </div>

  <!-- Hero -->
  <div class="hero">
    <div class="eyebrow">Research Report &mdash; E156 Format</div>
    <h1>Does trust predict significance?</h1>
    <p class="lede">
      Across {n_total:,} Cochrane meta-analyses, GRADE-style trust grades share
      <strong>{ov['mi']:.4f} bits</strong> of mutual information with statistical
      significance (NMI = {ov['nmi']:.4f}). Trust grade has moderate predictive
      content for significance, strongest in Neurological and Dermatology domains.
    </p>
    <div class="hero-stats">
      <div class="hstat">
        <div class="hstat-label">Mutual Information</div>
        <div class="hstat-value">{ov['mi']:.4f}</div>
        <div class="hstat-sub">bits (I(Trust; Sig))</div>
      </div>
      <div class="hstat">
        <div class="hstat-label">Normalized MI</div>
        <div class="hstat-value">{ov['nmi']:.4f}</div>
        <div class="hstat-sub">NMI = MI / sqrt(H(T)&middot;H(S))</div>
      </div>
      <div class="hstat">
        <div class="hstat-label">H(Trust)</div>
        <div class="hstat-value">{ov['h_trust']:.4f}</div>
        <div class="hstat-sub">bits (max {H_MAX:.4f})</div>
      </div>
      <div class="hstat">
        <div class="hstat-label">H(Sig | Trust)</div>
        <div class="hstat-value">{ov['h_sig_given_trust']:.4f}</div>
        <div class="hstat-sub">bits (H(Sig) = {ov['h_sig']:.4f})</div>
      </div>
    </div>
  </div>

  <!-- Entropy heatmap -->
  <div class="card">
    <h2>Section 1 &mdash; Entropy Heatmap by Domain</h2>
    <canvas id="heatmapCanvas" width="1100" height="360"></canvas>
  </div>

  <!-- MI bar chart -->
  <div class="card">
    <h2>Section 2 &mdash; Mutual Information by Domain</h2>
    <canvas id="miBarCanvas" width="1100" height="320"></canvas>

    <table class="domain-table">
      <thead>
        <tr>
          <th>Domain</th><th>N</th><th>H(Trust)</th><th>H(Sig)</th>
          <th>H(Sig|Trust)</th><th>MI (bits)</th><th>NMI</th>
          <th>Sig Rate</th><th>MI bar</th>
        </tr>
      </thead>
      <tbody id="domainTableBody"></tbody>
    </table>
  </div>

  <!-- Methodology -->
  <div class="card">
    <h2>Section 3 &mdash; Methodology</h2>
    <div class="method-grid">
      <div class="method-card">
        <h3>Shannon Entropy</h3>
        <p><code>H(X) = -&sum; p&middot;log&sub;2;(p)</code><br>
          Applied to the 6-grade trust distribution (A+,A,B,C,D,F).
          Maximum = log&sub;2;(6) = 2.585 bits (uniform).</p>
      </div>
      <div class="method-card">
        <h3>KL Divergence</h3>
        <p><code>D_KL(P||Q) = &sum; P&middot;log&sub;2;(P/Q)</code><br>
          Compares trust-weighted vs naive (grade-blind) significance distributions.
          KL(P||Q) = {kl_info['kl_pq']:.4f} bits.</p>
      </div>
      <div class="method-card">
        <h3>Mutual Information</h3>
        <p><code>I(T;S) = H(S) - H(S|T)</code><br>
          Built from 6&times;2 contingency table per domain.
          Overall I(T;S) = {ov['mi']:.4f} bits.</p>
      </div>
      <div class="method-card">
        <h3>Conditional Entropy</h3>
        <p><code>H(S|T) = -&sum;&sub;t&sub; p(t)&sum;&sub;s&sub; p(s|t)log&sub;2;p(s|t)</code><br>
          Residual uncertainty in significance after knowing trust grade.
          H(Sig|Trust) = {ov['h_sig_given_trust']:.4f} bits.</p>
      </div>
      <div class="method-card">
        <h3>Normalized MI</h3>
        <p><code>NMI = MI / &radic;(H(T)&middot;H(S))</code><br>
          Scale-free dependence in [0,1].
          Overall NMI = {ov['nmi']:.4f}.</p>
      </div>
      <div class="method-card">
        <h3>Data Sources</h3>
        <p>EvidenceScore (n={n_total:,} MAs, 6 grade levels), ActionableEvidence
          (significance verdicts), TrustGate review groups (14 Cochrane domains).
          All Cochrane open-access data.</p>
      </div>
    </div>
    <div class="findings">
      <div style="font-family:var(--sans);font-size:11px;text-transform:uppercase;
        letter-spacing:0.15em;color:var(--muted);margin:20px 0 8px">Key Findings</div>
"""

for finding in paper["key_findings"]:
    html += f'      <div class="finding">{finding}</div>\n'

html += f"""    </div>
  </div>

</div><!-- .page -->

<script>
// ---- Data ----
const domainNames = {json.dumps(domain_names)};
const domainMI    = {json.dumps(domain_mi)};
const domainNMI   = {json.dumps(domain_nmi)};
const domainN     = {json.dumps(domain_n)};
const domainHT    = {json.dumps(domain_ht)};
const domainHS    = {json.dumps(domain_hs)};
const domainSR    = {json.dumps(domain_sr)};
const domainHSGT  = {json.dumps([round(domains[d]['h_sig_given_trust'],4) for d in domain_names])};

const gradeLabels = {json.dumps(grade_labels)};
const gradeCounts = {json.dumps(grade_counts)};
const nDomains    = domainNames.length;

// ---- Entropy Heatmap (canvas grid) ----
(function() {{
  const canvas = document.getElementById('heatmapCanvas');
  const ctx    = canvas.getContext('2d');
  const W      = canvas.width;
  const H      = canvas.height;

  const metrics     = ['H(Trust)', 'H(Sig)', 'H(Sig|Trust)', 'MI', 'NMI'];
  const metricData  = [domainHT, domainHS, domainHSGT, domainMI, domainNMI];
  const nMetrics    = metrics.length;
  const padL = 140, padT = 40, padB = 36;
  const cellW = (W - padL - 16) / nDomains;
  const cellH = (H - padT - padB) / nMetrics;

  // Colour scale per metric (min-max normalise to [0,1])
  function cellColor(val, minV, maxV) {{
    const t = maxV > minV ? (val - minV) / (maxV - minV) : 0;
    // Blue-white-orange scale
    const r = Math.round(t < 0.5 ? 232 + (255 - 232) * (t * 2) : 255);
    const g = Math.round(t < 0.5 ? 240 - 40 * (t * 2) : 200 - 180 * ((t - 0.5) * 2));
    const b = Math.round(t < 0.5 ? 246 - 190 * (t * 2) : 56 - 40 * ((t - 0.5) * 2));
    return `rgb(${{r}},${{g}},${{b}})`;
  }}

  metricData.forEach((data, mi_) => {{
    const minV = Math.min(...data);
    const maxV = Math.max(...data);
    data.forEach((val, di) => {{
      const x = padL + di * cellW;
      const y = padT + mi_ * cellH;
      ctx.fillStyle = cellColor(val, minV, maxV);
      ctx.fillRect(x + 1, y + 1, cellW - 2, cellH - 2);
      ctx.fillStyle = '#222';
      ctx.font = '10px "Segoe UI", Arial, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(val.toFixed(3), x + cellW / 2, y + cellH / 2);
    }});
  }});

  // Row labels (metrics)
  ctx.fillStyle = '#5f5a53';
  ctx.font = 'bold 10px "Segoe UI", Arial, sans-serif';
  ctx.textAlign = 'right';
  metrics.forEach((m, mi_) => {{
    ctx.fillText(m, padL - 8, padT + mi_ * cellH + cellH / 2);
  }});

  // Column labels (domains)
  ctx.fillStyle = '#5f5a53';
  ctx.font = '10px "Segoe UI", Arial, sans-serif';
  ctx.textAlign = 'center';
  domainNames.forEach((d, di) => {{
    const x = padL + di * cellW + cellW / 2;
    const y = padT + nMetrics * cellH + 14;
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(-Math.PI / 5);
    ctx.fillText(d.replace('_', ' '), 0, 0);
    ctx.restore();
  }});

  // Title
  ctx.fillStyle = '#111';
  ctx.font = 'bold 12px "Segoe UI", Arial, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('Entropy metrics by domain (colour = relative value within row)', padL, 18);
}})();

// ---- MI Bar Chart ----
(function() {{
  const canvas = document.getElementById('miBarCanvas');
  const ctx    = canvas.getContext('2d');
  const W      = canvas.width;
  const H      = canvas.height;
  const padL = 28, padR = 24, padT = 36, padB = 60;
  const plotW  = W - padL - padR;
  const plotH  = H - padT - padB;
  const barW   = plotW / nDomains;
  const maxMI  = Math.max(...domainMI) * 1.15;

  // Bars
  domainMI.forEach((mi, i) => {{
    const bh   = (mi / maxMI) * plotH;
    const x    = padL + i * barW + barW * 0.1;
    const y    = padT + plotH - bh;
    const bwi  = barW * 0.8;
    // Colour by NMI
    const t = domainNMI[i] / Math.max(...domainNMI);
    const r = Math.round(50 + (160 - 50) * t);
    const g = Math.round(104 + (50 - 104) * t);
    const b = Math.round(145 + (30 - 145) * t);
    ctx.fillStyle = `rgb(${{r}},${{g}},${{b}})`;
    ctx.fillRect(x, y, bwi, bh);
    // Value label
    ctx.fillStyle = '#111';
    ctx.font = '10px "Segoe UI", Arial, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(mi.toFixed(3), x + bwi / 2, y - 4);
    // Domain label
    ctx.save();
    ctx.translate(x + bwi / 2, padT + plotH + 12);
    ctx.rotate(-Math.PI / 4);
    ctx.fillStyle = '#5f5a53';
    ctx.font = '11px "Segoe UI", Arial, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(domainNames[i].replace('_', ' '), 0, 0);
    ctx.restore();
  }});

  // Y axis label
  ctx.fillStyle = '#5f5a53';
  ctx.font = 'bold 11px "Segoe UI", Arial, sans-serif';
  ctx.save();
  ctx.translate(12, padT + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.fillText('Mutual Information (bits)', 0, 0);
  ctx.restore();

  // Title
  ctx.fillStyle = '#111';
  ctx.font = 'bold 12px "Segoe UI", Arial, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('I(Trust; Significance) per domain — colour intensity = NMI', padL, 18);
}})();

// ---- Domain Table ----
(function() {{
  const tbody   = document.getElementById('domainTableBody');
  const maxMI   = Math.max(...domainMI);
  domainNames.forEach((d, i) => {{
    const barPct = Math.round((domainMI[i] / maxMI) * 100);
    const tr = document.createElement('tr');
    tr.innerHTML =
      `<td><strong>${{d.replace('_', ' ')}}</strong></td>` +
      `<td>${{domainN[i].toLocaleString()}}</td>` +
      `<td>${{domainHT[i].toFixed(4)}}</td>` +
      `<td>${{domainHS[i].toFixed(4)}}</td>` +
      `<td>${{domainHSGT[i].toFixed(4)}}</td>` +
      `<td><strong>${{domainMI[i].toFixed(4)}}</strong></td>` +
      `<td>${{domainNMI[i].toFixed(4)}}</td>` +
      `<td>${{(domainSR[i] * 100).toFixed(1)}}%</td>` +
      `<td><span class="bar-wrap"><span class="bar-inner" style="width:${{barPct}}%"></span></span></td>`;
    tbody.appendChild(tr);
  }});
}})();
</script>
</body>
</html>"""

dashboard_path = "dashboard.html"
with open(dashboard_path, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Wrote {dashboard_path}")

# ---------------------------------------------------------------------------
# Summary to stdout
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print(f"EvidenceEntropy Results")
print("=" * 60)
print(f"N matched     : {n_total:,}")
print(f"H(Trust)      : {ov['h_trust']:.4f} bits  ({ov['h_trust']/H_MAX*100:.1f}% of max)")
print(f"H(Sig)        : {ov['h_sig']:.4f} bits")
print(f"H(Sig|Trust)  : {ov['h_sig_given_trust']:.4f} bits")
print(f"MI            : {ov['mi']:.4f} bits")
print(f"NMI           : {ov['nmi']:.4f}")
print(f"KL(P||Q)      : {kl_info['kl_pq']:.4f} bits")
print()
print("Top domains by NMI:")
ranked = sorted(domains.items(), key=lambda x: -x[1]['nmi'])
for d, dm in ranked[:5]:
    print(f"  {d:<20} n={dm['n']:5d}  MI={dm['mi']:.4f}  NMI={dm['nmi']:.4f}")
