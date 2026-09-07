import os
from pathlib import Path
ROOT = Path(os.environ.get('SPARSE_CODING_ROOT', Path(__file__).resolve().parents[1]))
"""Build final Figure 5: definition, replication, specificity, dynamics, decoupling."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from scipy.stats import spearmanr, t

HERE = Path(__file__).parent

OUT = ROOT / "outputs"
FINAL = ROOT / "figures" / "output"
sys.path.insert(0, str(HERE))
import style as st

st.setup()
RNG = np.random.default_rng(20260822)


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def mean_ci(values):
    a = np.asarray(values, float)
    m = np.nanmean(a)
    half = t.ppf(.975, max(len(a)-1, 1)) * np.nanstd(a, ddof=1) / np.sqrt(len(a))
    return m, m-half, m+half


def session_means(rows, value_key, filters=None):
    """Aggregate region observations to the independent session level."""
    grouped = defaultdict(list)
    for row in rows:
        if filters and any(row.get(k) != v for k, v in filters.items()):
            continue
        grouped[row["session"]].append(float(row[value_key]))
    return np.asarray([np.mean(v) for v in grouped.values()], float)


def session_deltas(rows, active_key, passive_key, filters=None):
    grouped = defaultdict(list)
    for row in rows:
        if filters and any(row.get(k) != v for k, v in filters.items()):
            continue
        grouped[row["session"]].append(float(row[active_key])-float(row[passive_key]))
    return np.asarray([np.mean(v) for v in grouped.values()], float)


def raincloud_h(ax, values, y, color, width=.25, seed=0, alpha=.22):
    """Compact horizontal raw-data distribution with a half violin."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    violin = ax.violinplot(values, positions=[y], vert=False, widths=width * 1.55,
                           showmeans=False, showmedians=False, showextrema=False)
    body = violin["bodies"][0]
    body.set_facecolor(color); body.set_edgecolor("none"); body.set_alpha(.16)
    # Keep only the upper half, leaving the lower half for raw observations.
    path = body.get_paths()[0]
    vertices = path.vertices
    vertices[:, 1] = np.maximum(vertices[:, 1], y)
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-width * .44, -width * .08, len(values))
    ax.scatter(values, y + jitter, s=4.2, color=color, alpha=alpha,
               edgecolor="none", rasterized=True, zorder=2)
    m, lo, hi = mean_ci(values)
    ax.errorbar(m, y + width * .11, xerr=[[m-lo], [hi-m]], fmt="o", color=color,
                ms=4.4, capsize=2.1, elinewidth=1.05, zorder=4)
    return m, lo, hi


def bootstrap_raincloud_h(ax, values, observed, y, color, marker="o",
                          width=.20, seed=0):
    """Horizontal raincloud for a cluster-bootstrap correlation distribution.

    The large marker is the observed correlation and its error bar is the
    percentile 95% cluster-bootstrap interval; bootstrap draws are not treated
    as independent observations.
    """
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    violin = ax.violinplot(values, positions=[y], vert=False,
                           widths=width * 1.55, showmeans=False,
                           showmedians=False, showextrema=False)
    body = violin["bodies"][0]
    body.set_facecolor(color); body.set_edgecolor("none"); body.set_alpha(.16)
    path = body.get_paths()[0]
    path.vertices[:, 1] = np.maximum(path.vertices[:, 1], y)
    rng = np.random.default_rng(seed)
    n_show = min(170, len(values))
    show = rng.choice(values, n_show, replace=False)
    jitter = rng.uniform(-width * .44, -width * .08, n_show)
    ax.scatter(show, y + jitter, s=4.0, color=color, alpha=.13,
               edgecolor="none", rasterized=True, zorder=2)
    lo, hi = np.quantile(values, [.025, .975])
    ax.errorbar(observed, y + width * .11,
                xerr=[[observed-lo], [hi-observed]], fmt=marker,
                color=color, markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=.4, ms=4.8, capsize=2.1,
                elinewidth=1.05, zorder=4)


def exact_crosspath_bootstrap(rows, axis_name, x_name, n_boot=3000, seed=0):
    """Reproduce P12's session-cluster bootstrap for one cross path."""
    selected = []
    for row in rows:
        if row["n_units"] != 20 or row["axis"] != axis_name:
            continue
        sharing_p = row["align_cv_p"] - row["align_cv_null_p"]
        sharing_a = row["align_cv_a"] - row["align_cv_null_a"]
        selected.append({
            "session": row["session"],
            "d_ps": row["ps_a"] - row["ps_p"],
            "d_lcr": row["lcr_a"] - row["lcr_p"],
            "d_sharing": sharing_a - sharing_p,
        })
    grouped = defaultdict(list)
    for row in selected:
        grouped[row["session"]].append(row)
    sessions = np.asarray(sorted(grouped), object)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, float)
    for i in range(n_boot):
        draw = rng.choice(sessions, len(sessions), replace=True)
        sample = [row for session in draw for row in grouped[session]]
        boots[i] = spearmanr(
            [row[x_name] for row in sample],
            [row["d_sharing"] for row in sample]).statistic
    observed = spearmanr(
        [row[x_name] for row in selected],
        [row["d_sharing"] for row in selected]).statistic
    return float(observed), boots


def save(fig):
    FINAL.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(FINAL / f"fig5_state_final.{ext}", dpi=450,
                    bbox_inches="tight", facecolor="white")


def add_cov_ellipse(ax, mean, cov, color, ls="-", scale=2.0):
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
    ax.add_patch(Ellipse(mean, 2*scale*np.sqrt(vals[0]), 2*scale*np.sqrt(vals[1]),
                         angle=angle, fill=False, edgecolor=color, lw=1.1, ls=ls))


def build():
    allen = load("allen_vbn_active_passive_redundancy.json")
    ibl = load("fig6_state_information_v2.json")
    high = load("nn_redundancy_high_dprime.json")
    tc = load("nn_redundancy_timecourse.json")
    exact = load("audit_fisher_same_task_exact.json")
    p12 = load("p12_partial_dissociation_exact.json")

    fig = plt.figure(figsize=(7.25, 5.45))
    gs = fig.add_gridspec(2, 3, left=.055, right=.985, bottom=.095, top=.96,
                          width_ratios=(1.12,.92,1.05), height_ratios=(1,1),
                          wspace=.43, hspace=.47)

    # a — Science-style measurement definition: covariance -> Fisher gap.
    host = fig.add_subplot(gs[0,0]); st.panel(host,"a",-.10,1.01); host.axis("off")
    ax = host.inset_axes([0,.10,.53,.82])
    means=[np.array([-.62,-.62]),np.array([.62,.62])]
    cov=np.array([[.18,.145],[.145,.18]]); cov_sh=np.diag(np.diag(cov))
    for mean,col in zip(means,(st.BIO,st.MODEL)):
        pts=RNG.multivariate_normal(mean,cov,95)
        ax.scatter(pts[:,0],pts[:,1],s=3,color=col,alpha=.18,edgecolor="none")
        ax.scatter(*mean,s=24,color=col,edgecolor="white",lw=.45,zorder=3)
        add_cov_ellipse(ax,mean,cov,st.ACTIVE,"-")
        add_cov_ellipse(ax,mean,cov_sh,"#E4A51C","--")
    ax.plot([-1.25,1.25],[-1.25,1.25],color=st.INK,lw=.7,ls=":")
    ax.set_xlabel("Neuron 1 response");ax.set_ylabel("Neuron 2 response")
    ax.set_xticks([]);ax.set_yticks([]);st.clean(ax)
    ax2 = host.inset_axes([.64,.14,.36,.74])
    n=np.asarray([4,8,16,24,32,48]); real=.010*np.sqrt(n); shuffled=.016*np.sqrt(n)
    ax2.plot(n,real,color=st.ACTIVE,marker="o",ms=2.7,label=r"$I_{real}$")
    ax2.plot(n,shuffled,color="#E4A51C",marker="o",ms=2.7,ls="--",label=r"$I_{shuffle}$")
    ax2.fill_between(n,real,shuffled,color=st.PALE_ACTIVE,alpha=.9,lw=0)
    ax2.set_xlabel("Population size");ax2.set_ylabel("Fisher information")
    ax2.set_xticks([]);ax2.set_yticks([]);st.clean(ax2)
    host.legend(handles=[Line2D([0],[0],color=st.ACTIVE,label="Intact covariance"),
                         Line2D([0],[0],color="#E4A51C",ls="--",label="Trial-shuffled"),
                         Line2D([],[],ls="",label=r"$I_{red}=I_{shuffle}-I_{real}$")],
                frameon=False,loc="lower center",bbox_to_anchor=(.50,.99),ncol=1,
                handletextpad=.4,fontsize=6.2)

    # b — common Fisher-redundancy state effect, with independent clusters shown.
    ax=fig.add_subplot(gs[0,1]);st.panel(ax,"b")
    qside=ibl["second_order"]["side"]["20"]["fisher"]
    qcon=ibl["second_order"]["contrast"]["20"]["fisher"]
    qall=allen["summary"]["20"]["fisher"]["animal"]
    ibl_side = session_deltas(exact["rows"], "fisher_a", "fisher_p",
                              {"axis":"side", "n_units":20})
    ibl_contrast = session_deltas(exact["rows"], "fisher_a", "fisher_p",
                                  {"axis":"contrast", "n_units":20})
    allen_by_animal = defaultdict(list)
    for row in allen["rows"]:
        if row["n_units"] == 20:
            allen_by_animal[row["animal"]].append(row["fisher_a"]-row["fisher_p"])
    allen_delta = np.asarray([np.mean(v) for v in allen_by_animal.values()])
    effects=[("IBL side",ibl_side), ("IBL contrast",ibl_contrast),
             ("Allen",allen_delta)]
    yy=np.arange(3)[::-1]
    for i,(y,(label,values)) in enumerate(zip(yy,effects)):
        raincloud_h(ax, values, y, st.ACTIVE, width=.34, seed=20+i,
                    alpha=.13 if len(values)>100 else .32)
    ax.axvline(0,color=st.INK,lw=.7,ls="--")
    ax.set_yticks(yy,[e[0] for e in effects]);ax.set_xlabel("Active - passive\nFisher redundancy")
    ax.legend(handles=[Line2D([0],[0],marker="o",ls="",color=st.ACTIVE,
                              label="Clusters; mean and 95% CI")],frameon=False,loc="lower right")
    ax.get_legend().set_bbox_to_anchor((1.0,1.01),transform=ax.transAxes)
    st.clean(ax)

    # c — actual passive and active animal means (not zero-referenced deltas).
    ax=fig.add_subplot(gs[0,2]);st.panel(ax,"c")
    by=defaultdict(lambda:[[],[]])
    for r in allen["rows"]:
        if r["n_units"]==20:
            by[r["animal"]][0].append(r["fisher_p"]);by[r["animal"]][1].append(r["fisher_a"])
    mat=np.asarray([[np.mean(v[0]),np.mean(v[1])] for _,v in sorted(by.items())])
    for row in mat: ax.plot([0,1],row,color=st.LIGHT,lw=.65,zorder=1)
    ax.scatter(np.zeros(len(mat)),mat[:,0],s=10,color=st.PASSIVE,alpha=.55,edgecolor="white",lw=.25,zorder=2)
    ax.scatter(np.ones(len(mat)),mat[:,1],s=10,color=st.ACTIVE,alpha=.55,edgecolor="white",lw=.25,zorder=2)
    for x,col in ((0,st.PASSIVE),(1,st.ACTIVE)):
        m,lo,hi=mean_ci(mat[:,x]);ax.errorbar(x,m,yerr=[[m-lo],[hi-m]],fmt="o",color=col,
                                              ms=5.2,capsize=2.4,zorder=4)
    ax.set_xticks([0,1],["Passive","Active"]);ax.set_ylabel("Fisher redundancy")
    ax.legend(handles=[Line2D([0],[0],color=st.LIGHT,label="Animals; N=20"),
                       Line2D([0],[0],marker="o",ls="",color=st.ACTIVE,label="Mean; 95% CI")],
              frameon=False,loc="upper left")
    st.clean(ax)

    # d — high- versus low-information session distributions.
    ax=fig.add_subplot(gs[1,0]);st.panel(ax,"d")
    entries=[]
    for axisn in ("side","contrast"):
        for group in ("high","low"):
            values=session_means(high["rows"], "fisher_delta",
                                 {"axis":axisn,"group":group})
            entries.append((axisn,group,values))
    yy=np.arange(4)[::-1]
    for i,(y,(axisn,group,values)) in enumerate(zip(yy,entries)):
        col=st.ACTIVE if group=="high" else st.CONTROL
        raincloud_h(ax,values,y,col,width=.30,seed=50+i,alpha=.12)
    ax.axvline(0,color=st.INK,lw=.7,ls="--")
    ax.set_yticks(yy,["Side high d′","Side low d′","Contrast high d′","Contrast low d′"])
    ax.set_xlabel("Active - passive\nFisher redundancy")
    ax.legend(handles=[Line2D([0],[0],marker="o",ls="",color=st.ACTIVE,label="High d′; 95% CI"),
                       Line2D([0],[0],marker="o",ls="",color=st.CONTROL,label="Low d′; 95% CI")],
              frameon=False,loc="lower right")
    st.clean(ax)

    # e — within-trial accumulation shown directly as active-minus-passive.
    ax=fig.add_subplot(gs[1,1]);st.panel(ax,"e")
    times=np.asarray([50,150,250,350])
    for axisn,col,marker in (("side",st.MODEL,"o"),("contrast",st.ACTIVE,"s")):
        means=[];los=[];his=[]
        for b in range(4):
            grouped=defaultdict(list)
            for r in tc["rows"]:
                if r["axis"]==axisn and r["bin"]==b: grouped[r["session"]].append(r["delta"])
            values=[np.mean(v) for v in grouped.values()]
            m,lo,hi=mean_ci(values);means.append(m);los.append(lo);his.append(hi)
        means,los,his=map(np.asarray,(means,los,his))
        ax.fill_between(times,los,his,color=col,alpha=.13,lw=0)
        ax.errorbar(times,means,yerr=[means-los,his-means],color=col,marker=marker,
                    ms=3.5,capsize=1.8,label=axisn.capitalize()+"; 95% CI")
    ax.axhline(0,color=st.INK,lw=.7,ls="--")
    ax.set_xlabel("Time from stimulus onset (ms)");ax.set_ylabel("Active - passive sharing")
    ax.legend(frameon=False,loc="upper left",bbox_to_anchor=(0,1.045));st.clean(ax)

    # f — exact-matched cross-path audit in the original raincloud idiom. This
    # replaces the superseded grouped-prediction cache while retaining the
    # approved visual structure. Bootstrap dots depict the session-cluster
    # resampling distribution, not additional independent sessions.
    ax=fig.add_subplot(gs[1,2]);st.panel(ax,"f")
    exact_rows = exact["rows"]
    ypos = {("side","d_ps"):1.16, ("side","d_lcr"):.84,
            ("contrast","d_ps"):.16, ("contrast","d_lcr"):-.16}
    for i, (axis_name, x_name, color, marker) in enumerate((
            ("side", "d_ps", st.MODEL, "o"),
            ("side", "d_lcr", st.PASSIVE, "s"),
            ("contrast", "d_ps", st.MODEL, "o"),
            ("contrast", "d_lcr", st.PASSIVE, "s"))):
        observed, boots = exact_crosspath_bootstrap(
            exact_rows, axis_name, x_name, n_boot=3000,
            seed=20260826 + i)
        bootstrap_raincloud_h(ax, boots, observed,
                              ypos[(axis_name, x_name)], color,
                              marker=marker, width=.22, seed=90+i)
    ax.axvline(0,color=st.INK,lw=.7,ls="--")
    ax.set_yticks([1,0], ["Side", "Contrast"])
    ax.set_ylim(-.48, 1.48)
    ax.set_xlabel("Correlation with Δ task-axis sharing")
    ax.legend(handles=[
        Line2D([0],[0],marker="o",ls="",color=st.MODEL,
               markeredgecolor="white",label="ΔPS; 95% CI"),
        Line2D([0],[0],marker="s",ls="",color=st.PASSIVE,
               markeredgecolor="white",label="ΔLCR; 95% CI")],
        frameon=False,loc="lower right",bbox_to_anchor=(1.0,1.01))
    st.clean(ax)

    save(fig);plt.close(fig)


if __name__ == "__main__":
    build()
