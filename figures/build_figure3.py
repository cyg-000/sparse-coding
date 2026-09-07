import os
from pathlib import Path
ROOT = Path(os.environ.get('SPARSE_CODING_ROOT', Path(__file__).resolve().parents[1]))
"""Build the final Figure 3: intervention -> phase trajectory -> counterfactual -> geometry.

The script uses cached model-level statistics for panels b/c and derives the
four matched cosine-distance matrices in panel d from saved checkpoints.  The
derived matrices are cached in figmain/derived so rerendering does not require
another model forward pass.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import t

HERE = Path(__file__).parent

FINAL = ROOT / "figures" / "output"
DERIVED = ROOT / "figures" / "derived"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "manu" / "02_analyses" / "00_shared"))
import style as st

st.setup()
RNG = np.random.default_rng(20260822)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mean_ci(values):
    a = np.asarray(values, float)
    m = np.nanmean(a)
    if len(a) < 2:
        return m, m, m
    half = t.ppf(.975, len(a)-1) * np.nanstd(a, ddof=1) / np.sqrt(len(a))
    return m, m-half, m+half


def save(fig):
    FINAL.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(FINAL / f"fig3_causality_final.{ext}", dpi=450,
                    bbox_inches="tight", facecolor="white")


def support_grid(ax, center, colors, sizes=None, dx=.050, dy=.064):
    cx, cy = center
    sizes = np.full(24, 31.0) if sizes is None else np.asarray(sizes, float)
    for i, (fc, size) in enumerate(zip(colors, sizes)):
        rr, cc = divmod(i, 6)
        ax.scatter(cx+(cc-2.5)*dx, cy+(1.5-rr)*dy, s=size, c=[fc],
                   edgecolors="white", linewidths=.35, zorder=3)


def cosine_distance(R):
    R = np.asarray(R, float)
    Rc = R - R.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(Rc, axis=1, keepdims=True)
    norm[norm == 0] = 1e-10
    return 1 - (Rc/norm) @ (Rc/norm).T


def condition_means(H, labels):
    return np.stack([H[labels == cat].mean(axis=0) for cat in np.unique(labels)])


def geometry_matrices(k=4, seed=42):
    cache = DERIVED / f"fig3_geometry_matrices_k{k}_seed{seed}.npz"
    if cache.exists():
        z = np.load(cache)
        return [z[name] for name in ("dense", "trained", "posthoc", "random")], z["order"]

    import torch
    from analysis_utils import load_k_gradient_model, load_stringer_data

    DERIVED.mkdir(parents=True, exist_ok=True)
    X, labels, _ = load_stringer_data()
    dense_model = load_k_gradient_model(256, seed)
    trained_model = load_k_gradient_model(k, seed)
    with torch.no_grad():
        _, H_dense_t, H2_dense_t = dense_model(X)
        _, H_trained_t, _ = trained_model(X)
    H_dense = H_dense_t.cpu().numpy()
    H2_dense = H2_dense_t.cpu().numpy()
    H_trained = H_trained_t.cpu().numpy()

    # Post-hoc: strongest K units from a densely trained representation.
    idx = np.argpartition(H2_dense, -k, axis=1)[:, -k:]
    H_post = np.zeros_like(H2_dense)
    np.put_along_axis(H_post, idx, np.take_along_axis(H2_dense, idx, axis=1), axis=1)

    # Random-K: average five independently masked geometry matrices, matching
    # the number of random repetitions used in cache_random_sparsity.json.
    random_dist = []
    for _ in range(5):
        H_rand = np.zeros_like(H2_dense)
        for i, row in enumerate(H2_dense):
            active = np.flatnonzero(row > 0)
            chosen = active if len(active) <= k else RNG.choice(active, k, replace=False)
            H_rand[i, chosen] = row[chosen]
        random_dist.append(cosine_distance(condition_means(H_rand, labels)))

    matrices = [cosine_distance(condition_means(H_dense, labels)),
                cosine_distance(condition_means(H_trained, labels)),
                cosine_distance(condition_means(H_post, labels)),
                np.mean(random_dist, axis=0)]
    # A single ordering is learned from the trained Top-K matrix and then held
    # fixed for all four panels; this is visualization only, not an inference.
    order = leaves_list(linkage(squareform(matrices[1], checks=False),
                                method="average", optimal_ordering=True))
    np.savez_compressed(cache, dense=matrices[0], trained=matrices[1],
                        posthoc=matrices[2], random=matrices[3], order=order)
    return matrices, order


def build():
    rows = load(ROOT / "models" / "k_gradient_all.json")
    post = load(ROOT / "cache" / "cache_entropy_control.json")["results"]
    rand = load(ROOT / "cache" / "cache_random_sparsity.json")["results"]
    ks = np.asarray(sorted({r["k"] for r in rows}))
    trained = {(r["k"], r["seed"]): r for r in rows}

    fig = plt.figure(figsize=(7.25, 5.35))
    gs = fig.add_gridspec(2, 2, left=.055, right=.985, bottom=.09, top=.965,
                          width_ratios=(.84, 1.16), height_ratios=(.92, 1.08),
                          wspace=.34, hspace=.42)

    # Manual panel labels in FIGURE coordinates: same-row cells share y1, so a/b
    # and c/d align exactly; d's label is placed at the gs[1,1] cell top-left
    # rather than a nested subplot's axes.
    def place_label(cell, letter, x_off=0.012, y_off=0.012):
        pos = cell.get_position(fig)
        fig.text(pos.x0 + x_off, pos.y1 - y_off, letter,
                 fontweight="bold", fontsize=9.5, va="top", ha="left")
    place_label(gs[0, 0], "a")
    place_label(gs[0, 1], "b")
    place_label(gs[1, 0], "c")
    place_label(gs[1, 1], "d")

    # a — matched intervention design (content shifted down ~2 letter-heights).
    ga = gs[0, 0].subgridspec(2, 1, height_ratios=(0.12, 1.0), hspace=0)
    ax = fig.add_subplot(ga[1]); ax.axis("off")
    dense_level = np.array([.16,.31,.68,.42,.83,.24,.55,.21,.76,.48,.64,.35,
                            .28,.72,.39,.89,.51,.18,.62,.37,.79,.30,.57,.44])
    dense_cols = [plt.cm.Greys(.18+.57*v) for v in dense_level]
    support_grid(ax, (.50,.78), dense_cols, 23+28*dense_level)
    strongest = set(np.argsort(dense_level)[-5:])
    trained_set = {2,4,8,15,20}
    random_set = {1,7,11,18,23}
    pale = "#EDF0F2"
    support_grid(ax, (.20,.28), [st.MODEL if i in trained_set else pale for i in range(24)])
    support_grid(ax, (.50,.28), [st.BIO if i in strongest else pale for i in range(24)])
    support_grid(ax, (.80,.28), [st.CONTROL if i in random_set else pale for i in range(24)])
    for x in (.20,.50,.80):
        ax.add_patch(FancyArrowPatch((.50,.64),(x,.42),arrowstyle="-|>",
                                     mutation_scale=8,color=st.INK,lw=.75))
    ax.legend(handles=[Line2D([0],[0],marker="o",ls="",color=st.MODEL,label="Trained Top-K"),
                       Line2D([0],[0],marker="o",ls="",color=st.BIO,label="Post-hoc Top-K"),
                       Line2D([0],[0],marker="o",ls="",color=st.CONTROL,label="Random-K")],
              frameon=False,loc="upper center",bbox_to_anchor=(.5,.03),ncol=3,
              columnspacing=.65,handletextpad=.25,fontsize=6.2)
    ax.set_xlim(0,1); ax.set_ylim(0,1)

    # b — joint dose-response trajectory in PS-LCR space (shifted down ~2 letter-heights).
    gb = gs[0, 1].subgridspec(2, 1, height_ratios=(0.12, 1.0), hspace=0)
    ax = fig.add_subplot(gb[1])
    pm=[]; lm=[]; plo=[]; phi=[]; llo=[]; lhi=[]
    for k in ks:
        p=[r["ps"] for r in rows if r["k"]==k]
        l=[r["nnr"] for r in rows if r["k"]==k]
        p0,p1,p2=mean_ci(p); l0,l1,l2=mean_ci(l)
        pm.append(p0);plo.append(p1);phi.append(p2);lm.append(l0);llo.append(l1);lhi.append(l2)
        ax.scatter(p,l,s=8,color=st.MODEL,alpha=.20,edgecolor="none",zorder=1)
    pm,lm,plo,phi,llo,lhi=map(np.asarray,(pm,lm,plo,phi,llo,lhi))
    logk=np.log2(ks)
    sc=ax.scatter(pm,lm,c=logk,cmap="viridis",s=36,edgecolor="white",lw=.45,zorder=4)
    ax.plot(pm,lm,color=st.MODEL,lw=1.0,zorder=2)
    ax.errorbar(pm,lm,xerr=[pm-plo,phi-pm],yerr=[lm-llo,lhi-lm],fmt="none",
                ecolor=st.MODEL,lw=.7,capsize=1.7,alpha=.72,zorder=3)
    ax.set_xlabel("Population sparseness"); ax.set_ylabel("Local clustering ratio")
    cb=fig.colorbar(sc,ax=ax,fraction=.042,pad=.025); cb.set_ticks(logk); cb.set_ticklabels(ks)
    cb.set_label("Active units, K")
    ax.legend(handles=[Line2D([0],[0],marker="o",ls="",color=st.MODEL,
                              label="K mean; 95% CI")],frameon=False,loc="upper right")
    st.clean(ax)

    # c — paired counterfactual effects at matched K (shifted down ~2 letter-heights).
    gc = gs[1, 0].subgridspec(2, 1, height_ratios=(0.12, 1.0), hspace=0)
    ax = fig.add_subplot(gc[1])
    common=np.asarray([2,4,8,16,32,64,128]); xpos=np.log2(common)
    post_by={(r["target_k"],r["seed"]):r["nnr"] for r in post}
    rand_by={(r["k"],r["seed"]):r["nnr_rand_mean"] for r in rand}
    for source,col,marker,label,offset in [
            (post_by,st.BIO,"o","Post-hoc - trained",-.055),
            (rand_by,st.CONTROL,"s","Random - trained",.055)]:
        means=[];los=[];his=[]
        for x,k in zip(xpos,common):
            vals=np.asarray([source[(k,seed)]-trained[(k,seed)]["nnr"]
                             for seed in sorted({r["seed"] for r in rows})])
            ax.scatter(np.full(len(vals),x+offset)+RNG.normal(0,.018,len(vals)),vals,
                       s=7,color=col,alpha=.28,edgecolor="none",zorder=1)
            m,lo,hi=mean_ci(vals);means.append(m);los.append(lo);his.append(hi)
        means,los,his=map(np.asarray,(means,los,his))
        ax.errorbar(xpos+offset,means,yerr=[means-los,his-means],fmt=marker+"-",
                    color=col,ms=3.8,lw=1,capsize=1.8,label=label+"; 95% CI",zorder=3)
    ax.axhline(0,color=st.INK,lw=.7,ls="--")
    ax.set_xticks(xpos,common); ax.set_xlabel("Active units, K")
    ax.set_ylabel("Control - trained LCR")
    ax.legend(frameon=False,loc="upper right")
    st.clean(ax)

    # d — matched representational distance matrices at K=4.
    outer = gs[1, 1].subgridspec(1, 5, width_ratios=(1,1,1,1,.075),wspace=.10)
    matrices, order = geometry_matrices(k=4, seed=42)
    ordered=[m[np.ix_(order,order)] for m in matrices]
    off=np.concatenate([m[~np.eye(m.shape[0],dtype=bool)] for m in ordered])
    vmin,vmax=np.quantile(off,[.03,.97])
    names=["Dense","Trained Top-K","Post-hoc","Random-K"]
    im=None
    for i,(m,name) in enumerate(zip(ordered,names)):
        sub=fig.add_subplot(outer[0,i]);

        im=sub.imshow(m,cmap="magma",vmin=vmin,vmax=vmax,interpolation="nearest",aspect="equal")
        sub.set_xticks([]);sub.set_yticks([]);sub.set_xlabel(name,labelpad=3)
        for spine in sub.spines.values(): spine.set_visible(False)
    cax=fig.add_subplot(outer[0,4]); cb=fig.colorbar(im,cax=cax);cb.set_label("Cosine distance")

    save(fig)
    plt.close(fig)


if __name__ == "__main__":
    build()
