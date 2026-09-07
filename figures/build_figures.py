import os
from pathlib import Path
ROOT = Path(os.environ.get('SPARSE_CODING_ROOT', Path(__file__).resolve().parents[1]))
import json
import base64
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from scipy.stats import t

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import style as st

st.setup()
ROOT = st.ROOT
OUT = ROOT / "outputs"
MAIN = ROOT / "figures" / "output"
RNG = np.random.default_rng(20260822)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mean_ci(values, axis=0):
    a = np.asarray(values, float)
    n = a.shape[axis]
    m = np.nanmean(a, axis=axis)
    se = np.nanstd(a, axis=axis, ddof=1) / np.sqrt(max(n, 1))
    q = t.ppf(.975, max(n - 1, 1))
    return m, m - q * se, m + q * se


def save(fig, name, match_svg_to_png=False):
    MAIN.mkdir(parents=True, exist_ok=True)
    png_path = MAIN / f"{name}.png"
    fig.savefig(png_path, bbox_inches="tight", facecolor="white", dpi=450)
    fig.savefig(MAIN / f"{name}.pdf", bbox_inches="tight", facecolor="white", dpi=450)
    if match_svg_to_png:
        # Keep an editable vector source, while making the official SVG a
        # pixel-exact wrapper around the approved PNG rendering.
        fig.savefig(MAIN / f"{name}_vector.svg", bbox_inches="tight", facecolor="white")
        from PIL import Image
        with Image.open(png_path) as im:
            width, height = im.size
        encoded = base64.b64encode(png_path.read_bytes()).decode("ascii")
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" '
               f'xmlns:xlink="http://www.w3.org/1999/xlink" width="{width}" '
               f'height="{height}" viewBox="0 0 {width} {height}">'
               f'<image width="{width}" height="{height}" '
               f'href="data:image/png;base64,{encoded}"/></svg>')
        (MAIN / f"{name}.svg").write_text(svg, encoding="utf-8")
    else:
        fig.savefig(MAIN / f"{name}.svg", bbox_inches="tight", facecolor="white")


def regression_ci(ax, x, y, color=st.INK, nboot=2500):
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    xx = np.linspace(x.min(), x.max(), 140)
    fits = []
    for _ in range(nboot):
        ii = RNG.integers(0, len(x), len(x))
        if np.ptp(x[ii]) > 0:
            fits.append(np.polyval(np.polyfit(x[ii], y[ii], 1), xx))
    fits = np.asarray(fits)
    lo, hi = np.quantile(fits, [.025, .975], axis=0)
    ax.fill_between(xx, lo, hi, color=color, alpha=.11, lw=0, zorder=1)
    ax.plot(xx, np.polyval(np.polyfit(x, y, 1), xx), color=color, lw=1.25, zorder=2)


def style_3d(ax):
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor(st.LIGHT)
    ax.tick_params(length=0, pad=-2)


def draw_support(ax, offset=(0, 0), scale=1.0, active_a=None, active_b=None, gap=None):
    active_a = set(active_a or [])
    active_b = set(active_b or [])
    ox, oy = offset
    gap = scale if gap is None else gap          # spacing can differ from circle size
    for i in range(24):
        r, c = divmod(i, 6)
        # anchor at top-left (ox, oy): reducing gap compresses the grid toward it
        x, y = ox + c * gap, oy + (3 - r) * gap
        if i in active_a and i in active_b:
            left = Circle((x, y), .18 * scale, fc=st.PASSIVE, ec=st.INK, lw=.35)
            right = Circle((x, y), .18 * scale, fc=st.ACTIVE, ec=st.INK, lw=.35)
            ax.add_patch(left); ax.add_patch(right)
            right.set_clip_path(Rectangle((x, y-.22*scale), .25*scale, .44*scale, transform=ax.transData))
        elif i in active_a:
            ax.add_patch(Circle((x, y), .18 * scale, fc=st.PASSIVE, ec=st.INK, lw=.35))
        elif i in active_b:
            ax.add_patch(Circle((x, y), .18 * scale, fc=st.ACTIVE, ec=st.INK, lw=.35))
        else:
            ax.add_patch(Circle((x, y), .18 * scale, fc=st.CONTROL, ec=st.INK, lw=.30))


def cloud3d(ax, center, color, cov=None, n=170, alpha=.25):
    cov = cov if cov is not None else np.diag([.07, .045, .05])
    pts = RNG.multivariate_normal(np.asarray(center), cov, n)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2.2, color=color, alpha=alpha, depthshade=False)
    ax.scatter(*center, s=28, color=color, edgecolor=st.INK, lw=.55, depthshade=False)


def covariance_shell(ax, center, cov, color, scale=2.15):
    u=np.linspace(0,2*np.pi,18); v=np.linspace(0,np.pi,10)
    sphere=np.stack([np.outer(np.cos(u),np.sin(v)),
                     np.outer(np.sin(u),np.sin(v)),
                     np.outer(np.ones_like(u),np.cos(v))],axis=-1)
    vals,vecs=np.linalg.eigh(np.asarray(cov))
    transform=vecs@np.diag(np.sqrt(np.maximum(vals,1e-8)))
    shell=sphere@transform.T*scale+np.asarray(center)
    ax.plot_wireframe(shell[:,:,0],shell[:,:,1],shell[:,:,2],rstride=3,cstride=3,
                      color=color,lw=.45,alpha=.48)


def fig1():
    fig = plt.figure(figsize=(7.80, 3.45))
    gs = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.28, 1.82), left=.025, right=.99,
                          bottom=.11, top=.96, wspace=.18)

    # Manual panel labels in FIGURE coordinates: a/b/c all share gs row 0, so the
    # y1 (top edge) is identical -> letters align on one horizontal line.
    def place_label(cell, letter, x_off=0.012, y_off=0.012):
        pos = cell.get_position(fig)
        fig.text(pos.x0 + x_off, pos.y1 - y_off, letter,
                 fontweight="bold", fontsize=9.5, va="top", ha="left")
    place_label(gs[0, 0], "a")
    place_label(gs[0, 1], "b")
    place_label(gs[0, 2], "c")

    # a: source-style sparse substrate plus compact population geometry
    sg = gs[0, 0].subgridspec(2, 1, height_ratios=(1.0, 1.2), hspace=.03)
    ax = fig.add_subplot(sg[0]); ax.axis("off")
    A={0,6,11,14,19}; B={2,8,11,14,20}
    draw_support(ax, (.15, .1), .58, A, B, gap=.435)   # spacing ×0.75, anchor top-left
    ax.set_xlim(-.25, 3.55); ax.set_ylim(-.2, 2.15)
    ax.legend(handles=[Line2D([0],[0],marker="o",ls="",color=st.PASSIVE,label="A"),
                       Line2D([0],[0],marker="o",ls="",color=st.ACTIVE,label="B"),
                       Line2D([0],[0],marker="o",ls="",markerfacecolor=st.MODEL,color=st.MODEL,label="A ∩ B")],
              loc="lower center", bbox_to_anchor=(.52,-.11), ncol=3, frameon=False,
              handletextpad=.25, columnspacing=.55)
    ax3 = fig.add_subplot(sg[1], projection="3d")
    lower_pos = ax3.get_position()
    ax3.set_position([lower_pos.x0, lower_pos.y0-.035,
                      lower_pos.width, lower_pos.height])
    ax3.patch.set_alpha(0)
    cloud3d(ax3, (-.55, 0, 0), st.PASSIVE, n=120)
    cloud3d(ax3, (.55, 0, 0), st.ACTIVE, n=120)
    ax3.plot([-.55,.55],[0,0],[0,0],color=st.INK,lw=1)
    ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_zticks([]); ax3.view_init(22,-55); style_3d(ax3)

    # b: multiple compact clusters plus their shared sparse supports
    sgb = gs[0, 1].subgridspec(2,1,height_ratios=(.78,.22),hspace=.01)
    ax = fig.add_subplot(sgb[0], projection="3d")
    centers=[(-.85,0,0),(.75,.05,.05),(-.15,.85,.45),(.15,-.82,-.3),(.88,.7,-.2),(-.8,-.65,.22)]
    colors=[st.PASSIVE,st.ACTIVE,st.MODEL,"#9BCB7A","#E7B6C7","#9FC4DA"]
    for c,col in zip(centers,colors): cloud3d(ax,c,col,n=130,alpha=.18)
    ax.plot([centers[0][0],centers[1][0]],[0,.05],[0,.05],color=st.INK,lw=1)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.view_init(23,-51); style_3d(ax)
    axs=fig.add_subplot(sgb[1]); axs.axis("off")
    support_sets=[({0,6,11,14},{2,8,11,14}),({1,7,12,18},{3,9,12,18}),
                  ({2,8,13,19},{4,10,13,19}),({3,9,14,20},{5,11,14,20})]
    for j,(aa,bb) in enumerate(support_sets):
        for idx in range(24):
            rr,cc=divmod(idx,6); x=.10+j*.78+cc*.075; y=.08+(3-rr)*.075
            fc=st.LIGHT
            if idx in aa and idx in bb: fc=st.MODEL
            elif idx in aa: fc=st.PASSIVE
            elif idx in bb: fc=st.ACTIVE
            axs.scatter(x,y,s=12,c=fc,edgecolors=st.INK,linewidths=.25)
    axs.set_xlim(-.02,3.05);axs.set_ylim(-.02,.43)

    # c: same means, passive spherical covariance versus active task-aligned covariance
    sgc = gs[0, 2].subgridspec(2, 2, width_ratios=(.42,1.0), hspace=.02, wspace=.02)
    for row,state in enumerate(("Passive","Active")):
        axs=fig.add_subplot(sgc[row,0]);
        pass  # c label placed in figure coords above
        draw_support(axs,(.1,.25),.45,A,B)
        axs.axis("off"); axs.set_xlim(-.25,2.65); axs.set_ylim(-.05,1.85)
        # Preserve the circular neuron glyphs; the wider panel prevents the
        # equal data aspect from squeezing the 6-by-4 array.
        axs.set_aspect("equal", adjustable="box")
        ax3=fig.add_subplot(sgc[row,1],projection="3d")
        if state=="Passive":
            cov=np.diag([.030,.026,.026])
        else:
            u=np.array([1,.8,.5]); u=u/np.linalg.norm(u); cov=.014*np.eye(3)+.34*np.outer(u,u)
        cloud3d(ax3,(-.52,0,0),st.PASSIVE,cov,n=170,alpha=.24)
        cloud3d(ax3,(.52,0,0),st.ACTIVE,cov,n=170,alpha=.24)
        covariance_shell(ax3,(-.52,0,0),cov,st.PASSIVE)
        covariance_shell(ax3,(.52,0,0),cov,st.ACTIVE)
        ax3.plot([-.52,.52],[0,0],[0,0],color=st.INK,lw=1)
        if state=="Active": ax3.plot([-1.0,1.0],[-.8,.8],[-.5,.5],color=st.MODEL,lw=1.1,ls="--")
        ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_zticks([]); ax3.view_init(22,-54); style_3d(ax3)
    # Directional flow is part of the concept, but carries no explanatory prose.
    fig.add_artist(FancyArrowPatch((.255,.50),(.285,.50),transform=fig.transFigure,
                                   arrowstyle="-|>",mutation_scale=10,color=st.INK,lw=.8))
    fig.add_artist(FancyArrowPatch((.565,.50),(.595,.50),transform=fig.transFigure,
                                   arrowstyle="-|>",mutation_scale=10,color=st.INK,lw=.8))
    fig.legend(handles=[Line2D([0],[0],color=st.PASSIVE,lw=5,label="Passive"),
                        Line2D([0],[0],color=st.ACTIVE,lw=5,label="Active"),
                        Line2D([0],[0],color=st.MODEL,lw=1.3,ls="--",label="Task axis")],
               loc="lower right", bbox_to_anchor=(.955,.028), ncol=3, frameon=False)
    save(fig,"fig1_framework_final",match_svg_to_png=True); plt.close(fig)

def geometry_data():
    ps3=load(ROOT/"cache_ps3.json")["per_region"]
    stein=(np.array([r["ps_behav"] for r in ps3]),np.array([r["sep_behav"] for r in ps3]),None,None,None)
    ibl=[r for r in load(ROOT/"ibl_passive_results.json") if r.get("epoch")=="full_stim"]
    by=defaultdict(list)
    for r in ibl:
        if np.isfinite(r.get("ps",np.nan)) and np.isfinite(r.get("nnr",np.nan)):
            by[r["region"]].append((r["ps"],r["nnr"]))
    x3=np.array([np.mean(v,axis=0)[0] for v in by.values()])
    y3=np.array([np.mean(v,axis=0)[1] for v in by.values()])
    x3e=np.array([np.std(v,axis=0,ddof=1)[0]/np.sqrt(len(v)) if len(v)>1 else 0 for v in by.values()])
    y3e=np.array([np.std(v,axis=0,ddof=1)[1]/np.sqrt(len(v)) if len(v)>1 else 0 for v in by.values()])
    ibld=(x3,y3,x3e,y3e,None)
    sr=load(ROOT/"cache_stringer.json")["per_category"]
    string=(np.array([r["ps_mean"] for r in sr]),np.array([r["sil_mean"] for r in sr]),
            np.array([r.get("ps_sem",0) for r in sr]),np.array([r.get("sil_sem",0) for r in sr]),
            np.array([r["recording"] for r in sr]))
    mr=load(ROOT/"cache_monkey.json")["macaque"]["per_region"]
    maca=(np.array([r["ps"] for r in mr]),np.array([1-r["nsr"] for r in mr]),None,None,None)
    return stein,ibld,string,maca


def empirical_panel(ax, data, letter, color):
    x,y,xe,ye,group=data; st.panel(ax,letter)
    if xe is not None:
        ax.errorbar(x,y,xerr=xe,yerr=ye,fmt="none",ecolor=color,alpha=.24,lw=.5,capsize=1)
    ax.scatter(x,y,s=16,color=color,alpha=.68,edgecolor="white",lw=.3,zorder=3)
    if group is None:
        regression_ci(ax,x,y,st.INK)
    else:
        for g in np.unique(group):
            k=group==g
            if k.sum()>2:
                o=np.argsort(x[k])
                ax.plot(x[k][o],np.polyval(np.polyfit(x[k],y[k],1),x[k][o]),color=st.INK,lw=.65,alpha=.42)
    ax.set_xlabel("Population sparseness"); ax.set_ylabel("Local clustering ratio"); st.clean(ax)


def fig3():
    fig=plt.figure(figsize=(7.25,4.65)); gs=fig.add_gridspec(2,3,width_ratios=(.72,1,1),left=.07,right=.99,bottom=.11,top=.95,wspace=.45,hspace=.48)
    ax=fig.add_subplot(gs[:,0]); st.panel(ax,"a",-.13,1.01)
    centers=np.array([[-.62,-.20],[.62,-.05],[-.08,.68]])
    covs=[[[.055,.014],[.014,.04]],[[.055,-.012],[-.012,.04]],[[.045,0],[0,.05]]]
    for c,cov,col in zip(centers,covs,[st.PASSIVE,st.ACTIVE,st.MODEL]):
        pts=RNG.multivariate_normal(c,cov,150)
        ax.scatter(pts[:,0],pts[:,1],s=5,color=col,alpha=.19,edgecolor="none")
        ax.scatter(*c,s=45,color=col,edgecolor=st.INK,lw=.6,zorder=4)
    ax.plot(centers[:2,0],centers[:2,1],color=st.CONTROL,ls="--",lw=1.15)
    ax.plot(centers[[0,2],0],centers[[0,2],1],color=st.MODEL,lw=1.35)
    ax.set_xlabel("Representation axis 1"); ax.set_ylabel("Representation axis 2")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal",adjustable="box"); st.clean(ax)
    ax.legend(handles=[Line2D([0],[0],color=st.MODEL,label=r"$d_{NN}$"),Line2D([0],[0],color=st.CONTROL,ls="--",label=r"$d_{other}$")],frameon=False,loc="lower center")
    datasets=geometry_data(); axes=[fig.add_subplot(gs[0,1]),fig.add_subplot(gs[0,2]),fig.add_subplot(gs[1,1]),fig.add_subplot(gs[1,2])]
    colors=[st.BIO,st.PASSIVE,st.BIO,st.BIO]
    for ax,d,l,c in zip(axes,datasets,"bcde",colors): empirical_panel(ax,d,l,c)
    axes[0].legend(handles=[Line2D([0],[0],marker="o",ls="",color=st.BIO,label="Steinmetz; N=54")],frameon=False,loc="upper right")
    axes[1].legend(handles=[Line2D([0],[0],marker="o",ls="",color=st.PASSIVE,label="IBL passive; N=321")],frameon=False,loc="upper right")
    axes[2].legend(handles=[Line2D([0],[0],marker="o",ls="",color=st.BIO,label="Stringer; N=5 recordings")],frameon=False,loc="upper right")
    axes[3].legend(handles=[Line2D([0],[0],marker="o",ls="",color=st.BIO,label="Macaque; N=6")],frameon=False,loc="upper right")
    save(fig,"fig3_geometry_v2"); plt.close(fig)


def error_curve(ax,x,groups,color,label,ls="-"):
    m=[];lo=[];hi=[]
    for g in groups:
        a,b,c=mean_ci(g);m.append(a);lo.append(b);hi.append(c)
    m,lo,hi=map(np.asarray,(m,lo,hi))
    ax.fill_between(x,lo,hi,color=color,alpha=.14,lw=0)
    ax.plot(x,m,color=color,marker="o",ms=3.3,ls=ls,label=label)
    ax.errorbar(x,m,yerr=[m-lo,hi-m],fmt="none",ecolor=color,lw=.7,capsize=1.8)


def fig4():
    rows=load(ROOT/"models"/"k_gradient_all.json"); ks=np.array(sorted({r["k"] for r in rows})); lx=np.log2(ks)
    p4=load(OUT/"p4_regime_capture.json"); p6=load(OUT/"p6_support_vs_magnitude.json"); abl=load(OUT/"p9_shared_ablation.json")
    p14=load(OUT/"p14_biological_null_model_v2.json")
    p15=load(OUT/"p15_biological_heldout_ablation.json")
    fig=plt.figure(figsize=(7.25,5.2)); gs=fig.add_gridspec(2,3,left=.06,right=.99,bottom=.09,top=.96,wspace=.40,hspace=.43,width_ratios=(.9,1,1))
    # a compact mechanism: dense activity -> hard Top-K support -> overlapping assemblies
    ax=fig.add_subplot(gs[0,0]); st.panel(ax,"a"); ax.axis("off")
    def support_grid(cx,cy,colors,dx=.043,dy=.061,size=36):
        for idx,fc in enumerate(colors):
            rr,cc=divmod(idx,6)
            ax.scatter(cx+(cc-2.5)*dx,cy+(1.5-rr)*dy,s=size,c=[fc],
                       edgecolors="white",linewidths=.35,zorder=3)
    # Dense: same population, graded activity rather than binary occupancy.
    dense_level=np.array([.18,.35,.62,.42,.78,.28,.55,.25,.83,.47,.68,.34,
                          .30,.72,.40,.88,.52,.20,.64,.38,.76,.32,.58,.46])
    dense_cols=[plt.cm.Greys(.18+.58*v) for v in dense_level]
    support_grid(.17,.53,dense_cols)
    # Top-K: a hard support mask over the identical neuron coordinates.
    selected={2,4,8,10,15,18,20,22}
    top_cols=[st.MODEL if i in selected else "#EDF0F2" for i in range(24)]
    support_grid(.50,.53,top_cols)
    # Two condition-specific supports share a stable subset of neurons.
    A={1,4,7,10,13,16,20,22}; B={2,4,8,10,14,17,20,23}; shared=A&B
    def assembly_cols(active,own):
        return [st.MODEL if i in shared else (own if i in active else "#EDF0F2") for i in range(24)]
    support_grid(.84,.67,assembly_cols(A,st.PASSIVE),dx=.034,dy=.043,size=25)
    support_grid(.84,.39,assembly_cols(B,st.ACTIVE),dx=.034,dy=.043,size=25)
    for x0,x1 in ((.29,.39),(.62,.70)):
        ax.add_patch(FancyArrowPatch((x0,.53),(x1,.53),arrowstyle="-|>",
                                     mutation_scale=9,color=st.INK,lw=.8))
    ax.legend(handles=[Line2D([0],[0],marker="o",ls="",markerfacecolor=st.CONTROL,
                              markeredgecolor="none",label="Dense activity"),
                       Line2D([0],[0],marker="o",ls="",color=st.MODEL,label="Top-K support"),
                       Line2D([0],[0],marker="o",ls="",color=st.PASSIVE,label="Assembly A"),
                       Line2D([0],[0],marker="o",ls="",color=st.ACTIVE,label="Assembly B"),
                       Line2D([0],[0],marker="o",ls="",color=st.MODEL,label="Shared")],
              frameon=False,loc="lower center",ncol=2,columnspacing=.7,handletextpad=.25)
    ax.set_xlim(0,1);ax.set_ylim(0,1)
    # b biological support-identity matched null
    ax=fig.add_subplot(gs[0,1]); st.panel(ax,"b")
    p14rows=p14["rows"]
    ps=np.asarray([r["ps"] for r in p14rows],float)
    real=np.asarray([r["lcr_real"] for r in p14rows],float)
    null=np.asarray([r["lcr_degree_null_mean"] for r in p14rows],float)
    sessions=np.asarray([r["session"] for r in p14rows],object)
    keep=np.isfinite(ps)&np.isfinite(real)&np.isfinite(null)
    offset=.0032
    for x,a,b in zip(ps[keep],real[keep],null[keep]):
        ax.plot([x-offset,x+offset],[a,b],color="#D5DBDF",lw=.55,alpha=.62,zorder=0)
    ax.scatter(ps[keep]-offset,real[keep],s=15,color=st.BIO,alpha=.80,
               edgecolor="white",linewidth=.25,label="Observed",zorder=3)
    ax.scatter(ps[keep]+offset,null[keep],s=15,color="#A9B0B7",alpha=.74,
               edgecolor="white",linewidth=.25,label="Matched null",zorder=2)
    def clustered_line(x,y,session,color):
        valid=np.isfinite(x)&np.isfinite(y)
        x,y,session=x[valid],y[valid],session[valid]
        grid=np.linspace(.47,.96,150)
        ax.plot(grid,np.polyval(np.polyfit(x,y,1),grid),color=color,lw=1.35,zorder=5)
        uniq=np.unique(session); by={s:np.flatnonzero(session==s) for s in uniq}; pred=[]
        for _ in range(2500):
            draw=RNG.choice(uniq,len(uniq),replace=True)
            idx=np.concatenate([by[s] for s in draw])
            if np.unique(x[idx]).size>=2:
                pred.append(np.polyval(np.polyfit(x[idx],y[idx],1),grid))
        lo,hi=np.quantile(np.asarray(pred),[.025,.975],axis=0)
        ax.fill_between(grid,lo,hi,color=color,alpha=.13,lw=0,zorder=1)
    clustered_line(ps,real,sessions,st.BIO)
    clustered_line(ps,null,sessions,"#A9B0B7")
    ax.set_xlim(.47,.96);ax.set_ylim(.18,.98)
    ax.set_xlabel("Population sparseness");ax.set_ylabel("Local clustering ratio");st.clean(ax)
    ax.legend(frameon=False,fontsize=6.4,loc="lower left",bbox_to_anchor=(0,1.01),
              ncol=2,borderaxespad=0,handletextpad=.3,columnspacing=.75)
    # c support/magnitude counterfactual
    ax=fig.add_subplot(gs[0,2]);st.panel(ax,"c")
    data=[[r[k] for r in p6] for k in ("nnr_orig","nnr_sup","nnr_mag")]; xpos=np.arange(3)
    for i,v in enumerate(data):
        jitter=RNG.normal(0,.035,len(v)); ax.scatter(np.full(len(v),i)+jitter,v,s=8,color=st.CONTROL,alpha=.3,edgecolor="none")
        m,lo,hi=mean_ci(v); ax.errorbar(i,m,yerr=[[m-lo],[hi-m]],fmt="o",color=[st.MODEL,st.PASSIVE,st.ACTIVE][i],ms=4.5,capsize=2.4)
    ax.set_xticks(xpos,["Original","Support\nshuffle","Magnitude\nshuffle"]);ax.set_ylabel("Local clustering ratio");st.clean(ax)
    # d progressive RDM recovery
    ax=fig.add_subplot(gs[1,0]);st.panel(ax,"d")
    retain=np.asarray(p4["retain"]); curves=np.asarray([r["rdm_recovery"] for r in p4["rows"]]);m,lo,hi=mean_ci(curves,axis=0)
    ax.fill_between(retain,lo,hi,color=st.MODEL,alpha=.15,lw=0);ax.plot(retain,m,color=st.MODEL,marker="o",ms=3);ax.errorbar(retain,m,yerr=[m-lo,hi-m],fmt="none",ecolor=st.MODEL,lw=.65,capsize=1.5)
    ax.set_xscale("log",base=2);ax.set_xlabel("Retained units");ax.set_ylabel("RDM recovery");st.clean(ax)
    # e tuning/support tiling heatmaps for three K regimes
    ax=fig.add_subplot(gs[1,1]);st.panel(ax,"e")
    mats=[np.asarray(p4["examples"][str(k)]) for k in (256,32,2)]
    tiled=np.concatenate([m[:6] for m in mats],axis=0)
    im=ax.imshow(tiled,aspect="auto",cmap="magma",vmin=0,vmax=1,interpolation="nearest")
    ax.axhline(5.5,color="white",lw=.7);ax.axhline(11.5,color="white",lw=.7)
    ax.set_xlabel("Condition");ax.set_ylabel("Locally tuned units");ax.set_xticks([0,7,14],["1","8","15"]);ax.set_yticks([])
    # f biological repeated bidirectional split-half targeted ablation
    ax=fig.add_subplot(gs[1,2]);st.panel(ax,"f")
    fields=["loss_unique","loss_random","loss_shared"]
    cats=["Unique","Matched\nrandom","Shared"]
    cols=["#A9B0B7",st.BIO,st.MODEL]
    session_matrix=np.column_stack([p15["summary"][f]["session_values"] for f in fields])
    for row in session_matrix:
        ax.plot(range(3),row,color="#DCE2E6",lw=.65,zorder=1)
        ax.scatter(range(3),row,s=8,color="#DCE2E6",edgecolor="white",lw=.25,zorder=2)
    for i,(field,col) in enumerate(zip(fields,cols)):
        stat=p15["summary"][field];m=stat["mean"];lo,hi=stat["session_cluster_boot_ci"]
        ax.errorbar(i,m,yerr=[[m-lo],[hi-m]],fmt="o",color=col,
                    ms=4.5,capsize=2.4,elinewidth=1.0,zorder=4)
    ax.set_xticks(range(3),cats);ax.set_ylabel("Held-out geometry loss");st.clean(ax)
    save(fig,"fig4_mechanism_final",match_svg_to_png=True);plt.close(fig)


def fig5():
    high=load(OUT/"nn_redundancy_high_dprime.json");tc=load(OUT/"nn_redundancy_timecourse.json");allen=load(OUT/"allen_vbn_active_passive_redundancy.json")
    fig=plt.figure(figsize=(7.25,5.25));gs=fig.add_gridspec(2,3,left=.065,right=.99,bottom=.10,top=.96,wspace=.42,hspace=.44,width_ratios=(.78,1,1))
    # a Fisher definition schematic
    ax=fig.add_subplot(gs[0,0]);st.panel(ax,"a");
    xx=np.linspace(0,1,6);real=.10*np.sqrt(xx+.03);sh=.17*np.sqrt(xx+.03)
    ax.plot(xx,real,color=st.ACTIVE,marker="o",ms=3,label=r"$I_{real}$")
    ax.plot(xx,sh,color="#E4A51C",marker="o",ms=3,label=r"$I_{shuffle}$")
    ax.fill_between(xx,real,sh,color=st.PALE_ACTIVE,alpha=.9)
    ax.set_xlabel("Population size");ax.set_ylabel("Linear Fisher information");ax.set_xticks([]);ax.set_yticks([]);st.clean(ax)
    ax.legend(handles=ax.get_legend_handles_labels()[0]+[Line2D([],[],ls="",label=r"$I_{red}=I_{shuffle}-I_{real}$")],labels=ax.get_legend_handles_labels()[1]+[r"$I_{red}=I_{shuffle}-I_{real}$"],frameon=False,loc="upper left")
    # b animal paired Allen redundancy
    ax=fig.add_subplot(gs[0,1]);st.panel(ax,"b")
    q=allen["summary"]["20"]["fisher"]["animal"]; ds=np.array(list(q["cluster_deltas"].values()));passive=np.zeros_like(ds);active=ds
    for i in range(len(ds)):ax.plot([0,1],[passive[i],active[i]],color=st.LIGHT,lw=.55,zorder=1)
    ax.scatter(np.zeros_like(ds),passive,s=10,color=st.PASSIVE,alpha=.55);ax.scatter(np.ones_like(ds),active,s=10,color=st.ACTIVE,alpha=.55)
    for x,v,col in [(0,passive,st.PASSIVE),(1,active,st.ACTIVE)]:
        m,lo,hi=mean_ci(v);ax.errorbar(x,m,yerr=[[m-lo],[hi-m]],fmt="o",color=col,ms=5,capsize=2.5,zorder=4)
    ax.axhline(0,color=st.INK,lw=.65,ls="--");ax.set_xticks([0,1],["Passive","Active"]);ax.set_ylabel("Fisher redundancy change");st.clean(ax)
    ax.legend(handles=[Line2D([0],[0],color=st.LIGHT,label="Animals; N=20"),Line2D([0],[0],marker="o",ls="",color=st.ACTIVE,label="Mean ± 95% CI")],frameon=False,loc="upper left")
    # c decomposition Allen
    ax=fig.add_subplot(gs[0,2]);st.panel(ax,"c")
    cats=["Fisher","Alignment","Sharing"];keys=["fisher","align_cv","sharing"]
    for i,k in enumerate(keys):
        z=allen["summary"]["20"][k]["animal"];m=z["delta_cluster_mean"];lo,hi=z["cluster_boot_ci"]
        ax.errorbar(i,m,yerr=[[m-lo],[hi-m]],fmt="o",color=st.ACTIVE,ms=4.5,capsize=2.4)
    ax.axhline(0,color=st.INK,lw=.65,ls="--");ax.set_xticks(range(3),cats);ax.set_ylabel("Active - passive");st.clean(ax)
    ax.legend(handles=[Line2D([0],[0],marker="o",ls="",color=st.ACTIVE,label="Mean ± 95% CI")],frameon=False,loc="upper left")
    # d high/low information units
    ax=fig.add_subplot(gs[1,0]);st.panel(ax,"d")
    entries=[]
    for axisn in ("side","contrast"):
        for group in ("high","low"):entries.append((axisn,group,high["summary"][axisn]["task_delta"][group]))
    yy=np.arange(4)[::-1]
    for y,(axisn,g,q) in zip(yy,entries):
        col=st.ACTIVE if g=="high" else st.CONTROL;m=q["mean"];lo,hi=q["ci"];ax.errorbar(m,y,xerr=[[m-lo],[hi-m]],fmt="o",color=col,ms=4,capsize=2.2)
    ax.axvline(0,color=st.INK,lw=.65,ls="--");ax.set_yticks(yy,["Side high d′","Side low d′","Contrast high d′","Contrast low d′"]);ax.set_xlabel("Active - passive sharing");st.clean(ax)
    # e/f within-trial trajectories
    for j,axisn in enumerate(("side","contrast")):
        ax=fig.add_subplot(gs[1,j+1]);st.panel(ax,"e" if j==0 else "f")
        by=defaultdict(lambda:{"passive":[[],[],[],[]],"active":[[],[],[],[]]})
        for r in tc["rows"]:
            if r["axis"]==axisn:
                by[r["session"]]["passive"][r["bin"]].append(r["passive"]);by[r["session"]]["active"][r["bin"]].append(r["active"])
        tt=np.array([50,150,250,350])
        for state,col in (("passive",st.PASSIVE),("active",st.ACTIVE)):
            mat=np.array([[np.mean(by[s][state][b]) for b in range(4)] for s in sorted(by)])
            m,lo,hi=mean_ci(mat,axis=0);ax.fill_between(tt,lo,hi,color=col,alpha=.15,lw=0);ax.plot(tt,m,color=col,marker="o",ms=3,label=f"{state.capitalize()}; 95% CI");ax.errorbar(tt,m,yerr=[m-lo,hi-m],fmt="none",ecolor=col,lw=.65,capsize=1.6)
        ax.axhline(0,color=st.INK,lw=.55,ls="--");ax.set_xlabel("Time from stimulus onset (ms)");ax.set_ylabel("Task-specific sharing");st.clean(ax);ax.legend(frameon=False,loc="upper left")
    save(fig,"fig5_state_v2");plt.close(fig)


if __name__ == "__main__":
    for function in (fig1,fig3,fig4,fig5):
        function()
    print(MAIN)
