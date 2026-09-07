import os
from pathlib import Path
ROOT = Path(os.environ.get('SPARSE_CODING_ROOT', Path(__file__).resolve().parents[1]))
"""Publication audit of the LCR/NNR aggregation rule.

Recomputes the primary six-session Steinmetz analysis and the 40 K-sparse
models using four aggregation rules over condition-wise nearest/other ratios:
arithmetic mean (canonical), median, 20% trimmed mean, and ratio of pooled
distance means. It also explicitly counts ratios > 1. By construction nearest
distances are order statistics and cannot exceed the mean of the remaining
distances except for numerical tolerance; no ratio is removed in this audit.
"""
import sys, os, json, time, collections, collections.abc
from pathlib import Path
collections.Callable = collections.abc.Callable

import numpy as np
import torch
from scipy.stats import spearmanr, trim_mean

HERE=Path(__file__).parent
ROOT=HERE.parent
OUT=HERE/"outputs"/"audit_lcr.json"

import analysis_utils as au

SESSIONS=["Richards_2017-10-30","Richards_2017-11-01","Forssmann_2017-11-05",
          "Hench_2017-06-15","Forssmann_2017-11-02","Hench_2017-06-16"]
EXTRACT=ROOT/"tmp"/"steinmetz_extracted"
MIN_UNITS=20; MIN_TRIALS=5

def ps(r): return au.population_sparseness(r)

def ratios_and_parts(R,k):
    n=len(R); Rc=R-R.mean(1,keepdims=True); nr=np.linalg.norm(Rc,axis=1,keepdims=True); nr[nr==0]=1e-10
    d=1-(Rc/nr)@(Rc/nr).T; rr=[]; dns=[]; dos=[]
    for i in range(n):
        idx=np.delete(np.arange(n),i); idx=idx[np.argsort(d[i,idx])]; ke=min(k,n-2)
        dn=float(d[i,idx[:ke]].mean()); do=float(d[i,idx[ke:]].mean())
        if do>0: rr.append(dn/do); dns.append(dn); dos.append(do)
    return np.asarray(rr),np.asarray(dns),np.asarray(dos)

def summaries(R,k):
    rr,dn,do=ratios_and_parts(R,k)
    return {"mean":float(np.mean(rr)),"median":float(np.median(rr)),
            "trim20":float(trim_mean(rr,.2)),"ratio_of_means":float(dn.mean()/do.mean()),
            "n_ratio":int(len(rr)),"n_gt1":int(np.sum(rr>1+1e-10)),
            "max_ratio":float(rr.max())}

def bin_contrast(c):
    if c==0 or np.isnan(c): return 0
    if c<=.25:return 1
    if c<=.5:return 2
    return 3

def trial_counts(st,sc,onsets,win,ncl):
    order=np.argsort(st); st=st[order]; sc=sc[order]; out=np.zeros((len(onsets),ncl),np.float32)
    for ti,o in enumerate(onsets):
        a=np.searchsorted(st,o+win[0]); b=np.searchsorted(st,o+win[1]); cw=sc[a:b]
        ok=(cw>=0)&(cw<ncl)
        if ok.any(): np.add.at(out[ti],cw[ok],1)
    return out/(win[1]-win[0])

def steinmetz_rows():
    rows=[]
    for sn in SESSIONS:
        sd=EXTRACT/sn
        pc=np.load(sd/"clusters.peakChannel.npy").ravel().astype(int); ncl=len(pc)
        lines=open(sd/"channels.brainLocation.tsv").read().strip().split('\n')
        crg=np.array([x.split('\t')[3] if len(x.split('\t'))>=4 else 'root' for x in lines[1:]])
        regions=np.array([crg[c-1] if 0<=c-1<len(crg) else 'root' for c in pc])
        st=np.load(sd/"spikes.times.npy").ravel(); sc=np.load(sd/"spikes.clusters.npy").ravel().astype(int)
        gc=np.load(sd/"trials.goCue_times.npy").ravel(); stim=np.load(sd/"trials.visualStim_times.npy").ravel()
        L=np.load(sd/"trials.visualStim_contrastLeft.npy").ravel(); Rr=np.load(sd/"trials.visualStim_contrastRight.npy").ravel()
        fb=np.load(sd/"trials.feedbackType.npy").ravel(); choice=np.load(sd/"trials.response_choice.npy").ravel(); nt=len(gc)
        beh=np.array([("R" if c>0 else "L" if c<0 else "N")+("_C" if f>0 else "_E" if f<0 else "_X") for c,f in zip(choice,fb)])
        stimlab=np.array([f"L{bin_contrast(a)}_R{bin_contrast(b)}" for a,b in zip(L,Rr)])
        tb=trial_counts(st,sc,gc,(0,.5),ncl); ts=trial_counts(st,sc,stim,(.025,.25),ncl)
        ru={}
        for i,r in enumerate(regions):
            if r!='root':ru.setdefault(r,[]).append(i)
        for reg,units in ru.items():
            if len(units)<MIN_UNITS:continue
            rec={"session":sn,"region":reg}
            for typ,lab,tc,k in [("behav",beh,tb,2),("stim",stimlab,ts,3)]:
                valid=[c for c in np.unique(lab) if c!="_X" and np.sum(lab==c)>=MIN_TRIALS]
                if len(valid)<k+2:continue
                M=np.maximum(np.array([tc[lab==c][:,units].mean(0) for c in valid]),0)
                rec[typ]={"ps":float(np.nanmean([ps(x) for x in M])),**summaries(M,k)}
            rows.append(rec)
        print("Steinmetz",sn,"regions",len(rows))
    return rows

def model_rows():
    X,cats,_=au.load_stringer_data(); rows=[]
    for k in au.K_VALS:
        for seed in au.SEEDS:
            m=au.load_k_gradient_model(k,seed)
            with torch.no_grad():_,H,_=m(X)
            H=H.cpu().numpy(); M=au.build_condition_mean_matrix(H,cats)
            rows.append({"k":int(k),"seed":int(seed),"ps":float(np.nanmean([ps(x) for x in H])),**summaries(M,2)})
    return rows

def correlations(rows,nested=None):
    out={}
    for metric in ["mean","median","trim20","ratio_of_means"]:
        if nested:
            z=[r[nested] for r in rows if nested in r]
            x=np.array([r["ps"] for r in z]); y=np.array([r[metric] for r in z])
        else:
            x=np.array([r["ps"] for r in rows]); y=np.array([r[metric] for r in rows])
        q=spearmanr(x,y); out[metric]={"r":float(q.statistic),"p":float(q.pvalue),"n":int(len(x))}
    return out

t=time.time(); sr=steinmetz_rows(); mr=model_rows()
summary={"definition_note":"NN distances are the k smallest order statistics; ratios >1 are mathematically impossible apart from numerical error. No clipping or exclusion was used by the canonical code.",
         "steinmetz_behav":correlations(sr,"behav"),"steinmetz_stim":correlations(sr,"stim"),
         "models":correlations(mr),
         "counts":{"steinmetz_behav_gt1":sum(r.get('behav',{}).get('n_gt1',0) for r in sr),
                   "steinmetz_stim_gt1":sum(r.get('stim',{}).get('n_gt1',0) for r in sr),
                   "models_gt1":sum(r['n_gt1'] for r in mr)}}
json.dump({"summary":summary,"steinmetz":sr,"models":mr},open(OUT,"w"),indent=1)
print(json.dumps(summary,indent=2)); print("saved",OUT,"seconds",time.time()-t)
