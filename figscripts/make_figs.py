import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import NullFormatter, FuncFormatter
import os
# Regenerates every figure of the paper from the CSV files in ../figdata
# (copies of the phase reports: P1_gap_vs_rho, P3b_ladder, P3c_collapse_points,
# P4b_crown_summary, P5c_mechanism, P7_crown_vs_summary, P7_lm_summary, P8_gpt2_compare).
HERE=os.path.dirname(os.path.abspath(__file__))
DATA=os.path.join(HERE,"..","figdata")+os.sep
OUT=os.path.join(HERE,"..","figs")+os.sep
C1,C2,C3,C4="#2a78d6","#eb6834","#1baf7a","#eda100"
INK,INK2,MUTED,GRID,AXIS="#0b0b0b","#52514e","#898781","#e1e0d9","#c3c2b7"
SEQ=["#cde2fb","#9ec5f4","#6da7ec","#3987e5","#256abf","#184f95","#0d366b"]
CMAP=LinearSegmentedColormap.from_list("seqblue",SEQ)
plt.rcParams.update({"font.size":8,"axes.titlesize":8.5,"axes.labelsize":8,"legend.fontsize":7.2,
 "xtick.labelsize":7.2,"ytick.labelsize":7.2,"axes.edgecolor":AXIS,"axes.linewidth":0.8,
 "xtick.color":INK2,"ytick.color":INK2,"axes.labelcolor":INK,"text.color":INK,
 "grid.color":GRID,"grid.linewidth":0.6,"grid.linestyle":"-","lines.linewidth":1.6,
 "legend.frameon":False,"pdf.fonttype":42,"savefig.bbox":"tight","savefig.pad_inches":0.02,
 "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150})
def style(ax):
    ax.grid(True,which="major"); ax.set_axisbelow(True)
    ax.xaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
def refline(ax,x0,y0,slope,xs,label,ls=(0,(1,0)),color=MUTED):
    xs=np.array(xs); ax.plot(xs,y0*(xs/x0)**slope,color=color,lw=0.9,ls="-",alpha=0.9)
    ax.text(xs[-1],y0*(xs[-1]/x0)**slope,"  "+label,color=INK2,fontsize=6.8,va="center")

# ---------- Fig A: synthetic slopes (Phase 1) ----------
d=pd.read_csv(DATA+"P1_gap_vs_rho.csv")
fig,axs=plt.subplots(1,2,figsize=(6.6,2.55))
meths=[("ibp_obj","interval pattern bounds",C4,"v"),("ps_auto","first-order (pattern polytope)",C2,"s"),
       ("sdp_1_unitdiag","level one, unit diagonal",C3,"^"),("sdp_1","level one (this paper)",C1,"o")]
for ax,fam,title in zip(axs,["F_random","F_ambiguous"],["random heads","ambiguous heads"]):
    for m,lab,col,mk in meths:
        s=d[(d.family==fam)&(d.set_type=="ball")&(d.method==m)&(d.rho<=0.2)].sort_values("rho")
        ax.fill_between(s.rho,s.q25,s.q75,color=col,alpha=0.10,lw=0)
        ax.plot(s.rho,s["median"],color=col,marker=mk,ms=3.6,label=lab,mec="white",mew=0.6)
    ax.set_xscale("log"); ax.set_yscale("log"); style(ax)
    ax.set_xlabel(r"perturbation radius $\rho$"); ax.set_title(title,loc="left")
    s=d[(d.family==fam)&(d.set_type=="ball")&(d.method=="sdp_1")&(d.rho<=0.2)].sort_values("rho")
    y0=s["median"].iloc[0]*0.35
    refline(ax,0.01,y0,2,[0.01,0.05],"slope 2")
    s2=d[(d.family==fam)&(d.set_type=="ball")&(d.method=="ibp_obj")&(d.rho<=0.2)].sort_values("rho")
    refline(ax,0.01,s2["median"].iloc[0]*3.2,1,[0.01,0.05],"slope 1")
axs[0].set_ylabel("certified gap $r-p^\\star$ (median, IQR)")
h,l=axs[0].get_legend_handles_labels(); fig.legend(h,l,loc="upper center",ncol=4,bbox_to_anchor=(0.5,1.07))
fig.savefig(OUT+"fig_synthetic_slopes.pdf"); plt.close(fig)

# ---------- Fig D: Theorem I ladder (Phase 3b) ----------
L=pd.read_csv(DATA+"P3b_ladder.csv")
rhos=[0.001,0.002,0.005,0.01,0.02]
fig,axs=plt.subplots(1,2,figsize=(6.6,2.55))
ax=axs[0]
for q,lab,col,mk in [("spread","range of motion",C3,"^"),("greedy","first-order gap",C2,"s"),("cert","certificate gap",C1,"o")]:
    M=np.array([L[f"{q}_{x}"].values for x in rhos])
    M=np.where(M>0,M,np.nan)
    med=np.nanmedian(M,axis=1); lo=np.nanpercentile(M,25,axis=1); hi=np.nanpercentile(M,75,axis=1)
    ax.fill_between(rhos,lo,hi,color=col,alpha=0.12,lw=0); ax.plot(rhos,med,color=col,marker=mk,ms=3.6,label=lab,mec="white",mew=0.6)
ax.set_xscale("log"); ax.set_yscale("log"); style(ax)
ax.set_xlabel(r"relative radius $\rho_{\rm rel}$"); ax.set_ylabel("value (median, IQR)")
ax.legend(loc="lower right",fontsize=6.6); ax.set_title("five-radius ladder, 720 units",loc="left")
ax=axs[1]
bins=np.linspace(0,3,61)
for q,lab,col in [("spread","range of motion",C3),("greedy","first-order gap",C2),("cert","certificate gap",C1)]:
    v=L[f"slope_{q}_small"].dropna().clip(0,3)
    ax.hist(v,bins=bins,histtype="step",color=col,lw=1.4,label=lab)
for x in (1,2): ax.axvline(x,color=MUTED,lw=0.8)
style(ax); ax.set_xlabel(r"slope $d\log(\cdot)/d\log\rho$ on $\rho_{\rm rel}\leq 0.005$"); ax.set_ylabel("units")
ax.set_title("per-unit slopes",loc="left")
fig.savefig(OUT+"fig_ladder.pdf"); plt.close(fig)

# ---------- Fig F: collapse ----------
Cc=pd.read_csv(DATA+"P3c_collapse_points.csv")
Cc=Cc[Cc.gap>1e-7*Cc.range]
Bk=pd.read_csv(DATA+"P4a_bound_terms.csv"); Bk=Bk[~Bk.floor][["source","unit","F0vf_term_knapsack"]]
Cc=Cc.merge(Bk,on=["source","unit"],how="inner")
fig,axs=plt.subplots(1,2,figsize=(6.6,2.6),sharey=True)
for ax,xcol,xlab,ttl in [(axs[0],"S",r"weighted secant slack $S$","raw slack"),(axs[1],"F0vf_term_knapsack",r"knapsack term $\mathrm{KN}(\chi,D,D_{\rm tot})$","value-weighted slack")]:
    sc=ax.scatter(Cc[xcol],Cc.gap,c=Cc.theta,cmap=CMAP,vmin=0,vmax=0.9,s=4,lw=0,alpha=0.8,rasterized=True)
    ax.set_xscale("log"); ax.set_yscale("log"); style(ax); ax.set_xlabel(xlab); ax.set_title(ttl,loc="left")
x=np.logspace(-7,1.5,10); axs[1].plot(x,x,color=MUTED,lw=0.9); axs[1].text(2e-6,6e-6,r"$y=x$",color=INK2,fontsize=6.8,rotation=0)
axs[1].set_xlim(1e-7,50)
axs[0].set_ylabel(r"certified gap $\mathrm{cert}-\mathrm{lb}$")
cb=fig.colorbar(sc,ax=axs,fraction=0.025,pad=0.02); cb.set_label(r"dispersion $1-\|a(\bar z)\|^2$"); cb.outline.set_edgecolor(AXIS)
fig.savefig(OUT+"fig_collapse.pdf",dpi=300); plt.close(fig)

# ---------- Fig H: comparison with linear bound propagation and the vertex bound (Phase 4b part D, Phase 7) ----------
# top row: the head's output (P7_crown_vs_summary = P4b_crown_summary plus the vertex bound);
# bottom row: the logit-margin specification (P7_lm_summary)
C5="#e87ba4"
rows=[("P7_crown_vs_summary.csv","head output"),("P7_lm_summary.csv","logit margin")]
meths=[("CROWN","CROWN",C4,"v"),("alpha-CROWN",r"$\alpha$-CROWN",C3,"^"),("VS","vertex bound",C5,"D"),
       ("greedy","first order (this paper)",C2,"s"),("cert","certificate (this paper)",C1,"o")]
fig,axs=plt.subplots(2,2,figsize=(6.6,5.0))
for r,(fname,spec) in enumerate(rows):
    Cs=pd.read_csv(DATA+fname)
    for c,(kind,qt) in enumerate([("fq","fixed query, exact reference"),("free","varying query, best feasible value")]):
        ax=axs[r,c]; s=Cs[Cs.kind==kind].sort_values("rho_rel")
        for m,lab,col,mk in meths:
            ax.fill_between(s.rho_rel,s[m+"_q25"],s[m+"_q75"],color=col,alpha=0.10,lw=0)
            ax.plot(s.rho_rel,s[m+"_median"],color=col,marker=mk,ms=3.6 if mk!="D" else 3.2,label=lab,mec="white",mew=0.6)
        ax.set_xscale("log"); ax.set_yscale("log"); style(ax)
        ax.set_title(spec+", "+qt,loc="left")
        ax.set_xticks(list(s.rho_rel)); ax.set_xticks([],minor=True)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p: "%g"%v))
        if r==1: ax.set_xlabel(r"relative radius $\rho_{\rm rel}$")
        if c==0: ax.set_ylabel("relative gap (median, IQR)")
fig.tight_layout(h_pad=1.2)
h,l=axs[0,0].get_legend_handles_labels(); fig.legend(h,l,loc="upper center",ncol=5,bbox_to_anchor=(0.5,1.045),columnspacing=1.0,handletextpad=0.4)
fig.savefig(OUT+"fig_crown.pdf"); plt.close(fig)

# ---------- Fig J: the per-operation mechanism on GPT-2 block remainders (Phase 5c) ----------
Mx=pd.read_csv(DATA+"P5c_mechanism.csv")
fig,ax=plt.subplots(1,1,figsize=(3.6,2.7))
for r,col,mk in [(0.0005,C1,"o"),(0.001,C3,"^"),(0.002,C2,"s")]:
    s=Mx[Mx.rho_rel==r]
    ax.scatter(s.P_T_p_complex/s.s_spec_p,s.v_tot/s.att_max,color=col,marker=mk,s=11,lw=0.4,edgecolor="white",label=r"$\rho_{\rm rel}=%g$"%r)
x=np.logspace(1,3.2,10); ax.plot(x,x,color=MUTED,lw=0.9); ax.text(14,10.5,r"$y=x$",color=INK2,fontsize=6.8)
ax.set_xscale("log"); ax.set_yscale("log"); style(ax)
ax.set_xlabel(r"predicted looseness $s_{\rm op}/s_{\rm spec}$"); ax.set_ylabel("CROWN bound / attained maximum")
ax.legend(loc="upper left",fontsize=6.4)
fig.savefig(OUT+"fig_mechanism.pdf"); plt.close(fig)
print("ok")

# ---------- Fig K: GPT-2 small and gpt-oss-20b at matched secant width (Phase 8) ----------
# P8_gpt2_compare: medians and quartiles of the relative gaps in bins of the secant width w_t,
# for the GPT-2 units of Table 6 and the gpt-oss-20b units of Parts F and V; only the bins that
# hold at least 30 units of each model are drawn, at the median w_t of the bin.
from matplotlib.lines import Line2D
Gc=pd.read_csv(DATA+"P8_gpt2_compare.csv")
meths=[("CROWN","CROWN",C4,"v"),("greedy","first order",C2,"s"),("cert","certificate",C1,"o")]
models=[("gpt-oss-20b","-",True),("gpt2",(0,(3.2,1.8)),False)]
gb=Gc.groupby(["kind","w_t_bin"]); both=(gb.units.transform("min")>=30)&(gb.model.transform("nunique")==2)
fig,axs=plt.subplots(1,2,figsize=(6.6,2.6),sharey=True)
for ax,(kind,qt) in zip(axs,[("fq","fixed query, exact reference"),("free","varying query, best feasible value")]):
    for mod,ls,filled in models:
        s=Gc[both&(Gc.kind==kind)&(Gc.model==mod)].sort_values("median_w_t")
        for m,lab,col,mk in meths:
            if filled: ax.fill_between(s.median_w_t,s[m+"_q25"],s[m+"_q75"],color=col,alpha=0.10,lw=0)
            ax.plot(s.median_w_t,s[m+"_median"],color=col,ls=ls,marker=mk,ms=3.6,
                    mfc=col if filled else "white",mec="white" if filled else col,mew=0.6 if filled else 0.9)
    ax.set_xscale("log"); ax.set_yscale("log"); style(ax)
    ax.set_xlabel(r"secant width $w_t$ (bin median)"); ax.set_title(qt,loc="left")
axs[0].set_ylabel("relative gap (median, IQR)")
hm=[Line2D([0],[0],color=col,marker=mk,ms=3.6,mec="white",mew=0.6,lw=1.6,label=lab) for m,lab,col,mk in meths]
hd=[Line2D([0],[0],color=INK2,ls="-",lw=1.3,label="gpt-oss-20b"),Line2D([0],[0],color=INK2,ls=(0,(3.2,1.8)),lw=1.3,label="GPT-2 small")]
fig.legend(handles=hm+hd,loc="upper center",ncol=5,bbox_to_anchor=(0.5,1.07),columnspacing=1.2,handletextpad=0.5)
fig.savefig(OUT+"fig_oss.pdf"); plt.close(fig)
