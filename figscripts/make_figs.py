import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import NullFormatter, FuncFormatter
import os
# Regenerates every figure of the paper from the CSV files in ../figdata
# (P1_gap_vs_rho, P3b_ladder, P3c_collapse_points,
# P4b_crown_summary, P5c_mechanism).
HERE=os.path.dirname(os.path.abspath(__file__))
DATA=os.path.join(HERE,"..","figdata")+os.sep
OUT=os.path.join(HERE,"..","figs")+os.sep
os.makedirs(OUT,exist_ok=True)
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

# ---------- Figure 1: synthetic slopes ----------
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

# ---------- Figure 2: the radius ladder on GPT-2 ----------
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

# ---------- Figure 3: collapse ----------
Cc=pd.read_csv(DATA+"P3c_collapse_points.csv")
Cc=Cc[Cc.gap>1e-7*Cc.range]
fig,axs=plt.subplots(1,2,figsize=(6.6,2.6),sharey=True)
for ax,xcol,xlab,ttl in [(axs[0],"S",r"weighted secant slack $S$","raw slack"),(axs[1],"Pi_v","value-weighted slack","value-weighted slack")]:
    sc=ax.scatter(Cc[xcol],Cc.gap,c=Cc.theta,cmap=CMAP,vmin=0,vmax=0.9,s=4,lw=0,alpha=0.8,rasterized=True)
    ax.set_xscale("log"); ax.set_yscale("log"); style(ax); ax.set_xlabel(xlab); ax.set_title(ttl,loc="left")
x=np.logspace(-7,1.5,10); axs[1].plot(x,x,color=MUTED,lw=0.9); axs[1].text(2e-6,6e-6,r"$y=x$",color=INK2,fontsize=6.8,rotation=0)
axs[1].set_xlim(1e-7,50)
axs[0].set_ylabel(r"certified gap $\mathrm{cert}-\mathrm{lb}$")
cb=fig.colorbar(sc,ax=axs,fraction=0.025,pad=0.02); cb.set_label(r"dispersion $1-\|a(\bar z)\|^2$"); cb.outline.set_edgecolor(AXIS)
fig.savefig(OUT+"fig_collapse.pdf",dpi=300); plt.close(fig)

# ---------- Figure 4: comparison with linear bound propagation ----------
Cs=pd.read_csv(DATA+"P4b_crown_summary.csv")
fig,axs=plt.subplots(1,2,figsize=(6.6,2.55))
for ax,kind,title in [(axs[0],"fq","fixed query (against the exact worst case)"),(axs[1],"free","varying query (against the best feasible value)")]:
    s=Cs[Cs.kind==kind].sort_values("rho_rel")
    for m,lab,col,mk in [("CROWN","CROWN",C4,"v"),("alpha-CROWN",r"$\alpha$-CROWN",C3,"^"),("greedy","first order (this paper)",C2,"s"),("cert","certificate (this paper)",C1,"o")]:
        ax.fill_between(s.rho_rel,s[m+"_q25"],s[m+"_q75"],color=col,alpha=0.10,lw=0)
        ax.plot(s.rho_rel,s[m+"_median"],color=col,marker=mk,ms=3.6,label=lab,mec="white",mew=0.6)
    ax.set_xscale("log"); ax.set_yscale("log"); style(ax)
    ax.set_xlabel(r"relative radius $\rho_{\rm rel}$"); ax.set_title(title,loc="left")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p: "%g"%v))
axs[0].set_ylabel("relative gap (median, IQR)")
h,l=axs[0].get_legend_handles_labels(); fig.legend(h,l,loc="upper center",ncol=4,bbox_to_anchor=(0.5,1.07))
fig.savefig(OUT+"fig_crown.pdf"); plt.close(fig)

# ---------- Figure 5: the per-operation mechanism on GPT-2 block remainders ----------
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
