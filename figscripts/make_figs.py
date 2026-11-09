import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import NullFormatter, FuncFormatter
from matplotlib.lines import Line2D
import os
# Regenerates every figure of the paper from the CSV files in ../figdata
# (copies of the phase reports: P1_gap_vs_rho, P3b_ladder, P3c_collapse_points, P4a_bound_terms,
# P5c_mechanism, P7_crown_vs_summary, P7_lm_summary, P8_gpt2_compare).
# Every figure is drawn at the size at which the paper prints it (the text width of the JMLR
# style, 6 inches, or 0.6 of it), so that all text in the figures is printed at 9 points, the
# minimum that the JMLR formatting instructions require. Fonts are DejaVu Sans, embedded as TrueType.
HERE=os.path.dirname(os.path.abspath(__file__))
DATA=os.path.join(HERE,"..","figdata")+os.sep
OUT=os.path.join(HERE,"..","figs")+os.sep
os.makedirs(OUT,exist_ok=True)
TEXTWIDTH=6.0   # inches, \textwidth of jmlr2e (433.62 pt)
FS=9            # points
C1,C2,C3,C4="#2a78d6","#eb6834","#1baf7a","#eda100"
INK,INK2,MUTED,GRID,AXIS="#0b0b0b","#52514e","#898781","#e1e0d9","#c3c2b7"
SEQ=["#cde2fb","#9ec5f4","#6da7ec","#3987e5","#256abf","#184f95","#0d366b"]
CMAP=LinearSegmentedColormap.from_list("seqblue",SEQ)
plt.rcParams.update({"font.family":"DejaVu Sans","mathtext.fontset":"dejavusans",
 "font.size":FS,"axes.titlesize":FS,"axes.labelsize":FS,"legend.fontsize":FS,
 "xtick.labelsize":FS,"ytick.labelsize":FS,"axes.edgecolor":AXIS,"axes.linewidth":0.8,
 "xtick.color":INK2,"ytick.color":INK2,"axes.labelcolor":INK,"text.color":INK,
 "grid.color":GRID,"grid.linewidth":0.6,"grid.linestyle":"-","lines.linewidth":1.6,
 "legend.frameon":False,"pdf.fonttype":42,"savefig.bbox":"standard",
 "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150,
 "axes.titlelocation":"left","figure.constrained_layout.h_pad":0.03,"figure.constrained_layout.w_pad":0.03})
def style(ax):
    ax.grid(True,which="major"); ax.set_axisbelow(True)
    ax.xaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
def refline(ax,x0,y0,slope,xs,label,va="center",dy=1.0):
    xs=np.array(xs); ax.plot(xs,y0*(xs/x0)**slope,color=MUTED,lw=0.9,ls="-",alpha=0.9)
    ax.text(xs[-1],dy*y0*(xs[-1]/x0)**slope," "+label,color=INK2,fontsize=FS,va=va)
def save(fig,name,**kw):
    fig.savefig(OUT+name,**kw); plt.close(fig)

# ---------- Figure 1: gaps against the radius on synthetic heads (Phase 1) ----------
d=pd.read_csv(DATA+"P1_gap_vs_rho.csv")
fig,axs=plt.subplots(1,2,figsize=(TEXTWIDTH,2.9),layout="constrained")
meths=[("ibp_obj","interval pattern bounds",C4,"v"),("ps_auto","first-order (pattern polytope)",C2,"s"),
       ("sdp_1_unitdiag","level one, unit diagonal",C3,"^"),("sdp_1","level one (this paper)",C1,"o")]
# reference lines of slopes one and two, placed clear of the medians and their bands
pos={"F_random":(6.0,0.12,"top",0.9),"F_ambiguous":(3.2,0.35,"center",1.0)}
for ax,fam,title in zip(axs,["F_random","F_ambiguous"],["random heads","ambiguous heads"]):
    for m,lab,col,mk in meths:
        s=d[(d.family==fam)&(d.set_type=="ball")&(d.method==m)&(d.rho<=0.2)].sort_values("rho")
        ax.fill_between(s.rho,s.q25,s.q75,color=col,alpha=0.10,lw=0)
        ax.plot(s.rho,s["median"],color=col,marker=mk,ms=4,label=lab,mec="white",mew=0.6)
    ax.set_xscale("log"); ax.set_yscale("log"); style(ax)
    ax.set_xlabel(r"perturbation radius $\rho$"); ax.set_title(title)
    s=d[(d.family==fam)&(d.set_type=="ball")&(d.method=="sdp_1")&(d.rho<=0.2)].sort_values("rho")
    k1,k2,va2,dy2=pos[fam]
    refline(ax,0.01,s["median"].iloc[0]*k2,2,[0.01,0.05],"slope 2",va=va2,dy=dy2)
    s2=d[(d.family==fam)&(d.set_type=="ball")&(d.method=="ibp_obj")&(d.rho<=0.2)].sort_values("rho")
    refline(ax,0.01,s2["median"].iloc[0]*k1,1,[0.01,0.05],"slope 1")
axs[0].set_ylabel(r"certified gap $r-p^\star$")
h,l=axs[0].get_legend_handles_labels()
fig.legend(h,l,loc="outside upper center",ncol=2,columnspacing=1.6,handlelength=1.8)
save(fig,"fig_synthetic_slopes.pdf")

# ---------- Figure 2: the five-radius ladder on GPT-2 small (Phase 3b) ----------
L=pd.read_csv(DATA+"P3b_ladder.csv")
rhos=[0.001,0.002,0.005,0.01,0.02]
fig,axs=plt.subplots(1,2,figsize=(TEXTWIDTH,2.75),layout="constrained")
ax=axs[0]
for q,lab,col,mk in [("spread","range of motion",C3,"^"),("greedy","first-order gap",C2,"s"),("cert","certificate gap",C1,"o")]:
    M=np.array([L[f"{q}_{x}"].values for x in rhos])
    M=np.where(M>0,M,np.nan)
    med=np.nanmedian(M,axis=1); lo=np.nanpercentile(M,25,axis=1); hi=np.nanpercentile(M,75,axis=1)
    ax.fill_between(rhos,lo,hi,color=col,alpha=0.12,lw=0); ax.plot(rhos,med,color=col,marker=mk,ms=4,label=lab,mec="white",mew=0.6)
ax.set_xscale("log"); ax.set_yscale("log"); style(ax)
ax.set_xlabel(r"relative radius $\rho_{\rm rel}$"); ax.set_ylabel("value (median, IQR)")
ax.legend(loc="lower right",handlelength=1.6,borderaxespad=0.2); ax.set_title("five-radius ladder, 720 units")
ax=axs[1]
bins=np.linspace(0,3,61)
for q,lab,col in [("spread","range of motion",C3),("greedy","first-order gap",C2),("cert","certificate gap",C1)]:
    v=L[f"slope_{q}_small"].dropna().clip(0,3)
    ax.hist(v,bins=bins,histtype="step",color=col,lw=1.4,label=lab)
for x in (1,2): ax.axvline(x,color=MUTED,lw=0.8)
style(ax); ax.set_xlabel(r"slope $d\log(\cdot)/d\log\rho$"); ax.set_ylabel("units")
ax.set_title(r"per-unit slopes, $\rho_{\rm rel}\leq 0.005$")
save(fig,"fig_ladder.pdf")

# ---------- Figure 3: the gap against the raw and the value-weighted slack ----------
Cc=pd.read_csv(DATA+"P3c_collapse_points.csv")
Cc=Cc[Cc.gap>1e-7*Cc.range]
Bk=pd.read_csv(DATA+"P4a_bound_terms.csv"); Bk=Bk[~Bk.floor][["source","unit","F0vf_term_knapsack"]]
Cc=Cc.merge(Bk,on=["source","unit"],how="inner")
fig,axs=plt.subplots(1,2,figsize=(TEXTWIDTH,2.75),sharey=True,layout="constrained")
for ax,xcol,xlab,ttl in [(axs[0],"S",r"weighted secant slack $S$","raw slack"),(axs[1],"F0vf_term_knapsack",r"knapsack term $\mathrm{KN}(\chi,D,D_{\rm tot})$","value-weighted slack")]:
    sc=ax.scatter(Cc[xcol],Cc.gap,c=Cc.theta,cmap=CMAP,vmin=0,vmax=0.9,s=4,lw=0,alpha=0.8,rasterized=True)
    ax.set_xscale("log"); ax.set_yscale("log"); style(ax); ax.set_xlabel(xlab); ax.set_title(ttl)
x=np.logspace(-7,1.5,10); axs[1].plot(x,x,color=MUTED,lw=0.9); axs[1].text(8e-7,6e-5,r"$y=x$",color=INK2,fontsize=FS)
axs[1].set_xlim(1e-7,50)
axs[0].set_xticks([1e-5,1e-3,1e-1,1e1,1e3]); axs[1].set_xticks([1e-7,1e-5,1e-3,1e-1,1e1])
axs[0].set_ylabel(r"certified gap $\mathrm{cert}-\mathrm{lb}$")
cb=fig.colorbar(sc,ax=axs,fraction=0.04,pad=0.02,aspect=25); cb.set_label(r"dispersion $1-\|a(\bar z)\|^2$"); cb.outline.set_edgecolor(AXIS)
save(fig,"fig_collapse.pdf",dpi=300)

# ---------- Figure 4: comparison with linear bound propagation and the vertex bound (Phases 4b, 7) ----------
# top row: the head's output (P7_crown_vs_summary); bottom row: the logit-margin specification (P7_lm_summary);
# left column: fixed query, exact reference; right column: varying query, best feasible value
C5="#e87ba4"
rows=[("P7_crown_vs_summary.csv","head output"),("P7_lm_summary.csv","logit margin")]
meths=[("CROWN","CROWN",C4,"v"),("alpha-CROWN",r"$\alpha$-CROWN",C3,"^"),("VS","vertex bound",C5,"D"),
       ("greedy","first order (this paper)",C2,"s"),("cert","certificate (this paper)",C1,"o")]
fig,axs=plt.subplots(2,2,figsize=(TEXTWIDTH,5.3),layout="constrained")
for r,(fname,spec) in enumerate(rows):
    Cs=pd.read_csv(DATA+fname)
    for c,(kind,qt) in enumerate([("fq","fixed query"),("free","varying query")]):
        ax=axs[r,c]; s=Cs[Cs.kind==kind].sort_values("rho_rel")
        for m,lab,col,mk in meths:
            ax.fill_between(s.rho_rel,s[m+"_q25"],s[m+"_q75"],color=col,alpha=0.10,lw=0)
            ax.plot(s.rho_rel,s[m+"_median"],color=col,marker=mk,ms=4 if mk!="D" else 3.5,label=lab,mec="white",mew=0.6)
        ax.set_xscale("log"); ax.set_yscale("log"); style(ax)
        ax.set_title(spec+", "+qt)
        ax.set_xticks(list(s.rho_rel)); ax.set_xticks([],minor=True)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p: "%g"%v))
        if r==1: ax.set_xlabel(r"relative radius $\rho_{\rm rel}$")
        if c==0: ax.set_ylabel("relative gap")
h,l=axs[0,0].get_legend_handles_labels()
order=[0,3,1,4,2]   # rows: the three baselines, then the two certificates of this paper
fig.legend([h[i] for i in order],[l[i] for i in order],loc="outside upper center",ncol=3,columnspacing=1.6,handlelength=1.8)
save(fig,"fig_crown.pdf")

# ---------- Figure 5: the per-operation mechanism on GPT-2 block remainders (Phase 5c) ----------
Mx=pd.read_csv(DATA+"P5c_mechanism.csv")
fig,ax=plt.subplots(1,1,figsize=(0.6*TEXTWIDTH,2.75),layout="constrained")
for r,col,mk in [(0.0005,C1,"o"),(0.001,C3,"^"),(0.002,C2,"s")]:
    s=Mx[Mx.rho_rel==r]
    ax.scatter(s.P_T_p_complex/s.s_spec_p,s.v_tot/s.att_max,color=col,marker=mk,s=13,lw=0.4,edgecolor="white",label=r"$\rho_{\rm rel}=%g$"%r)
x=np.logspace(1,3.2,10); ax.plot(x,x,color=MUTED,lw=0.9); ax.text(14,9.6,r"$y=x$",color=INK2,fontsize=FS)
ax.set_xscale("log"); ax.set_yscale("log"); style(ax)
ax.set_xlabel(r"predicted looseness $s_{\rm op}/s_{\rm spec}$"); ax.set_ylabel("CROWN bound / attained maximum")
ax.legend(loc="upper left",handletextpad=0.2,borderaxespad=0.2)
save(fig,"fig_mechanism.pdf")

# ---------- Figure 6: GPT-2 small and gpt-oss-20b at matched secant width (Phase 8) ----------
# P8_gpt2_compare: medians and quartiles of the relative gaps in bins of the secant width w_t,
# for the GPT-2 units of Table 6 and the gpt-oss-20b units of Parts F and V; only the bins that
# hold at least 30 units of each model are drawn, at the median w_t of the bin.
Gc=pd.read_csv(DATA+"P8_gpt2_compare.csv")
meths=[("CROWN","CROWN",C4,"v"),("greedy","first order",C2,"s"),("cert","certificate",C1,"o")]
models=[("gpt-oss-20b","-",True),("gpt2",(0,(3.2,1.8)),False)]
gb=Gc.groupby(["kind","w_t_bin"]); both=(gb.units.transform("min")>=30)&(gb.model.transform("nunique")==2)
fig,axs=plt.subplots(1,2,figsize=(TEXTWIDTH,2.9),sharey=True,layout="constrained")
for ax,(kind,qt) in zip(axs,[("fq","fixed query, exact reference"),("free","varying query, best feasible value")]):
    for mod,ls,filled in models:
        s=Gc[both&(Gc.kind==kind)&(Gc.model==mod)].sort_values("median_w_t")
        for m,lab,col,mk in meths:
            if filled: ax.fill_between(s.median_w_t,s[m+"_q25"],s[m+"_q75"],color=col,alpha=0.10,lw=0)
            ax.plot(s.median_w_t,s[m+"_median"],color=col,ls=ls,marker=mk,ms=4,
                    mfc=col if filled else "white",mec="white" if filled else col,mew=0.6 if filled else 0.9)
    ax.set_xscale("log"); ax.set_yscale("log"); style(ax)
    ax.set_xlabel(r"secant width $w_t$ (bin median)"); ax.set_title(qt)
axs[0].set_ylabel("relative gap")
hm=[Line2D([0],[0],color=col,marker=mk,ms=4,mec="white",mew=0.6,lw=1.6,label=lab) for m,lab,col,mk in meths]
hd=[Line2D([0],[0],color=INK2,ls="-",lw=1.3,label="gpt-oss-20b"),Line2D([0],[0],color=INK2,ls=(0,(3.2,1.8)),lw=1.3,label="GPT-2 small")]
fig.legend(handles=hm+hd,loc="outside upper center",ncol=5,columnspacing=1.0,handlelength=1.8,handletextpad=0.4)
save(fig,"fig_oss.pdf")
print("ok")
