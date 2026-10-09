import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, NullFormatter
from matplotlib.lines import Line2D

INK="#1a1a18"; MUTED="#6b6b65"; GRID="#e6e6e2"; SURF="#ffffff"
C={"3D":"#0072B2","2D":"#E69F00","1D":"#CC79A7"}
plt.rcParams.update({"figure.facecolor":SURF,"axes.facecolor":SURF,"savefig.facecolor":SURF,
    "font.family":"DejaVu Sans","font.size":10,"text.color":INK,"axes.edgecolor":MUTED,
    "axes.labelcolor":INK,"xtick.color":MUTED,"ytick.color":MUTED,"axes.linewidth":0.8})

# ZFP on Hurricane ISABEL temperature (real run, 2026-09-30)
D={
"3D":dict(ratio=[14.4,11.1,7.2,5.1,4.4],
          Er=[1.28e-5,6.49e-6,1.61e-6,4.02e-7,2.03e-7], Ec=[1.61e-10,5.87e-10,8.53e-9,2.51e-8,1.24e-8],
          Mr=[6.12e-6,3.08e-6,7.64e-7,1.89e-7,9.44e-8], Mc=[7.15e-11,3.00e-10,3.88e-9,1.25e-8,5.89e-9]),
"2D":dict(ratio=[8.0,5.5,4.1,3.3,3.0],
          Er=[5.98e-5,1.54e-5,3.81e-6,9.34e-7,4.95e-7], Ec=[1.65e-10,4.50e-12,1.92e-10,1.07e-9,7.32e-9],
          Mr=[2.90e-5,7.48e-6,1.85e-6,4.53e-7,2.41e-7], Mc=[8.20e-11,3.92e-12,9.28e-11,5.17e-10,3.65e-9]),
"1D":dict(ratio=[3.1,2.9,2.5,2.2,2.0],
          Er=[6.08e-4,1.50e-4,1.23e-5,1.78e-6,3.68e-6], Ec=[8.90e-9,9.37e-9,1.36e-8,4.48e-9,6.12e-8],
          Mr=[7.57e-5,6.58e-6,2.72e-5,6.58e-6,4.01e-6], Mc=[4.36e-9,3.37e-9,7.59e-9,1.26e-9,2.76e-8]),
}
fig,(axE,axM)=plt.subplots(1,2,figsize=(9.4,4.3),sharey=True)
def order(r,y):  # sort by ratio ascending for clean lines
    i=np.argsort(r); return np.array(r)[i],np.array(y)[i]
for dim,d in D.items():
    for ax,raw,cor in [(axE,"Er","Ec"),(axM,"Mr","Mc")]:
        rr,yr=order(d["ratio"],d[raw]); rc,yc=order(d["ratio"],d[cor])
        ax.plot(rr,yr,"-o",color=C[dim],lw=2,ms=5,label=f"{dim} — ZFP alone")
        ax.plot(rc,yc,"--o",color=C[dim],lw=2,ms=5,mfc=SURF,label=f"{dim} — + layer")
for ax,t in [(axE,"energy drift"),(axM,"mass drift")]:
    ax.set_yscale("log"); ax.set_xscale("log")
    ax.set_xticks([2,3,5,8,14]); ax.set_xticklabels(["2×","3×","5×","8×","14×"])
    ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("compression ratio"); ax.set_title(t,fontsize=11,color=INK,pad=8)
    ax.grid(True,color=GRID,lw=0.7);
    for s in ("top","right"): ax.spines[s].set_visible(False)
axE.set_ylabel("relative invariant drift")
leg=[Line2D([0],[0],color=C[k],lw=2,marker='o',ms=5,label=k) for k in ["1D","2D","3D"]]
leg+=[Line2D([0],[0],color=MUTED,lw=2,marker='o',ms=5,label="ZFP alone"),
      Line2D([0],[0],color=MUTED,lw=2,ls="--",marker='o',ms=5,mfc=SURF,label="+ layer")]
axM.legend(handles=leg,frameon=False,fontsize=8.5,loc="lower left",ncol=1)
fig.suptitle("Conservation layer across dimensionality — ZFP on Hurricane ISABEL (temperature)",
             fontsize=11.5,y=0.99)
fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig("fig3_backbone_zfp.png",dpi=150)
print("saved fig3_backbone_zfp.png")
