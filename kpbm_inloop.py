"""
In-loop compression test on a D2Q9 LBM (compact cylinder flow, faithful to
validated_lbm.py at reduced scale). Question: does a conservation correction
keep an LBM stable when its distribution functions are lossily compressed
every step, where uncorrected compression drifts?

Compression is modeled as truncation quantization of f each step (a biased
lossy step, like ZFP's default truncation), injecting a systematic error.
Three modes:
  baseline      : no compression
  lossy         : truncate f each step, no correction
  lossy+correct : truncate f, then restore each node's mass and momentum
                  (the LBM conserved moments) via a weighted additive correction
"""
import numpy as np
np.seterr(all='ignore')

cx=np.array([0,1,0,-1,0,1,-1,-1,1]); cy=np.array([0,0,1,0,-1,1,1,-1,-1])
w=np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36]); opp=np.array([0,3,4,1,2,7,8,5,6])
cs2=1/3

NX,NY=260,130; D=16; R=D//2; CXC=NX//5; CYC=NY//2; U0=0.05
obst=np.zeros((NY,NX),bool)
yy,xx=np.ogrid[:NY,:NX]; obst=((xx-CXC)**2+(yy-CYC)**2)<=R**2
wall=np.zeros((NY,NX),bool); wall[0,:]=True; wall[-1,:]=True
solid=obst|wall

def eq(rho,ux,uy):
    feq=np.zeros((9,NY,NX)); usq=ux*ux+uy*uy
    for i in range(9):
        cu=cx[i]*ux+cy[i]*uy
        feq[i]=w[i]*rho*(1+cu/cs2+0.5*cu*cu/cs2**2-0.5*usq/cs2)
    return feq

def moments(f):
    rho=np.sum(f,axis=0); rho=np.maximum(rho,1e-10)
    jx=np.sum(f*cx[:,None,None],axis=0); jy=np.sum(f*cy[:,None,None],axis=0)
    return rho,jx,jy

def truncate(f,q):
    return np.floor(f/q)*q            # biased lossy step (like default truncation)

def restore_moments(fq, rho0, jx0, jy0):
    """Per-node weighted additive correction so mass & momentum match the
    pre-compression values exactly. g_i = fq_i + w_i[(dρ) + dJx*cx_i/cs2 + dJy*cy_i/cs2]."""
    rho1,jx1,jy1=moments(fq)
    dR=rho0-rho1; dJx=jx0-jx1; dJy=jy0-jy1
    g=fq.copy()
    for i in range(9):
        g[i]=fq[i]+w[i]*(dR+dJx*cx[i]/cs2+dJy*cy[i]/cs2)
    return g

def run(mode, Re=10, N=2000, q=4e-3):
    # Re=10 keeps the BGK base solver comfortably stable (tau=0.74). At Re>~50
    # with this D=16 resolution, BGK is unstable on its OWN -- baseline blows up
    # with no compression at all -- and such a regime is not a valid test of the
    # correction. Keep tau >~ 0.55 (raise Re only with more resolution/MRT).
    nu=U0*D/Re; tau=nu/cs2+0.5; om=1/tau
    rho=np.ones((NY,NX)); ux=np.ones((NY,NX))*U0; uy=np.zeros((NY,NX))
    np.random.seed(1); uy+=0.001*U0*np.random.randn(NY,NX)
    ux[solid]=0; uy[solid]=0; f=eq(rho,ux,uy)
    yv=np.arange(NY); uin=U0*4*yv*(NY-1-yv)/((NY-1)**2); uin[0]=0; uin[-1]=0
    M0=f.sum(); mass_hist=[]; umax_hist=[]; steps_rec=[]
    for step in range(N):
        rho,jx,jy=moments(f); ux=jx/rho; uy=jy/rho
        feq=eq(rho,ux,uy); fp=f-om*(f-feq)
        fs=np.zeros_like(fp)
        for i in range(9): fs[i]=np.roll(np.roll(fp[i],cx[i],1),cy[i],0)
        for i in range(9): fs[i][obst]=fp[opp[i]][obst]
        for i in range(9): fs[i][wall]=fp[opp[i]][wall]
        rin=(1/(1-uin))*(fs[0,:,0]+fs[2,:,0]+fs[4,:,0]+2*(fs[3,:,0]+fs[6,:,0]+fs[7,:,0]))
        fs[1,:,0]=fs[3,:,0]+(2/3)*rin*uin
        fs[5,:,0]=fs[7,:,0]+(1/6)*rin*uin+0.5*(fs[4,:,0]-fs[2,:,0])
        fs[8,:,0]=fs[6,:,0]+(1/6)*rin*uin-0.5*(fs[4,:,0]-fs[2,:,0])
        fs[:,:,-1]=fs[:,:,-2]
        f=fs
        # ---- in-loop compression hook ----
        if mode!='baseline':
            r0,jx0,jy0=moments(f)      # true moments (pre-compression)
            fq=truncate(f,q)
            f = restore_moments(fq,r0,jx0,jy0) if mode=='lossy+correct' else fq
        if step%50==0:
            umax=np.sqrt((ux*ux+uy*uy)).max()
            mass_hist.append(float(f.sum())); umax_hist.append(float(umax)); steps_rec.append(step)
            if not np.isfinite(umax) or umax>1.0:   # blow-up
                return dict(mode=mode,blewup=True,step=step,field=None,M0=M0)
    rr,jjx,jjy=moments(f)
    return dict(mode=mode,blewup=False,step=N,field=np.stack([jjx/rr,jjy/rr]),M0=M0)

if __name__ == "__main__":
    print("2D D2Q9 cylinder, Re=10 (stable BGK base), in-loop compression, 2000 steps")
    print("uncorrected truncation diverges; per-node moment restore keeps it stable.\n")
    base=run('baseline')
    assert not base['blewup'], "base solver unstable -- raise tau before testing"
    mask=~solid; ub=base['field']; den=np.sqrt(np.sum(ub[:,mask]**2))
    print("%-10s %-16s %-16s %-16s"%("q","uncorrected","corrected","corrected err"))
    for q in (5e-4,1e-3,2e-3,4e-3):
        ru=run('lossy',q=q); rc=run('lossy+correct',q=q)
        us=('blows up @%d'%ru['step']) if ru['blewup'] else 'stable'
        if rc['blewup']:
            cs=('blows up @%d'%rc['step']); err='-'
        else:
            cs='stable'
            err='%.1f%%'%(100*np.sqrt(np.sum((rc['field']-ub)[:,mask]**2))/den)
        print("%-10.0e %-16s %-16s %-16s"%(q,us,cs,err))
