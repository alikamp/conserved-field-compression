/* conserve_test.cpp -- correctness + bound-preservation test for conserve.h.
   Builds with: g++ -O2 conserve_test.cpp -o ctest && ./ctest
   Checks (1) invariants are restored, and (2) the corrected field respects the
   codec's pointwise bound in both ABS and REL modes. */
#include "conserve.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

static double frand(){ return (double)rand()/RAND_MAX; }           /* [0,1] */
static double urand(double a,double b){ return a+(b-a)*frand(); }  /* [a,b] */

int main(){
  const long N=200000;
  float* x   =new float[N];
  float* orig=new float[N];
  srand(7);

  /* ---- 1. invariants restored (quantization as a stand-in codec) ---- */
  for(long i=0;i<N;i++){ double t=(double)i/N;
    orig[i]=(float)(sin(20*t)+0.3*t*t+0.05*frand()); x[i]=orig[i]; }
  double M0=0,S0=0; for(long i=0;i<N;i++){ M0+=orig[i]; S0+=(double)orig[i]*orig[i]; }
  double q=0.02; for(long i=0;i<N;i++) x[i]=(float)(round(x[i]/q)*q);
  double Mq=0,Sq=0; for(long i=0;i<N;i++){ Mq+=x[i]; Sq+=(double)x[i]*x[i]; }
  conserve_mass_energy(x,N,M0,S0);
  double Mc=0,Sc=0; for(long i=0;i<N;i++){ Mc+=x[i]; Sc+=(double)x[i]*x[i]; }
  printf("[invariants] mass  %.3e -> %.3e\n", fabs(Mq-M0)/fabs(M0), fabs(Mc-M0)/fabs(M0));
  printf("[invariants] energy %.3e -> %.3e\n", fabs(Sq-S0)/fabs(S0), fabs(Sc-S0)/fabs(S0));

  /* ---- precompute path: store (a,b), single-pass decode == two-pass ---- */
  { float* y=new float[N]; for(long i=0;i<N;i++) y[i]=(float)(round(orig[i]/q)*q); /* fresh xhat */
    float* z=new float[N]; for(long i=0;i<N;i++) z[i]=y[i];
    conserve_mass_energy(y,N,M0,S0);                 /* two-pass reference */
    double a,b; conserve_mass_energy_params(z,N,M0,S0,&a,&b);
    conserve_apply_affine(z,N,a,b);                  /* single-pass from stored (a,b) */
    double md=0; for(long i=0;i<N;i++){ double d=fabs((double)y[i]-z[i]); if(d>md)md=d; }
    printf("[precompute] store (a=%.6f,b=%.4g); single-pass vs two-pass max diff = %.2e  %s\n",
           a,b,md, md==0.0?"identical":"(float rounding)");
    delete[] y; delete[] z; }

  /* ---- precompute path, linear invariants: params (1,b) + single pass == two-pass ---- */
  { float* y=new float[N]; float* z=new float[N]; float* w=new float[N];
    for(long i=0;i<N;i++) w[i]=(float)(1.0+0.5*sin(0.37*i));      /* density-like weight in [0.5,1.5]; no rand() so later sections are unchanged */
    double a,b,md;
    /* unweighted (mass): conserve_linear vs conserve_linear_params(w=NULL) */
    for(long i=0;i<N;i++){ y[i]=(float)(round(orig[i]/q)*q); z[i]=y[i]; }
    conserve_linear(y,N,M0);                                       /* two-pass reference */
    conserve_linear_params(z,N,M0,NULL,&a,&b);                     /* encoder: (1,b) */
    conserve_apply_affine(z,N,a,b);                                /* decoder: one pass */
    md=0; for(long i=0;i<N;i++){ double d=fabs((double)y[i]-z[i]); if(d>md)md=d; }
    double Ml=0; for(long i=0;i<N;i++) Ml+=z[i];
    printf("[precompute linear] mass:     single vs two-pass max diff = %.2e  %s; mass drift -> %.2e\n",
           md, md==0.0?"identical":"(float rounding)", fabs(Ml-M0)/fabs(M0));
    /* weighted (momentum, w = density): conserve_linear_w vs conserve_linear_params(w) */
    double L0=0; for(long i=0;i<N;i++) L0+=(double)w[i]*orig[i];   /* true momentum */
    for(long i=0;i<N;i++){ y[i]=(float)(round(orig[i]/q)*q); z[i]=y[i]; }
    double Lq=0; for(long i=0;i<N;i++) Lq+=(double)w[i]*z[i];
    conserve_linear_w(y,N,L0,w);                                   /* two-pass reference */
    conserve_linear_params(z,N,L0,w,&a,&b);                        /* encoder: (1,b) */
    conserve_apply_affine(z,N,a,b);                                /* decoder: one pass, no w */
    md=0; for(long i=0;i<N;i++){ double d=fabs((double)y[i]-z[i]); if(d>md)md=d; }
    double Lc=0; for(long i=0;i<N;i++) Lc+=(double)w[i]*z[i];
    printf("[precompute linear] momentum: single vs two-pass max diff = %.2e  %s; momentum drift %.2e -> %.2e\n",
           md, md==0.0?"identical":"(float rounding)", fabs(Lq-L0)/fabs(L0), fabs(Lc-L0)/fabs(L0));
    delete[] y; delete[] z; delete[] w; }

  /* ---- 2a. ABS: field with far-from-mean outliers; error within b'=(1-e)eps ---- */
  for(long i=0;i<N;i++) orig[i]=(float)urand(9.0,11.0);          /* mean ~10 */
  for(long i=0;i<2000;i++) orig[i]=(float)(10.0+(frand()<.5?-1:1)*urand(6,9)); /* outliers */
  M0=0;S0=0; for(long i=0;i<N;i++){ M0+=orig[i]; S0+=(double)orig[i]*orig[i]; }
  double rng=0; { double mn=orig[0],mx=orig[0];
    for(long i=0;i<N;i++){ if(orig[i]<mn)mn=orig[i]; if(orig[i]>mx)mx=orig[i]; } rng=mx-mn; }
  double eps=0.01*rng, e=0.10, bp=(1-e)*eps;
  /* linear (mass) under ABS */
  for(long i=0;i<N;i++) x[i]=(float)(orig[i]+urand(-bp,bp));
  conserve_linear(x,N,M0);
  double mdl=0; for(long i=0;i<N;i++){ double d=fabs((double)x[i]-orig[i]); if(d>mdl)mdl=d; }
  printf("[ABS mass ] max|C-x|/eps = %.3f  %s\n", mdl/eps, mdl<=eps?"OK":"VIOLATES");
  /* mass+energy under ABS: flat tightening vs. maxdisp-aware tightening */
  for(long i=0;i<N;i++) x[i]=(float)(orig[i]+urand(-bp,bp));
  double md_flat = conserve_mass_energy_maxdisp(x,N,M0,S0);   /* measured before correcting */
  conserve_mass_energy(x,N,M0,S0);
  double mde=0; for(long i=0;i<N;i++){ double d=fabs((double)x[i]-orig[i]); if(d>mde)mde=d; }
  printf("[ABS mass+E] flat tighten: max|C-x|/eps = %.3f  %s  (maxdisp=%.3g, b'+maxdisp=%.3g vs eps=%.3g)\n",
         mde/eps, mde<=eps?"OK":"VIOLATES (expected: tighten by maxdisp)", md_flat, bp+md_flat, eps);
  /* re-tighten so b'' + maxdisp <= eps, recompress, re-check */
  double bpp = eps - md_flat; if(bpp<0) bpp=0;
  for(long i=0;i<N;i++) x[i]=(float)(orig[i]+urand(-bpp,bpp));
  conserve_mass_energy(x,N,M0,S0);
  double mde2=0; for(long i=0;i<N;i++){ double d=fabs((double)x[i]-orig[i]); if(d>mde2)mde2=d; }
  printf("[ABS mass+E] maxdisp-aware: max|C-x|/eps = %.3f  %s\n", mde2/eps, mde2<=eps?"OK":"VIOLATES");

  /* ---- 2b. REL: single-sign field with near-zero values ---- */
  for(long i=0;i<N;i++){ double g=urand(-1,1); orig[i]=(float)(g*g+1e-6); } /* >=1e-6 */
  M0=0;S0=0; for(long i=0;i<N;i++){ M0+=orig[i]; S0+=(double)orig[i]*orig[i]; }
  double r=1e-3, rp=(1-e)*r;
  /* additive offset is WRONG under REL -> should blow up near zero */
  for(long i=0;i<N;i++) x[i]=(float)(orig[i]*(1+urand(-rp,rp)));
  conserve_linear(x,N,M0);
  double rl=0; for(long i=0;i<N;i++){ double d=fabs(1-(double)x[i]/orig[i]); if(d>rl)rl=d; }
  printf("[REL mass (additive, WRONG)] max rel/r = %.3g  %s\n", rl/r, rl<=r?"OK":"VIOLATES (expected)");
  /* multiplicative mass scale -> preserves REL */
  for(long i=0;i<N;i++) x[i]=(float)(orig[i]*(1+urand(-rp,rp)));
  conserve_apply_rel(x,N,CONS_MASS,M0,S0);
  double rm=0; for(long i=0;i<N;i++){ double d=fabs(1-(double)x[i]/orig[i]); if(d>rm)rm=d; }
  printf("[REL mass (scale)         ] max rel/r = %.3f  %s\n", rm/r, rm<=r?"OK":"VIOLATES");
  /* multiplicative energy scale -> preserves REL */
  for(long i=0;i<N;i++) x[i]=(float)(orig[i]*(1+urand(-rp,rp)));
  conserve_apply_rel(x,N,CONS_ENERGY,M0,S0);
  double re=0; for(long i=0;i<N;i++){ double d=fabs(1-(double)x[i]/orig[i]); if(d>re)re=d; }
  printf("[REL energy (scale)       ] max rel/r = %.3f  %s\n", re/r, re<=r?"OK":"VIOLATES");
  /* joint under REL is unsupported as a single affine -> returns 0, no-op */
  int ok = conserve_apply_rel(x,N,CONS_MASS|CONS_ENERGY,M0,S0);
  printf("[REL mass+energy          ] conserve_apply_rel returned %d (0 = correctly refused)\n", ok);

  /* ---- 3. CAPPED correction on a heavy-tailed field, bound-filling codec ----
     HARSH synthetic case: i.i.d. lognormal values (huge range, no spatial smoothness)
     and errors that fill the bound like SZ3/SLEEK: xhat = x + U(-b', b').
     Plain mass+energy (10% headroom) overshoots badly; the capped correction (20%
     headroom) cuts the overshoot by ~15x, but at the loosest bound this worst case
     still exceeds slightly -- a fixed headroom is not a guarantee for every field.
     On the real ISABEL/NYX/Miranda data, capped + 20% headroom held the bound.
     For a strict guarantee, verify at encode (decompress, check, retighten). */
  for(long i=0;i<N;i++){ double u1=frand()+1e-12, u2=frand();
    double z=sqrt(-2*log(u1))*cos(2*M_PI*u2); orig[i]=(float)exp(2.0*z); }   /* lognormal: huge range */
  M0=0;S0=0; double mn=orig[0],mx=orig[0];
  for(long i=0;i<N;i++){ M0+=orig[i]; S0+=(double)orig[i]*orig[i]; if(orig[i]<mn)mn=orig[i]; if(orig[i]>mx)mx=orig[i]; }
  double cap=conserve_cap_from_original(orig,N,M0/N,97.0), mu0=M0/N;
  float* y=new float[N]; float* z=new float[N];
  const double fracs[2]={1e-2, 1e-3};
  for(int t=0;t<2;t++){
    double frac=fracs[t], bb=frac*(mx-mn);
    /* plain, 10% headroom */
    for(long i=0;i<N;i++) x[i]=(float)(orig[i]+urand(-0.9*bb,0.9*bb));
    conserve_mass_energy(x,N,M0,S0);
    double mp=0; for(long i=0;i<N;i++){ double d=fabs((double)x[i]-orig[i]); if(d>mp)mp=d; }
    /* capped, 20% headroom: two-pass decode */
    for(long i=0;i<N;i++){ x[i]=(float)(orig[i]+urand(-0.8*bb,0.8*bb)); y[i]=x[i]; z[i]=x[i]; }
    conserve_mass_energy_capped(x,N,M0,S0,cap);
    double mc=0,Mc=0,Sc=0; for(long i=0;i<N;i++){ double d=fabs((double)x[i]-orig[i]); if(d>mc)mc=d; Mc+=x[i]; Sc+=(double)x[i]*x[i]; }
    /* capped, precompute path: params at encode, one-pass apply at decode == two-pass */
    double a=0,b=0,cu=0; conserve_mass_energy_capped_params(y,N,M0,S0,cap,&a,&b,&cu);
    conserve_apply_capped(y,N,a,b,mu0,cu);
    double dd=0; for(long i=0;i<N;i++){ double d=fabs((double)y[i]-x[i]); if(d>dd)dd=d; }
    /* capped, energy only */
    conserve_energy_capped(z,N,S0,mu0,cap);
    double me=0,Se=0; for(long i=0;i<N;i++){ double d=fabs((double)z[i]-orig[i]); if(d>me)me=d; Se+=(double)z[i]*z[i]; }
    printf("[capped %.0e] plain M+E (10%%): max|C-x|/eps = %.2f %s | capped M+E (20%%): %.2f %s, drift E %.1e M %.1e | precompute == two-pass: %s | capped E only: %.2f %s, drift E %.1e\n",
           frac, mp/bb, mp<=bb?"OK":"VIOLATES",
           mc/bb, mc<=bb?"OK":"exceeds (harsh case)", fabs(Sc-S0)/S0, fabs(Mc-M0)/fabs(M0),
           dd==0.0?"identical":"differs",
           me/bb, me<=bb?"OK":"exceeds (harsh case)", fabs(Se-S0)/S0);
  }
  delete[] y; delete[] z;

  delete[] x; delete[] orig; return 0;
}
