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

  delete[] x; delete[] orig; return 0;
}
