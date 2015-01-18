/*******************************************************************************
* Piotr's Computer Vision Matlab Toolbox      Version 3.24
* Copyright 2014 Piotr Dollar.  [pdollar-at-gmail.com]
* Licensed under the Simplified BSD License [see external/bsd.txt]
*******************************************************************************/

#include <cstring>
#include <cstdint>
#include <cmath>
#include <iostream>
using namespace std;

#include "_random_forests.h"

#define gini(p) p*p
#define entropy(p) (-p*flog2(float(p)))


// fast approximate log2(x) from Paul Mineiro <paul@mineiro.com>
inline float flog2( float x ) {
  union { float f; uint32_t i; } vx = { x };
  union { uint32_t i; float f; } mx = { (vx.i & 0x007FFFFF) | 0x3f000000 };
  float y = float(vx.i); y *= 1.1920928955078125e-7f;
  return y - 124.22551499f - 1.498030302f * mx.f
    - 1.72587999f / (0.3520887068f + mx.f);
}

// perform actual computation
void forestFindThr(int H, int N, int F, const float *data, const int *hs,
                   const float* ws, const int *order, const int split,
                   int &fid, float &thr, double &gain)
{
  double *Wl, *Wr, *W; float *data1; int *order1;
  int i, j, j1, j2, h; double vBst, vInit, v, w, wl, wr, g, gl, gr;
  Wl=new double[H]; Wr=new double[H]; W=new double[H];
  // perform initialization
  vBst = vInit = 0; g = 0; w = 0; fid = 0; thr = 0;
  for( i=0; i<H; i++ ) W[i] = 0;
  for( j=0; j<N; j++ ) { w+=ws[j]; W[hs[j]]+=ws[j]; }
  if( split==0 ) { for( i=0; i<H; i++ ) g+=gini(W[i]); vBst=vInit=(1-g/w/w); }
  if( split==1 ) { for( i=0; i<H; i++ ) g+=entropy(W[i]); vBst=vInit=g/w; }
  // loop over features, then thresholds (data is sorted by feature value)
  for( i=0; i<F; i++ ) {
    order1=(int*) order+i*N; data1=(float*) data+i*size_t(N);
    for( j=0; j<H; j++ ) { Wl[j]=0; Wr[j]=W[j]; } gl=wl=0; gr=g; wr=w;
    for( j=0; j<N-1; j++ ) {
      j1=order1[j]; j2=order1[j+1]; h=hs[j1];
      if(split==0) {
        // gini = 1-\sum_h p_h^2; v = gini_l*pl + gini_r*pr
        wl+=ws[j1]; gl-=gini(Wl[h]); Wl[h]+=ws[j1]; gl+=gini(Wl[h]);
        wr-=ws[j1]; gr-=gini(Wr[h]); Wr[h]-=ws[j1]; gr+=gini(Wr[h]);
        v = (wl-gl/wl)/w + (wr-gr/wr)/w;
      } else if (split==1) {
        // entropy = -\sum_h p_h log(p_h); v = entropy_l*pl + entropy_r*pr
        gl+=entropy(wl); wl+=ws[j1]; gl-=entropy(wl);
        gr+=entropy(wr); wr-=ws[j1]; gr-=entropy(wr);
        gl-=entropy(Wl[h]); Wl[h]+=ws[j1]; gl+=entropy(Wl[h]);
        gr-=entropy(Wr[h]); Wr[h]-=ws[j1]; gr+=entropy(Wr[h]);
        v = gl/w + gr/w;
      } else {
        // twoing: v = pl*pr*\sum_h(|p_h_left - p_h_right|)^2 [slow if H>>0]
        j1=order1[j]; j2=order1[j+1]; h=hs[j1];
        wl+=ws[j1]; Wl[h]+=ws[j1]; wr-=ws[j1]; Wr[h]-=ws[j1];
        g=0; for( int h1=0; h1<H; h1++ ) g+=fabs(Wl[h1]/wl-Wr[h1]/wr);
        v = - wl/w*wr/w*g*g;
      }
      if( v<vBst && data1[j2]-data1[j1]>=1e-6f ) {
        vBst=v; fid=i; thr=0.5f*(data1[j1]+data1[j2]); }
    }
  }
  delete [] Wl; delete [] Wr; delete [] W; gain = vInit-vBst;
}