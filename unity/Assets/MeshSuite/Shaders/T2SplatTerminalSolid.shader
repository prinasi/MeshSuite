// T²-Splat terminal deployment pass.
//
// T²-Splat's Stage I mesh is a MeshSplatting surface (sigma -> 1e-4, rendered with
// opacity floor 0.999), so, as in MeshSplatTerminalSolid, the engine depth buffer
// stands in for the limiting front-to-back compositor and each vertex keeps its
// learned SH colour (noperspective screen-space barycentrics, as natively).
// Stage II adds a residual RGB texture per face, sampled exactly as the CUDA
// rasterizer does: bilinear at (wB*(R-1), wC*(R-1)) where wB/wC weight the
// face's second/third vertex, trilinear across the Mip pyramid with a constant
// per-triangle LOD = log2(max screen-space texel footprint), measured on the
// native 4x-supersampled pixel grid (_LodPixelScale).
Shader "MeshSuite/T2SplatTerminalSolid"
{
 SubShader { Tags { "Queue"="Geometry" "RenderType"="Opaque" } Pass {
  Cull Off ZWrite On ZTest LEqual Blend Off
  CGPROGRAM
  #pragma target 5.0
  #pragma only_renderers metal vulkan
  #pragma vertex Vert
  #pragma fragment Frag
  #include "UnityCG.cginc"
  ByteAddressBuffer _Positions,_Indices,_ShDc,_ShRest,_FaceMip;
  int _PrimitiveCount,_ShDegree,_RawCode,_HasFaceTexture,_FaceTexRes,_FaceMipLevels,_FaceTexelsPerFace;
  float _LodPixelScale; float3 _CameraWorldPos;
  float F(ByteAddressBuffer b,uint i){return asfloat(b.Load(i*4));} uint I(ByteAddressBuffer b,uint i){return b.Load(i*4);}
  float3 Pos(uint i){uint k=i*3;return float3(F(_Positions,k),F(_Positions,k+1),F(_Positions,k+2));}
  float3 Dc(uint e){uint k=e*3;return float3(F(_ShDc,k),F(_ShDc,k+1),F(_ShDc,k+2));}
  float3 Rest(uint e,uint c){uint k=e*45+c*3;return float3(F(_ShRest,k),F(_ShRest,k+1),F(_ShRest,k+2));}
  float3 Sh(uint e,float3 d){ float3 r=0.28209479177387814*Dc(e); if(_ShDegree<1)return max(r+.5,0); float x=d.x,y=d.y,z=d.z,xx=x*x,yy=y*y,zz=z*z,xy=x*y,yz=y*z,xz=x*z; r+=-.4886025119*y*Rest(e,0)+.4886025119*z*Rest(e,1)-.4886025119*x*Rest(e,2); if(_ShDegree>1)r+=1.09254843*xy*Rest(e,3)-1.09254843*yz*Rest(e,4)+.315391565*(2*zz-xx-yy)*Rest(e,5)-1.09254843*xz*Rest(e,6)+.546274215*(xx-yy)*Rest(e,7); if(_ShDegree>2)r+=-.59004359*y*(3*xx-yy)*Rest(e,8)+2.89061144*xy*z*Rest(e,9)-.4570458*y*(4*zz-xx-yy)*Rest(e,10)+.37317633*z*(2*zz-3*xx-3*yy)*Rest(e,11)-.4570458*x*(4*zz-xx-yy)*Rest(e,12)+1.44530572*z*(xx-yy)*Rest(e,13)-.59004359*x*(xx-3*yy)*Rest(e,14); return max(r+.5,0); }

  // Residual Mip pyramid: per face, levels R, R>>1, ..., 1 stored row-major
  // (row = wC, column = wB), RGB IEEE halves packed two per uint32 word.
  float Half(uint i){uint w=_FaceMip.Load((i>>1)*4);return f16tof32((i&1)!=0?(w>>16):(w&0xffff));}
  int LevelRes(int level){return max(_FaceTexRes>>level,1);}
  uint LevelOffset(int level){uint o=0;for(int l=0;l<level;++l){uint r=(uint)LevelRes(l);o+=r*r;}return o;}
  float3 Texel(uint face,uint base,int res,int x,int y){uint k=(face*(uint)_FaceTexelsPerFace+base+(uint)(y*res+x))*3;return float3(Half(k),Half(k+1),Half(k+2));}
  float3 Bilinear(uint face,int level,float wB,float wC){
   int res=LevelRes(level); uint base=LevelOffset(level);
   float lx=wB*(res-1),ly=wC*(res-1);
   int x0=clamp((int)floor(lx),0,res-1),y0=clamp((int)floor(ly),0,res-1),x1=min(x0+1,res-1),y1=min(y0+1,res-1);
   float fx=saturate(lx-x0),fy=saturate(ly-y0);
   return (1-fx)*(1-fy)*Texel(face,base,res,x0,y0)+fx*(1-fy)*Texel(face,base,res,x1,y0)
        +(1-fx)*fy*Texel(face,base,res,x0,y1)+fx*fy*Texel(face,base,res,x1,y1); }
  float3 Residual(uint face,float wB,float wC,float lod){
   lod=clamp(lod,0,(float)(_FaceMipLevels-1)); int lo=(int)floor(lod),hi=min(lo+1,_FaceMipLevels-1); float t=lod-lo;
   return (1-t)*Bilinear(face,lo,wB,wC)+t*Bilinear(face,hi,wB,wC); }
  float2 Pix(float3 p){float4 c=mul(UNITY_MATRIX_VP,float4(p,1));return c.xy/c.w*0.5*_ScreenParams.xy*_LodPixelScale;}
  float TriangleLod(float3 a,float3 b,float3 c){
   if(_FaceTexRes<=1)return 0;
   float2 p0=Pix(a),e0=Pix(b)-p0,e1=Pix(c)-p0; float inv=1.0/(e0.x*e1.y-e1.x*e0.y+1e-12);
   float dudx=e1.y*inv,dudy=-e1.x*inv,dvdx=-e0.y*inv,dvdy=e0.x*inv;
   float rho=max(sqrt(dudx*dudx+dvdx*dvdx),sqrt(dudy*dudy+dvdy*dvdy))*_FaceTexRes;
   return log2(max(rho,1e-8)); }

  struct V{float4 p:SV_POSITION;noperspective float3 c:TEXCOORD0;noperspective float2 w:TEXCOORD1;nointerpolation uint face:TEXCOORD2;nointerpolation float lod:TEXCOORD3;};
  V Vert(uint id:SV_VertexID){
   V o;uint t=id/3,corner=id-t*3,ib=t*3;uint vi=I(_Indices,ib+corner);float3 p=Pos(vi);
   o.p=mul(UNITY_MATRIX_VP,float4(p,1));o.c=Sh(vi,normalize(p-_CameraWorldPos));
   o.w=corner==1?float2(1,0):(corner==2?float2(0,1):float2(0,0));o.face=t;
   o.lod=_HasFaceTexture!=0?TriangleLod(Pos(I(_Indices,ib)),Pos(I(_Indices,ib+1)),Pos(I(_Indices,ib+2))):0;
   return o; }
  float4 Frag(V i):SV_Target{
   float3 c=i.c; if(_HasFaceTexture!=0)c+=Residual(i.face,i.w.x,i.w.y,i.lod);
   c=_RawCode!=0?c:GammaToLinearSpace(max(c,0));return float4(c,1); }
  ENDCG
 } }
}
