// Mesh-Splatting terminal deployment pass.
//
// The native trainer anneals sigma to 1e-4 and the opacity floor to 0.9999.
// Thus its front-to-back splat compositor converges to a solid indexed surface:
// each pixel shows the first covering triangle in the compositor's order, which
// sorts triangles by the view depth of their centroids.  Pass 0 reproduces that
// visibility rule with the hardware depth test by giving every fragment of a
// triangle its centroid's depth; it also applies the native centroid near cull
// (view depth <= 0.2) and retains the learned degree-3 SH appearance.  The
// noperspective qualifiers intentionally match the native rasterizer's
// screen-space barycentric color interpolation rather than flat
// (provoking-vertex) shading; centroid sampling keeps MSAA from extrapolating
// colors outside small triangles.
//
// The native renderer also draws at native_render_scaling x the output
// resolution and area-averages down.  MethodSpecificSplatRenderer reproduces it
// by drawing pass 0 into an enlarged target and resolving it with pass 1.
Shader "MeshSuite/MeshSplatTerminalSolid"
{
 Properties { _MainTex ("Supersampled target", 2D) = "black" {} }
 SubShader { Tags { "Queue"="Geometry" "RenderType"="Opaque" } Pass {
  Cull Off ZWrite On ZTest LEqual Blend Off
  CGPROGRAM
  #pragma target 5.0
  #pragma only_renderers metal vulkan
  #pragma vertex Vert
  #pragma fragment Frag
  #include "UnityCG.cginc"
  ByteAddressBuffer _Positions,_Indices,_ShDc,_ShRest;
  int _PrimitiveCount,_ShDegree,_RawCode; float3 _CameraWorldPos;
  float F(ByteAddressBuffer b,uint i){return asfloat(b.Load(i*4));} uint I(ByteAddressBuffer b,uint i){return b.Load(i*4);}
  float3 Pos(uint i){uint k=i*3;return float3(F(_Positions,k),F(_Positions,k+1),F(_Positions,k+2));}
  float3 Dc(uint e){uint k=e*3;return float3(F(_ShDc,k),F(_ShDc,k+1),F(_ShDc,k+2));}
  float3 Rest(uint e,uint c){uint k=e*45+c*3;return float3(F(_ShRest,k),F(_ShRest,k+1),F(_ShRest,k+2));}
  float3 Sh(uint e,float3 d){ float3 r=0.28209479177387814*Dc(e); if(_ShDegree<1)return max(r+.5,0); float x=d.x,y=d.y,z=d.z,xx=x*x,yy=y*y,zz=z*z,xy=x*y,yz=y*z,xz=x*z; r+=-.4886025119*y*Rest(e,0)+.4886025119*z*Rest(e,1)-.4886025119*x*Rest(e,2); if(_ShDegree>1)r+=1.09254843*xy*Rest(e,3)-1.09254843*yz*Rest(e,4)+.315391565*(2*zz-xx-yy)*Rest(e,5)-1.09254843*xz*Rest(e,6)+.546274215*(xx-yy)*Rest(e,7); if(_ShDegree>2)r+=-.59004359*y*(3*xx-yy)*Rest(e,8)+2.89061144*xy*z*Rest(e,9)-.4570458*y*(4*zz-xx-yy)*Rest(e,10)+.37317633*z*(2*zz-3*xx-3*yy)*Rest(e,11)-.4570458*x*(4*zz-xx-yy)*Rest(e,12)+1.44530572*z*(xx-yy)*Rest(e,13)-.59004359*x*(xx-3*yy)*Rest(e,14); return max(r+.5,0); }
  struct V{float4 p:SV_POSITION;noperspective centroid float3 c:TEXCOORD0;};
  V Vert(uint id:SV_VertexID){
   V o;uint t=id/3,corner=id-t*3,ib=t*3;uint vi=I(_Indices,ib+corner);float3 p=Pos(vi);o.p=mul(UNITY_MATRIX_VP,float4(p,1));
   // Native visibility: one depth per triangle, taken at its centroid.
   float3 cen=(Pos(I(_Indices,ib))+Pos(I(_Indices,ib+1))+Pos(I(_Indices,ib+2)))/3.0;
   float4 cc=mul(UNITY_MATRIX_VP,float4(cen,1));
   if(cc.w<=0.2) o.p=float4(2,2,2,1); // native near cull on the centroid; outside the frustum
   else o.p.z=cc.z/cc.w*o.p.w;
   o.c=Sh(vi,normalize(p-_CameraWorldPos));return o;}
  float4 Frag(V i):SV_Target{float3 c=_RawCode!=0?i.c:GammaToLinearSpace(i.c);return float4(c,1);}
  ENDCG
 }
 Pass {
  // Box-filter resolve of the native_render_scaling x target (area downsampling).
  Cull Off ZWrite Off ZTest Always Blend Off
  CGPROGRAM
  #pragma target 5.0
  #pragma only_renderers metal vulkan
  #pragma vertex vert_img
  #pragma fragment Down
  #include "UnityCG.cginc"
  Texture2D _MainTex; float4 _MainTex_TexelSize; int _SupersampleFactor;
  float4 Down(v2f_img i):SV_Target{
   int k=max(_SupersampleFactor,1); float2 dst=_MainTex_TexelSize.zw/k; int2 q=int2(floor(i.uv*dst)); float4 s=0;
   for(int y=0;y<k;y++) for(int x=0;x<k;x++) s+=_MainTex.Load(int3(q*k+int2(x,y),0));
   return s/(k*k); }
  ENDCG
 } }
}
