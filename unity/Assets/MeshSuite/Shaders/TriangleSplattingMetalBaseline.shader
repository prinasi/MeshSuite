Shader "MeshSuite/TriangleSplattingMetalBaseline"
{
    SubShader
    {
        Tags { "Queue"="Transparent" "RenderType"="Transparent" }
        Pass
        {
            Cull Off
            ZWrite Off
            ZTest Always
            Blend OneMinusDstAlpha One

            CGPROGRAM
            #pragma target 5.0
            #pragma only_renderers metal vulkan
            #pragma vertex Vert
            #pragma fragment Frag
            #include "UnityCG.cginc"

            ByteAddressBuffer _Positions;
            ByteAddressBuffer _Indices;
            ByteAddressBuffer _OpacityLogits;
            ByteAddressBuffer _SigmaLogits;
            ByteAddressBuffer _ShDc;
            ByteAddressBuffer _ShRest;
            ByteAddressBuffer _TriangleOrder;
            uint _PrimitiveCount;
            int _OutputRawCodeValues;

            float PositionValue(uint index) { return asfloat(_Positions.Load(index * 4)); }
            uint IndexValue(uint index) { return _Indices.Load(index * 4); }
            float OpacityValue(uint tri) { return asfloat(_OpacityLogits.Load(tri * 4)); }
            float SigmaValue(uint tri) { return asfloat(_SigmaLogits.Load(tri * 4)); }
            float ShDcValue(uint index) { return asfloat(_ShDc.Load(index * 4)); }
            float ShRestValue(uint index) { return asfloat(_ShRest.Load(index * 4)); }

            float3 LoadPosition(uint vertexIndex)
            {
                uint b = vertexIndex * 3;
                return float3(PositionValue(b), PositionValue(b + 1), PositionValue(b + 2));
            }

            float3 LoadCoefficient(uint tri, uint coefficient)
            {
                if (coefficient == 0)
                {
                    uint b = tri * 3;
                    return float3(ShDcValue(b), ShDcValue(b + 1), ShDcValue(b + 2));
                }
                uint r = tri * 45 + (coefficient - 1) * 3;
                return float3(ShRestValue(r), ShRestValue(r + 1), ShRestValue(r + 2));
            }

            float3 EvalSh3(uint tri, float3 direction)
            {
                const float C0 = 0.28209479177387814;
                const float C1 = 0.4886025119029199;
                float x = direction.x, y = direction.y, z = direction.z;
                float xx = x*x, yy = y*y, zz = z*z;
                float xy = x*y, yz = y*z, xz = x*z;
                float3 result = C0 * LoadCoefficient(tri, 0);
                result += -C1*y*LoadCoefficient(tri,1) + C1*z*LoadCoefficient(tri,2) - C1*x*LoadCoefficient(tri,3);
                result += 1.0925484305920792*xy*LoadCoefficient(tri,4);
                result += -1.0925484305920792*yz*LoadCoefficient(tri,5);
                result += 0.31539156525252005*(2.0*zz-xx-yy)*LoadCoefficient(tri,6);
                result += -1.0925484305920792*xz*LoadCoefficient(tri,7);
                result += 0.5462742152960396*(xx-yy)*LoadCoefficient(tri,8);
                result += -0.5900435899266435*y*(3.0*xx-yy)*LoadCoefficient(tri,9);
                result += 2.890611442640554*xy*z*LoadCoefficient(tri,10);
                result += -0.4570457994644658*y*(4.0*zz-xx-yy)*LoadCoefficient(tri,11);
                result += 0.3731763325901154*z*(2.0*zz-3.0*xx-3.0*yy)*LoadCoefficient(tri,12);
                result += -0.4570457994644658*x*(4.0*zz-xx-yy)*LoadCoefficient(tri,13);
                result += 1.445305721320277*z*(xx-yy)*LoadCoefficient(tri,14);
                result += -0.5900435899266435*x*(xx-3.0*yy)*LoadCoefficient(tri,15);
                return max(result + 0.5, 0.0);
            }

            struct Varyings
            {
                float4 position : SV_POSITION;
                noperspective float3 barycentric : TEXCOORD0;
                nointerpolation float3 color : TEXCOORD1;
                nointerpolation float opacity : TEXCOORD2;
                nointerpolation float sigma : TEXCOORD3;
                nointerpolation float3 edgeScale : TEXCOORD4;
            };

            Varyings Vert(uint vertexId : SV_VertexID)
            {
                Varyings o;
                uint drawTriangle = vertexId / 3;
                uint corner = vertexId - drawTriangle * 3;
                uint tri = _TriangleOrder.Load(drawTriangle * 4);
                uint ib = tri * 3;
                float3 p0 = LoadPosition(IndexValue(ib));
                float3 p1 = LoadPosition(IndexValue(ib + 1));
                float3 p2 = LoadPosition(IndexValue(ib + 2));
                float3 world = corner == 0 ? p0 : (corner == 1 ? p1 : p2);
                float3 center = (p0 + p1 + p2) / 3.0;
                o.position = mul(UNITY_MATRIX_VP, float4(world, 1.0));
                o.barycentric = corner == 0 ? float3(1,0,0) : (corner == 1 ? float3(0,1,0) : float3(0,0,1));
                o.opacity = rcp(1.0 + exp(-OpacityValue(tri)));
                o.sigma = 0.01 + exp(clamp(SigmaValue(tri), -20.0, 20.0));
                // The native rasterizer's SH position is the comma expression
                // (glm::vec3)(center.x, center.y, center.z), which is vec3(center.z)
                // (triangle_splatting_rasterization forward.cu). Training optimized
                // the SH coefficients under that input, so deployment reproduces it.
                o.color = EvalSh3(tri, normalize(center.zzz - _WorldSpaceCameraPos));

                float4 clip0 = mul(UNITY_MATRIX_VP, float4(p0, 1.0));
                float4 clip1 = mul(UNITY_MATRIX_VP, float4(p1, 1.0));
                float4 clip2 = mul(UNITY_MATRIX_VP, float4(p2, 1.0));
                float4 clipCenter = mul(UNITY_MATRIX_VP, float4(center, 1.0));
                float2 s0 = (clip0.xy / clip0.w) * 0.5 * _ScreenParams.xy;
                float2 s1 = (clip1.xy / clip1.w) * 0.5 * _ScreenParams.xy;
                float2 s2 = (clip2.xy / clip2.w) * 0.5 * _ScreenParams.xy;
                float2 sc = (clipCenter.xy / clipCenter.w) * 0.5 * _ScreenParams.xy;
                float a = length(s1 - s2), b = length(s0 - s2), c = length(s0 - s1);
                float perimeter = a + b + c;
                // The native window is the edge distance over the screen-space inradius,
                // so it peaks at the incenter. barycentric_i = d_i / h_i and h_i / r = perimeter / side_i.
                o.edgeScale = perimeter / max(float3(a, b, c), 1e-6);

                // The native rasterizer's culling rules (forward.cu).
                float2 e1 = s1 - s0, e2 = s2 - s0;
                float inradius = abs(e1.x * e2.y - e1.y * e2.x) / max(perimeter, 1e-6);
                float extent = max(length(s0 - sc), max(length(s1 - sc), length(s2 - sc)));
                float3 normal = normalize(cross(p1 - p0, p2 - p0));
                float cosine = abs(dot(normal, normalize(center - _WorldSpaceCameraPos)));
                if (clipCenter.w <= 0.2 || o.opacity < 0.01 || inradius < 1.0 || extent < 1.0 || extent > 1600.0 || cosine < 0.001)
                    o.position = float4(2, 2, 2, 1);
                return o;
            }

            float4 Frag(Varyings i) : SV_Target
            {
                float3 scaled = i.barycentric * i.edgeScale;
                float phi = saturate(min(scaled.x, min(scaled.y, scaled.z)));
                float coverage = pow(max(phi, 1e-6), i.sigma);
                float alpha = min(0.99, i.opacity * coverage);
                clip(alpha - (1.0 / 255.0));
                // The native trainer compares directly against PIL RGB / 255,
                // i.e. image code values rather than a linear-light decode.
                // Evaluation therefore renders into a non-sRGB target and keeps
                // these values untouched. The display path converts them once
                // for Unity's Linear Color Space output.
                float3 color = _OutputRawCodeValues != 0 ? i.color : GammaToLinearSpace(i.color);
                return float4(color * alpha, alpha);
            }
            ENDCG
        }
    }
}
