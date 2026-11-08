Shader "MeshSuite/StandardTextureCutoutRaw"
{
    Properties { _MainTex ("Baked RGBA atlas", 2D) = "white" {} }
    SubShader
    {
        Tags { "RenderType"="TransparentCutout" "Queue"="AlphaTest" }
        Pass
        {
            // Same raster state as StandardVertexColorRaw; only the appearance differs.
            Cull Off
            ZWrite On
            ZTest LEqual
            Blend Off
            CGPROGRAM
            #pragma target 3.5
            #pragma vertex Vert
            #pragma fragment Frag
            #pragma only_renderers metal vulkan
            #include "UnityCG.cginc"
            sampler2D _MainTex;
            struct AppData { float4 vertex : POSITION; float2 uv : TEXCOORD0; };
            // Atlas tiles are packed tightly: with MSAA, centroid sampling keeps
            // partially covered pixels from reading a neighbouring triangle's tile.
            struct Varyings { float4 position : SV_POSITION; centroid float2 uv : TEXCOORD0; };
            Varyings Vert(AppData v) { Varyings o; o.position = UnityObjectToClipPos(v.vertex); o.uv = v.uv; return o; }
            float4 Frag(Varyings i) : SV_Target { float4 c = tex2D(_MainTex, i.uv); clip(c.a - 0.5); return float4(c.rgb, 1); }
            ENDCG
        }
    }
}
