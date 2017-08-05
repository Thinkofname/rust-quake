#version 150 core

in vec3 a_position;
in uint a_light;
in uvec2 a_tex;
in ivec4 a_texInfo;
in ivec2 a_lightInfo;
in uint a_lightType;

uniform Transform {
    mat4 pMat;
    mat4 uMat;
    // float lightStyles[11];
};

const float lightStyles[11] = float[11](
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0
);

out vec2 v_tex;
out vec4 v_texInfo;
out float v_light;
out vec2 v_lightInfo;
out float v_lightType;
out vec2 v_pos;

const float invTextureSize = 1.0 / 1024.0;
const float invPackSize = 1.0;

void main() {
    gl_Position = pMat * uMat * vec4(a_position, 1.0);
    v_tex = vec2(a_tex);
    v_texInfo = vec4(a_texInfo) * invPackSize;
    v_light = float(a_light) / 255.0;
    v_lightInfo = vec2(a_lightInfo) * invTextureSize;
    v_pos = a_position.xy / 4096.0;
    v_lightType = a_lightType;
}