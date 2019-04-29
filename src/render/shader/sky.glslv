#version 450

layout(location = 0) in vec3 a_position;
layout(location = 1) in uvec2 a_tex;
layout(location = 2) in ivec4 a_texInfo;
layout(location = 3) in ivec2 a_lightInfo;
layout(location = 4) in uint a_light;
layout(location = 5) in uint a_lightType;

layout(push_constant) uniform Transform {
    mat4 matrix;
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

layout(location = 0) out vec2 v_tex;
layout(location = 1) out vec4 v_texInfo;
layout(location = 2) out float v_light;
layout(location = 3) out vec2 v_lightInfo;
layout(location = 4) out float v_lightType;
layout(location = 5) out vec2 v_pos;

const float invTextureSize = 1.0 / 1024.0;
const float invPackSize = 1.0;

void main() {
    gl_Position = matrix * vec4(a_position, 1.0);
    v_tex = vec2(a_tex);
    v_texInfo = vec4(a_texInfo) * invPackSize;
    v_light = float(a_light) / 255.0;
    v_lightInfo = vec2(a_lightInfo) * invTextureSize;
    v_pos = a_position.xy / 4096.0;
    v_lightType = a_lightType;
}