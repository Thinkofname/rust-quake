#version 450

layout(set = 0, binding = 0) uniform texture2D colourMap;
layout(set = 0, binding = 1) uniform sampler colourMapSamp;
layout(set = 0, binding = 2) uniform texture2D palette;
layout(set = 0, binding = 3) uniform sampler paletteSamp;

layout(set = 0, binding = 4) uniform texture2D textureLight;
layout(set = 0, binding = 5) uniform sampler textureLightSamp;
layout(set = 0, binding = 6) uniform texture2D textures;
layout(set = 0, binding = 7) uniform sampler texturesSamp;

// uniform float timeOffset;
// const float timeOffset = 0.0;
layout(push_constant) uniform Transform {
    layout(offset = 64) float timeOffset;
};

layout(location = 0) in vec2 v_tex;
layout(location = 1) in vec4 v_texInfo;
layout(location = 2) in float v_light;
layout(location = 3) in vec2 v_lightInfo;
layout(location = 4) in float v_lightType;
layout(location = 5) in vec2 v_pos;

layout(location = 0) out vec4 fragColor;

const float invTextureSize = 1.0 / 1024.0;

vec3 lookupColour(float col, float light);

void main() {
  if (timeOffset < 0.0) {
    fragColor = vec4(1.0, 0.0, 0.0, 1.0);
    return;
  }
  float light = 0.5;
  vec2 offset = mod(v_pos * 1024.0 + timeOffset * v_texInfo.z * (2.0 - v_lightType), v_texInfo.zw);
  float col = texture(sampler2D(textures, texturesSamp), (v_tex.xy + offset) * invTextureSize).r;
  fragColor = vec4(lookupColour(col, light), 1.0);
}
vec3 lookupColour(float col, float light) {
  float index = texture(sampler2D(colourMap, colourMapSamp), vec2(col, light)).r * 255.0;
  if (index < 1.0) discard;
  float x = floor(mod(index, 16.0)) / 16.0;
  float y = floor(index / 16.0) / 16.0;
  return texture(sampler2D(palette, paletteSamp), vec2(x, y)).rgb;
}