#version 450
#define COLOR_RENDER

#ifndef DEPTH_ONLY

layout(set = 0, binding = 0) uniform texture2D colourMap;
layout(set = 0, binding = 1) uniform sampler colourMapSamp;
layout(set = 0, binding = 2) uniform texture2D palette;
layout(set = 0, binding = 3) uniform sampler paletteSamp;

layout(set = 0, binding = 4) uniform texture2D textureLight;
layout(set = 0, binding = 5) uniform sampler textureLightSamp;
layout(set = 0, binding = 6) uniform texture2D textures;
layout(set = 0, binding = 7) uniform sampler texturesSamp;

layout(location = 0) in vec2 v_tex;
layout(location = 1) in vec4 v_texInfo;
layout(location = 2) in float v_light;
layout(location = 3) in vec2 v_lightInfo;
layout(location = 4) in float v_lightType;

layout(location = 0) out vec4 fragColor;

const float invTextureSize = 1.0 / 1024.0;

vec3 lookupColour(float col, float light);

void main() {
  float light = 1.0 - v_light;
  if (v_lightInfo.x >= 0.0) {
    light = light - texture(sampler2D(textureLight, textureLightSamp), v_lightInfo).r;
  }
  light *= v_lightType;
  vec2 offset = mod(v_texInfo.xy, v_texInfo.zw);
  // float col = float(textureLod(textures, (v_tex.xy + offset) * invTextureSize, 4.0 - gl_FragCoord.w * 3000.0).r) / 255.0;
  float col = texture(sampler2D(textures, texturesSamp), (v_tex.xy + offset) * invTextureSize).r;
  fragColor = vec4(lookupColour(col, light), 1.0);
}

vec3 lookupColour(float col, float light) {
  float index = texture(sampler2D(colourMap, colourMapSamp), vec2(col, light)).r * 255.0;
  float x = floor(mod(index, 16.0)) / 16.0;
  float y = floor(index / 16.0) / 16.0;
  return texture(sampler2D(palette, paletteSamp), vec2(x, y)).rgb;
}
#else
void main() {
}
#endif