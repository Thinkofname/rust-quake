#version 150 core

uniform sampler2D palette;
uniform usampler2D colourMap;
uniform usampler2D textures;
uniform sampler2D textureLight;
uniform float timeOffset;

in vec2 v_tex;
in vec4 v_texInfo;
in float v_light;
in vec2 v_lightInfo;
in float v_lightType;
in vec2 v_pos;

out vec4 fragColor;

const float invTextureSize = 1.0 / 1024.0;

vec3 lookupColour(float col, float light);

void main() {
  if (timeOffset < 0.0) {
    fragColor = vec4(1.0, 0.0, 0.0, 1.0);
    return;
  }
  float light = 0.5;
  vec2 offset = mod(v_pos * 1024.0 + timeOffset * v_texInfo.z * (2.0 - v_lightType), v_texInfo.zw);
  float col = float(textureLod(textures, (v_tex.xy + offset) * invTextureSize, 4.0 - gl_FragCoord.w * 3000.0).r) / 255.0;
  fragColor = vec4(lookupColour(col, light), 1.0);
}

vec3 lookupColour(float col, float light) {
  float index = float(texture(colourMap, vec2(col, light)).r);
  if (index < 1.0) discard;
  float x = floor(mod(index, 16.0)) / 16.0;
  float y = floor(index / 16.0) / 16.0;
  return texture(palette, vec2(x, y)).rgb;
}