#version 330 core

in vec2 uv;
out vec4 fragColor;

uniform sampler2D tex;
uniform sampler2D ascii_map;
uniform bool colored;

uniform int downscale_factor = 10;
uniform int ascii_size = 8;

vec3 get_ascii(float lum)
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    ivec2 ascii_map_size = textureSize(ascii_map, 0);
    int num_glyphs = textureSize(ascii_map, 0).x / ascii_size;

    // --- UV inside a single glyph (0–1 range within an 8×8 cell) ---
    vec2 cell_uv = vec2(
        float(pixel.x % ascii_size) / float(ascii_size),
        float(pixel.y % ascii_size) / float(ascii_size)
    );

    // --- Pick which glyph column to use based on luminance ---
    int glyph_index = int(lum * float(num_glyphs - 1));

    // --- Compute glyph offset (each glyph is 1/N wide) ---
    float glyph_width = 1.0 / float(num_glyphs);
    float glyph_offset = glyph_index * glyph_width;

    // --- Combine into final ASCII map UV ---
    vec2 ascii_uv = vec2(
        glyph_offset + cell_uv.x * glyph_width,
        cell_uv.y
    );

    return texture(ascii_map, ascii_uv).r;
}

void main()
{
    vec2 texSize = vec2(textureSize(tex, 0));

    // Sample original texture in blocks (downscale)
    vec2 scaledUV = floor(uv * texSize / float(downscale_factor)) * float(downscale_factor) / texSize;

    vec3 color = texture(tex, scaledUV).rgb;

    int quantization_levels = textureSize(ascii_map, 0).x / ascii_size;
    float gray = dot(color, vec3(0.2126, 0.7152, 0.0722));

    // Quantize luminance to discrete ASCII levels
    float quantized_lum = floor(gray * float(quantization_levels)) / float(quantization_levels - 1);

    fragColor = colored ? vec4(get_ascii(quantized_lum) * color, 1.0) : vec4(get_ascii(quantized_lum), 1.0);
}