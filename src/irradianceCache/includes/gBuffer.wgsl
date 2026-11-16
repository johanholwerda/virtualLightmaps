//the normalDepth target of the gBuffer is rg32uint, 8 bytes instead of the 16 of rgba32float:
//x the analytic distance along the camera ray as f32 bits, y the world normal octahedral encoded in 2x16 snorm.
//needs webGPU-framework/shaders/normalPacking.wgsl
fn packNormalDepth(normal: vec3<f32>, depth: f32) -> vec2<u32> {
	return vec2(bitcast<u32>(depth), packSnorm2x16(encodeNormal(normal)));
}

//xyz the normal, w the depth, like the old rgba32float target
fn unpackNormalDepth(packed: vec2<u32>) -> vec4<f32> {
	return vec4(decodeNormal(unpackSnorm2x16(packed.y)), bitcast<f32>(packed.x));
}

//pack2x16snorm and unpack2x16snorm by hand: on macOS (Metal) the builtin pack2x16snorm wrote 0, so every gBuffer
//normal read as (0, 0, 1). x in the low 16 bits, like the builtin
fn packSnorm2x16(v: vec2<f32>) -> u32 {
	let s = vec2<i32>(round(clamp(v, vec2(-1.), vec2(1.)) * 32767.));
	return (bitcast<u32>(s.x) & 0xFFFFu) | (bitcast<u32>(s.y) << 16u);
}

fn unpackSnorm2x16(packed: u32) -> vec2<f32> {
	//the arithmetic shift right extends the sign of the 16 bits
	let s = vec2(bitcast<i32>(packed << 16u) >> 16u, bitcast<i32>(packed) >> 16u);
	return max(vec2<f32>(s) / 32767., vec2(-1.));
}

fn loadNormalDepth(normalDepthTexture: texture_2d<u32>, coord: vec2<u32>) -> vec4<f32> {
	return unpackNormalDepth(textureLoad(normalDepthTexture, coord, 0).xy);
}

//the model target of the gBuffer is r32uint: the index in sceneBuffer in the high 16 bits, the triangle of a mesh in the low
//16 (0 for the analytic primitives). Scene.init checks that both fit
const GBUFFER_TRIANGLE_BITS = 16u;

fn packModelTriangle(modelIndex: u32, triangle: u32) -> u32 {
	return (modelIndex << GBUFFER_TRIANGLE_BITS) | triangle;
}

//x the model index, y the triangle
fn unpackModelTriangle(packed: u32) -> vec2<u32> {
	return vec2(packed >> GBUFFER_TRIANGLE_BITS, packed & ((1u << GBUFFER_TRIANGLE_BITS) - 1u));
}
