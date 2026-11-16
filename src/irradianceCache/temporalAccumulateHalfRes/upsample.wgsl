@group(0) @binding(0) var textureSampler: sampler;
//the half res running average of the light arriving at the pixels, without their albedo (rgb) and its sample count (a),
//see accumulate.wgsl
@group(0) @binding(1) var texture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> uniforms : Uniforms;
//the full res gBuffer, alpha 0 of the albedo is the sky
@group(0) @binding(3) var albedoTexture: texture_2d<f32>;
@group(0) @binding(4) var normalDepthTexture: texture_2d<u32>;
//the packed normalDepth of the full res pixels the half res pixels stand for, see accumulate.wgsl
@group(0) @binding(5) var halfNormalDepthTexture: texture_2d<u32>;

struct Uniforms {
	exposure : f32,
	skyColor : vec3<f32>,
	inverseProjection : mat4x4<f32>,
	inverseView : mat4x4<f32>,
	cameraPosition : vec3<f32>,
	//of the gBuffer
	resolution : vec2<f32>,
	//the distance of a half res pixel to the plane of the full res pixel, relative to its depth, up to which it counts
	//as the same surface
	planeTolerance : f32,
	showSampleCount : f32,
	//the sample count that shows white
	sampleCountMax : f32,
}

struct VertexInput {
	@location(0) position : vec3<f32>,
	@location(1) uv : vec2<f32>
}

struct VertexOutput {
	@builtin(position) position : vec4<f32>,
	@location(0) uv : vec2<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
	var output : VertexOutput;
	output.position = vec4(input.position, 1.);
	output.uv = input.uv;
	return output;
}

//the position of the full res pixel that half res pixel c stands for, the top left one of its 2x2
fn getHalfResPosition(c: vec2<i32>, d: f32) -> vec3<f32>{
	return depthToPosition(d, (vec2<f32>(c * 2) + .5) / uniforms.resolution * 2. - 1.);
}

//the lighting of the full res pixel at coord on the surface p, n: bilinear over the 2x2 half res pixels around it, of
//the ones on the same surface. A full res pixel at an even coordinate is one the half res pixels stand for, it gets
//that one's lighting. When none of the 2x2 is on this surface (thin geometry, an edge, a sky pixel the half res pixel
//stands for), the half res pixel of the 3x3 around it that matches best
fn upsample(coord: vec2<i32>, p: vec3<f32>, n: vec3<f32>, d: f32) -> vec4<f32>{
	let size = vec2<i32>(textureDimensions(texture));
	//in half res pixels, where half res pixel c is at c: its position is the center of full res pixel 2c
	let base = vec2<f32>(coord) * .5;
	let c0 = vec2<i32>(floor(base));
	let f = base - floor(base);
	var sum = vec4(0.);
	var weightSum = 0.;
	var best = vec4(0.);
	var bestScore = 1e30;
	for(var i = 0; i < 9; i++){
		let offset = vec2(i % 3, i / 3);
		let c = c0 + offset - vec2(select(1, 0, f.x > 0.), select(1, 0, f.y > 0.));
		if(any(c < vec2(0)) || any(c >= size)){
			continue;
		}
		let s = textureLoad(texture, c, 0);
		if(s.a == 0.){
			continue;
		}
		let normalDepth = unpackNormalDepth(textureLoad(halfNormalDepthTexture, c, 0).xy);
		let q = getHalfResPosition(c, normalDepth.w);
		let planeDistance = abs(dot(q - p, n)) / d;
		let normalWeight = pow(saturate(dot(n, normalDepth.xyz)), 8.);
		let score = planeDistance / uniforms.planeTolerance + 1. - normalWeight;
		if(score < bestScore){
			best = s;
			bestScore = score;
		}
		//the bilinear weight, 0 outside the 2x2. The 3x3 is centered on the pixel for an even coordinate, and covers the
		//2x2 plus a row or column for an odd one
		let t = vec2<f32>(c) - base;
		let bilinear = saturate(1. - abs(t.x)) * saturate(1. - abs(t.y));
		if(bilinear > 0. && planeDistance < uniforms.planeTolerance){
			let weight = bilinear * normalWeight;
			sum += s * weight;
			weightSum += weight;
		}
	}
	if(weightSum > 1e-4){
		return sum / weightSum;
	}
	if(bestScore < 1e30){
		return best;
	}
	return vec4(.5, .5, .5, 0.);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
	let coord = vec2<i32>(input.position.xy);
	let albedo = textureLoad(albedoTexture, coord, 0);

	var col: vec3<f32>;
	var lighting = vec4(0.);
	//the sky is emissive, it has no lighting
	if(albedo.a == 0.){
		col = uniforms.skyColor;
	}else{
		let normalDepth = loadNormalDepth(normalDepthTexture, vec2<u32>(coord));
		let p = depthToPosition(normalDepth.w, (vec2<f32>(coord) + .5) / uniforms.resolution * 2. - 1.);
		lighting = upsample(coord, p, normalDepth.xyz, normalDepth.w);
		col = albedo.xyz * lighting.rgb;
	}

	col *= uniforms.exposure;

	col = toneMap(col);

	if(uniforms.showSampleCount > .5){
		col = vec3(lighting.a / uniforms.sampleCountMax);
	}

	return vec4(col, 1.);
}
