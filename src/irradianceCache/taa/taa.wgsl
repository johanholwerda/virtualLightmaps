//temporal anti aliasing of the image of a draw mode, after deferredRenderer/taa/taa.wgsl. The camera is jittered with
//the halton sequence. The scene is static, so the history is reprojected with the camera alone: the jittered matrices
//reconstruct the position of the gBuffer sample exactly, the ones without jitter give the camera motion. The history is
//sampled with catmull-rom and clipped to the variance of the 3x3 neighbourhood in YCoCg against ghosting. The blend
//weighs by 1 / (1 + luma), which keeps the bright pixels from flickering on their edges

struct Uniforms {
	inverseProjection : mat4x4<f32>,
	inverseView : mat4x4<f32>,
	cameraPosition : vec3<f32>,
	viewProjectionNoJitter : mat4x4<f32>,
	viewProjectionOldNoJitter : mat4x4<f32>,
	resolution : vec2<f32>,
	historyValid : f32,
	feedback : f32,
	clipGamma : f32,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var normalDepthTexture : texture_2d<u32>;
@group(0) @binding(2) var source : texture_2d<f32>;
@group(0) @binding(3) var history : texture_2d<f32>;
@group(0) @binding(4) var linearSampler : sampler;

struct VertexInput {
	@location(0) position : vec3<f32>,
	@location(1) uv : vec2<f32>,
}

struct VertexOutput {
	@builtin(position) position : vec4<f32>,
}

@vertex
fn vs_main(input : VertexInput) -> VertexOutput {
	var output : VertexOutput;
	output.position = vec4(input.position, 1.);
	return output;
}

//the uv of a world position, 0 to 1, y down
fn projectToUV(viewProjection : mat4x4<f32>, p : vec3<f32>) -> vec2<f32> {
	let clip = viewProjection * vec4(p, 1.);
	let ndc = clip.xy / clip.w;
	return vec2(ndc.x * .5 + .5, .5 - ndc.y * .5);
}

fn rgbToYCoCg(c : vec3<f32>) -> vec3<f32> {
	return vec3(dot(c, vec3(.25, .5, .25)), dot(c, vec3(.5, 0., -.5)), dot(c, vec3(-.25, .5, -.25)));
}

fn yCoCgToRgb(c : vec3<f32>) -> vec3<f32> {
	return vec3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

//catmull-rom in 5 bilinear taps (the corners are dropped), https://gist.github.com/TheRealMJP/c83b8c0f46b63f3a88a5986f4fa982b5
fn sampleHistory(uv : vec2<f32>) -> vec3<f32> {
	let samplePosition = uv * uniforms.resolution;
	let texPos1 = floor(samplePosition - .5) + .5;
	let f = samplePosition - texPos1;
	let w0 = f * (-.5 + f * (1. - .5 * f));
	let w1 = 1. + f * f * (-2.5 + 1.5 * f);
	let w2 = f * (.5 + f * (2. - 1.5 * f));
	let w3 = f * f * (-.5 + .5 * f);
	let w12 = w1 + w2;
	let offset12 = w2 / w12;
	let texPos0 = (texPos1 - 1.) / uniforms.resolution;
	let texPos3 = (texPos1 + 2.) / uniforms.resolution;
	let texPos12 = (texPos1 + offset12) / uniforms.resolution;
	var result = vec3(0.);
	result += textureSampleLevel(history, linearSampler, vec2(texPos12.x, texPos0.y), 0.).rgb * w12.x * w0.y;
	result += textureSampleLevel(history, linearSampler, vec2(texPos0.x, texPos12.y), 0.).rgb * w0.x * w12.y;
	result += textureSampleLevel(history, linearSampler, vec2(texPos12.x, texPos12.y), 0.).rgb * w12.x * w12.y;
	result += textureSampleLevel(history, linearSampler, vec2(texPos3.x, texPos12.y), 0.).rgb * w3.x * w12.y;
	result += textureSampleLevel(history, linearSampler, vec2(texPos12.x, texPos3.y), 0.).rgb * w12.x * w3.y;
	let weightSum = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
	return max(result / weightSum, vec3(0.));
}

fn luma(c : vec3<f32>) -> f32 {
	return dot(c, vec3(.2126, .7152, .0722));
}

@fragment
fn fs_main(input : VertexOutput) -> @location(0) vec4<f32> {
	let coord = vec2<i32>(input.position.xy);
	let current = textureLoad(source, coord, 0).rgb;
	if(uniforms.historyValid < .5){
		return vec4(current, 1.);
	}

	//the sky is a sphere around the scene, it has a depth like the other models
	let uv = (vec2<f32>(coord) + .5) / uniforms.resolution;
	let depth = loadNormalDepth(normalDepthTexture, vec2<u32>(coord)).w;
	let p = depthToPosition(depth, uv * 2. - 1.);
	let uvOld = uv + projectToUV(uniforms.viewProjectionOldNoJitter, p) - projectToUV(uniforms.viewProjectionNoJitter, p);
	if(any(uvOld < vec2(0.)) || any(uvOld >= vec2(1.))){
		return vec4(current, 1.);
	}

	let maxCoord = vec2<i32>(uniforms.resolution) - 1;
	var m1 = vec3(0.);
	var m2 = vec3(0.);
	for(var i = 0; i < 9; i++){
		let c = clamp(coord + vec2(i % 3 - 1, i / 3 - 1), vec2(0), maxCoord);
		let s = rgbToYCoCg(textureLoad(source, c, 0).rgb);
		m1 += s;
		m2 += s * s;
	}
	m1 /= 9.;
	let sigma = sqrt(max(m2 / 9. - m1 * m1, vec3(0.)));
	let historyYCoCg = clamp(rgbToYCoCg(sampleHistory(uvOld)), m1 - sigma * uniforms.clipGamma, m1 + sigma * uniforms.clipGamma);
	let previous = max(yCoCgToRgb(historyYCoCg), vec3(0.));

	let wCurrent = (1. - uniforms.feedback) / (1. + luma(current));
	let wPrevious = uniforms.feedback / (1. + luma(previous));
	return vec4((current * wCurrent + previous * wPrevious) / (wCurrent + wPrevious), 1.);
}
