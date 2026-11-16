override WORK_GROUP_SIZE_X: u32 = 8u;
override WORK_GROUP_SIZE_Y: u32 = 8u;

//TemporalAccumulate at half resolution, reproject and fill in one pass: every half res pixel gets
//1 sample per frame, so there are no new and refine pixel lists to compact. Half res pixel c stands for full res pixel
//2c of the gBuffer, the top left one of its 2x2: always the same one, so the surface under a history pixel stays the
//same while the camera is static. An average of the 2x2 would invent surfaces at the edges. Every position below is
//that of the full res pixel, at uv (2c + .5) / fullResolution, see getUV. upsample.wgsl brings it back to full res
struct Uniforms {
	viewProjectionOld : mat4x4<f32>,
	inverseProjection : mat4x4<f32>,
	inverseView : mat4x4<f32>,
	//the camera of the previous frame, for the positions of the pixels of normalDepthOld
	inverseProjectionOld : mat4x4<f32>,
	inverseViewOld : mat4x4<f32>,
	cameraPositionOld : vec3<f32>,
	cameraPosition : vec3<f32>,
	//of the gBuffer, the half res textures are ceil(resolution / 2)
	resolution : vec2<f32>,
	halfResolution : vec2<f32>,
	depthThreshold : f32,
	filterLinear : f32,
	//catmull-rom instead of bilinear: bilinear blurs the history a little every frame the camera moves
	filterSharp : f32,
	//the frames the exponential average is about long while the camera moves, and the samples at which a pixel stops
	//getting them while it is static
	accumulationMax : f32,
	//1: the camera didn't move since the previous frame, the pixels at accumulationMax stop getting samples
	isStatic : f32,
	lightDirection : vec3<f32>,
	//angular radius of the sun, see sampleSunDirection
	sunRadius : f32,
	directionalColor : vec3<f32>,
	modelCount : f32,
	//the first models are static and in the BVH, see traceRay
	bvhModelCount : f32,
	bounceCount : i32,
	frame : f32,
	directionalBrightness : f32,
	directionalProbability : f32,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
//the full res gBuffer, alpha 0 of the albedo is the sky
@group(0) @binding(1) var normalDepthTexture: texture_2d<u32>;
@group(0) @binding(2) var albedoTexture: texture_2d<f32>;
@group(0) @binding(3) var<storage, read> sceneBuffer : array<Model>;
@group(0) @binding(4) var<storage, read> bvhBuffer : array<BVHNode>;
//the packed normalDepth of the full res pixels the half res pixels stand for, of the previous frame and of this one.
//upsample.wgsl reads it too
@group(0) @binding(5) var normalDepthOld: texture_2d<u32>;
@group(0) @binding(6) var normalDepthTarget: texture_storage_2d<rg32uint, write>;
//the running average of the light arriving at the pixels, without their albedo (rgb) and its sample count (a)
@group(0) @binding(7) var lightingOld: texture_2d<f32>;
@group(0) @binding(8) var lighting: texture_storage_2d<rgba16float, write>;

//no radiance cache, the paths go on until the sky, the sun or the bounce limit. See pathTrace.wgsl
fn getCachedLighting(p: vec3<f32>, n: vec3<f32>) -> vec4<f32> {
	return vec4(0.);
}

//-1 to 1, of the full res pixel that half res pixel c stands for
fn getUV(c: vec2<i32>) -> vec2<f32>{
	return (vec2<f32>(c * 2) + .5) / uniforms.resolution * 2. - 1.;
}

//getRay of cameraRay.wgsl with the camera of the previous frame
fn getRayOld(uv: vec2<f32>) -> vec3<f32>{
	var dirEye = uniforms.inverseProjectionOld * vec4(uv.x, -uv.y, 1., 1.);
	dirEye.w = 0.;
	return normalize((uniforms.inverseViewOld * dirEye).xyz);
}

//the lighting of a half res pixel of the previous frame, if it has samples of the surface at p
fn loadHistory(c: vec2<i32>, p: vec3<f32>) -> vec4<f32>{
	if(any(c < vec2(0)) || any(c >= vec2<i32>(uniforms.halfResolution))){
		return vec4(0.);
	}
	let history = textureLoad(lightingOld, c, 0);
	if(history.a == 0.){
		return vec4(0.);
	}
	//the position of the pixel in the previous frame, as depthToPosition gave it then
	let dOld = bitcast<f32>(textureLoad(normalDepthOld, c, 0).x);
	let pOld = uniforms.cameraPositionOld + getRayOld(getUV(c)) * dOld;
	//relative to the distance to the camera: the depth precision and the spacing of the pixels grow with it
	if(distance(p, pOld) > uniforms.depthThreshold * dOld){
		return vec4(0.);
	}
	return history;
}

fn catmullRomWeights(t: f32) -> vec4<f32>{
	let t2 = t * t;
	let t3 = t2 * t;
	return vec4(-.5 * t3 + t2 - .5 * t, 1.5 * t3 - 2.5 * t2 + 1., -1.5 * t3 + 2. * t2 + .5 * t, .5 * t3 - .5 * t2);
}

//the history at p, from the half res pixels around its position in the previous frame. See shared/reproject.wgsl,
//the same filter
fn reproject(p: vec3<f32>) -> vec4<f32>{
	let screenOld = uniforms.viewProjectionOld * vec4(p, 1.);
	var uv = screenOld.xy / screenOld.w;
	uv = uv * .5 + .5;
	uv.y = 1. - uv.y;
	//in half res pixels, where half res pixel c is at c: its position is the center of full res pixel 2c
	let base = (uv * uniforms.resolution - .5) * .5;
	if(uniforms.filterLinear < .5){
		return loadHistory(vec2<i32>(round(base)), p);
	}
	var old = vec4(0.);
	//bilinear over the taps that have samples of this surface only: an unsampled pixel or another surface would darken it
	let c0 = vec2<i32>(floor(base));
	//a static camera reprojects onto the pixel centers up to float error. Filtering that error every frame shifts
	//the converged pixels a little each frame, by a different amount per row and column: it shows as lines
	var f = base - floor(base);
	f = select(f, round(f), abs(f - round(f)) < vec2(.01));
	//catmull-rom over the 4x4 taps around it when they all have samples of this surface, clamped to the inner 2x2
	//against its overshoot. Otherwise the outer taps are skipped once one of them fails
	let wx = catmullRomWeights(f.x);
	let wy = catmullRomWeights(f.y);
	var sharpValid = uniforms.filterSharp > .5;
	var sharp = vec3(0.);
	var minimum = vec3(65504.);
	var maximum = vec3(0.);
	var weightSum = 0.;
	for(var i = 0; i < 16; i++){
		let offset = vec2(i % 4, i / 4);
		let inner = all(offset >= vec2(1)) && all(offset <= vec2(2));
		if(!inner && !sharpValid){
			continue;
		}
		let history = loadHistory(c0 + offset - 1, p);
		if(history.a == 0.){
			sharpValid = false;
			continue;
		}
		sharp += history.rgb * wx[offset.x] * wy[offset.y];
		if(inner){
			minimum = min(minimum, history.rgb);
			maximum = max(maximum, history.rgb);
			let weight = select(1. - f.x, f.x, offset.x == 2) * select(1. - f.y, f.y, offset.y == 2);
			if(weight > 0.){
				old += history * weight;
				weightSum += weight;
			}
		}
	}
	if(weightSum > 0.){
		//the sample count is scaled by the weight of the valid taps, see shared/reproject.wgsl
		old = vec4(old.rgb / weightSum, old.a);
		if(sharpValid){
			old = vec4(clamp(sharp, minimum, maximum), old.a);
		}
	}
	return old;
}

@compute @workgroup_size(WORK_GROUP_SIZE_X, WORK_GROUP_SIZE_Y)
fn compute_main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
	let coord = GlobalInvocationID.xy;
	if(coord.x >= u32(uniforms.halfResolution.x) || coord.y >= u32(uniforms.halfResolution.y)){
		return;
	}
	//inside the gBuffer: halfResolution is ceil(resolution / 2)
	let fullCoord = coord * 2u;

	//the sky is emissive, upsample.wgsl skips the pixels with a lighting of 0. So does loadHistory
	if(textureLoad(albedoTexture, fullCoord, 0).a == 0.){
		textureStore(normalDepthTarget, coord, vec4(0u));
		textureStore(lighting, coord, vec4(0.));
		return;
	}

	let packed = textureLoad(normalDepthTexture, fullCoord, 0).xy;
	textureStore(normalDepthTarget, coord, vec4(packed, 0u, 0u));
	let normalDepth = unpackNormalDepth(packed);
	let p = depthToPosition(normalDepth.w, getUV(vec2<i32>(coord)));

	var old = reproject(p);
	//when accumulationMax was lowered. While the camera moves the count stays at accumulationMax, which makes the
	//running average an exponential one
	old.a = min(old.a, uniforms.accumulationMax);
	if(uniforms.isStatic > .5 && old.a >= uniforms.accumulationMax){
		textureStore(lighting, coord, old);
		return;
	}

	//without the albedo, it is applied per full res pixel in upsample.wgsl. The seed is the half res pixel: the blue
	//noise is spread over the pixels that get samples
	let lightingSample = getLighting(p, normalDepth.xyz, coord, 1u);
	let count = old.a + 1.;
	textureStore(lighting, coord, vec4(mix(old.rgb, lightingSample, 1. / count), min(count, uniforms.accumulationMax)));
}
