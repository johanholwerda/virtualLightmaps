override WORK_GROUP_SIZE_X: u32 = 16u;
override WORK_GROUP_SIZE_Y: u32 = 16u;

struct Uniforms {
	viewProjectionOld : mat4x4<f32>,
	inverseProjection : mat4x4<f32>,
	inverseView : mat4x4<f32>,
	//the camera of the previous frame, for the positions of the pixels of depthOld
	inverseProjectionOld : mat4x4<f32>,
	inverseViewOld : mat4x4<f32>,
	cameraPositionOld : vec3<f32>,
	cameraPosition : vec3<f32>,
	resolution : vec2<f32>,
	depthThreshold : f32,
	filterLinear : f32,
	//the sample count at which a pixel stops getting samples, see fill.wgsl
	accumulationMax : f32,
	//catmull-rom instead of bilinear: bilinear blurs the history a little every frame the camera moves
	filterSharp : f32,
	//the pixels with fewer samples get the refine budget first, see fill.wgsl
	prioritySamples : f32,
	//1 in newPixelStride pixels without history is added to newPixels per frame, see compute_main
	newPixelStride : f32,
	frame : f32,
	//1: the light moves, the pixels at accumulationMax are refined too (last), the fill keeps their count there, which
	//makes their running average an exponential one
	refineAtMax : f32,
}

struct PixelCounts {
	newCount : atomic<u32>,
	priorityCount : atomic<u32>,
	refineCount : atomic<u32>,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var normalDepthTexture: texture_2d<u32>;
//alpha 0 is the sky
@group(0) @binding(2) var albedoTexture: texture_2d<f32>;
//the depth of normalDepthTexture, of the previous frame and of this one
@group(0) @binding(3) var depthOld: texture_2d<f32>;
@group(0) @binding(4) var depthTarget: texture_storage_2d<r32float, write>;
//the running average of the light arriving at the pixels, without their albedo (rgb) and its sample count (a)
@group(0) @binding(5) var lightingOld: texture_2d<f32>;
@group(0) @binding(6) var lighting: texture_storage_2d<rgba16float, write>;
//the pixels without a valid history, fill.wgsl samples them first
@group(0) @binding(7) var<storage, read_write> newPixels : array<u32>;
//the pixels that have fewer than accumulationMax samples. The priority pixels from the start, the others from the end
@group(0) @binding(8) var<storage, read_write> refinePixels : array<u32>;
@group(0) @binding(9) var<storage, read_write> pixelCounts : PixelCounts;

//getRay of cameraRay.wgsl with the camera of the previous frame
fn getRayOld(uv: vec2<f32>) -> vec3<f32>{
	var dirEye = uniforms.inverseProjectionOld * vec4(uv.x, -uv.y, 1., 1.);
	dirEye.w = 0.;
	return normalize((uniforms.inverseViewOld * dirEye).xyz);
}

//the lighting of a pixel of the previous frame, if it has samples of the surface at p with normal n
fn loadHistory(c: vec2<i32>, p: vec3<f32>, n: vec3<f32>) -> vec4<f32>{
	if(any(c < vec2(0)) || any(c >= vec2<i32>(uniforms.resolution))){
		return vec4(0.);
	}
	let history = textureLoad(lightingOld, c, 0);
	if(history.a == 0.){
		return vec4(0.);
	}
	//the position of the pixel in the previous frame, as depthToPosition gave it then
	let uv = (vec2<f32>(c) + .5) / uniforms.resolution * 2. - 1.;
	let pOld = uniforms.cameraPositionOld + getRayOld(uv) * textureLoad(depthOld, c, 0).x;
	//the distance to the plane of p instead of to p: on a distant surface seen at a grazing angle a pixel covers more
	//than depthThreshold along the surface. The neighbouring taps were rejected, often all of them: the pixels became
	//new every frame the camera moved and showed as 1 spp noise
	if(abs(dot(n, p - pOld)) > uniforms.depthThreshold){
		return vec4(0.);
	}
	return history;
}

fn catmullRomWeights(t: f32) -> vec4<f32>{
	let t2 = t * t;
	let t3 = t2 * t;
	return vec4(-.5 * t3 + t2 - .5 * t, 1.5 * t3 - 2.5 * t2 + 1., -1.5 * t3 + 2. * t2 + .5 * t, .5 * t3 - .5 * t2);
}

@compute @workgroup_size(WORK_GROUP_SIZE_X, WORK_GROUP_SIZE_Y)
fn compute_main(
  @builtin(global_invocation_id) GlobalInvocationID : vec3<u32>
) {
	let coord = GlobalInvocationID.xy;

	if(coord.x >= u32(uniforms.resolution.x) || coord.y >= u32(uniforms.resolution.y)){
		return;
	}

	//the sky is emissive, the resolve shows the sky color there. Its depth isn't written: loadHistory skips it on
	//its lighting of 0
	if(textureLoad(albedoTexture, coord, 0).a == 0.){
		textureStore(lighting, coord, vec4(0.));
		return;
	}

	//gather the lighting at the position of this pixel in the previous frame
	let normalDepth = loadNormalDepth(normalDepthTexture, coord);
	let n = normalDepth.xyz;
	let d = normalDepth.w;
	textureStore(depthTarget, coord, vec4(d));
	var uv = (vec2<f32>(coord) + .5) / uniforms.resolution;
	let p = depthToPosition(d, uv * 2. - 1.);
	let screenOld = uniforms.viewProjectionOld * vec4(p, 1.);
	uv = screenOld.xy / screenOld.w;
	uv = uv * .5 + .5;
	uv.y = 1. - uv.y;
	let pixelOld = uv * uniforms.resolution;
	var old = vec4(0.);
	if(uniforms.filterLinear > .5){
		//bilinear over the taps that have samples of this surface only: an unsampled pixel or another surface would darken it
		let base = pixelOld - .5;
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
			let history = loadHistory(c0 + offset - 1, p, n);
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
			//the sample count is scaled by the weight of the valid taps: a history rebuilt from part of the taps is
			//noisier than its count says. When the plane test rejects some taps every frame the camera moves (at the
			//edges of a surface), the history gets copied from a neighbor over and over and shows as streaks once the
			//camera stops. The lower count keeps those pixels getting samples
			old = vec4(old.rgb / weightSum, old.a);
			if(sharpValid){
				old = vec4(clamp(sharp, minimum, maximum), old.a);
			}
		}
	}else{
		old = loadHistory(vec2<i32>(floor(pixelOld)), p, n);
	}

	let pixelIndex = coord.x + coord.y * u32(uniforms.resolution.x);
	//disoccluded or off screen in the previous frame
	if(old.a == 0.){
		textureStore(lighting, coord, vec4(0.));
		//1 in newPixelStride of them is path traced this frame, the resolve fills in the others from their neighbours.
		//Selected here and not in the fill, so its dispatch stays compact: an early return there leaves the warps as
		//long as before. Interleaved gradient noise ranks the pixels, so the selected ones are spread evenly. With a
		//static camera every pixel gets its turn within newPixelStride frames. While it moves, an unselected pixel
		//next to a traced one usually reprojects a part of its count and joins the priority refine pixels instead
		let stride = u32(uniforms.newPixelStride);
		if(stride > 1u){
			let noise = fract(52.9829189 * fract(.06711056 * f32(coord.x) + .00583715 * f32(coord.y)));
			if((u32(noise * f32(stride)) + u32(uniforms.frame)) % stride != 0u){
				return;
			}
		}
		newPixels[atomicAdd(&pixelCounts.newCount, 1u)] = pixelIndex;
		return;
	}

	//when accumulationMax was lowered
	old.a = min(old.a, uniforms.accumulationMax);
	textureStore(lighting, coord, old);
	if(old.a < uniforms.prioritySamples){
		refinePixels[atomicAdd(&pixelCounts.priorityCount, 1u)] = pixelIndex;
	}else if(old.a < uniforms.accumulationMax || uniforms.refineAtMax > .5){
		let pixelTotal = u32(uniforms.resolution.x) * u32(uniforms.resolution.y);
		refinePixels[pixelTotal - 1u - atomicAdd(&pixelCounts.refineCount, 1u)] = pixelIndex;
	}
}
