override WORK_GROUP_SIZE: u32 = 64u;

struct Uniforms {
	resolution : vec2<f32>,
	//the samples a new pixel gets, every new pixel gets them in the frame it appears
	newSampleCount : u32,
	//the samples a refined pixel gets per frame
	sampleCount : u32,
	//the refine pixels that get samples per frame
	refineBudget : u32,
	//the sample count at which a pixel stops getting samples
	accumulationMax : f32,
	//rotates the refine pixels that get samples when the budget doesn't cover all of them
	refineOffset : f32,
	inverseProjection : mat4x4<f32>,
	inverseView : mat4x4<f32>,
	lightDirection : vec3<f32>,
	//angular radius of the sun, see sampleSunDirection
	sunRadius : f32,
	cameraPosition : vec3<f32>,
	directionalColor : vec3<f32>,
	modelCount : f32,
	//the first models are static and in the BVH, see traceRay
	bvhModelCount : f32,
	bounceCount : i32,
	frame : f32,
	directionalBrightness : f32,
	directionalProbability : f32,
}

struct PixelCounts {
	newCount : u32,
	priorityCount : u32,
	refineCount : u32,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var normalDepthTexture: texture_2d<u32>;
@group(0) @binding(2) var<storage, read> sceneBuffer : array<Model>;
@group(0) @binding(3) var<storage, read> bvhBuffer : array<BVHNode>;
@group(0) @binding(4) var<storage, read> newPixels : array<u32>;
@group(0) @binding(5) var<storage, read> refinePixels : array<u32>;
@group(0) @binding(6) var<storage, read> pixelCounts : PixelCounts;
//the reprojected lighting, see reproject.wgsl
@group(0) @binding(7) var lightingReprojected: texture_2d<f32>;
@group(0) @binding(8) var lighting: texture_storage_2d<rgba16float, write>;

//no radiance cache, the paths go on until the sky, the sun or the bounce limit. See pathTrace.wgsl
fn getCachedLighting(p: vec3<f32>, n: vec3<f32>) -> vec4<f32> {
	return vec4(0.);
}

//the entry of a refine list (the priority or the other pixels, count long) that refined pixel j takes, of the taken
//ones this frame. The lists are in about screen order, the reprojection appends to them per work group: a contiguous
//slice of them is a band of rows across the screen, which showed as a scanline that swept down at 4K. When the budget
//doesn't cover the list, the entries are spread over it with the golden ratio sequence (a Weyl sequence in 0.32 fixed
//point, like getR2): the refined pixels spread over the screen every frame and refineOffset continues the sequence
//in the next one. Not a bijection, a pixel can be taken twice in a frame: one of the writes wins, which is harmless.
//When the budget covers the list it takes every entry exactly once, TemporalAccumulate relies on that
fn getRefineEntry(j: u32, count: u32, taken: u32) -> u32 {
	let n = j + u32(uniforms.refineOffset);
	if(taken >= count){
		return n % count;
	}
	let x = n * 2654435769u;
	return min(u32(f32(x) * 2.3283064e-10 * f32(count)), count - 1u);
}

//dispatched for all pixels plus the refine budget, in 2d: the group count of a dimension is limited to 65535.
//The new pixels get newSampleCount samples, at most a 1 spp path trace of the screen when everything got disoccluded,
//less with newPixelStride (see reproject.wgsl). The refine budget is fixed: it adds samples to the pixels that have fewer than accumulationMax, the
//priority pixels (fewer than prioritySamples, see reproject.wgsl) first: a pixel with a few samples gains much more
//from new ones than an almost converged one, and the resolve blurs it until it has them
@compute @workgroup_size(WORK_GROUP_SIZE)
fn compute_main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>, @builtin(num_workgroups) numWorkgroups : vec3<u32>) {
	let i = GlobalInvocationID.x + GlobalInvocationID.y * numWorkgroups.x * WORK_GROUP_SIZE;
	let rx = u32(uniforms.resolution.x);
	var pixelIndex = 0u;
	var sampleCount = uniforms.sampleCount;
	if(i < pixelCounts.newCount){
		pixelIndex = newPixels[i];
		sampleCount = uniforms.newSampleCount;
	}else{
		var j = i - pixelCounts.newCount;
		let priorityTaken = min(pixelCounts.priorityCount, uniforms.refineBudget);
		if(j < priorityTaken){
			pixelIndex = refinePixels[getRefineEntry(j, pixelCounts.priorityCount, priorityTaken)];
		}else{
			j -= priorityTaken;
			let refineTaken = min(pixelCounts.refineCount, uniforms.refineBudget - priorityTaken);
			if(j >= refineTaken){
				return;
			}
			let pixelTotal = rx * u32(uniforms.resolution.y);
			pixelIndex = refinePixels[pixelTotal - 1u - getRefineEntry(j, pixelCounts.refineCount, refineTaken)];
		}
	}
	let coord = vec2<u32>(pixelIndex % rx, pixelIndex / rx);

	let normalDepth = loadNormalDepth(normalDepthTexture, coord);
	let uv = (vec2<f32>(coord) + .5) / uniforms.resolution;
	let position = depthToPosition(normalDepth.w, uv * 2. - 1.);

	//without the albedo, it is applied per pixel in resolve.wgsl, like the octree does
	let lightingSample = getLighting(position, normalDepth.xyz, coord, sampleCount);
	let old = textureLoad(lightingReprojected, coord, 0);
	let n = f32(sampleCount);
	let count = old.a + n;
	textureStore(lighting, coord, vec4(mix(old.rgb, lightingSample, n / count), min(count, uniforms.accumulationMax)));
}
