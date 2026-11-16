override WORK_GROUP_SIZE_X: u32;
override WORK_GROUP_SIZE_Y: u32;

struct Uniforms {
	resolution : vec2<f32>,
	inverseProjection : mat4x4<f32>,
	inverseView : mat4x4<f32>,
	cameraPosition : vec3<f32>,
	frame : f32,
	lightDirection : vec3<f32>,
	//angular radius of the sun, see sampleSunDirection
	sunRadius : f32,
	directionalColor : vec3<f32>,
	modelCount : f32,
	//the first models are static and in the BVH, see traceRay
	bvhModelCount : f32,
	bounceCount : i32,
	directionalBrightness : f32,
	directionalProbability : f32,
	sampleCount : u32,
	skyColor : vec3<f32>,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
//per pixel: the sum of the color (rgb) and the frame count (a)
@group(0) @binding(1) var<storage, read_write> accumulationBuffer : array<vec4<f32>>;
@group(0) @binding(2) var normalDepthTexture: texture_2d<u32>;
@group(0) @binding(3) var<storage, read> sceneBuffer : array<Model>;
@group(0) @binding(4) var<storage, read> bvhBuffer : array<BVHNode>;
//alpha 0 is the sky
@group(0) @binding(5) var albedoTexture: texture_2d<f32>;

//no radiance cache, the paths go on until the sky, the sun or the bounce limit. See pathTrace.wgsl
fn getCachedLighting(p: vec3<f32>, n: vec3<f32>) -> vec4<f32> {
	return vec4(0.);
}

@compute @workgroup_size(WORK_GROUP_SIZE_X, WORK_GROUP_SIZE_Y, 1)
fn compute_main(
  @builtin(global_invocation_id) GlobalInvocationID : vec3<u32>
) {
	let coord = GlobalInvocationID.xy;
	if(coord.x >= u32(uniforms.resolution.x) || coord.y >= u32(uniforms.resolution.y)){
		return;
	}

	//the color with the albedo, not the lighting alone: with the TAA the camera is jittered, the pixel averages the
	//surfaces under it, its edges are anti aliased. The sky is emissive, it has no lighting
	let index = coord.y * u32(uniforms.resolution.x) + coord.x;
	let albedo = textureLoad(albedoTexture, coord, 0);
	if(albedo.a == 0.){
		accumulationBuffer[index] += vec4(uniforms.skyColor, 1.);
		return;
	}

	let normalDepth = loadNormalDepth(normalDepthTexture, coord);
	let uv = (vec2<f32>(coord) + .5) / uniforms.resolution;
	let worldPosition = depthToPosition(normalDepth.w, uv * 2. - 1.);
	//the lighting has no albedo, see getLighting in pathTrace.wgsl
	let lighting = getLighting(worldPosition, normalDepth.xyz, coord, uniforms.sampleCount);
	accumulationBuffer[index] += vec4(albedo.rgb * lighting, 1.);
}
