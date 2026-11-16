override WORK_GROUP_SIZE_X: u32;
override WORK_GROUP_SIZE_Y: u32;
override CELL_SIZE: u32;

struct Uniforms {
	resolution : vec2<f32>,
	inverseProjection : mat4x4<f32>,
	inverseView : mat4x4<f32>,
	cameraPosition : vec3<f32>,
	frame : f32,
	//the world size of a pixel at distance 1
	pixelSize : f32,
	//the size of the level 0 cells
	sizeMin : f32,
	//the size of a cell in pixels
	cellSize : f32,
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
	accumulationMax : f32,
	//1: the paths end at their first hit in the lighting the hash map has there, see getCachedLighting
	cacheBounces : u32,
	//coarser levels that accumulate the samples too, at most ANCESTOR_LEVELS_MAX
	ancestorLevels : u32,
	//the samples of the coarser lighting a new cell starts with
	seedCount : f32,
	//the frames after which a cell no pixel uses can be taken by another one
	evictFrames : f32,
	searchCount : u32,
}

const ANCESTOR_LEVELS_MAX = 4u;

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage, read_write> hashBuffer : array<atomic<u32>>;
//the frame a cell was last used in
@group(0) @binding(2) var<storage, read_write> hashTimeBuffer : array<atomic<u32>>;
//per cell: running average of the lighting (rgb) and the sample count it holds (a), as 4 x f16
@group(0) @binding(3) var<storage, read_write> lightingBuffer : array<vec2<u32>>;
//the slots in use, for the log
@group(0) @binding(4) var<storage, read_write> cellCount : atomic<u32>;
@group(0) @binding(5) var normalDepthTexture: texture_2d<u32>;
@group(0) @binding(7) var<storage, read> sceneBuffer : array<Model>;
@group(0) @binding(10) var<storage, read> bvhBuffer : array<BVHNode>;

//linear probing. A cell that wasn't used for evictFrames is taken over, its lighting cleared. The compare exchange makes sure only 1 thread takes it.
//First the search for the key, up to the first empty slot (the key is never after one: slots are only taken over, never emptied).
//Evicting during the search gave a cell that was further on in the probe sequence a second, empty slot, so it restarted from its seed
fn findOrInsert(p: vec3<f32>, direction: vec3<f32>, level: u32) -> u32 {
	let keySlot = getKey(vec3<i32>(floor(getCellPosition(p, direction, level))), direction, level);
	let key = keySlot.x;
	let frame = u32(uniforms.frame);
	var index = keySlot.y;
	var evictIndex = HASH_NONE;
	var evictKey = 0u;
	var i = 0u;
	for(; i < uniforms.searchCount; i++){
		let existing = atomicLoad(&hashBuffer[index]);
		if(existing == key){
			atomicStore(&hashTimeBuffer[index], frame);
			return index;
		}
		if(existing == 0u){
			break;
		}
		if(evictIndex == HASH_NONE && f32(frame - atomicLoad(&hashTimeBuffer[index])) > uniforms.evictFrames){
			evictIndex = index;
			evictKey = existing;
		}
		index = (index + 1u) % HASHMAP_SIZE;
	}

	//the first stale slot before an empty one: keeps the probe sequences short. Retried while the weak compare exchange fails spuriously
	if(evictIndex != HASH_NONE){
		for(var j = 0u; j < 4u; j++){
			let cmp = atomicCompareExchangeWeak(&hashBuffer[evictIndex], evictKey, key);
			if(cmp.exchanged){
				atomicStore(&hashTimeBuffer[evictIndex], frame);
				lightingBuffer[evictIndex] = vec2(0u);
				return evictIndex;
			}
			//another thread inserted the same cell there
			if(cmp.old_value == key){
				atomicStore(&hashTimeBuffer[evictIndex], frame);
				return evictIndex;
			}
			if(cmp.old_value != evictKey){
				break;
			}
		}
	}

	//the empty slot, or the next ones if other threads filled it meanwhile. A spurious failure retries the same slot:
	//skipping it would leave an empty slot before the key, where find stops
	for(; i < uniforms.searchCount; i++){
		let cmp = atomicCompareExchangeWeak(&hashBuffer[index], 0u, key);
		if(cmp.exchanged){
			atomicAdd(&cellCount, 1u);
		}
		if(cmp.exchanged || cmp.old_value == key){
			atomicStore(&hashTimeBuffer[index], frame);
			return index;
		}
		if(cmp.old_value != 0u){
			index = (index + 1u) % HASHMAP_SIZE;
		}
	}
	return HASH_NONE;
}

fn find(p: vec3<f32>, direction: vec3<f32>, level: u32) -> u32 {
	let keySlot = getKey(vec3<i32>(floor(getCellPosition(p, direction, level))), direction, level);
	let key = keySlot.x;
	var index = keySlot.y;
	for(var i = 0u; i < uniforms.searchCount; i++){
		let existing = atomicLoad(&hashBuffer[index]);
		if(existing == key){
			return index;
		}
		if(existing == 0u){
			return HASH_NONE;
		}
		index = (index + 1u) % HASHMAP_SIZE;
	}
	return HASH_NONE;
}

//the count is capped, so the average keeps following changes in the scene
fn addSample(accumulated: vec4<f32>, lighting: vec3<f32>) -> vec4<f32> {
	return vec4(mix(accumulated.rgb, lighting, 1. / (accumulated.a + 1.)), min(accumulated.a + 1., uniforms.accumulationMax));
}

//the lighting the hash map has at a hit point: the cell at the level the camera sees it at, else the coarser ones.
//The hash map is a radiance cache for sampleRay of pathTrace.wgsl, like the octree of updateOctree.wgsl
fn getCachedLighting(p: vec3<f32>, n: vec3<f32>) -> vec4<f32> {
	if(uniforms.cacheBounces == 0u){
		return vec4(0.);
	}
	let level = u32(getLevel(distance(p, uniforms.cameraPosition)));
	let direction = getDirection(n);
	for(var i = 0u; i <= min(uniforms.ancestorLevels, ANCESTOR_LEVELS_MAX); i++){
		let index = find(p, direction, level + i);
		if(index != HASH_NONE){
			let lighting = unpackLighting(lightingBuffer[index]);
			if(lighting.a > 0.){
				return lighting;
			}
		}
	}
	return vec4(0.);
}

@compute @workgroup_size(WORK_GROUP_SIZE_X, WORK_GROUP_SIZE_Y, 1)
fn compute_main(
  @builtin(global_invocation_id) GlobalInvocationID : vec3<u32>
) {
	let cellID = GlobalInvocationID.xy;
	//not the first 2 numbers of the hash: getLighting takes those as its white noise, the cell is its seed
	let rand = hash34uf(vec3<u32>(cellID, u32(uniforms.frame))).zw;
	//random pixel in the cell
	let coord = cellID * CELL_SIZE + vec2<u32>(rand * f32(CELL_SIZE));
	if(coord.x >= u32(uniforms.resolution.x) || coord.y >= u32(uniforms.resolution.y)){
		return;
	}

	let normalDepth = loadNormalDepth(normalDepthTexture, coord);
	let n = normalDepth.xyz;
	let uv = (vec2<f32>(coord) + .5) / uniforms.resolution;
	let worldPosition = depthToPosition(normalDepth.w, uv * 2. - 1.);
	//the sky gets no cells, it is drawn directly (see resolveSpatialHash.wgsl)
	if(isSky(worldPosition)){
		return;
	}

	let level = u32(getLevel(normalDepth.w));
	let ancestorLevels = min(uniforms.ancestorLevels, ANCESTOR_LEVELS_MAX);
	let direction = getDirection(n);
	var indices: array<u32, ANCESTOR_LEVELS_MAX + 1u>;
	for(var i = 0u; i <= ancestorLevels; i++){
		indices[i] = findOrInsert(worldPosition, direction, level + i);
	}
	//the hash map is full here
	if(indices[0] == HASH_NONE){
		return;
	}

	//the blue noise of the cell, not of the pixel: the pixel is random in the cell, so the noise of the pixels that
	//neighbouring cells pick would be white
	let lighting = getLighting(worldPosition, n, cellID, uniforms.sampleCount);

	//cells that the same frame samples more than once race here. One of the samples gets lost, which is harmless.
	//From coarse to fine: the coarser cells cover more pixels, so they converge faster. A new cell starts from the lighting of the
	//coarser one, as seedCount samples, instead of from its first noisy sample. The resolve uses them as the prior of the finer ones
	var prior = vec4(0.);
	for(var i = i32(ancestorLevels); i >= 0; i--){
		let index = indices[i];
		if(index == HASH_NONE){
			continue;
		}
		var cell = unpackLighting(lightingBuffer[index]);
		if(cell.a == 0. && prior.a > 0.){
			cell = vec4(prior.rgb, min(prior.a, uniforms.seedCount));
		}
		cell = addSample(cell, lighting);
		lightingBuffer[index] = packLighting(cell);
		prior = cell;
	}
}
