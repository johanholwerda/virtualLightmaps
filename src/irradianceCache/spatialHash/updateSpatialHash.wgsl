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
	//the paths a feed cell holds, 0: no feed. See getCachedLighting
	rayFeedPaths : f32,
	//the size of the feed cells in m, see getFeedLevel
	feedCellSize : f32,
	//the part of the workgroups whose paths feed full feed cells too, see feedRefreshing
	feedRefresh : f32,
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
//keySlot: of getCellKey or getFeedKey
fn findOrInsert(keySlot: vec2<u32>) -> u32 {
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

fn find(keySlot: vec2<u32>) -> u32 {
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

//the count is capped at countMax, so the average keeps following changes in the scene
fn addSample(accumulated: vec4<f32>, lighting: vec3<f32>, countMax: f32) -> vec4<f32> {
	return vec4(mix(accumulated.rgb, lighting, 1. / (accumulated.a + 1.)), min(accumulated.a + 1., countMax));
}

//the feed cell of the hit point of the last getCachedLighting that found no lighting in the cells of the screen.
//feedOn: there is a feed (rayFeedPaths), the rest is only set then
var<private> feedOn: bool;
//what the feed cell holds, alpha 0 when it has no paths or no slot yet
var<private> feedLighting: vec4<f32>;
//its key, and its slot (HASH_NONE when it has none yet), see getFeedCell
var<private> feedKeySlot: vec2<u32>;
var<private> feedIndex: u32;

/*
the paths of this workgroup all feed the cell of their first hit, also when it is full: set by compute_main for
feedRefresh of the workgroups. So a full feed cell keeps following the lighting of the cells its paths end in: its
first samples are from when the feed around it was still empty, with less bounce light, and the sun moves.
Per workgroup and not per path, like feedRefreshing of updateVirtualLightmap.wgsl: a path that feeds takes 1 or 2 rays
more, and the threads of a workgroup wait on the one that takes the most
 */
var<private> feedRefreshing: bool;

/*
the lighting the hash map has at a hit point: the cell at the level the camera sees it at, else the coarser ones.
The hash map is a radiance cache for the paths of getIndirectPathSum, like the octree of updateOctree.wgsl.

The feed (rayFeedPaths), like the one of updateVirtualLightmap.wgsl: the screen only fills the cells of what is in view,
so a path that hits something else found no lighting and went on for up to bounceCount bounces. The feed cells are a
second set of cells that the rays fill themselves, for the hit points the cells of the screen have nothing for.
-cells of their own (FEED_LEVEL in the key): a cell of the screen that the camera doesn't see would hold the few noisy
 paths of the rays, and the resolve blends it with its neighbours
-at 1 level for the whole scene, see getFeedLevel: the level of the screen cells follows the distance to the camera,
 so a surface would get other feed cells whenever the camera moves, each starting empty
When the cells of the screen have no lighting this returns none, and sets the feed cell of the hit point: feedOn and the
vars with it. A feed cell is kept while rays look at it, see evictFrames
 */
fn getCachedLighting(p: vec3<f32>, n: vec3<f32>) -> vec4<f32> {
	feedOn = false;
	if(uniforms.cacheBounces == 0u){
		return vec4(0.);
	}
	let level = u32(getLevel(distance(p, uniforms.cameraPosition)));
	let direction = getDirection(n);
	for(var i = 0u; i <= min(uniforms.ancestorLevels, ANCESTOR_LEVELS_MAX); i++){
		let index = find(getCellKey(p, direction, level + i));
		if(index != HASH_NONE){
			let lighting = unpackLighting(lightingBuffer[index]);
			if(lighting.a > 0.){
				return lighting;
			}
		}
	}
	if(uniforms.rayFeedPaths <= 0.){
		return vec4(0.);
	}

	feedOn = true;
	feedLighting = vec4(0.);
	feedKeySlot = getFeedKey(p, direction);
	feedIndex = find(feedKeySlot);
	if(feedIndex == HASH_NONE){
		return vec4(0.);
	}
	//the cell stays while rays look at it. Only written when it changes: the rays of many threads hit the same cells
	let frame = u32(uniforms.frame);
	if(atomicLoad(&hashTimeBuffer[feedIndex]) != frame){
		atomicStore(&hashTimeBuffer[feedIndex], frame);
	}
	feedLighting = unpackLighting(lightingBuffer[feedIndex]);
	return vec4(0.);
}

//the slot of the feed cell of the hit point of the last getCachedLighting, which set feedOn, to add a sample to. It is
//added when it is not in the hash map yet. HASH_NONE when the hash map is full there
fn getFeedCell() -> u32 {
	if(feedIndex != HASH_NONE){
		return feedIndex;
	}
	return findOrInsert(feedKeySlot);
}

/*
the indirect lighting of getLighting of pathTrace.wgsl: the average of its sampleCount paths, but as 1 loop over the rays
of all of them instead of a loop over the bounces in a loop over the paths, like getIndirectPathSum of
updateVirtualLightmap.wgsl. A thread whose path ended starts on its next path in the same step, while the others bounce.
The sun ray of a hit is a step of the loop too, a traceRay to the sky instead of getShadow.
This is the only traceRay call site of this shader: sampleRay is not used.

A path ends at the first hit that the cells of the screen have lighting for (getCachedLighting). Where they have none:
-without the feed (rayFeedPaths 0) it goes on like sampleRay, for up to bounceCount bounces
-with the feed it is at most 2 rays and a sun ray, whatever bounceCount:
 1 the first hit ends in its feed cell, with the chance of the part of rayFeedPaths paths that cell holds. Else the
   path feeds it: it goes on for 1 more ray (the sun sample with directionalProbability, else a bounce), and the light
   that brings to the hit, without its albedo, is added to the cell as a sample of 1 path. A full cell is still fed
   by the workgroups that refresh (feedRefreshing). The cell is added here
 2 the second hit ends in its feed cell when that holds anything, however few paths: the noise averages out in the
   cell of the first hit. Else in its sun light only, the sun ray times directionalProbability, without a bounce.
   No cell is added here: the feed is what the surfaces in view see, not the whole scene
 So the light of a surface that neither the camera nor a surface in view sees is its sun light only, the rest has all
 its bounces: a feed cell is fed by paths that end in the other cells, over the frames
 */
fn getIndirectPathSum(p: vec3<f32>, n: vec3<f32>, seed: vec2<u32>, sampleCount: u32) -> vec3<f32>{
	var sum = vec3(0.);
	if(sampleCount == 0u || uniforms.bounceCount <= 0){
		return sum;
	}
	let sunColor = uniforms.directionalColor * uniforms.directionalBrightness;
	var i = 0u;
	var ro = p + n * MIN_DIST;
	var rd = getIndirectDirection(n, seed, i, sampleCount);
	var col = vec3(1.);
	var bounce = 0;
	//the ray is the sun ray of the last hit of the path
	var sunRay = false;
	//the feed cell the path feeds, and what the path lets through from that hit on: col without what is before it
	var feedCell = HASH_NONE;
	var feedCol = vec3(1.);

	loop {
		let hit = traceRay(ro + rd * MIN_DIST, rd);
		let hitPoint = ro + rd * hit.distance;
		let sky = isSky(hitPoint);
		var ended = true;
		//the light the path ends in
		var light = vec3(0.);
		if(sunRay){
			//col has the cosine at the hit
			if(sky){
				light = sunColor;
			}
		}else if(sky){
			light = hit.color;
		}else{
			let hitNormal = hit.normal;
			col *= hit.color;
			feedCol *= hit.color;
			let cached = getCachedLighting(hitPoint, hitNormal);
			if(cached.a > 0.){
				light = cached.rgb;
			}else{
				let r = hash44ff(vec4(hitPoint, f32(i)));
				//the path goes on with a sun ray that counts for this much, or with a bounce
				var sunWeight = 0.;
				var bounces = false;
				if(!feedOn){
					if(r.z < uniforms.directionalProbability){
						sunWeight = 1.;
					}else{
						bounce++;
						bounces = bounce < uniforms.bounceCount;
					}
				}else if(bounce == 0){
					let endChance = select(feedLighting.a / uniforms.rayFeedPaths, 0., feedRefreshing);
					if(r.w < endChance){
						light = feedLighting.rgb;
					}else{
						feedCell = getFeedCell();
						feedCol = vec3(1.);
						if(r.z < uniforms.directionalProbability){
							sunWeight = 1.;
						}else{
							bounce = 1;
							bounces = true;
						}
					}
				}else if(feedLighting.a > 0.){
					light = feedLighting.rgb;
				}else{
					sunWeight = uniforms.directionalProbability;
				}

				if(sunWeight > 0.){
					let l = sampleSunDirection(r.xy);
					let cosine = dot(hitNormal, l);
					//a hit that faces away from the sun gets no sun ray
					if(cosine > 0.){
						col *= cosine * sunWeight;
						feedCol *= cosine * sunWeight;
						ro = hitPoint + hitNormal * MIN_DIST;
						rd = l;
						sunRay = true;
						ended = false;
					}
				}else if(bounces){
					ro = hitPoint;
					rd = cosWeightedRandomHemisphereDirection(hitNormal, r.xy);
					ended = false;
				}
			}
		}

		if(ended){
			sum += col * light;
			//threads that feed the same cell race, a sample gets lost, which is harmless
			if(feedCell != HASH_NONE){
				lightingBuffer[feedCell] = packLighting(addSample(unpackLighting(lightingBuffer[feedCell]), feedCol * light, uniforms.rayFeedPaths));
				feedCell = HASH_NONE;
			}
			i++;
			if(i >= sampleCount){
				break;
			}
			ro = p + n * MIN_DIST;
			rd = getIndirectDirection(n, seed, i, sampleCount);
			col = vec3(1.);
			bounce = 0;
			sunRay = false;
		}
	}
	return sum / f32(sampleCount);
}

@compute @workgroup_size(WORK_GROUP_SIZE_X, WORK_GROUP_SIZE_Y, 1)
fn compute_main(
  @builtin(global_invocation_id) GlobalInvocationID : vec3<u32>,
  @builtin(workgroup_id) workgroupID : vec3<u32>
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
		indices[i] = findOrInsert(getCellKey(worldPosition, direction, level + i));
	}
	//the hash map is full here
	if(indices[0] == HASH_NONE){
		return;
	}

	//the blue noise of the cell, not of the pixel: the pixel is random in the cell, so the noise of the pixels that
	//neighbouring cells pick would be white
	//the workgroups that refresh the feed, see feedRefreshing
	let group = workgroupID.x ^ (workgroupID.y << 16u);
	feedRefreshing = f32(hashU32(group ^ hashU32(u32(uniforms.frame))) >> 8u) / 16777216. < uniforms.feedRefresh;
	//getLighting of pathTrace.wgsl, with the paths of getIndirectPathSum
	let lighting = combineLighting(getIndirectPathSum(worldPosition, n, cellID, uniforms.sampleCount), getSeededDirectLighting(worldPosition, n, cellID));

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
		cell = addSample(cell, lighting, uniforms.accumulationMax);
		lightingBuffer[index] = packLighting(cell);
		prior = cell;
	}
}
