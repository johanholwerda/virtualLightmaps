override WORK_GROUP_SIZE_X: u32;
override WORK_GROUP_SIZE_Y: u32;
override CELL_SIZE: u32;
//the slots of the feed hash map, see getFeedKey
override FEED_HASHMAP_SIZE: u32;

struct Uniforms {
	resolution : vec2<f32>,
	inverseProjection : mat4x4<f32>,
	inverseView : mat4x4<f32>,
	viewProjection : mat4x4<f32>,
	cameraPosition : vec3<f32>,
	scale : f32,
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
	accumulationMax : f32,
	edgeNormalThreshold : f32,
	edgeSplitDepth : u32,
	//1: the paths end at their first hit in the lighting the octree has there, see getCachedLighting
	cacheBounces : u32,
	//ancestors of the leaf that accumulate its samples too, at most ANCESTOR_LEVELS_MAX
	ancestorLevels : u32,
	//the samples the nearest ancestor counts as in the lighting shown for a leaf
	ancestorWeight : f32,
	//the samples of the ancestor lighting a new leaf starts with
	seedCount : f32,
	//the paths a feed cell holds, 0: no feed. See getCachedLighting
	rayFeedPaths : f32,
	//the size of the feed cells in m, see getFeedKey
	feedCellSize : f32,
	//the part of the workgroups whose paths feed full feed cells too, see feedRefreshing
	feedRefresh : f32,
	//the frames after which a feed cell no ray looked at can be taken by another one
	feedEvictFrames : f32,
}

const ANCESTOR_LEVELS_MAX = 4u;

//the node containing p, and its ancestors: ancestors[0] is the parent
struct NodePath {
	result: NodeResult,
	ancestors: array<u32, ANCESTOR_LEVELS_MAX>,
	ancestorCount: u32,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(3) var<storage, read_write> octreeBuffer : array<OctreeNode>;
@group(0) @binding(4) var<storage, read_write> lastWrittenIndex : atomic<u32>;
@group(0) @binding(5) var normalDepthTexture: texture_2d<u32>;
//after the nodes: the feed hash map, FEED_WORDS per slot, see getFeedWord. In this buffer as the shader is at the limit
//of 10 storage buffers
@group(0) @binding(6) var<storage, read_write> nodeSplitCount: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read> sceneBuffer : array<Model>;
@group(0) @binding(10) var<storage, read> bvhBuffer : array<BVHNode>;
//per node: running average of the lighting (rgb) and the frame count it holds (a), as 4 x f16
@group(0) @binding(9) var<storage, read_write> lightingBuffer: array<vec2<u32>>;
//the first indices of the blocks of 8 children that collapseOctree.wgsl freed
@group(0) @binding(11) var<storage, read_write> freeBlocks: array<u32>;
@group(0) @binding(12) var<storage, read_write> freeBlockCount: atomic<i32>;
//per node: the frame a sample last landed in it. collapseOctree.wgsl collapses the nodes that are not seen when the nodes run out
@group(0) @binding(13) var<storage, read_write> lastSeen: array<u32>;

fn getNodeCenter(p: vec3<f32>, octreeDepth: u32) -> vec3<f32>{
	let nodeCount = f32(1u << octreeDepth);
	return (floor(p * nodeCount) + .5) / nodeCount;
}

//a block of 8 children: a freed one, else a new one at the end. 0 when the octree is full (0 is the root, never a child)
//The blocks are empty: new ones were cleared at the reset, freed ones by collapseOctree.wgsl
fn allocateChildren() -> u32 {
	let freeIndex = atomicSub(&freeBlockCount, 1) - 1;
	if(freeIndex >= 0){
		return freeBlocks[freeIndex];
	}
	//none free: undo. Nothing is freed during this pass, so every thread that took the count below 0 adds its 1 back
	atomicAdd(&freeBlockCount, 1);
	let nodeCount = arrayLength(&octreeBuffer);
	if(atomicLoad(&lastWrittenIndex) + 8u > nodeCount){
		return 0u;
	}
	let lastIndex = atomicAdd(&lastWrittenIndex, 8u);
	return select(0u, lastIndex, lastIndex + 8u <= nodeCount);
}

//getOctreeNode, but it also keeps the last ANCESTOR_LEVELS_MAX nodes on the path
fn getOctreeNodePath(p: vec3<f32>) -> NodePath {
	var path: NodePath;
	var n: NodeResult;
	n.node = octreeBuffer[0];
	var nodeCount = 1.;
	for(var i = 1u; i <= OCTREE_DEPTH_MAX; i++){
		if(n.node.firstChildIndex == 0u){
			break;
		}
		for(var j = ANCESTOR_LEVELS_MAX - 1u; j > 0u; j--){
			path.ancestors[j] = path.ancestors[j - 1u];
		}
		path.ancestors[0] = n.index;
		nodeCount *= 2.;
		let coord = vec3<u32>(p * nodeCount) & vec3<u32>(1u);
		n.index = n.node.firstChildIndex + (coord.x | (coord.y << 1u) | (coord.z << 2u));
		n.node = octreeBuffer[n.index];
		n.depth = i;
	}
	path.result = n;
	path.ancestorCount = min(n.depth, ANCESTOR_LEVELS_MAX);
	return path;
}

//the running average of the lighting (rgb) and the sample count it holds (a)
fn unpackLighting(v: vec2<u32>) -> vec4<f32> {
	return vec4(unpack2x16float(v.x), unpack2x16float(v.y));
}

fn packLighting(l: vec4<f32>) -> vec2<u32> {
	return vec2(pack2x16float(l.rg), pack2x16float(l.ba));
}

//the count is capped, so the average keeps following changes in the scene
fn addSample(accumulated: vec4<f32>, lighting: vec3<f32>) -> vec4<f32> {
	return vec4(mix(accumulated.rgb, lighting, 1. / (accumulated.a + 1.)), min(accumulated.a + 1., uniforms.accumulationMax));
}

fn getOctreeDepth(distance: f32) -> u32{
	return min(u32(floor(log2(uniforms.scale / distance))), OCTREE_DEPTH_MAX);
}

/*
the feed hash map: world space cells of feedCellSize that the rays fill themselves, for the hit points the octree has no
lighting for, like the feed cells of updateSpatialHash.wgsl. A hash map and not nodes of the octree: the octree only
has the nodes of what the screen sees, a split from a ray would race with the splits of the cells, and collapseOctree.wgsl
would free what is not seen.
Per slot FEED_WORDS words: the checksum of the cell (0: empty), the frame a ray last looked at it, and its running
average of the lighting (rgb) and the paths it holds (a) as 4 x f16
 */
const FEED_WORDS = 4u;
const FEED_NONE = 0xFFFFFFFFu;
//slots probed before a feed cell gives up
const FEED_SEARCH_COUNT = 16u;

//the index in nodeSplitCount of a word of a feed slot: after the split counts of the nodes
fn getFeedWord(slot: u32, word: u32) -> u32 {
	return arrayLength(&octreeBuffer) + slot * FEED_WORDS + word;
}

fn feedHash(x: u32) -> u32 {
	let state = x * 747796405u + 2891336453u;
	let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

/*
the checksum and the first slot of the feed cell at p, of the face n is closest to: the axis with its sign, so the 2
sides of a wall never share a cell. Offset along that axis, so a surface exactly on a cell boundary is in 1 cell.
The same size for the whole scene and wherever the camera is, so the feed of a surface stays as the camera moves.
The slot and the checksum are independent hashes, so 2 cells with the same checksum still need the same probe sequence
 */
fn getFeedKey(p: vec3<f32>, n: vec3<f32>) -> vec2<u32> {
	let a = abs(n);
	var axis = 2u;
	if(a.x >= a.y && a.x >= a.z){
		axis = 0u;
	}else if(a.y >= a.z){
		axis = 1u;
	}
	var direction = vec3(0.);
	direction[axis] = select(-1., 1., n[axis] > 0.);
	let face = axis * 2u + select(0u, 1u, n[axis] > 0.);
	let cell = bitcast<vec3<u32>>(vec3<i32>(floor((p + direction * (uniforms.feedCellSize * .1)) / uniforms.feedCellSize)));
	let slot = feedHash(cell.x + feedHash(cell.y + feedHash(cell.z + feedHash(face))));
	let key = max(feedHash((cell.x * 73856093u) ^ feedHash((cell.y * 19349663u) ^ feedHash((cell.z * 83492791u) ^ ((face + 1u) * 0x9e3779b9u)))), 1u);
	return vec2(key, slot % FEED_HASHMAP_SIZE);
}

fn findFeed(keySlot: vec2<u32>) -> u32 {
	var slot = keySlot.y;
	for(var i = 0u; i < FEED_SEARCH_COUNT; i++){
		let existing = atomicLoad(&nodeSplitCount[getFeedWord(slot, 0u)]);
		if(existing == keySlot.x){
			return slot;
		}
		if(existing == 0u){
			return FEED_NONE;
		}
		slot = (slot + 1u) % FEED_HASHMAP_SIZE;
	}
	return FEED_NONE;
}

//linear probing, like findOrInsert of updateSpatialHash.wgsl: up to the first empty slot, which a cell that is already
//in the map is never after, then the first stale slot on the way (not looked at for feedEvictFrames), else the empty one
fn findOrInsertFeed(keySlot: vec2<u32>) -> u32 {
	let key = keySlot.x;
	let frame = u32(uniforms.frame);
	var slot = keySlot.y;
	var evictSlot = FEED_NONE;
	var evictKey = 0u;
	var i = 0u;
	for(; i < FEED_SEARCH_COUNT; i++){
		let existing = atomicLoad(&nodeSplitCount[getFeedWord(slot, 0u)]);
		if(existing == key){
			atomicStore(&nodeSplitCount[getFeedWord(slot, 1u)], frame);
			return slot;
		}
		if(existing == 0u){
			break;
		}
		if(evictSlot == FEED_NONE && f32(frame - atomicLoad(&nodeSplitCount[getFeedWord(slot, 1u)])) > uniforms.feedEvictFrames){
			evictSlot = slot;
			evictKey = existing;
		}
		slot = (slot + 1u) % FEED_HASHMAP_SIZE;
	}

	if(evictSlot != FEED_NONE){
		for(var j = 0u; j < 4u; j++){
			let cmp = atomicCompareExchangeWeak(&nodeSplitCount[getFeedWord(evictSlot, 0u)], evictKey, key);
			if(cmp.exchanged){
				atomicStore(&nodeSplitCount[getFeedWord(evictSlot, 1u)], frame);
				atomicStore(&nodeSplitCount[getFeedWord(evictSlot, 2u)], 0u);
				atomicStore(&nodeSplitCount[getFeedWord(evictSlot, 3u)], 0u);
				return evictSlot;
			}
			//another thread inserted the same cell there
			if(cmp.old_value == key){
				return evictSlot;
			}
			if(cmp.old_value != evictKey){
				break;
			}
		}
	}

	//a spurious failure retries the same slot: skipping it would leave an empty slot before the key, where find stops
	for(; i < FEED_SEARCH_COUNT; i++){
		let cmp = atomicCompareExchangeWeak(&nodeSplitCount[getFeedWord(slot, 0u)], 0u, key);
		if(cmp.exchanged || cmp.old_value == key){
			atomicStore(&nodeSplitCount[getFeedWord(slot, 1u)], frame);
			return slot;
		}
		if(cmp.old_value != 0u){
			slot = (slot + 1u) % FEED_HASHMAP_SIZE;
		}
	}
	return FEED_NONE;
}

fn loadFeedLighting(slot: u32) -> vec4<f32> {
	return unpackLighting(vec2(atomicLoad(&nodeSplitCount[getFeedWord(slot, 2u)]), atomicLoad(&nodeSplitCount[getFeedWord(slot, 3u)])));
}

//a path of light (without the albedo of the feed cell) added to a feed cell, which holds up to rayFeedPaths of them.
//Threads that feed the same cell race, a sample gets lost, which is harmless
fn addFeedSample(slot: u32, lighting: vec3<f32>) {
	let accumulated = loadFeedLighting(slot);
	let packed = packLighting(vec4(mix(accumulated.rgb, lighting, 1. / (accumulated.a + 1.)), min(accumulated.a + 1., uniforms.rayFeedPaths)));
	atomicStore(&nodeSplitCount[getFeedWord(slot, 2u)], packed.x);
	atomicStore(&nodeSplitCount[getFeedWord(slot, 3u)], packed.y);
}

//the feed cell of the hit point of the last getCachedLighting that found no lighting in the octree.
//feedOn: there is a feed (rayFeedPaths), the rest is only set then
var<private> feedOn: bool;
//what the feed cell holds, alpha 0 when it has no paths or no slot yet
var<private> feedLighting: vec4<f32>;
//its key, and its slot (FEED_NONE when it has none yet), see getFeedSlot
var<private> feedKeySlot: vec2<u32>;
var<private> feedSlot: u32;

/*
the paths of this workgroup all feed the cell of their first hit, also when it is full: set by compute_main for
feedRefresh of the workgroups. So a full feed cell keeps following the lighting of what its paths end in: its first
samples are from when the feed around it was still empty, with less bounce light, and the sun moves.
Per workgroup and not per path: a path that feeds takes 1 or 2 rays more, and the threads of a workgroup wait on the
one that takes the most
 */
var<private> feedRefreshing: bool;

/*
the lighting of the leaf at a hit point, for getIndirectPathSum. The octree is a radiance cache.
Only a filled leaf of the hit surface counts: an interior node only has the lighting of one sample, taken when it was split,
which biased large blocks.
Elsewhere (out of view, never seen, a leaf filled from another face, outside the root) this returns none, and with the
feed (rayFeedPaths) it sets the feed cell of the hit point: feedOn and the vars with it. A feed cell is kept while rays
look at it, see feedEvictFrames
 */
fn getCachedLighting(hitPoint: vec3<f32>, n: vec3<f32>) -> vec4<f32> {
	feedOn = false;
	if(uniforms.cacheBounces == 0u){
		return vec4(0.);
	}
	let p = hitPoint / OCTREE_ROOT_NODE_SIZE;
	if(all(abs(p) < vec3(1.))){
		let leaf = getOctreeNode(p * .5 + .5);
		if(nodeIsFilled(leaf.node) && dot(getNormal(leaf.node), n) >= uniforms.edgeNormalThreshold){
			let accumulated = lightingBuffer[leaf.index];
			return vec4(unpack2x16float(accumulated.x), unpack2x16float(accumulated.y).x, 1.);
		}
	}
	if(uniforms.rayFeedPaths <= 0.){
		return vec4(0.);
	}

	feedOn = true;
	feedLighting = vec4(0.);
	feedKeySlot = getFeedKey(hitPoint, n);
	feedSlot = findFeed(feedKeySlot);
	if(feedSlot == FEED_NONE){
		return vec4(0.);
	}
	//the cell stays while rays look at it. Only written when it changes: the rays of many threads hit the same cells
	let frame = u32(uniforms.frame);
	let time = &nodeSplitCount[getFeedWord(feedSlot, 1u)];
	if(atomicLoad(time) != frame){
		atomicStore(time, frame);
	}
	feedLighting = loadFeedLighting(feedSlot);
	return vec4(0.);
}

//the slot of the feed cell of the hit point of the last getCachedLighting, which set feedOn, to add a sample to. It is
//added when it is not in the hash map yet. FEED_NONE when the hash map is full there
fn getFeedSlot() -> u32 {
	if(feedSlot != FEED_NONE){
		return feedSlot;
	}
	return findOrInsertFeed(feedKeySlot);
}

/*
the indirect lighting of getLighting of pathTrace.wgsl: the average of its sampleCount paths, as 1 loop over the rays of
all of them, like getIndirectPathSum of updateSpatialHash.wgsl. A thread whose path ended starts on its next path in
the same step, while the others bounce. The sun ray of a hit is a step of the loop too, a traceRay to the sky.
This is the only traceRay call site of this shader: sampleRay is not used.

A path ends at the first hit that the octree has lighting for (getCachedLighting). Where it has none:
-without the feed (rayFeedPaths 0) it goes on like sampleRay, for up to bounceCount bounces
-with the feed it is at most 2 rays and a sun ray, whatever bounceCount:
 1 the first hit ends in its feed cell, with the chance of the part of rayFeedPaths paths that cell holds. Else the
   path feeds it: it goes on for 1 more ray (the sun sample with directionalProbability, else a bounce), and the light
   that brings to the hit, without its albedo, is added to the cell as a sample of 1 path. A full cell is still fed
   by the workgroups that refresh (feedRefreshing). The cell is added here
 2 the second hit ends in its feed cell when that holds anything, however few paths. Else in its sun light only, the
   sun ray times directionalProbability, without a bounce. No cell is added here: the feed is what the surfaces in view
   see, not the whole scene
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
	var feedCell = FEED_NONE;
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
						feedCell = getFeedSlot();
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
			if(feedCell != FEED_NONE){
				addFeedSample(feedCell, feedCol * light);
				feedCell = FEED_NONE;
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

/*
fn getLocalIndex(p: vec3<f32>, octreeDepth: u32) -> u32{
	let nodeCount = f32(1u << octreeDepth);
//	let coord = vec3<u32>((p * nodeCount) % 2);
	let coord = vec3<u32>(p * nodeCount) & vec3<u32>(1u);
//	return coord.x + coord.y * 2 + coord.z * 4;
	return coord.x | (coord.y << 1u) | (coord.z << 2u);
}
*/

@compute @workgroup_size(WORK_GROUP_SIZE_X, WORK_GROUP_SIZE_Y, 1)
fn compute_main(
  @builtin(global_invocation_id) GlobalInvocationID : vec3<u32>,
  @builtin(workgroup_id) workgroupID : vec3<u32>
) {
	let cellID = GlobalInvocationID.xy;
	//not the first 2 numbers of the hash: getLighting takes those as its white noise, the cell is its seed
	let rand = hash34uf(vec3<u32>(cellID, u32(uniforms.frame))).zw;
//	let coord = cellID * CELL_SIZE + CELL_SIZE / 2;
	//random pixel in the cell
	let coord = cellID * CELL_SIZE + vec2<u32>(rand * f32(CELL_SIZE));
	if(coord.x >= u32(uniforms.resolution.x) || coord.y >= u32(uniforms.resolution.y)){
		return;
	}

	let normalDepth = loadNormalDepth(normalDepthTexture, coord);

	let uv = (vec2<f32>(coord) + .5) / uniforms.resolution;
	let worldPosition = depthToPosition(normalDepth.w, uv * 2. - 1.);
	//the sky gets no nodes, it is drawn directly (see octreeSplat/normalize.wgsl)
	if(isSky(worldPosition)){
		return;
	}
	let targetOctreeDepth = getOctreeDepth(distance(worldPosition, uniforms.cameraPosition));
	//to local space
	var p = worldPosition / OCTREE_ROOT_NODE_SIZE;
	if(any(abs(p) >= vec3(1.))){
		return;
	}
	p = p * .5 + .5;

//	let nodeResult = getOctreeNode(p, targetOctreeDepth);
	let path = getOctreeNodePath(p);
	let nodeResult = path.result;
	let nodeIndex = nodeResult.index;
	//before the returns below: the node is visible, whatever happens to it. Cells that land in the same node write the same frame
	lastSeen[nodeIndex] = u32(uniforms.frame);

	//the node straddles a hard edge: the sample is on a different face than the one the node was filled with.
	//Edge nodes get split deeper than the target depth, and samples of the other face are not accumulated in them
	let onEdge = nodeIsFilled(nodeResult.node) && dot(getNormal(nodeResult.node), normalDepth.xyz) < uniforms.edgeNormalThreshold;
	let edgeOctreeDepth = min(targetOctreeDepth + uniforms.edgeSplitDepth, OCTREE_DEPTH_MAX);

	//the node was split here, and gets the lighting of this sample as it never had any: see below
	var split = false;
	var fillSplit = false;
	if(nodeResult.depth < targetOctreeDepth || (onEdge && nodeResult.depth < edgeOctreeDepth)){
		//Only the first thread that brings nodeSplitCount[nodeIndex] from 0 to 1 passes
 		let splitCount = atomicAdd(&nodeSplitCount[nodeIndex], 1);
		if(splitCount > 0){
			return;
		}

		//The children are empty already. Only the fields are written, not the node: other cells of this pass can be
		//filling the node, or its children, and a whole node write would undo theirs
		let firstChildIndex = allocateChildren();
		if(firstChildIndex != 0u){
			//the center and size, used by collapseOctree.wgsl
			octreeBuffer[nodeIndex].worldPosition = packPositionAndSize(getNodeCenter(p, nodeResult.depth), 1. / f32(1u << nodeResult.depth));
			octreeBuffer[nodeIndex].normal = 0u;
			octreeBuffer[nodeIndex].firstChildIndex = firstChildIndex;
			//only the colorNormal of a filled node: a node without color is not on an edge
			if(onEdge){
				octreeBuffer[nodeIndex].colorNormal = nodeResult.node.colorNormal | EDGE_SPLIT_FLAG;
			}
			split = true;
			fillSplit = nodeResult.node.color == 0u;
			if(!fillSplit){
				return;
			}
		}else{
			//the octree is full: the node stays a leaf and keeps accumulating, until collapseOctree.wgsl frees nodes
			atomicStore(&nodeSplitCount[nodeIndex], 0u);
		}
	}

	{
		//the requested node exists. If it is a empty leaf node, then fill it. Then add lighting samples to it
		if(!split && (atomicLoad(&nodeSplitCount[nodeIndex]) > 0u || onEdge)){
			return;
		}

		//the workgroups that refresh the feed, see feedRefreshing
		let group = workgroupID.x ^ (workgroupID.y << 16u);
		feedRefreshing = f32(feedHash(group ^ feedHash(u32(uniforms.frame))) >> 8u) / 16777216. < uniforms.feedRefresh;
		//getLighting of pathTrace.wgsl with the paths of getIndirectPathSum, called once: a second call site of the
		//path loop made this shader slower.
		//The blue noise of the cell, not of the pixel: the pixel is random in the cell, so the noise of the pixels that
		//neighbouring cells pick would be white
		let lighting = combineLighting(getIndirectPathSum(worldPosition, normalDepth.xyz, cellID, uniforms.sampleCount), getSeededDirectLighting(worldPosition, normalDepth.xyz, cellID));

		//the color of the split node stays as a fallback for the pixels that the empty children don't cover yet (see
		//getFallbackLighting). A node that never had lighting gets the lighting of this sample, else a chain of splits
		//leaves nothing to fall back on. Its colorNormal flagged as an edge above stays
		if(split){
			octreeBuffer[nodeIndex].color = packColor(lighting);
			if(!onEdge){
				octreeBuffer[nodeIndex].colorNormal = packNormal(normalDepth.xyz);
			}
			return;
		}

		//cells that hit the same node in the same frame race here and below. One of their samples gets lost, which is harmless

		//the ancestors lit from the face of this sample accumulate it too. They cover more cells than the leaf, so they converge
		//faster, and the nearest one is the prior of the leaf. Ancestors lit from another face (at a hard edge) are skipped.
		//Their color is updated as well, it is the fallback for the pixels no leaf covers (see getFallbackLighting)
		var prior = vec4(0.);
		for(var i = 0u; i < min(path.ancestorCount, uniforms.ancestorLevels); i++){
			let index = path.ancestors[i];
			let colorNormal = unpackNormal(octreeBuffer[index].colorNormal);
			if(octreeBuffer[index].color == 0u || dot(colorNormal, normalDepth.xyz) < uniforms.edgeNormalThreshold){
				continue;
			}
			let ancestor = addSample(unpackLighting(lightingBuffer[index]), lighting);
			lightingBuffer[index] = packLighting(ancestor);
			octreeBuffer[index].color = packColor(ancestor.rgb);
			if(prior.a == 0.){
				prior = ancestor;
			}
		}

		var leaf = unpackLighting(lightingBuffer[nodeIndex]);
		//a new leaf starts from the lighting of its ancestor, as seedCount samples, instead of from its first noisy sample
		if(leaf.a == 0. && prior.a > 0.){
			leaf = vec4(prior.rgb, min(prior.a, uniforms.seedCount));
		}
		leaf = addSample(leaf, lighting);
		lightingBuffer[nodeIndex] = packLighting(leaf);

		//the lighting shown: while the leaf has few samples it leans on the smoother lighting of the ancestor.
		//lightingBuffer keeps the leaf's own average, the blend is only in the color
		var average = leaf.rgb;
		if(prior.a > 0.){
			average = (leaf.rgb * leaf.a + prior.rgb * uniforms.ancestorWeight) / (leaf.a + uniforms.ancestorWeight);
		}

		if(!nodeIsFilled(nodeResult.node)){
			octreeBuffer[nodeIndex].normal = packNormal (normalDepth.xyz);
			octreeBuffer[nodeIndex].colorNormal = packNormal (normalDepth.xyz);
			octreeBuffer[nodeIndex].worldPosition = packPositionAndSize(p.xyz, 1. / f32(1u << nodeResult.depth));
		}
		octreeBuffer[nodeIndex].color = packColor(average);
	}

}
