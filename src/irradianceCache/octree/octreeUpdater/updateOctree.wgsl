override WORK_GROUP_SIZE_X: u32;
override WORK_GROUP_SIZE_Y: u32;
override CELL_SIZE: u32;

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

//the lighting of the leaf at a hit point, for sampleRay of pathTrace.wgsl. The octree is a radiance cache.
//Only a filled leaf of the hit surface counts: an interior node only has the lighting of one sample, taken when it was split,
//which biased large blocks. Elsewhere (never seen, or a leaf filled from another face) the path goes on
fn getCachedLighting(hitPoint: vec3<f32>, n: vec3<f32>) -> vec4<f32> {
	let p = hitPoint / OCTREE_ROOT_NODE_SIZE;
	if(uniforms.cacheBounces != 0u && all(abs(p) < vec3(1.))){
		let leaf = getOctreeNode(p * .5 + .5);
		if(nodeIsFilled(leaf.node) && dot(getNormal(leaf.node), n) >= uniforms.edgeNormalThreshold){
			let accumulated = lightingBuffer[leaf.index];
			return vec4(unpack2x16float(accumulated.x), unpack2x16float(accumulated.y).x, 1.);
		}
	}
	return vec4(0.);
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
  @builtin(global_invocation_id) GlobalInvocationID : vec3<u32>
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
			//the color stays as a fallback for the pixels that the empty children don't cover yet (see getFallbackLighting).
			//A node that never had lighting gets the lighting of this sample, else a chain of splits leaves nothing to fall back on
			if(nodeResult.node.color == 0u){
				octreeBuffer[nodeIndex].color = packColor(getLighting(worldPosition, normalDepth.xyz, cellID, uniforms.sampleCount));
				octreeBuffer[nodeIndex].colorNormal = packNormal(normalDepth.xyz);
			}
			//only the colorNormal of a filled node: a node without color is not on an edge
			if(onEdge){
				octreeBuffer[nodeIndex].colorNormal = nodeResult.node.colorNormal | EDGE_SPLIT_FLAG;
			}
			return;
		}
		//the octree is full: the node stays a leaf and keeps accumulating, until collapseOctree.wgsl frees nodes
		atomicStore(&nodeSplitCount[nodeIndex], 0u);
	}

	{
		//the requested node exists. If it is a empty leaf node, then fill it. Then add lighting samples to it
		let splitCount = atomicLoad(&nodeSplitCount[nodeIndex]);
		if(splitCount > 0u || onEdge){
			return;
		}

		//the blue noise of the cell, not of the pixel: the pixel is random in the cell, so the noise of the pixels that
		//neighbouring cells pick would be white
		let lighting = getLighting(worldPosition, normalDepth.xyz, cellID, uniforms.sampleCount);

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
