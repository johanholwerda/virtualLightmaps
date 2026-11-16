override WORK_GROUP_SIZE: u32;

struct Uniforms {
	cameraPosition : vec3<f32>,
	scale : f32,
	edgeNormalThreshold : f32,
	edgeSplitDepth : u32,
	accumulationMax : f32,
	//levels finer than the view needs that are kept, so zooming back in doesn't split them again
	keepLevels : u32,
	//levels coarser than the view needs for the nodes that no sample landed in for unseenFrames, see isUnderPressure
	offscreenLevels : u32,
	//frames without a sample after which a node counts as unseen: offscreen or occluded
	unseenFrames : u32,
	//the fraction of the blocks that is kept free once the end of octreeBuffer is reached, see isUnderPressure
	freeReserve : f32,
	//0 or 1: the depth parity of the nodes collapsed this frame
	parity : u32,
	frame : u32,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage, read_write> octreeBuffer : array<OctreeNode>;
@group(0) @binding(3) var<storage, read_write> nodeSplitCount: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> lightingBuffer: array<vec2<u32>>;
//the first indices of the blocks of 8 children that are free, see allocateChildren in updateOctree.wgsl
@group(0) @binding(5) var<storage, read_write> freeBlocks: array<u32>;
@group(0) @binding(6) var<storage, read_write> freeBlockCount: atomic<i32>;
//per node: the frame a g-buffer sample of updateOctree.wgsl last landed in it
@group(0) @binding(7) var<storage, read_write> lastSeen: array<u32>;
//not atomic: read only here, the same value for all threads
@group(0) @binding(8) var<storage, read> lastWrittenIndex: u32;
//freeBlockCount after updateOctree.wgsl, copied: freeBlockCount itself is atomic here, and an atomic load per thread serializes
@group(0) @binding(9) var<storage, read> freeBlockCountSnapshot: i32;

//same as in updateOctree.wgsl
fn getOctreeDepth(distance: f32) -> u32{
	return min(u32(floor(log2(uniforms.scale / distance))), OCTREE_DEPTH_MAX);
}

fn getNodeCenter(p: vec3<f32>, octreeDepth: u32) -> vec3<f32>{
	let nodeCount = f32(1u << octreeDepth);
	return (floor(p * nodeCount) + .5) / nodeCount;
}

/*
The unseen nodes are only collapsed when the nodes run out: the end of octreeBuffer is reached and the free list is below the reserve.
Until then the nodes of what was looked at stay, also when it is looked away from or hidden behind something for a long time
 */
fn isUnderPressure() -> bool {
	let nodeCount = arrayLength(&octreeBuffer);
	return lastWrittenIndex + 8u > nodeCount && f32(freeBlockCountSnapshot) < uniforms.freeReserve * f32(nodeCount / 8u);
}

//the last frame a sample landed in the node or its children. The node got the frame of its split as a leaf, so new children
//that no sample landed in yet count as seen
fn getLastSeen(nodeIndex: u32, firstChildIndex: u32) -> u32 {
	var seen = lastSeen[nodeIndex];
	for(var i = 0u; i < 8u; i++){
		seen = max(seen, lastSeen[firstChildIndex + i]);
	}
	return seen;
}

/*
One thread per node. A node whose 8 children are all leaves becomes a leaf when the children are finer than the view needs.
Its children are cleared and their block goes on the free list. A branch collapses one level at a time, from the bottom up.
Only the nodes of one depth parity collapse per frame, so a node and its parent never collapse in the same pass
 */
@compute @workgroup_size(WORK_GROUP_SIZE, 1, 1)
fn compute_main(
  @builtin(global_invocation_id) GlobalInvocationID : vec3<u32>
) {
	let nodeIndex = GlobalInvocationID.x;
	//not lastWrittenIndex: an atomic load per thread of the same address serializes, and makes the pass 20x slower.
	//The nodes that were never allocated are empty, so they return below
	if(nodeIndex >= arrayLength(&octreeBuffer)){
		return;
	}

	//only the fields that are needed: most nodes are leaves, and most of the rest are kept
	let firstChildIndex = octreeBuffer[nodeIndex].firstChildIndex;
	if(firstChildIndex == 0u){
		return;
	}

	//the size of an interior node is set when it is split
	let worldPosition = octreeBuffer[nodeIndex].worldPosition;
	let positionAndSize = vec4(unpack2x16unorm(worldPosition.x), unpack2x16unorm(worldPosition.y));
	let size = positionAndSize.w;
	let depth = u32(round(-log2(size)));
	if((depth & 1u) != uniforms.parity){
		return;
	}

	for(var i = 0u; i < 8u; i++){
		if(octreeBuffer[firstChildIndex + i].firstChildIndex != 0u){
			return;
		}
	}

	//the box of the node in world space
	let center = (getNodeCenter(positionAndSize.xyz, depth) * 2. - 1.) * OCTREE_ROOT_NODE_SIZE;
	let halfSize = size * OCTREE_ROOT_NODE_SIZE;
	let boxDistance = length(max(abs(uniforms.cameraPosition - center) - halfSize, vec3(0.)));
	let targetDepth = i32(getOctreeDepth(max(boxDistance, 1e-4)));

	var allowedDepth = targetDepth + i32(uniforms.keepLevels);
	//children on a hard edge (see below) may be up to edgeSplitDepth deeper. Nodes that are fine even then are kept without reading their children
	if(i32(depth) + 1 <= allowedDepth){
		//fine enough for the view. Only when the nodes run out, the ones that are offscreen or occluded get coarser.
		//The samples come from the g-buffer, so a node that none landed in for a while is not visible
		if(!isUnderPressure() || getLastSeen(nodeIndex, firstChildIndex) + uniforms.unseenFrames > uniforms.frame){
			return;
		}
		allowedDepth = targetDepth - i32(uniforms.offscreenLevels);
		if(i32(depth) + 1 <= allowedDepth){
			return;
		}
	}
	//split on an edge. The children of the face that came second are filled by the few samples that land on it,
	//often a line of single pixels in the distance. Collapsed before that (the check of the children below misses it),
	//the node was split again by the next sample of that face: the edge never kept its lighting, and flickered
	let edgeSplit = (octreeBuffer[nodeIndex].colorNormal & EDGE_SPLIT_FLAG) != 0u;
	if(edgeSplit && i32(depth) + 1 <= allowedDepth + i32(uniforms.edgeSplitDepth)){
		return;
	}

	//the child with the most samples gives the normal and the position
	var best = 8u;
	var bestCount = 0.;
	for(var i = 0u; i < 8u; i++){
		if(octreeBuffer[firstChildIndex + i].normal > 0u){
			let count = unpack2x16float(lightingBuffer[firstChildIndex + i].y).y;
			if(best == 8u || count > bestCount){
				best = i;
				bestCount = count;
			}
		}
	}

	//children that face different ways are on a hard edge, which the update splits deeper on purpose.
	//The lighting is the average of the children that face the same way as the best one, weighted by their sample counts
	var parent: OctreeNode;
	parent.worldPosition = worldPosition;
	var lighting = vec2(0u);
	if(best < 8u){
		let bestChild = octreeBuffer[firstChildIndex + best];
		let n = getNormal(bestChild);
		var onEdge = false;
		var sum = vec3(0.);
		var count = 0.;
		for(var i = 0u; i < 8u; i++){
			let normal = octreeBuffer[firstChildIndex + i].normal;
			if(normal > 0u){
				if(dot(unpackNormal(normal), n) < uniforms.edgeNormalThreshold){
					onEdge = true;
				}else{
					let accumulated = lightingBuffer[firstChildIndex + i];
					let rg = unpack2x16float(accumulated.x);
					let ba = unpack2x16float(accumulated.y);
					sum += vec3(rg, ba.x) * ba.y;
					count += ba.y;
				}
			}
		}
		if(onEdge && i32(depth) + 1 <= allowedDepth + i32(uniforms.edgeSplitDepth)){
			return;
		}
		let average = sum / max(count, 1.);
		parent.normal = bestChild.normal;
		parent.colorNormal = bestChild.colorNormal;
		parent.worldPosition = packPositionAndSize(getPositionAndSize(bestChild).xyz, size);
		parent.color = packColor(average);
		lighting = vec2(pack2x16float(average.rg), pack2x16float(vec2(average.b, min(count, uniforms.accumulationMax))));
	}
	//firstChildIndex is 0: a leaf. It keeps the last frame its children were seen
	lastSeen[nodeIndex] = getLastSeen(nodeIndex, firstChildIndex);
	octreeBuffer[nodeIndex] = parent;
	lightingBuffer[nodeIndex] = lighting;
	atomicStore(&nodeSplitCount[nodeIndex], 0u);

	//cleared, so a split that reuses the block gets empty children
	var empty: OctreeNode;
	for(var i = 0u; i < 8u; i++){
		octreeBuffer[firstChildIndex + i] = empty;
		lightingBuffer[firstChildIndex + i] = vec2(0u);
		lastSeen[firstChildIndex + i] = 0u;
		atomicStore(&nodeSplitCount[firstChildIndex + i], 0u);
	}
	freeBlocks[atomicAdd(&freeBlockCount, 1)] = firstChildIndex;
}
