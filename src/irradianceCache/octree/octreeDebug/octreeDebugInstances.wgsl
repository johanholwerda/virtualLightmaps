override WORK_GROUP_SIZE: u32;

struct Uniforms {
	//only the leaves get a box, else the interior nodes too
	leavesOnly : f32,
	//the nodes without lighting (red) get no box
	hideEmpty : f32,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage, read> octreeBuffer : array<OctreeNode>;
//x the index of the node, y the index of its parent (ROOT_PARENT for the root). Unfilled children have no position of their own,
//their box comes from the parent, see octreeDebug.wgsl
@group(0) @binding(2) var<storage, read_write> instances : array<vec2<u32>>;
//drawIndirect: vertexCount, instanceCount, firstVertex, firstInstance. Cleared before this pass
@group(0) @binding(3) var<storage, read_write> drawArgs : array<atomic<u32>>;

const ROOT_PARENT = 0xffffffffu;

fn isDrawn(index: u32) -> bool{
	if(uniforms.leavesOnly > .5 && octreeBuffer[index].firstChildIndex != 0u){
		return false;
	}
	return uniforms.hideEmpty < .5 || octreeBuffer[index].color != 0u;
}

/*
One thread per node. An interior node adds its 8 children to the instances. Going over the parents, not the nodes themselves,
skips the blocks on the free list and gives the empty children a box
 */
@compute @workgroup_size(WORK_GROUP_SIZE, 1, 1)
fn compute_main(
  @builtin(global_invocation_id) GlobalInvocationID : vec3<u32>
) {
	let nodeIndex = GlobalInvocationID.x;
	if(nodeIndex >= arrayLength(&octreeBuffer)){
		return;
	}
	let firstChildIndex = octreeBuffer[nodeIndex].firstChildIndex;

	if(nodeIndex == 0u){
		//12 lines of 2 vertices per box
		atomicStore(&drawArgs[0], 24u);
		//the root has no parent to add it
		if(isDrawn(0u)){
			let slot = atomicAdd(&drawArgs[1], 1u);
			instances[slot] = vec2(0u, ROOT_PARENT);
		}
	}

	if(firstChildIndex == 0u){
		return;
	}

	var mask = 0u;
	for(var i = 0u; i < 8u; i++){
		if(isDrawn(firstChildIndex + i)){
			mask |= 1u << i;
		}
	}
	if(mask == 0u){
		return;
	}
	var slot = atomicAdd(&drawArgs[1], countOneBits(mask));
	for(var i = 0u; i < 8u; i++){
		if((mask & (1u << i)) != 0u){
			instances[slot] = vec2(firstChildIndex + i, nodeIndex);
			slot++;
		}
	}
}
