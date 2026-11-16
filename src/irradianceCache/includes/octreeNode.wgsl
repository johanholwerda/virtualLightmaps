override OCTREE_DEPTH_MAX: u32;
override OCTREE_ROOT_NODE_SIZE: f32;

struct NodeResult{
	node: OctreeNode,
	depth: u32,
	index: u32,
}

struct OctreeNode {
	//could be 3 bytes
	//packed normal of the surface the color was sampled on. Unlike normal it stays when the node is split,
	//so getFallbackLighting can skip ancestors lit from another face. The 4th byte is EDGE_SPLIT_FLAG
	colorNormal: u32,
	//could be 3 bytes
	//needed for lighting
	normal: u32,
	//could be 3 bytes
	color: u32,
	//could be 3 bytes
	//is 0 when it does not have children
	firstChildIndex: u32,
	//packed normalized world position, node size
	worldPosition: vec2<u32>,
};

//the 4th byte of colorNormal (packNormal leaves it 0): the node was split because it straddles a hard edge, see updateOctree.wgsl.
//collapseOctree.wgsl keeps its children up to edgeSplitDepth deeper than the view needs, also before the other face has filled any
const EDGE_SPLIT_FLAG = 0xff000000u;

fn nodeIsFilled(node: OctreeNode) -> bool{
	return node.normal > 0u;
}

fn getOctreeNode(p: vec3<f32>) -> NodeResult {
	var n: NodeResult;
	n.depth = 0u;
	n.node = octreeBuffer[0];
	var nodeCount = 1.;
	for(var i = 1u; i <= OCTREE_DEPTH_MAX; i++){
		if(n.node.firstChildIndex > 0){
			nodeCount *= 2.;
			let coord = vec3<u32>(p * nodeCount) & vec3<u32>(1u);
			let localIndex = coord.x | (coord.y << 1u) | (coord.z << 2u);
			n.index = n.node.firstChildIndex + localIndex;
			n.node = octreeBuffer[n.index];
			n.depth = i;
		}else{
			//is a leaf node
			break;
		}
	}
	return n;
}

fn getPositionAndSize(node:OctreeNode)-> vec4<f32>{
	let data0 = unpack2x16unorm(node.worldPosition.x);
	let data1 = unpack2x16unorm(node.worldPosition.y);
	return vec4(data0, data1);
}

fn packPositionAndSize(position: vec3<f32>, size: f32) -> vec2<u32> {
	return vec2(pack2x16unorm(vec2(position.xy)), pack2x16unorm(vec2(position.z, size)));
}

//color is HDR lighting, reinhard compressed to fit 8 bits per channel. See packColor
fn getColor(node:OctreeNode) -> vec3<f32>{
	return unpackColor(node.color);
}

fn unpackColor(color: u32) -> vec3<f32>{
	let c = unpack4x8unorm(color).xyz;
	return c / max(1. - c, vec3(1. / 255.));
}

//the minimum dot of the pixel normal and colorNormal for the fallback lighting
const FALLBACK_NORMAL_THRESHOLD = .7;

/*
The lighting of the deepest node containing p that has lighting of a surface facing like n: the leaf, else the ancestor it was split from.
A split keeps the color of the node (see updateOctree.wgsl), so it fills the pixels no leaf node covers yet.
Ancestors lit from another face are skipped: at a hard edge the face that filled the node first would leak onto the other one.
Only up to levelsMax levels above the leaf: a larger ancestor spans other surfaces facing the same way, like the sunlit side of a pillar
10 cm from the shadowed side of a molding on it. Above the levels that accumulate samples (ancestorLevels of updateOctree.wgsl)
its color is the one sample of its split, or the lighting from when it was a large leaf far away: bright specks in the shadowed
corners whose leaves were filled by the other face.
p is in octree space (0 -> 1). Alpha is 0 when no node on the path has matching lighting
 */
fn getFallbackLighting(p: vec3<f32>, n: vec3<f32>, levelsMax: u32) -> vec4<f32>{
	var index = 0u;
	var color = 0u;
	var colorDepth = 0u;
	var depth = 0u;
	for(; depth <= OCTREE_DEPTH_MAX; depth++){
		//only the 3 fields, not the whole node
		let nodeColor = octreeBuffer[index].color;
		if(nodeColor != 0u && dot(unpackNormal(octreeBuffer[index].colorNormal), n) >= FALLBACK_NORMAL_THRESHOLD){
			color = nodeColor;
			colorDepth = depth;
		}
		let firstChildIndex = octreeBuffer[index].firstChildIndex;
		if(firstChildIndex == 0u){
			break;
		}
		let c = vec3<u32>(p * f32(1u << (depth + 1u))) & vec3(1u);
		index = firstChildIndex + (c.x | (c.y << 1u) | (c.z << 2u));
	}
	if(color == 0u || depth - colorDepth > levelsMax){
		return vec4(0.);
	}
	return vec4(unpackColor(color), 1.);
}

/*
The average lighting of the filled leaves one leaf size away from p along the surface, in the 4 directions of the tangents of n,
that face like n. For the pixels without any matching node on their path: the face that came second at a hard edge,
whose samples the edge leaf drops (see onEdge in updateOctree.wgsl). Their own face continues in the leaf next to it.
The tangents are axis aligned for an axis aligned n, like the faces of the boxes.
p is in octree space (0 -> 1). Alpha is the number of leaves found
 */
fn getNeighbourLighting(p: vec3<f32>, n: vec3<f32>) -> vec4<f32>{
	let size = 1. / f32(1u << getOctreeNode(p).depth);
	let up = select(vec3(0.,1,0), vec3(1.,0,0), abs(n.y) > .99);
	let b = normalize(cross(n, up));
	let t = cross(n, b);
	let offsets = array(b, -b, t, -t);
	var sum = vec3(0.);
	var count = 0.;
	for(var i = 0u; i < 4u; i++){
		let q = p + offsets[i] * size;
		if(any(q <= vec3(0.)) || any(q >= vec3(1.))){
			continue;
		}
		let node = getOctreeNode(q).node;
		if(nodeIsFilled(node) && dot(getNormal(node), n) >= FALLBACK_NORMAL_THRESHOLD){
			sum += getColor(node);
			count += 1.;
		}
	}
	return vec4(sum / max(count, 1.), count);
}

/*
A leaf coarser than the depth updateOctree.wgsl splits it to, at the distance of its farthest point.
A leaf that a g-buffer sample lands in gets split to that depth, and collapseOctree.wgsl keeps the leaves on screen at least that fine,
so such a leaf has no visible surface in it: it was collapsed while it was far away or offscreen, and nothing splits it again.
It is larger than its geometry and its lighting is stale, so the draw modes that draw whole leaves skip it.
center and halfSize of the leaf are in world space, scale is the one of updateOctree.wgsl
 */
fn isTooCoarse(depth: u32, center: vec3<f32>, halfSize: f32, cameraPosition: vec3<f32>, scale: f32) -> bool{
	let farthest = length(abs(cameraPosition - center) + halfSize);
	return i32(depth) < min(i32(floor(log2(scale / farthest))), i32(OCTREE_DEPTH_MAX));
}

//TODO: use all bits
fn getNormal(node:OctreeNode) -> vec3<f32>{
	return unpackNormal(node.normal);
}

fn unpackNormal(packed: u32) -> vec3<f32>{
	return unpack4x8snorm(packed).xyz;
}

//snorm, not unorm: unorm can't store 0, it came back as 1 / 255. The normal of a floor was (.004, 1, .004), so the discs of a lit floor
//had a small weight on the pixels of a wall facing +x or +z right above it. Where the wall had no lighting of its own,
//fillHole of octreeSplat/normalize.wgsl divided by that weight: single fully lit pixels along the corners.
//Any unit normal has a component of at least .57, so a packed normal is never 0, see nodeIsFilled
fn packNormal(n: vec3<f32>) -> u32 {
	return pack4x8snorm(vec4(n.xyz, 0.));
}

//alpha is 1, so a node with lighting never has color 0, even when it is black. See getFallbackLighting
fn packColor(c: vec3<f32>)-> u32{
	return pack4x8unorm (vec4(c / (1. + c), 1.));
}
