
/*
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
			n.node = octreeBuffer[n.node.firstChildIndex + localIndex];
			n.depth = i;
		}else{
			//is a leaf node
			break;
		}
	}
	return n;
}
*/

/*
The lighting of the first filled leaf along the camera ray around the surface it hits (from the g-buffer) that faces like the surface.
Only near it, from 2 leaf sizes in front of the surface to 1 behind it: in front are the leaves of other surfaces the ray passes
close to, behind it the ones of the far side of the object.
Leaves facing another way are skipped: at an edge the leaves of the other face stick out of this one, a staircase of their lighting,
and at the corner of a shadowed and a lit face the lit leaves showed as specks on the shadowed one.
With skipCoarseLeaves the ray passes through the leaves that are too coarse for their distance, see isTooCoarse.
ro is the camera and tSurface the distance to the surface, in octree space (0 -> 1). n is the normal of the surface.
Returns the lighting, alpha 0 when no leaf matched
 */
fn rayMarchOctree(ro: vec3<f32>, rd: vec3<f32>, tSurface: f32, n: vec3<f32>, scale: f32, skipCoarseLeaves: bool) -> vec4<f32> {
	let cameraPosition = (ro * 2. - 1.) * OCTREE_ROOT_NODE_SIZE;
	let maxSteps = 100u;
	//the side each axis travels to. A component of 0 counts as positive, and is made tiny instead of 0: 0 * inf was NaN
	//when the ray was exactly on a node boundary, which ended the march (a horizontal ray at a camera height like 1.5)
	let travel = select(vec3(-1.), vec3(1.), rd >= vec3(0.));
	let invRD = vec3(1.) / select(travel * 1e-12, rd, abs(rd) >= vec3(1e-12));
	//the nodes are looked up this far towards where the ray goes. A ray exactly on a node boundary, going to the node below it,
	//was put in the node above it by floor: its exit was where it already was, and the march stalled there without reaching the surface
	let bias = travel * 1e-6;

	let surface = ro + rd * tSurface;
	if(any(abs(surface - .5) >= vec3(.5))){
		return vec4(0.);
	}
	let surfaceLeafSize = 1. / f32(1u << getOctreeNode(surface).depth);

	//where the ray enters the octree, 0 when the camera is inside it
	let t1 = min(-ro * invRD, (vec3(1.) - ro) * invRD);
	let tEnter = max(max(max(t1.x, t1.y), t1.z), 0.);

	var t = max(tSurface - 2. * surfaceLeafSize, tEnter);
	let tEnd = tSurface + surfaceLeafSize;
	for (var i = 0u; i < maxSteps && t <= tEnd; i++) {
		let p = ro + rd * t;
		if (any(abs(p - .5) >= vec3(.5))) {
			break;
		}

		let q = clamp(p + bias, vec3(0.), vec3(1. - 1e-7));
		let nodeResult = getOctreeNode(q);
		let nodeCount = f32(1u << nodeResult.depth);
		let nodeSize = 1.0 / nodeCount;
		let nodePos = floor(q * nodeCount) / nodeCount;

		if (nodeIsFilled(nodeResult.node) && dot(getNormal(nodeResult.node), n) >= FALLBACK_NORMAL_THRESHOLD) {
			let center = ((nodePos + nodeSize * .5) * 2. - 1.) * OCTREE_ROOT_NODE_SIZE;
			if (!skipCoarseLeaves || !isTooCoarse(nodeResult.depth, center, nodeSize * OCTREE_ROOT_NODE_SIZE, cameraPosition, scale)) {
				return vec4(getColor(nodeResult.node), 1.);
			}
		}

		//to just past the side where the ray leaves the node. A small fraction of the node: a fixed 1e-4 was 40% of the
		//smallest leaves, the ray skipped the corners of nodes
		//the faces on the side the ray travels to
		let exitFace = nodePos + select(vec3(0.), vec3(nodeSize), travel > vec3(0.));
		let t2 = max((exitFace - p) * invRD, vec3(0.));
		t += min(min(t2.x, t2.y), t2.z) + nodeSize * 1e-3;
	}

	return vec4(0.);
}
