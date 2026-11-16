
struct Uniforms {
	cameraPosition : vec3<f32>,
	inverseProjection : mat4x4<f32>,
	inverseView : mat4x4<f32>,
	resolution : vec2<f32>,
	radius : f32,
	exposure : f32,
	skyColor : vec3<f32>,
	//the gathered weight below which the fallback lighting fades in, see fillHole
	fallbackWeight : f32,
	showHoles : f32,
	//the levels above the leaf that getFallbackLighting may take the lighting from
	fallbackLevels : f32,
}

struct VertexInput {
  @location(0) position : vec3<f32>,
  @location(1) uv : vec2<f32>,
};

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  @location(0) uv : vec2<f32>,
}

struct FragmentOutput {
  @location(0) color : vec4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage, read> octreeBuffer : array<OctreeNode>;
@group(0) @binding(2) var normalDepthTexture: texture_2d<u32>;
//alpha 0 is the sky
@group(0) @binding(3) var albedoTexture: texture_2d<f32>;

@vertex
fn vs_main(input: VertexInput, @builtin(instance_index) instanceIndex: u32) -> VertexOutput {
	var output : VertexOutput;

	output.position = vec4(input.position, 1.0);
	output.uv = input.uv;
	return output;
}

//descends from a node to the leaf node containing p, no deeper than depthMax. Reads only the child indices, the node itself is read once at the end
fn descend(p: vec3<f32>, startIndex: u32, startDepth: u32, depthMax: u32) -> vec2<u32> {
	var index = startIndex;
	var depth = startDepth;
	for(; depth < depthMax; depth++){
		let firstChildIndex = octreeBuffer[index].firstChildIndex;
		if(firstChildIndex == 0u){
			break;
		}
		let c = vec3<u32>(p * f32(1u << (depth + 1u))) & vec3(1u);
		index = firstChildIndex + (c.x | (c.y << 1u) | (c.z << 2u));
	}
	return vec2(index, depth);
}

/*
gathers the leaf nodes around the pixel with the weights of octreeSplat, so it is stable in world space and has the resolution of the nodes.
The 2 * 2 * 2 nodes nearest to the pixel at the size of the leaf node it is in, like the corners of a trilinear lookup.
A node leaves that set when the pixel is a node size away along an axis, so a radius of 1 node size fades it out before that.
p is in octree space (0 -> 1). Returns the lighting sum and the weight sum
 */
fn gatherLighting(p: vec3<f32>, n: vec3<f32>) -> vec4<f32> {
	//the leaf node of the pixel, remembering the last 3 ancestors
	var index = 0u;
	var depth = 0u;
	var parent1 = 0u;
	var parent2 = 0u;
	var parent3 = 0u;
	for(; depth < OCTREE_DEPTH_MAX; depth++){
		let firstChildIndex = octreeBuffer[index].firstChildIndex;
		if(firstChildIndex == 0u){
			break;
		}
		parent3 = parent2;
		parent2 = parent1;
		parent1 = index;
		let c = vec3<u32>(p * f32(1u << (depth + 1u))) & vec3(1u);
		index = firstChildIndex + (c.x | (c.y << 1u) | (c.z << 2u));
	}

	let cellCount = i32(1u << depth);
	let nodeSize = 1. / f32(cellCount);
	//the cell of corner 0 at the depth of the leaf node
	let base = vec3<i32>(floor(p / nodeSize - .5));
	let ownCell = vec3<i32>(p * f32(cellCount));

	//the corners share the path from the root down to the smallest node containing all of them, often only 1 or 2 levels up.
	//Its depth is where the cell coordinates of the 2 corners of each axis become the same
	var ancestorIndex = 0u;
	var ancestorDepth = 0u;
	if(all(base >= vec3(0)) && all(base + 1 < vec3(cellCount))){
		let differentBits = vec3<u32>(base ^ (base + 1));
		let levelsUp = firstLeadingBit(max(differentBits.x, max(differentBits.y, differentBits.z))) + 1u;
		ancestorDepth = depth - levelsUp;
		if(levelsUp == 1u){
			ancestorIndex = parent1;
		}else if(levelsUp == 2u){
			ancestorIndex = parent2;
		}else if(levelsUp == 3u){
			ancestorIndex = parent3;
		}else{
			ancestorIndex = descend(p, 0u, 0u, ancestorDepth).x;
		}
	}

	var lightingSum = vec3(0.0);
	var weightSum = 0.0;
	for(var i = 0u; i < 8u; i++){
		let offset = vec3(i & 1u, (i >> 1u) & 1u, i >> 2u);
		let samplePosition = (vec3<f32>(base + vec3<i32>(offset)) + .5) * nodeSize;
		if(any(abs(samplePosition - .5) >= vec3(.5))){
			continue;
		}
		//the corner of the cell of the pixel is the leaf node found above
		var result = vec2(index, depth);
		if(any(base + vec3<i32>(offset) != ownCell)){
			result = descend(samplePosition, ancestorIndex, ancestorDepth, OCTREE_DEPTH_MAX);
		}

		//a node coarser than the leaf of the pixel can contain more than one corner. Only the corner with the lowest index uses it:
		//this one is a duplicate when the corner below it on an axis (offset 0 instead of 1) is in the same node
		if(result.y < depth){
			let shift = vec3(depth - result.y);
			let sameNode = (base >> shift) == ((base + 1) >> shift);
			if(any(sameNode & (offset == vec3(1u)))){
				continue;
			}
		}

		let node = octreeBuffer[result.x];
		if(!nodeIsFilled(node)){
			continue;
		}
		let ps = getPositionAndSize(node);
		var weight = saturate(dot(n, normalize(getNormal(node))));
		weight *= smoothstep(ps.w * uniforms.radius, 0., distance(p, ps.xyz));
		weightSum += weight;
		lightingSum += getColor(node) * weight;
	}
	return vec4(lightingSum, weightSum);
}

/*
Pixels with little or no weight get the lighting of the nearest ancestor node that has it for a surface facing the same way,
faded in below fallbackWeight so there is no seam.
It is the lighting of the parent of the empty node, so it is stable in world space and close to what the children get.
Only up to fallbackLevels above the leaf, see getFallbackLighting. Without one (at a hard edge whose ancestors were lit from
the other face) the leaves next to it on the same face, then the gathered lighting alone,
and without that, which is only the case for a few frames after a reset, the albedo with a dim light
 */
fn fillHole(lighting: vec4<f32>, p: vec3<f32>, n: vec3<f32>, albedo: vec3<f32>) -> vec3<f32>{
	let prior = uniforms.fallbackWeight - lighting.w;
	if(prior <= 0. && lighting.w > 0.){
		return albedo * lighting.xyz / lighting.w;
	}
	var fallback = getFallbackLighting(p, n, u32(uniforms.fallbackLevels));
	if(fallback.w == 0.){
		//no near ancestor faces this way: the leaves next to it on the same face, see fillHole of octreeSplat/normalize.wgsl
		fallback = getNeighbourLighting(p, n);
	}
	if(fallback.w == 0.){
		//nothing facing this way around: the little gathered lighting there is beats lighting from another face
		if(lighting.w > 0.){
			return albedo * lighting.xyz / lighting.w;
		}
		fallback = vec4(.5, .5, .5, 1.);
	}
	//the node lighting has no albedo, see getLighting in updateOctree.wgsl
	return albedo * (lighting.xyz + fallback.xyz * max(prior, 1e-4)) / (lighting.w + max(prior, 1e-4));
}

@fragment
fn fs_main(input: VertexOutput) -> FragmentOutput {
	var output: FragmentOutput;

	let coord = vec2<u32>(input.uv * uniforms.resolution);

	//the sky has no nodes, it is emissive
	let albedo = textureLoad(albedoTexture, coord, 0);
	if(albedo.a == 0.){
		output.color = vec4(toneMap(uniforms.skyColor * uniforms.exposure), 1.);
		return output;
	}

	let normalDepth = loadNormalDepth(normalDepthTexture, coord);
	let uv = (vec2<f32>(coord) + .5) / uniforms.resolution;
	let worldPosition = depthToPosition(normalDepth.w, uv * 2. - 1.);
	//to local space
	var p = worldPosition / OCTREE_ROOT_NODE_SIZE;
	if(any(abs(p) >= vec3(1.))){
		discard;
	}
	p = p * .5 + .5;

	let lighting = gatherLighting(p, normalDepth.xyz);

	var c = toneMap(fillHole(lighting, p, normalDepth.xyz, albedo.xyz) * uniforms.exposure);
	if(uniforms.showHoles > .5 && lighting.w == 0.){
		c = vec3(1., 0., 0.);
	}

	output.color = vec4(c, 1.0);
	return output;
}
