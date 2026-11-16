struct Uniforms {
	viewProjection : mat4x4<f32>,
	resolution : vec2<f32>,
	cameraPosition  : vec3<f32>,
	inverseProjection : mat4x4<f32>,
	inverseView : mat4x4<f32>,
	pointSize : f32,
	aspectRatio : f32,
	solidDisc : f32,
	//the one of updateOctree.wgsl, see isTooCoarse
	scale : f32,
	skipCoarseLeaves : f32,
	//the distance from the plane of a disc at which its weight is 0, as a fraction of the disc radius
	planeDistance : f32,
}

struct VertexInput {
  @location(0) position : vec3<f32>,
  @location(1) uv : vec2<f32>,
};

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  @location(0) uv : vec2<f32>,
  @location(1) normal : vec3<f32>,
  @location(2) color : vec3<f32>,
  @location(3) center : vec3<f32>,
  @location(4) size : f32,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var normalDepthTexture: texture_2d<u32>;
@group(0) @binding(2) var<storage, read> octreeBuffer : array<OctreeNode>;

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex : u32) -> VertexOutput {

	let pos = array(
      vec2( -1., -1.),
      vec2(1., -1.),
      vec2(-1., 1.),
      vec2(1., 1.),
    );

	let node = octreeBuffer[vertexIndex / 4];
	var out : VertexOutput;

	let ps = getPositionAndSize(node);
	var visible = node.firstChildIndex == 0u && nodeIsFilled(node);
	if(visible && uniforms.skipCoarseLeaves > .5){
		//the box of the leaf, ps.xyz is the position of its sample
		let depth = u32(round(-log2(ps.w)));
		let nodeCount = f32(1u << depth);
		let boxCenter = ((floor(ps.xyz * nodeCount) + .5) / nodeCount * 2. - 1.) * OCTREE_ROOT_NODE_SIZE;
		visible = !isTooCoarse(depth, boxCenter, ps.w * OCTREE_ROOT_NODE_SIZE, uniforms.cameraPosition, uniforms.scale);
	}

	if(visible){
		let uv = pos[vertexIndex % 4];

		let center = (ps.xyz * 2. - 1.) * OCTREE_ROOT_NODE_SIZE;

		let n = getNormal(node);

		let up = select(vec3(0.,1,0), vec3(1.,0,0), abs(n.y) > .99);
		let b = normalize(cross(n, up));
		let t = normalize(cross(n, b));

		out.size = ps.w * uniforms.pointSize * OCTREE_ROOT_NODE_SIZE;

		let world = center + (b * uv.x + t * uv.y) * out.size;

		out.position = uniforms.viewProjection * vec4(world, 1.);
//		out.position = uniforms.viewProjection * vec4(center, 1.);

//		let offset = uv * ps.w * uniforms.pointSize / out.position.w;
//
//		out.position /= out.position.w;
//		out.position.x += offset.x;
//		out.position.y += offset.y * uniforms.aspectRatio;
		out.color = getColor(node);
		out.normal = getNormal(node);
		out.center = center;
		out.uv = uv;
	}
	return out;
}

fn getWeight(p: vec3<f32>, n: vec3<f32>, surfelP: vec3<f32>, surfelN: vec3<f32>, depth: f32, size:f32)-> f32{
	var weight = saturate(dot(n, surfelN));
	let d = distance(p, surfelP);
//	weight /= max(d * 100., 1.);
	weight *= smoothstep(size, 0.,d);
	//only the surface the disc lies on: a parallel surface just in front of or behind it (another step of a molding) has other lighting
	weight *= smoothstep(max(size * uniforms.planeDistance, 1e-6), 0., abs(dot(p - surfelP, surfelN)));
	return weight;
}

@fragment
fn fs_main(in : VertexOutput) -> @location(0) vec4<f32> {
	let d = dot(in.uv, in.uv);
	if(d > 1.){
		discard;
	}
	let uv = in.position.xy / uniforms.resolution;
	let coord = vec2<u32>(in.position.xy);
	let nd = loadNormalDepth(normalDepthTexture, coord);
	let n = nd.xyz;
	let p = depthToPosition(nd.w, uv * 2. - 1.);
	let c = in.color.rgb;
	var a = getWeight(p, n, in.center, in.normal, nd.w, in.size);

	if(uniforms.solidDisc > .5){
		a = 1.;
	}
	return vec4(c * a, a);
}

