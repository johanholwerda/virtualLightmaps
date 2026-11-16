struct Uniforms {
	viewProjection : mat4x4<f32>,
	cameraPosition : vec3<f32>,
	//multiplies the HDR lighting before the reinhard compression, so the nodes in the shadow show
	brightness : f32,
	//hides the lines behind the scene surfaces of the g-buffer
	depthTest : f32,
	//the distance behind the surface, as a fraction of its distance, at which a line is hidden
	depthBias : f32,
}

struct VertexOutput {
	@builtin(position) position : vec4<f32>,
	@location(0) color : vec3<f32>,
	@location(1) world : vec3<f32>,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var normalDepthTexture: texture_2d<u32>;
@group(0) @binding(2) var<storage, read> octreeBuffer : array<OctreeNode>;
//see octreeDebugInstances.wgsl
@group(0) @binding(3) var<storage, read> instances : array<vec2<u32>>;

const ROOT_PARENT = 0xffffffffu;

//the corner of edge vertexIndex / 2 of a box, line-list: 4 edges along each axis. -1 -> 1
fn getCorner(vertexIndex: u32) -> vec3<f32>{
	let edge = vertexIndex / 2u;
	let end = f32(vertexIndex & 1u);
	let axis = edge / 4u;
	let a = f32(edge & 1u);
	let b = f32((edge >> 1u) & 1u);
	var c = vec3(end, a, b);
	if(axis == 1u){
		c = vec3(a, end, b);
	}else if(axis == 2u){
		c = vec3(a, b, end);
	}
	return c * 2. - 1.;
}

@vertex
fn vs_main(
	@builtin(vertex_index) vertexIndex : u32,
	@builtin(instance_index) instanceIndex : u32
) -> VertexOutput {
	let instance = instances[instanceIndex];
	let node = octreeBuffer[instance.x];

	//the box in octree space (0 -> 1). The position of a leaf is the one of its sample (and empty leaves have none),
	//the one of an interior node is its center, so the box comes from the parent
	var center = vec3(.5);
	var halfSize = .5;
	if(instance.y != ROOT_PARENT){
		let parent = octreeBuffer[instance.y];
		let ps = getPositionAndSize(parent);
		let depth = u32(round(-log2(ps.w)));
		let childIndex = instance.x - parent.firstChildIndex;
		let childCount = f32(2u << depth);
		let childCell = floor(ps.xyz * f32(1u << depth)) * 2. + vec3(f32(childIndex & 1u), f32((childIndex >> 1u) & 1u), f32((childIndex >> 2u) & 1u));
		center = (childCell + .5) / childCount;
		halfSize = .5 / childCount;
	}

	let world = (center * 2. - 1. + getCorner(vertexIndex) * halfSize * 2.) * OCTREE_ROOT_NODE_SIZE;

	var out : VertexOutput;
	out.position = uniforms.viewProjection * vec4(world, 1.);
	out.world = world;
	//red: the node has no lighting yet
	out.color = vec3(1., 0., 0.);
	if(node.color != 0u){
		let c = getColor(node) * uniforms.brightness;
		out.color = c / (1. + c);
	}
	return out;
}

@fragment
fn fs_main(in : VertexOutput) -> @location(0) vec4<f32> {
	if(uniforms.depthTest > .5){
		let sceneDistance = loadNormalDepth(normalDepthTexture, vec2<u32>(in.position.xy)).w;
		if(distance(in.world, uniforms.cameraPosition) > sceneDistance * (1. + uniforms.depthBias)){
			discard;
		}
	}
	return vec4(in.color, 1.);
}
