//the MESH models into the gBuffer, like forward.wgsl. No vertex buffer: the vertices are pulled from meshBuffer, the
//triangles of the ray tracer (see getMeshVertex in rayTraceScene.wgsl), so the drawn and the traced mesh are the same
//and the triangle index goes in the gBuffer for the virtual lightmap
struct Uniforms {
	viewProjection : mat4x4<f32>,
	lightDirection : vec3<f32>,
	cameraPosition : vec3<f32>,
	modelCount : f32,
	bvhModelCount : f32,
	//the first MESH instance in modelIndicesByType
	instanceOffset : f32,
}

struct VertexOutput {
	@builtin(position) position : vec4<f32>,
	@location(0) normal : vec3<f32>,
	@location(1) world : vec3<f32>,
	@location(2) @interpolate(flat) modelIndex : u32,
	@location(3) @interpolate(flat) triangle : u32,
}

struct FragmentOutput {
	@location(0) color : vec4<f32>,
	//see includes/gBuffer.wgsl
	@location(1) normalDepth : vec2<u32>,
	//the model and the triangle, see forward.wgsl
	@location(2) modelTriangle : u32,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage, read> sceneBuffer : array<Model>;
@group(0) @binding(2) var<storage, read> bvhBuffer : array<BVHNode>;
@group(0) @binding(3) var<storage, read> modelIndicesByType : array<u32>;

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex : u32, @builtin(instance_index) instanceIndex : u32) -> VertexOutput {
	var output : VertexOutput;
	let modelIndex = modelIndicesByType[u32(uniforms.instanceOffset) + instanceIndex];
	let model = sceneBuffer[modelIndex];
	let triangle = vertexIndex / 3u;
	let vertex = getMeshVertex(triangle, vertexIndex % 3u);
	let world = model.worldMatrix * vec4(vertex.xyz, 1.);
	output.position = uniforms.viewProjection * world;
	output.normal = rotateQuaternion(model.quaternion, meshDecodeNormal(vertex.w));
	output.world = world.xyz;
	output.modelIndex = modelIndex;
	output.triangle = triangle;
	return output;
}

@fragment
fn fs_main(input: VertexOutput) -> FragmentOutput {
	var output: FragmentOutput;
	//drawn with back face culling. The normal is still put on the side the camera is on, like traceMesh:
	//from the geometric normal, the smooth one can face away near the silhouette
	let t = getMeshTriangle(input.triangle);
	let geometric = rotateQuaternion(sceneBuffer[input.modelIndex].quaternion, cross(t.v1 - t.v0, t.v2 - t.v0));
	let n = normalize(input.normal) * select(1., -1., dot(geometric, input.world - uniforms.cameraPosition) > 0.);
	output.color = vec4(sceneBuffer[input.modelIndex].color, 1.);
	//the distance along the camera ray, like the analytic hits of forward.wgsl
	output.normalDepth = packNormalDepth(n, distance(input.world, uniforms.cameraPosition));
	output.modelTriangle = packModelTriangle(input.modelIndex, input.triangle);
	return output;
}
