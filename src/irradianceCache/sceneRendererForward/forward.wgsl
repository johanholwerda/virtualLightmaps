struct Uniforms {
	viewProjection : mat4x4<f32>,
	lightDirection : vec3<f32>,
	//angular radius of the sun, see sampleSunDirection
	sunRadius : f32,
	cameraPosition : vec3<f32>,
	directionalColor : vec3<f32>,
	modelCount : f32,
	//the first models are static and in the BVH, see traceRay
	bvhModelCount : f32,
	//the first instance of the primitive type in modelIndicesByType
	instanceOffset : f32,
	bounceCount : i32,
	frame : f32,
	directionalBrightness : f32,
	skyBrightness : f32,
	directionalProbability : f32,
//	world : mat4x4<f32>,
//	color:vec3<f32>,
}

struct VertexInput {
  @location(0) position : vec3<f32>,
  @location(1) uv : vec2<f32>,
  @location(2) normal : vec3<f32>,
};

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  @location(0) uv : vec2<f32>,
  @location(1) normal : vec3<f32>,
  @location(2) world : vec3<f32>,
  @location(3) color : vec3<f32>,
  @location(4) @interpolate(flat) modelIndex : u32,
}

struct FragmentOutput {
  @location(0) color : vec4<f32>,
  //see includes/gBuffer.wgsl
  @location(1) normalDepth : vec2<u32>,
  //the index in sceneBuffer and the triangle of a mesh, packed by packModelTriangle (see includes/gBuffer.wgsl), for the
  //virtual lightmap: its texels are on the faces of the models and on the charts of the meshes, see getSurface in virtualLightmap.wgsl
  @location(2) modelTriangle : u32,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage, read> sceneBuffer : array<Model>;
@group(0) @binding(2) var<storage, read> bvhBuffer : array<BVHNode>;
@group(0) @binding(3) var<storage, read> modelIndicesByType : array<u32>;

@vertex
fn vs_main(input: VertexInput, @builtin(instance_index) instanceIndex : u32) -> VertexOutput {
  var output : VertexOutput;
  let modelIndex = modelIndicesByType[u32(uniforms.instanceOffset) + instanceIndex];
  let model = sceneBuffer[modelIndex];
  let world = model.worldMatrix * vec4(input.position, 1.);
//  let world = uniforms.world * vec4(input.position, 1.);
  output.position = uniforms.viewProjection * world;
  output.uv = input.uv;
  output.normal = rotateQuaternion(model.quaternion, input.normal);
  output.world = world.xyz;
  output.color = model.color;
  output.modelIndex = modelIndex;
  return output;
}

@fragment
fn fs_main(input: VertexOutput) -> FragmentOutput {
	var output: FragmentOutput;
	let n = normalize(input.normal);
	//gets analytical world position to avoid artifacts caused by position interpolation of triangles
	let rd = normalize(input.world - uniforms.cameraPosition);
	let ro = uniforms.cameraPosition;
	let i = input.modelIndex;

	//hit is retrieved analytically to prevent a mismatch between triangles and analytical primitives
	let hit = getModelHit(ro, rd, i, MAX_DIST);
	let p = ro + rd * hit.w;

//	let c = getColor(ro, rd, input.color, p, n);

//	let c = sampleRay(ro, rd, u32(uniforms.frame));
	let worldNormal = normalize(rotateQuaternion(sceneBuffer[i].quaternion, hit.xyz));

	//alpha 0 marks the sky, which the octree doesn't store
	output.color = vec4(input.color, select(1., 0., i == u32(uniforms.modelCount) - 1u));
	output.normalDepth = packNormalDepth(worldNormal, hit.w);
	output.modelTriangle = packModelTriangle(i, 0u);
	return output;
}
