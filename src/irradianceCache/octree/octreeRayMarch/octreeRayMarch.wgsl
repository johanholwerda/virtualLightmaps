
struct Uniforms {
	viewProjection : mat4x4<f32>,
	cameraPosition : vec3<f32>,
	skyColor : vec3<f32>,
	exposure : f32,
	//the one of updateOctree.wgsl, see isTooCoarse
	scale : f32,
	skipCoarseLeaves : f32,
	//the levels above the leaf that getFallbackLighting may take the lighting from
	fallbackLevels : f32,
}

struct VertexInput {
  @location(0) position : vec3<f32>,
};

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  @location(0) local : vec3<f32>,
}

struct FragmentOutput {
  @location(0) color : vec4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage, read> octreeBuffer : array<OctreeNode>;
//alpha 0 is the sky
@group(0) @binding(2) var albedoTexture: texture_2d<f32>;
@group(0) @binding(3) var normalDepthTexture: texture_2d<u32>;

@vertex
fn vs_main(input: VertexInput, @builtin(instance_index) instanceIndex: u32) -> VertexOutput {
	var output : VertexOutput;

	let local = input.position + uniforms.cameraPosition;
	output.position = uniforms.viewProjection * vec4(local, 1.);

	output.local = input.position;
	return output;
}

@fragment
fn fs_main(input: VertexOutput) -> FragmentOutput {
	var output: FragmentOutput;

	//the sky has no nodes, it is emissive
	let albedo = textureLoad(albedoTexture, vec2<u32>(input.position.xy), 0);
	if(albedo.a == 0.){
		output.color = vec4(toneMap(uniforms.skyColor * uniforms.exposure), 1.);
		return output;
	}

	let normalDepth = loadNormalDepth(normalDepthTexture, vec2<u32>(input.position.xy));
	let n = normalDepth.xyz;
	//normalized (0 -> 1)
	let ro = (uniforms.cameraPosition / OCTREE_ROOT_NODE_SIZE) * .5 + .5;
	let rd = normalize(input.local);
	let tSurface = normalDepth.w / (2. * OCTREE_ROOT_NODE_SIZE);
	var lighting = rayMarchOctree(ro, rd, tSurface, n, uniforms.scale, uniforms.skipCoarseLeaves > .5);
	//no leaf of this face around the surface: the face that came second at a hard edge, or a leaf not filled yet.
	//Like fillHole of octreeResolve.wgsl, without the blend
	let surface = ro + rd * tSurface;
	if(lighting.w == 0. && all(abs(surface - .5) < vec3(.5))){
		lighting = getFallbackLighting(surface, n, u32(uniforms.fallbackLevels));
		if(lighting.w == 0.){
			lighting = getNeighbourLighting(surface, n);
		}
	}
	if(lighting.w == 0.){
		lighting = vec4(.5);
	}
	//the node lighting has no albedo, see getLighting in updateOctree.wgsl
	let c = albedo.xyz * lighting.xyz;

	output.color = vec4(toneMap(c * uniforms.exposure), 1.0);
	return output;
}
