
@group(0) @binding(0) var textureSampler: sampler;
@group(0) @binding(1) var texture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> uniforms : Uniforms;
//alpha 0 is the sky
@group(0) @binding(3) var albedoTexture: texture_2d<f32>;
@group(0) @binding(4) var normalDepthTexture: texture_2d<u32>;
@group(0) @binding(5) var<storage, read> octreeBuffer : array<OctreeNode>;

struct Uniforms {
	exposure : f32,
	showDensity : f32,
	skyColor : vec3<f32>,
	cameraPosition : vec3<f32>,
	inverseProjection : mat4x4<f32>,
	inverseView : mat4x4<f32>,
	resolution : vec2<f32>,
	//the splat weight below which the fallback lighting fades in, see fillHole
	fallbackWeight : f32,
	showHoles : f32,
	//the levels above the leaf that getFallbackLighting may take the lighting from
	fallbackLevels : f32,
	//the way fillHole lit each pixel in a color, see fs_main
	showFill : f32,
}

struct VertexInput {
	@location(0) position : vec3<f32>,
	@location(1) uv : vec2<f32>
}

struct VertexOutput {
	@builtin(position) position : vec4<f32>,
	@location(0) uv : vec2<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
	var output : VertexOutput;
	output.position = vec4(input.position, 1.);
	output.uv = input.uv;
	return output;
}

//the way fillHole lit the pixel, for showFill
const FILL_SPLAT = 0u;
const FILL_ANCESTOR = 1u;
const FILL_SPLAT_ONLY = 2u;
const FILL_NEIGHBOURS = 3u;
const FILL_CONSTANT = 4u;
var<private> fillPath = FILL_SPLAT;

//same as in octreeResolve.wgsl: pixels with little or no splat weight get the lighting of the nearest ancestor node that has it
fn fillHole(lighting: vec4<f32>, coord: vec2<u32>, albedo: vec3<f32>) -> vec3<f32>{
	let prior = uniforms.fallbackWeight - lighting.w;
	if(prior <= 0. && lighting.w > 0.){
		return albedo * lighting.xyz / lighting.w;
	}
	let uv = (vec2<f32>(coord) + .5) / uniforms.resolution;
	let normalDepth = loadNormalDepth(normalDepthTexture, coord);
	let worldPosition = depthToPosition(normalDepth.w, uv * 2. - 1.);
	let p = worldPosition / OCTREE_ROOT_NODE_SIZE;
	let inOctree = all(abs(p) < vec3(1.));
	var fallback = vec4(0.);
	fillPath = FILL_ANCESTOR;
	if(inOctree){
		fallback = getFallbackLighting(p * .5 + .5, normalDepth.xyz, u32(uniforms.fallbackLevels));
	}
	if(fallback.w == 0.){
		//no near ancestor faces this way, e.g. the face that came second at a hard edge: its lighting is in the leaves next to the
		//edge leaf. Before the little splat lighting there is: that is a few discs with a small weight, often of another surface
		//facing the same way, which the division by the weight made as bright as a fully lit pixel.
		//Not a constant: that was many times brighter than the shadows, single lit pixels along the edges
		fillPath = FILL_NEIGHBOURS;
		if(inOctree){
			fallback = getNeighbourLighting(p * .5 + .5, normalDepth.xyz);
		}
	}
	if(fallback.w == 0.){
		//nothing facing this way around: the little splat lighting there is beats lighting from another face
		if(lighting.w > 0.){
			fillPath = FILL_SPLAT_ONLY;
			return albedo * lighting.xyz / lighting.w;
		}
		//only in the first frames after a reset
		fillPath = FILL_CONSTANT;
		fallback = vec4(.5, .5, .5, 1.);
	}
	//the node lighting has no albedo, see getLighting in updateOctree.wgsl
	return albedo * (lighting.xyz + fallback.xyz * max(prior, 1e-4)) / (lighting.w + max(prior, 1e-4));
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
	let c = textureSample(texture, textureSampler, input.uv);
	let coord = vec2<u32>(input.position.xy);
	let albedo = textureLoad(albedoTexture, coord, 0);

	var col: vec3<f32>;
    //the sky has no surfels, it is emissive
    if(albedo.a == 0.){
    	col = uniforms.skyColor;
    }else{
		col = fillHole(c, coord, albedo.xyz);
    }

    col *= uniforms.exposure;

	col = toneMap(col);

	if(uniforms.showHoles > .5 && c.w == 0. && albedo.a > 0.){
		col = vec3(1, 0, 0);
	}

	//how fillHole lit the pixels: unchanged when by the splats alone, magenta a near ancestor, yellow neighbouring leaves without one,
	//cyan little splat weight without either, red the constant
	if(uniforms.showFill > .5 && albedo.a > 0. && fillPath != FILL_SPLAT){
		let fillColors = array(vec3(0.), vec3(1., 0., 1.), vec3(0., 1., 1.), vec3(1., 1., 0.), vec3(1., 0., 0.));
		col = fillColors[fillPath];
	}

	if(uniforms.showDensity > 0.5){
		col = vec3(c.w * .2);
	}

//	col = pow(col, vec3(1. / 2.2));
	return vec4(col, 1.);
}
