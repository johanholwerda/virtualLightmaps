@group(0) @binding(0) var textureSampler: sampler;
//the running average of the light arriving at the pixels, without their albedo (rgb) and its sample count (a), see fill.wgsl
@group(0) @binding(1) var texture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> uniforms : Uniforms;
//alpha 0 is the sky
@group(0) @binding(3) var albedoTexture: texture_2d<f32>;
@group(0) @binding(4) var normalDepthTexture: texture_2d<u32>;

struct Uniforms {
	exposure : f32,
	skyColor : vec3<f32>,
	//the sample count at which the denoise blur has faded out
	denoiseSamples : f32,
	//the distance of a neighbour to the plane of the pixel, relative to the pixel's depth, up to which it counts as the
	//same surface, see denoise
	depthTolerance : f32,
	showSampleCount : f32,
	//the sample count that shows white
	sampleCountMax : f32,
	//for the positions of the pixels, see cameraRay.wgsl
	inverseProjection : mat4x4<f32>,
	inverseView : mat4x4<f32>,
	cameraPosition : vec3<f32>,
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

//pixels with few samples get the lighting of the pixels around them on the same surface: a fast camera rotation
//disoccludes large areas that have only a few samples (see fill.wgsl). The neighbours are weighted by their sample
//count and only make up the samples the pixel misses to denoiseSamples, the inner rings first: the blur shrinks and
//fades out gradually while the pixel accumulates samples, and stays small next to pixels that have many
fn denoise(coord: vec2<u32>, lighting: vec4<f32>) -> vec3<f32>{
	var budget = uniforms.denoiseSamples - lighting.a;
	if(budget <= 0.){
		return lighting.rgb;
	}
	let normalDepth = loadNormalDepth(normalDepthTexture, coord);
	let size = vec2<i32>(textureDimensions(texture));
	let p = depthToPosition(normalDepth.w, (vec2<f32>(coord) + .5) / vec2<f32>(size) * 2. - 1.);
	let planeTolerance = normalDepth.w * uniforms.depthTolerance;
	//interleaved gradient noise rotates the taps per pixel, a fixed pattern shows
	let angle = fract(52.9829189 * fract(.06711056 * f32(coord.x) + .00583715 * f32(coord.y))) * 6.2831853;
	var sum = vec4(lighting.rgb * lighting.a, lighting.a);
	var radius = 1.5;
	for(var ring = 0; ring < 5; ring++){
		var ringSum = vec4(0.);
		for(var i = 0; i < 8; i++){
			let a = angle + f32(i) * .7853982;
			let c = vec2<i32>(coord) + vec2<i32>(round(vec2(cos(a), sin(a)) * radius));
			if(any(c < vec2(0)) || any(c >= size)){
				continue;
			}
			let s = textureLoad(texture, c, 0);
			if(s.a == 0.){
				continue;
			}
			let neighbour = loadNormalDepth(normalDepthTexture, vec2<u32>(c));
			let normalWeight = pow(saturate(dot(normalDepth.xyz, neighbour.xyz)), 8.);
			//the distance to the plane of the pixel instead of the depth difference: on a surface seen at a grazing
			//angle the depth changes fast across the screen, the outer rings were rejected where the noise is worst
			let pNeighbour = depthToPosition(neighbour.w, (vec2<f32>(c) + .5) / vec2<f32>(size) * 2. - 1.);
			let depthWeight = select(0., 1., abs(dot(normalDepth.xyz, pNeighbour - p)) < planeTolerance);
			let weight = s.a * normalWeight * depthWeight;
			ringSum += vec4(s.rgb, 1.) * weight;
		}
		if(ringSum.w >= budget){
			sum += ringSum * (budget / ringSum.w);
			break;
		}
		sum += ringSum;
		budget -= ringSum.w;
		radius *= 2.;
	}
	if(sum.w == 0.){
		return vec3(.5);
	}
	return sum.rgb / sum.w;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
	let coord = vec2<u32>(input.position.xy);
	let lighting = textureLoad(texture, coord, 0);
	let albedo = textureLoad(albedoTexture, coord, 0);

	var col: vec3<f32>;
	//the sky is emissive, it has no lighting
	if(albedo.a == 0.){
		col = uniforms.skyColor;
	}else{
		col = albedo.xyz * denoise(coord, lighting);
	}

	col *= uniforms.exposure;

	col = toneMap(col);

	if(uniforms.showSampleCount > .5){
		col = vec3(lighting.a / uniforms.sampleCountMax);
	}

	return vec4(col, 1.);
}
