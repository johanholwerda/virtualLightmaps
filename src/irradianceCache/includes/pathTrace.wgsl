//diffuse path tracing on top of rayTraceScene.wgsl, needs webGPU-framework/shaders/hash.wgsl.
//The including shader defines the radiance cache that the paths end in:
//fn getCachedLighting(p: vec3<f32>, n: vec3<f32>) -> vec4<f32>
//the light arriving at a hit point (rgb) when the cache has it there (a > 0). Without a cache it returns vec4(0.)
//Bind blueNoiseTexture with Scene.bindPathTrace

const TAU : f32 = 6.28318530718;

//the random numbers of getLighting. false: white noise, the hash, to compare with
override BLUE_NOISE: bool = true;

//a tiling 2 channel blue noise texture (BlueNoiseTexture2D16Bits), see getBlueNoise. Declared here instead of in the
//including shaders, at a binding they leave free
@group(0) @binding(15) var blueNoiseTexture: texture_2d<f32>;

//2 uniform random numbers, for sample index of the pixel at coord. Over the indices a pixel walks the R2 sequence, so the
//samples it accumulates cover the domain evenly instead of clumping. The sequence starts at the blue noise of the pixel, so
//neighbouring pixels are far apart in it and their errors differ: a blur averages that out (see denoise in resolve.wgsl)
//and what remains is fine grained instead of blotchy. The 2 numbers are 1 texel of a 2 channel blue noise, so the pairs
//of neighbouring pixels are far apart as 2d points: 2 reads of a 1 channel blue noise are blue per number only.
//dimension: another start, for another pair of the same sample: the texture at another offset, so they are not correlated
fn getBlueNoise(coord: vec2<u32>, index: u32, dimension: u32) -> vec2<f32> {
	let size = textureDimensions(blueNoiseTexture);
	let offset = vec2<u32>(getR2(dimension + 1u) * vec2<f32>(size));
	let start = textureLoad(blueNoiseTexture, (coord + offset) % size, 0).xy;
	return fract(start + getR2(index));
}

//the number of the sample the seed takes now, from which the index of getBlueNoise follows: its sampleCount paths are
//sampleNumber * sampleCount + i. -1: the frame, which is that number when the seed gets a sample every frame. A seed that
//gets one in some frames only (a texel of updateVirtualLightmap.wgsl) sets the samples it took instead, so it walks its
//R2 sequence without gaps
var<private> sampleNumber: i32 = -1;

fn getSampleNumber() -> u32 {
	return select(u32(uniforms.frame), u32(sampleNumber), sampleNumber >= 0);
}

//a direction to the sun disc, which has an angular radius of uniforms.sunRadius (radians). r: uniform random numbers
fn sampleSunDirection(r: vec2<f32>) -> vec3<f32> {
	let l = uniforms.lightDirection;
	let cosTheta = 1. - r.x * (1. - cos(uniforms.sunRadius));
	let sinTheta = sqrt(max(1. - cosTheta * cosTheta, 0.));
	let phi = TAU * r.y;
	let t = normalize(cross(l, select(vec3(0., 1., 0.), vec3(1., 0., 0.), abs(l.y) > .5)));
	let b = cross(l, t);
	return normalize((t * cos(phi) + b * sin(phi)) * sinTheta + l * cosTheta);
}

//towards a random point on the sun disc, so the shadows get a penumbra when averaged
fn getShadow(p:vec3<f32>, n: vec3<f32>, r: vec2<f32>) -> f32 {
	let l = sampleSunDirection(r);
	return select(max(dot(n, l), 0.), 0., isOccluded(p, l));
}

fn cosWeightedRandomHemisphereDirection(n: vec3<f32>, r:vec2<f32>) -> vec3<f32> {
    let uu = normalize(cross(n, select(vec3<f32>(0.0, 1.0, 0.0),vec3<f32>(1.0, 0.0, 0.0),  abs(n.y) > 0.5)));
    let vv = cross(uu, n);
    let ra = sqrt(r.y);
    let rx = ra * cos(TAU * r.x);
    let ry = ra * sin(TAU * r.x);
    let rz = sqrt(1.0 - r.y);
    let rr = vec3<f32>(rx * uu + ry * vv + rz * n);
    return normalize(rr);
}

//the light arriving along rd0 at ro0, only diffuse. Includes the primary ray.
//A path ends at the first hit that getCachedLighting has lighting for: the light leaving the hit point is its albedo times
//that lighting. The cache is filled from these paths, so the further bounces add up over the frames, without a bounce limit.
//Keep this the only traceRay call site of a shader: a second one (a cached copy of this loop) made updateOctree 1.75x slower
fn sampleRay(ro0: vec3<f32>, rd0: vec3<f32>, rayIndex: u32) -> vec3<f32> {
	var col = vec3(1.0);
	var ro = ro0;
	var rd = rd0;

	for (var i = 0; i < uniforms.bounceCount; i++) {
		let hit = traceRay(ro + rd * MIN_DIST, rd);
		let hitPoint = ro + rd * hit.distance;
		if (isSky(hitPoint)) {
			return col * hit.color;
		}
		ro = hitPoint;
		let n = hit.normal;
		col *= hit.color;

		let cached = getCachedLighting(hitPoint, n);
		if(cached.a > 0.){
			return col * cached.rgb;
		}

		let r = hash43ff(vec4(ro, f32(rayIndex)));
		if(r.z < uniforms.directionalProbability){
			let directional = getShadow(ro + n * MIN_DIST, n, r.xy);
			return col * directional * uniforms.directionalColor * uniforms.directionalBrightness;
		}
		rd = cosWeightedRandomHemisphereDirection(n, r.xy);
	}
	return vec3(0.);
}

//the light arriving at p, without the albedo of p: radiance leaving p towards the camera is albedo * getLighting.
//The albedo is applied per pixel when resolving, so lighting that leaks onto a neighbouring surface with another albedo
//(e.g. at an edge) only leaks its lighting, not its color.
//Uses the same direct / indirect split as sampleRay does per bounce, so it converges to the path traced reference.
//seed: the pixel the samples are for, its blue noise (see getBlueNoise). Only the first bounce and the sun sample use it,
//the further bounces of sampleRay stay white noise: the more dimensions, the less the blue noise helps
fn getLighting(p: vec3<f32>, n: vec3<f32>, seed: vec2<u32>, sampleCount: u32) -> vec3<f32>{
	let indirect = getIndirectLighting(p, n, seed, sampleCount);
	return combineLighting(indirect, getSeededDirectLighting(p, n, seed));
}

//the indirect and the direct part of getLighting together
fn combineLighting(indirect: vec3<f32>, direct: vec3<f32>) -> vec3<f32>{
	return mix(indirect, direct, uniforms.directionalProbability);
}

//the sun sample of getLighting, with the random numbers of the seed
fn getSeededDirectLighting(p: vec3<f32>, n: vec3<f32>, seed: vec2<u32>) -> vec3<f32>{
	return getSeededDirectSample(p, n, seed, 0u, 1u);
}

//sun sample i of sampleCount per frame, each to another point on the sun disc: their average is the direct lighting.
//On its own, so the samples of a point can be spread over threads, like getIndirectPath
fn getSeededDirectSample(p: vec3<f32>, n: vec3<f32>, seed: vec2<u32>, i: u32, sampleCount: u32) -> vec3<f32>{
	let index = getSampleNumber() * sampleCount + i;
	//~index: a seed that differs from the ones of the indirect samples
	var rSun = hash32uf(vec3<u32>(seed, ~index));
	if(BLUE_NOISE){
		rSun = getBlueNoise(seed, index, 1u);
	}
	return getDirectLighting(p, n, rSun);
}

//the paths of getLighting, the light that arrives at p after a bounce
fn getIndirectLighting(p: vec3<f32>, n: vec3<f32>, seed: vec2<u32>, sampleCount: u32) -> vec3<f32>{
	var indirect = vec3(0.);
	for(var i = 0u; i < sampleCount; i++){
		indirect += getIndirectPath(p, n, seed, i, sampleCount);
	}
	return indirect / f32(sampleCount);
}

//path i of the sampleCount paths of getIndirectLighting. On its own, so the paths of a point can be spread over threads
fn getIndirectPath(p: vec3<f32>, n: vec3<f32>, seed: vec2<u32>, i: u32, sampleCount: u32) -> vec3<f32>{
	return sampleRay(p + n * MIN_DIST, getIndirectDirection(n, seed, i, sampleCount), i);
}

//the direction of the first ray of path i
fn getIndirectDirection(n: vec3<f32>, seed: vec2<u32>, i: u32, sampleCount: u32) -> vec3<f32>{
	let index = getSampleNumber() * sampleCount + i;
	var r = hash32uf(vec3<u32>(seed, index));
	if(BLUE_NOISE){
		r = getBlueNoise(seed, index, 0u);
	}
	return cosWeightedRandomHemisphereDirection(n, r);
}

//the sun sample of getLighting, the light that arrives at p from the sun. r: uniform random numbers, a point on the sun disc
fn getDirectLighting(p: vec3<f32>, n: vec3<f32>, r: vec2<f32>) -> vec3<f32>{
	return getShadow(p + n * MIN_DIST, n, r) * uniforms.directionalColor * uniforms.directionalBrightness;
}
