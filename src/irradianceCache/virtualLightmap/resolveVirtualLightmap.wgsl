
struct Uniforms {
	cameraPosition : vec3<f32>,
	inverseProjection : mat4x4<f32>,
	inverseView : mat4x4<f32>,
	resolution : vec2<f32>,
	exposure : f32,
	skyColor : vec3<f32>,
	//the world size of a pixel at distance 1
	pixelSize : f32,
	//lightmap texels per screen pixel, along a side
	texelsPerPixel : f32,
	//the size of the level 0 texels
	texelSizeMin : f32,
	//how much the texels grow at a grazing view angle, see getLevel
	levelAngle : f32,
	searchCount : u32,
	//the paths the coarser level counts as in the lighting shown for a level, see combine
	ancestorWeight : f32,
	showHoles : f32,
	showLevels : f32,
	showTiles : f32,
	//the texels that hold less than accumulationMax paths in yellow
	showConverging : f32,
	accumulationMax : f32,
	//the lightmap itself instead of the scene, see getLightmapColor
	showLightmap : f32,
	//a pixel reads 1 of the 2 levels around its level instead of both, see getLightingStochastic. Only with the TAA pass
	stochasticLevel : f32,
	//for the noise of stochasticLevel
	frame : f32,
	//for rayTraceScene.wgsl, see Scene.bind
	lightDirection : vec3<f32>,
	modelCount : f32,
	bvhModelCount : f32,
}

struct VertexInput {
  @location(0) position : vec3<f32>,
  @location(1) uv : vec2<f32>,
};

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  @location(0) uv : vec2<f32>,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage, read> tileBuffer : array<u32>;
@group(0) @binding(3) var<storage, read> lightingBuffer : array<vec2<u32>>;
@group(0) @binding(4) var normalDepthTexture: texture_2d<u32>;
//alpha 0 is the sky
@group(0) @binding(5) var albedoTexture: texture_2d<f32>;
@group(0) @binding(6) var modelTexture: texture_2d<u32>;
@group(0) @binding(7) var<storage, read> sceneBuffer : array<Model>;
@group(0) @binding(10) var<storage, read> bvhBuffer : array<BVHNode>;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
	var output : VertexOutput;
	output.position = vec4(input.position, 1.0);
	output.uv = input.uv;
	return output;
}

//the slot of a tile, TILE_NONE when it doesn't exist. Only reads, the resolve never inserts tiles
fn findTile(surface: u32, level: u32, tile: vec2<u32>) -> u32 {
	let keySlot = getTileKey(surface, level, tile);
	let key = keySlot.x;
	var index = keySlot.y;
	for(var i = 0u; i < uniforms.searchCount; i++){
		let existing = tileBuffer[index];
		if(existing == key){
			return index;
		}
		if(existing == 0u){
			break;
		}
		index = (index + 1u) % TILE_COUNT;
	}
	return TILE_NONE;
}

/*
the lighting of the 4 texels of a level nearest to the surface point, bilinearly. The texels are clamped to the face, so the
lighting of another face never blends in. The 4 texels are mostly in 1 tile, which is then looked up once: only at the
border of a tile the other tiles are looked up too, each once.
The weight of a texel is also its path count, so a new texel with a few noisy paths fades in. Up to GATHER_COUNT_MAX of
accumulationMax: above that the texels count the same, else the lighting would stay close to that of the texel with the
most paths until they all are resolved, as blocks instead of a gradient.
Returns the lighting times the path count, and the path count: the paths of the texels, interpolated
 */
const GATHER_COUNT_MAX = .25;

//the lighting of a texel, at its place in its tile. No lighting and no paths when the tile doesn't exist
fn loadTexel(tileIndex: u32, tileTexel: vec2<u32>) -> vec4<f32> {
	if(tileIndex == TILE_NONE){
		return vec4(0.);
	}
	return unpackLighting(lightingBuffer[(tileIndex * TILE_SIZE + tileTexel.y) * TILE_SIZE + tileTexel.x]);
}

fn gather(surface: Surface, level: u32) -> vec4<f32> {
	let q = surface.position / getTexelSize(level) - .5;
	let base = vec2<i32>(floor(q));
	let f = fract(q);
	let last = getTexelCount(surface, level) - 1;
	let texel0 = vec2<u32>(clamp(base, vec2(0), last));
	let texel1 = vec2<u32>(clamp(base + 1, vec2(0), last));

	//the tiles of the 4 texels: 00, 10, 01 and 11
	let t0 = texel0 / TILE_SIZE;
	let t1 = texel1 / TILE_SIZE;
	let i00 = findTile(surface.id, level, t0);
	var i10 = i00;
	var i01 = i00;
	var i11 = i00;
	if(t1.x != t0.x){
		i10 = findTile(surface.id, level, vec2(t1.x, t0.y));
		i11 = i10;
	}
	if(t1.y != t0.y){
		i01 = findTile(surface.id, level, vec2(t0.x, t1.y));
		i11 = i01;
		if(t1.x != t0.x){
			i11 = findTile(surface.id, level, t1);
		}
	}

	//written out per texel, a loop over the 4 is a little slower
	let p0 = texel0 % TILE_SIZE;
	let p1 = texel1 % TILE_SIZE;
	let l00 = loadTexel(i00, p0);
	let l10 = loadTexel(i10, vec2(p1.x, p0.y));
	let l01 = loadTexel(i01, vec2(p0.x, p1.y));
	let l11 = loadTexel(i11, p1);

	let bilinear = vec4((1. - f.x) * (1. - f.y), f.x * (1. - f.y), (1. - f.x) * f.y, f.x * f.y);
	let counts = vec4(l00.a, l10.a, l01.a, l11.a);
	let weights = bilinear * min(counts, vec4(uniforms.accumulationMax * GATHER_COUNT_MAX));
	let weight = weights.x + weights.y + weights.z + weights.w;
	if(weight == 0.){
		return vec4(0.);
	}
	let count = dot(bilinear, counts);
	let sum = l00.rgb * weights.x + l10.rgb * weights.y + l01.rgb * weights.z + l11.rgb * weights.w;
	return vec4(sum / weight * count, count);
}

//the paths of a level at which the coarser level no longer counts, in ancestorWeights: see combine
const PRIOR_FADE = 8.;
//and as a part of accumulationMax, whichever is less. A resolved texel holds accumulationMax paths: when PRIOR_FADE
//ancestorWeights is that many or more (the low quality) it would never be without the coarser level, which is a
//lookup of a level more for every pixel
const PRIOR_FADE_ACCUMULATION = .75;

//the paths from which a level is shown as it is, without the coarser level
fn getPriorEnd() -> f32 {
	return min(uniforms.ancestorWeight * PRIOR_FADE, uniforms.accumulationMax * PRIOR_FADE_ACCUMULATION);
}

//the lighting of a level, leaning on the smoother coarser level while it has few samples, like combine of resolveSpatialHash.wgsl.
//The weight of the coarser level fades out towards getPriorEnd paths: a level that holds that many is not noisy, the coarser
//one would only blur it. Alpha 0 when neither has lighting
fn combine(fine: vec4<f32>, coarse: vec4<f32>) -> vec4<f32> {
	let priorEnd = getPriorEnd();
	if(fine.w >= priorEnd || coarse.w == 0.){
		return fine;
	}
	let priorWeight = min(coarse.w, uniforms.ancestorWeight) * (1. - fine.w / priorEnd);
	return vec4(fine.xyz + coarse.xyz / coarse.w * priorWeight, fine.w + priorWeight);
}

//between the 2 levels around the level of the pixel, so there is no seam where the level changes
fn getLighting(surface: Surface, levelF: f32) -> vec4<f32> {
	let level = u32(levelF);
	let l0 = gather(surface, level);
	let l1 = gather(surface, level + 1u);
	let c0 = combine(l0, l1);
	//the third level is only the prior of the second: not looked up where that needs none, which is most pixels once
	//the lighting has converged
	var c1 = l1;
	if(l1.w < getPriorEnd()){
		c1 = combine(l1, gather(surface, level + 2u));
	}
	let t = fract(levelF);
	let lighting0 = select(vec3(0.), c0.xyz / c0.w, c0.w > 0.);
	let lighting1 = select(vec3(0.), c1.xyz / c1.w, c1.w > 0.);
	//a level without lighting doesn't fade to black
	let w0 = select(0., 1. - t, c0.w > 0.);
	let w1 = select(0., t, c1.w > 0.);
	if(w0 + w1 == 0.){
		return vec4(select(lighting1, lighting0, c0.w > 0.), max(c0.w, c1.w));
	}
	return vec4((lighting0 * w0 + lighting1 * w1) / (w0 + w1), max(c0.w, c1.w));
}

/*
1 of the 2 levels around the level of the pixel instead of both, the coarser one with the chance of its weight in
getLighting. Over the pixels and the frames that is the same blend, the TAA pass averages it: so only with that pass,
without it the noise shows where the levels differ (at the edge of a shadow). A level less to look up for every pixel,
the resolve is about 1.5 times faster.
noise: 0 to 1, different per pixel and per frame
 */
fn getLightingStochastic(surface: Surface, levelF: f32, noise: f32) -> vec4<f32> {
	let levelBase = u32(levelF);
	let coarser = noise < fract(levelF);
	let level = levelBase + select(0u, 1u, coarser);
	var c = gather(surface, level);
	//the level above it is only its prior, like the third level in getLighting
	if(c.w < getPriorEnd()){
		c = combine(c, gather(surface, level + 1u));
	}
	//a level without lighting doesn't show as a hole when the other one has it. The finer one already had the coarser as its prior
	if(c.w == 0. && coarser){
		c = gather(surface, levelBase);
	}
	return vec4(select(vec3(0.), c.xyz / c.w, c.w > 0.), c.w);
}

//interleaved gradient noise, moved every frame: a repeat of 64 frames
fn getNoise(coord: vec2<u32>) -> f32 {
	let p = vec2<f32>(coord) + 5.588238 * (uniforms.frame % 64.);
	return fract(52.9829189 * fract(dot(p, vec2(.06711056, .00583715))));
}

//the lightmap as it is stored: the tile slots in rows, a square in the middle of the screen with black borders next to it.
//The lighting without the albedo. A slot is where the hash of its tile put it, so neighbouring tiles are not neighbours on a face.
//Red tiles are empty slots, black texels are in a tile but no pixel sampled them yet
fn getLightmapColor(uv: vec2<f32>) -> vec3<f32> {
	let side = min(uniforms.resolution.x, uniforms.resolution.y);
	let p = (uv * uniforms.resolution - (uniforms.resolution - side) * .5) / side;
	if(any(p < vec2(0.)) || any(p >= vec2(1.))){
		return vec3(0.);
	}
	let tilesPerSide = LIGHTMAP_SIZE / TILE_SIZE;
	let texel = vec2<u32>(p * f32(LIGHTMAP_SIZE));
	let tile = texel / TILE_SIZE;
	let tileIndex = tile.y * tilesPerSide + tile.x;
	if(tileBuffer[tileIndex] == 0u){
		return vec3(.5, 0., 0.);
	}
	let lighting = unpackLighting(lightingBuffer[getTexelIndex(tileIndex, texel)]);
	return toneMap(lighting.rgb * uniforms.exposure);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
	if(uniforms.showLightmap > .5){
		return vec4(getLightmapColor(input.uv), 1.);
	}
	let coord = vec2<u32>(input.uv * uniforms.resolution);

	//the sky has no texels, it is emissive
	let albedo = textureLoad(albedoTexture, coord, 0);
	if(albedo.a == 0.){
		return vec4(toneMap(uniforms.skyColor * uniforms.exposure), 1.);
	}

	let normalDepth = loadNormalDepth(normalDepthTexture, coord);
	let uv = (vec2<f32>(coord) + .5) / uniforms.resolution;
	let worldPosition = depthToPosition(normalDepth.w, uv * 2. - 1.);
	let model = unpackModelTriangle(textureLoad(modelTexture, coord, 0).x);
	let surface = getSurface(model.x, model.y, worldPosition, normalDepth.xyz);

	let levelF = getLevel(worldPosition, normalDepth.xyz);
	var lighting: vec4<f32>;
	if(uniforms.stochasticLevel > .5){
		lighting = getLightingStochastic(surface, levelF, getNoise(coord));
	}else{
		lighting = getLighting(surface, levelF);
	}
	//no texel yet, only the case for a few frames after a reset or a disocclusion: the albedo with a dim light, like resolveSpatialHash.wgsl
	let l = select(vec3(.5), lighting.xyz, lighting.w > 0.);
	//the lighting has no albedo, see getLighting in pathTrace.wgsl
	var c = toneMap(albedo.xyz * l * uniforms.exposure);
	if(uniforms.showHoles > .5 && lighting.w == 0.){
		c = vec3(1., 0., 0.);
	}
	let level = u32(levelF);
	if(uniforms.showLevels > .5){
		c = mix(c, fract(vec3(f32(level)) * vec3(.31, .57, .73)), .5);
	}
	//the tiles of the level of the pixel, and its texels as a checkerboard
	if(uniforms.showTiles > .5){
		let texel = getTexel(surface, level);
		let tile = texel / TILE_SIZE;
		let tileColor = vec3<f32>((vec3(tileHash(tile.x ^ tileHash(tile.y ^ tileHash(surface.id)))) >> vec3(0u, 8u, 16u)) & vec3(255u)) / 255.;
		c = mix(c, tileColor * select(.7, 1., (texel.x + texel.y) % 2u == 0u), .5);
	}
	if(uniforms.showConverging > .5){
		let texel = getTexel(surface, level);
		let tileIndex = findTile(surface.id, level, texel / TILE_SIZE);
		if(tileIndex != TILE_NONE && unpackLighting(lightingBuffer[getTexelIndex(tileIndex, texel)]).a < uniforms.accumulationMax){
			c = mix(c, vec3(1., 1., 0.), .5);
		}
	}
	return vec4(c, 1.);
}
