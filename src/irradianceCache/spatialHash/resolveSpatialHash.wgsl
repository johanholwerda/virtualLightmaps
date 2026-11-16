
struct Uniforms {
	cameraPosition : vec3<f32>,
	inverseProjection : mat4x4<f32>,
	inverseView : mat4x4<f32>,
	resolution : vec2<f32>,
	exposure : f32,
	skyColor : vec3<f32>,
	//the world size of a pixel at distance 1
	pixelSize : f32,
	//the size of the level 0 cells
	sizeMin : f32,
	//the size of a cell in pixels
	cellSize : f32,
	searchCount : u32,
	//the samples the coarser level counts as in the lighting shown for a level, see combine
	ancestorWeight : f32,
	//the ratio of the 2 largest normal components from which the 2 nearest directions blend, see getLighting
	directionBlend : f32,
	showHoles : f32,
	showLevels : f32,
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
@group(0) @binding(1) var<storage, read> hashBuffer : array<u32>;
@group(0) @binding(3) var<storage, read> lightingBuffer : array<vec2<u32>>;
@group(0) @binding(4) var normalDepthTexture: texture_2d<u32>;
//alpha 0 is the sky
@group(0) @binding(5) var albedoTexture: texture_2d<f32>;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
	var output : VertexOutput;
	output.position = vec4(input.position, 1.0);
	output.uv = input.uv;
	return output;
}

//the lighting of a cell, alpha 0 when it doesn't exist. Only reads, the resolve never inserts cells
fn getCell(cell: vec3<i32>, direction: vec3<f32>, level: u32) -> vec4<f32> {
	let keySlot = getKey(cell, direction, level);
	let key = keySlot.x;
	var index = keySlot.y;
	for(var i = 0u; i < uniforms.searchCount; i++){
		let existing = hashBuffer[index];
		if(existing == key){
			return unpackLighting(lightingBuffer[index]);
		}
		if(existing == 0u){
			break;
		}
		index = (index + 1u) % HASHMAP_SIZE;
	}
	return vec4(0.);
}

/*
the lighting of the 4 cells of a level nearest to p, bilinearly in the plane of the surface. Interpolating instead of the
stochastic jitter of a single cell lookup, so the result is smooth and stable. Not along the axis of the direction:
the cell next to it on that axis is a parallel surface, like the other side of a thin wall.
The weight of a cell is also its sample count, so a new cell with a few noisy samples fades in.
Returns the lighting sum and the weight sum
 */
fn gather(p: vec3<f32>, direction: vec3<f32>, level: u32) -> vec4<f32> {
	let q = getCellPosition(p, direction, level);
	//the 2 axes of the plane of the direction
	var axisU = vec3(0, 1, 0);
	var axisV = vec3(0, 0, 1);
	if(direction.y != 0.){
		axisU = vec3(1, 0, 0);
	}else if(direction.z != 0.){
		axisU = vec3(1, 0, 0);
		axisV = vec3(0, 1, 0);
	}
	//floor(q - .5) on the 2 tangent axes, floor(q) on the axis of the direction
	let shifted = q - vec3<f32>(axisU + axisV) * .5;
	let base = vec3<i32>(floor(shifted));
	let f = fract(shifted);
	let fu = dot(f, vec3<f32>(axisU));
	let fv = dot(f, vec3<f32>(axisV));

	var sum = vec4(0.);
	for(var i = 0; i < 4; i++){
		let offset = vec2(i & 1, i >> 1);
		let cell = getCell(base + axisU * offset.x + axisV * offset.y, direction, level);
		let weight = select(1. - fu, fu, offset.x == 1) * select(1. - fv, fv, offset.y == 1) * cell.a;
		sum += vec4(cell.rgb * weight, weight);
	}
	return sum;
}

//the lighting of a level, leaning on the smoother coarser level while it has few samples, like the ancestor blend of updateOctree.wgsl.
//Alpha 0 when neither has lighting
fn combine(fine: vec4<f32>, coarse: vec4<f32>) -> vec4<f32> {
	let priorWeight = min(coarse.w, uniforms.ancestorWeight);
	if(priorWeight == 0.){
		return fine;
	}
	return vec4(fine.xyz + coarse.xyz / coarse.w * priorWeight, fine.w + priorWeight);
}

//the lighting at p of the cells facing a direction: between the 2 levels around the level of p, so there is no seam where the level changes
fn getDirectionLighting(p: vec3<f32>, direction: vec3<f32>, d: f32) -> vec4<f32> {
	let levelF = getLevel(d);
	let level = u32(levelF);
	let l0 = gather(p, direction, level);
	let l1 = gather(p, direction, level + 1u);
	let l2 = gather(p, direction, level + 2u);
	let c0 = combine(l0, l1);
	let c1 = combine(l1, l2);
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

//the lighting of the direction of n, blended with the second nearest direction where n is between them (on a curved surface),
//so there is no seam where the direction changes. A flat face of a box has 1 direction
fn getLighting(p: vec3<f32>, n: vec3<f32>, d: f32) -> vec4<f32> {
	let first = getDirection(n);
	let lighting0 = getDirectionLighting(p, first, d);
	let second = getDirection(n * (1. - abs(first)));
	//1 when n is halfway between the 2 directions
	let ratio = dot(abs(n), abs(second)) / dot(abs(n), abs(first));
	let w = .5 * smoothstep(uniforms.directionBlend, 1., ratio);
	if(w == 0.){
		return lighting0;
	}
	let lighting1 = getDirectionLighting(p, second, d);
	if(lighting1.w == 0.){
		return lighting0;
	}
	if(lighting0.w == 0.){
		return lighting1;
	}
	return vec4(mix(lighting0.xyz, lighting1.xyz, w), max(lighting0.w, lighting1.w));
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
	let coord = vec2<u32>(input.uv * uniforms.resolution);

	//the sky has no cells, it is emissive
	let albedo = textureLoad(albedoTexture, coord, 0);
	if(albedo.a == 0.){
		return vec4(toneMap(uniforms.skyColor * uniforms.exposure), 1.);
	}

	let normalDepth = loadNormalDepth(normalDepthTexture, coord);
	let uv = (vec2<f32>(coord) + .5) / uniforms.resolution;
	let worldPosition = depthToPosition(normalDepth.w, uv * 2. - 1.);

	let lighting = getLighting(worldPosition, normalDepth.xyz, normalDepth.w);
	//no cell yet, only the case for a few frames after a reset or a disocclusion: the albedo with a dim light, like octreeResolve.wgsl
	let l = select(vec3(.5), lighting.xyz, lighting.w > 0.);
	//the cell lighting has no albedo, see getLighting in updateSpatialHash.wgsl
	var c = toneMap(albedo.xyz * l * uniforms.exposure);
	if(uniforms.showHoles > .5 && lighting.w == 0.){
		c = vec3(1., 0., 0.);
	}
	if(uniforms.showLevels > .5){
		let level = u32(getLevel(normalDepth.w));
		c = mix(c, fract(vec3(f32(level)) * vec3(.31, .57, .73)), .5);
	}
	return vec4(c, 1.);
}
