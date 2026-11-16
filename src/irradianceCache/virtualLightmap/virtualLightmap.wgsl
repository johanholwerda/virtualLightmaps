//the surfaces, texels and tiles of the virtual lightmap, see VirtualLightmap.ts. Needs rayTraceScene.wgsl (Model),
//sceneBuffer and uniforms.cameraPosition, pixelSize, texelsPerPixel, texelSizeMin and levelAngle

//texels along the side of the lightmap, which is square
override LIGHTMAP_SIZE: u32;
//the tiles the lightmap has room for
override TILE_COUNT: u32;
//texels along the side of a tile
override TILE_SIZE: u32;

//no tile: not found, or the probe sequence is full of tiles that are in use
const TILE_NONE = 0xFFFFFFFFu;
//the texel size doubles per level, so this is far more than a scene needs. Keeps 1 << level in range
const LEVEL_MAX = 20u;

//a point on a face of a model. Every primitive has 6 faces and a mesh a face per UV island (chart), which are the charts
//of the lightmap: the lighting is never interpolated from one face to another
struct Surface {
	//model * 6 + face, MESH_SURFACE | model << 16 | chart for a mesh
	id : u32,
	//on the face in m, from its corner
	position : vec2<f32>,
	//of the face in m
	size : vec2<f32>,
}

fn tileHash(x: u32) -> u32 {
	var a = x;
	a = (a ^ 61u) ^ (a >> 16u);
	a = a + (a << 3u);
	a = a ^ (a >> 4u);
	a = a * 0x27d4eb2du;
	a = a ^ (a >> 15u);
	return a;
}

//independent of tileHash, for the slot of getTileKey
fn tilePcg(x: u32) -> u32 {
	let state = x * 747796405u + 2891336453u;
	let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

//the bit of the surface ids of meshes, so they never are the id of a face of a primitive
const MESH_SURFACE = 0x80000000u;

/*
the face of the model that p is on, and where p is on it.
-cube: the face p is on, the largest component of the local position
-sphere: projected on the cube around it, so the 6 faces are those of a cube map
-cylinder: the 2 caps, and the side projected on the 4 sides of the box around it
-mesh: the chart of the triangle, at the barycentrics of p. A chart is in local units (see LightmapMesh.layoutCharts),
 the scale of the model is uniform
The face of a primitive follows from the local position alone, not from n, so it doesn't depend on the precision of the
packed gBuffer normal (on macOS that once read as (0, 0, 1), see packSnorm2x16 in gBuffer.wgsl)
p: world position, n: world normal (not used), triangle: the triangle of a mesh, from the gBuffer or hitTriangle
 */
fn getSurface(modelIndex: u32, triangle: u32, p: vec3<f32>, n: vec3<f32>) -> Surface {
	let scale = sceneBuffer[modelIndex].scale;
	let primitiveType = u32(sceneBuffer[modelIndex].primitiveType);
	var local = (sceneBuffer[modelIndex].inverseModel * vec4(p, 1.)).xyz;

	if(primitiveType == PRIMITIVE_MESH){
		let t = getMeshTriangle(triangle);
		let b = getMeshBarycentrics(t, local);
		var surface: Surface;
		surface.id = MESH_SURFACE | ((modelIndex & 0x7FFFu) << 16u) | (t.chart & 0xFFFFu);
		surface.size = getMeshChartSize(t.chart) * scale.x;
		surface.position = (t.c0 + (t.c1 - t.c0) * b.x + (t.c2 - t.c0) * b.y) * scale.x;
		return surface;
	}
	//its largest component is the axis of the face. The local position is on the -1 to 1 cube for a cube and on the
	//cap of a cylinder; a sphere and the side of a cylinder are projected onto it below
	var d = local;
	let onSide = primitiveType == 2u && abs(local.y) < length(local.xz);
	if(onSide){
		d = vec3(local.x, 0., local.z);
	}
	let a = abs(d);
	//with selects instead of indexing the vectors by the axis, which is slower: this runs for every pixel of the resolve
	let isX = a.x >= a.y && a.x >= a.z;
	let isY = !isX && a.y >= a.z;
	let axis = select(select(2u, 1u, isY), 0u, isX);
	let aAxis = select(select(a.z, a.y, isY), a.x, isX);
	let dAxis = select(select(d.z, d.y, isY), d.x, isX);

	//onto the cube
	let scaleToCube = 1. / max(aAxis, 1e-6);
	if(primitiveType == 0u){
		local *= scaleToCube;
	}else if(onSide){
		local = vec3(local.x * scaleToCube, local.y, local.z * scaleToCube);
	}

	//the 2 other axes, in order after the axis of the face: yz, zx or xy
	let faceLocal = select(select(local.xy, local.zx, isY), local.yz, isX);
	let faceScale = select(select(scale.xy, scale.zx, isY), scale.yz, isX);

	var surface: Surface;
	surface.id = modelIndex * 6u + axis * 2u + select(0u, 1u, dAxis > 0.);
	//the primitives are 2 wide
	surface.size = 2. * faceScale;
	surface.position = (clamp(faceLocal, vec2(-1.), vec2(1.)) * .5 + .5) * surface.size;
	return surface;
}

//the most levels a grazing view angle adds, see getLevel
const LEVEL_ANGLE_MAX = 2.;

//the level of the texels at p with normal n, so there are texelsPerPixel texels per screen pixel.
//Level 0 texels are texelSizeMin, each level doubles the size. Fractional, so the resolve can blend between 2 levels.
//A pixel that sees the surface at an angle covers 1 / cos of it along the view direction, so texels at the size of the
//distance alone are far smaller than the pixel that way: a floor seen at a low angle would take many times the texels,
//which each get a sample far less often. levelAngle is the part of that the texels grow with: at .5 by 1 / sqrt(cos),
//which keeps the texels per pixel the same at every angle, for some blur across the view direction.
//From the camera position only, not its rotation, and the same for a pixel and for a ray hit (see getCachedLighting)
fn getLevel(p: vec3<f32>, n: vec3<f32>) -> f32 {
	let toCamera = uniforms.cameraPosition - p;
	let d = length(toCamera);
	let cosAngle = abs(dot(n, toCamera)) / max(d, 1e-6);
	let angleLevels = min(-log2(max(cosAngle, 1e-6)) * uniforms.levelAngle, LEVEL_ANGLE_MAX);
	let size = uniforms.pixelSize * d / uniforms.texelsPerPixel;
	return clamp(log2(size / uniforms.texelSizeMin) + angleLevels, 0., f32(LEVEL_MAX));
}

fn getTexelSize(level: u32) -> f32 {
	return uniforms.texelSizeMin * f32(1u << level);
}

//the texels of a face at a level, at least 1
fn getTexelCount(surface: Surface, level: u32) -> vec2<i32> {
	return max(vec2<i32>(ceil(surface.size / getTexelSize(level))), vec2(1));
}

//the texel of the surface point. A texel of a level is 2 x 2 texels of the level below it
fn getTexel(surface: Surface, level: u32) -> vec2<u32> {
	let texel = vec2<i32>(floor(surface.position / getTexelSize(level)));
	return vec2<u32>(clamp(texel, vec2(0), getTexelCount(surface, level) - 1));
}

//the checksum and the first slot of a tile of a face at a level. The checksum is never 0, which is an empty slot.
//The slot is a second, independent hash, see getKey in spatialHash.wgsl
fn getTileKey(surface: u32, level: u32, tile: vec2<u32>) -> vec2<u32> {
	let key = max(tileHash(tile.x ^ tileHash(tile.y ^ tileHash(surface ^ tileHash(level)))), 1u);
	let slot = tilePcg(tile.x + tilePcg(tile.y + tilePcg(surface + tilePcg(level))));
	return vec2(key, slot % TILE_COUNT);
}

//the index in lightingBuffer of a texel of the face, in the tile at tileIndex
fn getTexelIndex(tileIndex: u32, texel: vec2<u32>) -> u32 {
	let t = texel % TILE_SIZE;
	return (tileIndex * TILE_SIZE + t.y) * TILE_SIZE + t.x;
}

//the running average of the lighting (rgb) and the sample count it holds (a), as 4 x f16
fn unpackLighting(v: vec2<u32>) -> vec4<f32> {
	return vec4(unpack2x16float(v.x), unpack2x16float(v.y));
}

fn packLighting(l: vec4<f32>) -> vec2<u32> {
	return vec2(pack2x16float(l.rg), pack2x16float(l.ba));
}
