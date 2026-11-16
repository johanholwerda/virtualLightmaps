//the hashing of the spatial hash map. A cell is a world space cube at a level of detail,
//for one normal direction. Needs uniforms.sizeMin, uniforms.cellSize, uniforms.pixelSize and uniforms.feedCellSize
override HASHMAP_SIZE: u32;

//no entry: not found, or the probe sequence is full
const HASH_NONE = 0xFFFFFFFFu;

fn hashU32(x: u32) -> u32 {
	var a = x;
	a = (a ^ 61u) ^ (a >> 16u);
	a = a + (a << 3u);
	a = a ^ (a >> 4u);
	a = a * 0x27d4eb2du;
	a = a ^ (a >> 15u);
	return a;
}

fn hash3d(p: vec3<i32>) -> u32 {
	return hashU32(u32(p.x) ^ hashU32(u32(p.y) ^ hashU32(u32(p.z))));
}

//the pcg hash, independent of hashU32, for the slot of getKey
fn pcgU32(x: u32) -> u32 {
	let state = x * 747796405u + 2891336453u;
	let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

//the level of the cells at a distance d from the camera, so a cell covers about cellSize pixels.
//Level 0 cells are sizeMin, each level doubles the size. Fractional, so the resolve can blend between 2 levels
fn getLevel(d: f32) -> f32 {
	let size = uniforms.cellSize * uniforms.pixelSize * d;
	return max(0., log2(size / uniforms.sizeMin));
}

fn getCellSize(level: u32) -> f32 {
	return uniforms.sizeMin * f32(1u << level);
}

/*
the axis the normal is closest to, with its sign: the 6 directions a cell can face.
Keying on the normal quantized in 7 steps per axis didn't work: on a curved surface (the sphere) every step got its own cells,
with nothing to interpolate between them: the lighting showed the facets. The resolve blends the 2 nearest directions instead
 */
fn getDirection(n: vec3<f32>) -> vec3<f32> {
	let a = abs(n);
	if(a.x >= a.y && a.x >= a.z){
		return vec3(sign(n.x), 0., 0.);
	}
	if(a.y >= a.z){
		return vec3(0., sign(n.y), 0.);
	}
	return vec3(0., 0., sign(n.z));
}

//the position in cells. Offset along the direction, so a surface that is exactly on a cell boundary (which the axis aligned
//boxes of the building are) is in 1 cell, not flickering between 2
fn getCellPosition(p: vec3<f32>, direction: vec3<f32>, level: u32) -> vec3<f32> {
	let cellSize = getCellSize(level);
	return (p + direction * (cellSize * .1)) / cellSize;
}

//the checksum and the first slot of the cell at p, see getKey
fn getCellKey(p: vec3<f32>, direction: vec3<f32>, level: u32) -> vec2<u32> {
	return getKey(vec3<i32>(floor(getCellPosition(p, direction, level))), direction, level);
}

//the level bit of the feed cells in their key: cells of their own, apart from the cells of the screen at the same level.
//See getCachedLighting in updateSpatialHash.wgsl
const FEED_LEVEL = 0x10000u;
//the highest level of the feed cells, so the shift of getCellSize stays in range
const FEED_LEVEL_MAX = 24.;

//the level of the feed cells: the one with the cells nearest to feedCellSize, the same for the whole scene and wherever
//the camera is
fn getFeedLevel() -> u32 {
	return u32(clamp(round(log2(uniforms.feedCellSize / uniforms.sizeMin)), 0., FEED_LEVEL_MAX));
}

//the checksum and the first slot of the feed cell at p
fn getFeedKey(p: vec3<f32>, direction: vec3<f32>) -> vec2<u32> {
	let level = getFeedLevel();
	return getKey(vec3<i32>(floor(getCellPosition(p, direction, level))), direction, level | FEED_LEVEL);
}

//the checksum and the first slot of the cell. The direction is part of the key, so the 2 sides of a wall,
//or 2 faces at a hard edge, never share lighting. The level is too, so the levels don't collide. The checksum is never 0, which is an empty slot.
//The slot is a second, independent hash instead of the checksum modulo HASHMAP_SIZE: then 2 cells with the same 32 bit checksum had the
//same slot too, and shared their lighting. With millions of cells that was 1 in ~900 new cells (a lit cell in the shade, a dark one in
//the sun), and it took over the full sample count of the other cell, so it hardly faded. Now they also need the same probe sequence
fn getKey(cell: vec3<i32>, direction: vec3<f32>, level: u32) -> vec2<u32> {
	let directionHash = hash3d(vec3<i32>(direction));
	let key = max(hash3d(cell) ^ (directionHash * 0x9e3779b9u) ^ hashU32(level), 1u);
	let d = vec3<u32>(vec3<i32>(direction) + 1);
	let slot = pcgU32(u32(cell.x) + pcgU32(u32(cell.y) + pcgU32(u32(cell.z) + pcgU32(d.x + d.y * 3u + d.z * 9u + level * 27u))));
	return vec2(key, slot % HASHMAP_SIZE);
}

//the running average of the lighting (rgb) and the sample count it holds (a), as 4 x f16
fn unpackLighting(v: vec2<u32>) -> vec4<f32> {
	return vec4(unpack2x16float(v.x), unpack2x16float(v.y));
}

fn packLighting(l: vec4<f32>) -> vec2<u32> {
	return vec2(pack2x16float(l.rg), pack2x16float(l.ba));
}
