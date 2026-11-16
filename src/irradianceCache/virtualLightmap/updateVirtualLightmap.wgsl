override WORK_GROUP_SIZE_X: u32;
override WORK_GROUP_SIZE_Y: u32;
override CELL_SIZE: u32;
//the threads of a workgroup of trace_main
override TRACE_GROUP_SIZE: u32;
//true: a new tile takes the place of the first unused tile of its probe sequence instead of the one that was unused the
//longest, see findOrInsert
override EVICT_FIRST: bool = false;

struct Uniforms {
	resolution : vec2<f32>,
	inverseProjection : mat4x4<f32>,
	inverseView : mat4x4<f32>,
	cameraPosition : vec3<f32>,
	frame : f32,
	//the world size of a pixel at distance 1
	pixelSize : f32,
	//lightmap texels per screen pixel, along a side
	texelsPerPixel : f32,
	//the size of the level 0 texels: 1 / the most texels per m
	texelSizeMin : f32,
	//how much the texels grow at a grazing view angle, see getLevel
	levelAngle : f32,
	lightDirection : vec3<f32>,
	//angular radius of the sun, see sampleSunDirection
	sunRadius : f32,
	directionalColor : vec3<f32>,
	modelCount : f32,
	//the first models are static and in the BVH, see traceRay
	bvhModelCount : f32,
	bounceCount : i32,
	directionalBrightness : f32,
	directionalProbability : f32,
	sampleCount : u32,
	//the paths the running average of a texel holds, see addSample
	accumulationMax : f32,
	//1: the sun or the sky moves (the day cycle), so a texel that holds accumulationMax paths keeps taking them. 0: it
	//takes no more, it is resolved. Not for a model that is dragged, see dragState
	lightChanging : u32,
	//the paths a resolved texel goes back to when the light stops changing, see refresh_main
	refreshCount : f32,
	//the paths a texel that took no samples since the last refresh goes back to, see refresh_main
	refreshCountIdle : f32,
	//1: the paths end at their first hit in the lighting the lightmap has there, see getCachedLighting
	cacheBounces : u32,
	//coarser levels that accumulate the samples too, at most ANCESTOR_LEVELS_MAX
	ancestorLevels : u32,
	//the paths of the coarser lighting a new texel starts with
	seedCount : f32,
	//the frames a tile has to be unused before another one can take its place, when there is no empty slot
	evictFrames : f32,
	searchCount : u32,
	//the threads of trace_main per cell, a divisor of TRACE_GROUP_SIZE. Follows sampleCount, see VirtualLightmap.getTraceLanes
	traceLanes : u32,
	//the frames of light a texel holds while the light changes, at every level: see getChangingCountMax
	changeFrames : f32,
	//the screen height as a part of the one pixelSize is for
	pixelScale : f32,
	//the first and the last frame of the last time the light changed, the last is this frame while it does. Both 0 when
	//it never did. See getStaleFrames
	lightChangeStart : u32,
	lightChangeEnd : u32,
	//the model that is dragged as a capsule: its center at the start of the drag and now, and its radius. See getDragInfluence
	dragFrom : vec3<f32>,
	dragRadius : f32,
	dragTo : vec3<f32>,
	//1: a model is dragged, see getCountMax
	dragState : u32,
	//the index of the dragged model in sceneBuffer
	dragModel : u32,
	//how far the dragged model reaches: times the part of the hemisphere of a texel it covers
	dragInfluence : f32,
	//the highest reset frame of the strokes, 0 when there is none
	dragStrokeEnd : u32,
	//the last DRAG_STROKE_COUNT drags, each as the capsule around the whole drag: A is its start and radius, B its end
	//and its reset frame, 0 for no stroke: a texel that was last sampled before that frame is reset, see resetStaleTexel.
	//The frame after the drag ended, or some frames ago for the drag that is going on
	dragStrokeA0 : vec4<f32>,
	dragStrokeB0 : vec4<f32>,
	dragStrokeA1 : vec4<f32>,
	dragStrokeB1 : vec4<f32>,
	dragStrokeA2 : vec4<f32>,
	dragStrokeB2 : vec4<f32>,
	dragStrokeA3 : vec4<f32>,
	dragStrokeB3 : vec4<f32>,
	//the model of each stroke
	dragStrokeModels : vec4<f32>,
	//1: a dragged model changes the lighting where it has influence only. 0: everywhere, see getCapsuleInfluence
	dragLocal : u32,
	//the most cells that trace in a frame, see select_main
	traceCellsMax : u32,
	//the paths a feed texel holds, 0: no feed. See getCachedLighting
	rayFeedPaths : f32,
	//the size of the feed texels in m, see getFeedLevel
	feedTexelSize : f32,
}

const DRAG_STROKE_COUNT = 4u;
//the paths a texel keeps when it is reset where a stroke has all the influence, see resetStaleTexel. Next to nothing: the
//lighting it holds can be that of the sun where there is a shadow now, many times brighter, so even a percent of it
//left in the texel shows
const DRAG_RETURN_PATHS = .25;
//a ray hit in a tile that is out of view since a stroke with more influence than this is not cached, see getCachedLighting
const DRAG_BOUNCE_INFLUENCE = .05;
//the least a surface counts as facing a dragged model, see getCapsuleInfluence
const DRAG_BOUNCE_FACING = .25;

const ANCESTOR_LEVELS_MAX = 4u;
//the highest path count of a texel while the light changes: an f16 has steps of 8 up to here, less than a sample weighs
const CHANGING_COUNT_MAX = 16384.;
//the paths a texel holds at least while the light changes, in samples
const CHANGING_SAMPLES_MIN = 4.;
//a ray hit in a tile that missed more than this many times changeFrames of the changing light is not cached, see getCachedLighting
const STALE_BOUNCE = 4.;

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
//per tile slot: the checksum of the tile, 0 when empty
@group(0) @binding(1) var<storage, read_write> tileBuffer : array<atomic<u32>>;
//the frame a tile was last used in
@group(0) @binding(2) var<storage, read_write> tileTimeBuffer : array<atomic<u32>>;
//per tile slot TILE_SIZE x TILE_SIZE texels: running average of the lighting (rgb) and the sample count it holds (a), as 4 x f16.
//2 words per texel (rg and ba, see packLighting), atomic for the lock of addTexelSample
@group(0) @binding(3) var<storage, read_write> lightingBuffer : array<atomic<u32>>;
//the slots in use, for the log
@group(0) @binding(4) var<storage, read_write> tileCount : atomic<u32>;
@group(0) @binding(5) var normalDepthTexture: texture_2d<u32>;
@group(0) @binding(6) var modelTexture: texture_2d<u32>;
@group(0) @binding(7) var<storage, read> sceneBuffer : array<Model>;
@group(0) @binding(10) var<storage, read> bvhBuffer : array<BVHNode>;
//per texel: the highest bid of the cells that sample it this frame, see claim_main. Its frame is also when the texel was
//last sampled, see resetStaleTexel
@group(0) @binding(11) var<storage, read_write> claimBuffer : array<atomic<u32>>;
//2 lists with a slot per cell, in the 1 buffer as the shader is at the limit of 10 storage buffers.
//-first the cells that won the bid of their texel, x | y << 16
//-then per cell the texel it bids on this frame (a texel index of getTexelIndex), TILE_NONE when it has none. From
// claim_main to select_main, which then doesn't have to look up the texel again. See getCellTexelSlot
@group(0) @binding(12) var<storage, read_write> traceCells : array<u32>;
//the workgroups of trace_main (x, y, z) and the count of traceCells. Copied to the indirect buffer, which can't be bound
//as storage in the same dispatch.
//4 and 5: the cells that won the bid of their texel, of the even and of the odd frames. So of this frame and of the
//previous one, which select_main uses to stay within traceCellsMax
@group(0) @binding(13) var<storage, read_write> traceArgs : array<atomic<u32>, 8>;

//the frames the light changed in since a tile was last used, in lastUsed: a tile only gets samples while it is in view,
//so this is how far its lighting is behind. 0 when the light never changed, so then nothing here does anything
fn getStaleFrames(lastUsed: u32) -> f32 {
	let first = max(lastUsed, uniforms.lightChangeStart);
	return f32(max(uniforms.lightChangeEnd, first) - first);
}

//the surface point of the cell whose texels are looked up, for resetStaleTexel. Set by getCellTexels
var<private> cellPosition: vec3<f32>;
var<private> cellNormal: vec3<f32>;
var<private> cellModel: u32;

//the path counts of the texels of a tile times scale, at least 1 so they keep their lighting as a start. The texels then
//follow the new samples fast, and the resolve shows the coarser level until they have some (see combine)
fn ageTile(tileIndex: u32, scale: f32) {
	let start = tileIndex * TILE_SIZE * TILE_SIZE * 2u;
	for(var t = 0u; t < TILE_SIZE * TILE_SIZE; t++){
		//blue and the count
		let ba = &lightingBuffer[start + t * 2u + 1u];
		let v = unpack2x16float(atomicLoad(ba));
		if(v.y > 1.){
			atomicStore(ba, pack2x16float(vec2(v.x, max(v.y * scale, 1.))));
		}
	}
}

//the tile is used this frame. Most cells of a tile find it already marked: a store by all of them would queue on the 1 address.
//A tile that comes back in view after the sun moved holds the lighting of when it left: its texels count for less
//the longer ago that is, see ageTile. Only by the thread that replaces the old time.
//frame: this frame for a tile that is in view, the previous one for a tile the rays feed (see getFeedStamp)
fn markUsed(index: u32, frame: u32) {
	let last = atomicLoad(&tileTimeBuffer[index]);
	if(last >= frame){
		return;
	}
	let stale = getStaleFrames(last);
	if(last == 0u || stale <= uniforms.changeFrames){
		atomicStore(&tileTimeBuffer[index], frame);
	}else if(atomicExchange(&tileTimeBuffer[index], frame) == last){
		ageTile(index, uniforms.changeFrames / stale);
	}
}

//the slots from its first one a new tile can take the place of an unused tile in, while there still is an empty slot
//further on. See findOrInsert
const EVICT_NEAR = 4u;

/*
the slot of a tile, which is also where its texels are in lightingBuffer. Linear probing, like findOrInsert of updateSpatialHash.wgsl.
-first the search for the key, up to the first empty slot (the key is never after one: slots are only taken over, never emptied)
-not found: the empty slot, so a tile that is out of view keeps its lighting while there is room
-no empty slot: the tile of the probe sequence that was unused the longest (at least evictFrames) makes place, its lighting cleared.
 The compare exchange makes sure only 1 thread takes it. The ages are only read when the key is not found
Every cell searches its tiles every frame, and a tile at the end of the probe sequence takes searchCount loads to find.
In a lightmap that is full the tiles ended up anywhere in their sequence, half of searchCount loads on average, which
made claim_main 2.5 times slower than in an empty one. So a new tile stays near its first slot:
-an unused tile in the first EVICT_NEAR slots makes place before an empty slot further on is taken
-without an empty slot the unused ones in the first EVICT_NEAR slots go first, the rest only when there is none
So some tiles that are out of view lose their lighting before the lightmap is full.
EVICT_FIRST: the first unused tile of the sequence makes place, whatever its age over evictFrames. The tiles are then as
near their first slot as they can be, a full lightmap is as fast as an empty one (2 loads per search instead of 3).
But a tile that is out of view for a second is as likely to go as one that is for a minute.
Not the same as an EVICT_NEAR of 1: that looks at the first slot only and then takes the oldest of all the others,
which is anywhere in the sequence (4.3 loads)
 */
//stamp: the frame the tile is marked as used in, see markUsed
fn findOrInsert(surface: u32, level: u32, tile: vec2<u32>, stamp: u32) -> u32 {
	let keySlot = getTileKey(surface, level, tile);
	let key = keySlot.x;
	let frame = u32(uniforms.frame);
	var index = keySlot.y;
	var i = 0u;
	for(; i < uniforms.searchCount; i++){
		let existing = atomicLoad(&tileBuffer[index]);
		if(existing == key){
			markUsed(index, stamp);
			return index;
		}
		if(existing == 0u){
			break;
		}
		index = (index + 1u) % TILE_COUNT;
	}

	//no key, and no empty slot near: only now the ages are worth reading, so the hit path doesn't load tileTimeBuffer.
	//Scan the same slots again from the first one. With an empty slot only the near ones, else all, but the far ones
	//only when no near one can go
	let near = min(EVICT_NEAR, uniforms.searchCount);
	var evictIndex = TILE_NONE;
	var evictKey = 0u;
	if(i >= near){
		var evictAge = u32(uniforms.evictFrames);
		let scanCount = select(near, uniforms.searchCount, i >= uniforms.searchCount);
		var scanIndex = keySlot.y;
		for(var s = 0u; s < scanCount; s++){
			if(evictIndex != TILE_NONE && (EVICT_FIRST || s == near)){
				break;
			}
			let existing = atomicLoad(&tileBuffer[scanIndex]);
			if(existing == key){
				markUsed(scanIndex, stamp);
				return scanIndex;
			}
			if(existing != 0u){
				let age = frame - min(atomicLoad(&tileTimeBuffer[scanIndex]), frame);
				if(age > evictAge){
					evictAge = age;
					evictIndex = scanIndex;
					evictKey = existing;
				}
			}
			scanIndex = (scanIndex + 1u) % TILE_COUNT;
		}
	}

	//retried while the weak compare exchange fails spuriously
	if(evictIndex != TILE_NONE){
		for(var j = 0u; j < 4u; j++){
			let cmp = atomicCompareExchangeWeak(&tileBuffer[evictIndex], evictKey, key);
			if(cmp.exchanged){
				markUsed(evictIndex, stamp);
				let start = evictIndex * TILE_SIZE * TILE_SIZE * 2u;
				for(var t = 0u; t < TILE_SIZE * TILE_SIZE * 2u; t++){
					atomicStore(&lightingBuffer[start + t], 0u);
				}
				return evictIndex;
			}
			//another thread inserted the same tile there
			if(cmp.old_value == key){
				markUsed(evictIndex, stamp);
				return evictIndex;
			}
			//another tile took it: the empty slot below, if there is one
			if(cmp.old_value != evictKey){
				break;
			}
		}
	}

	//the empty slot, or the next ones if other threads filled it meanwhile. A spurious failure retries the same slot:
	//skipping it would leave an empty slot before the key, where find stops
	for(; i < uniforms.searchCount; i++){
		let cmp = atomicCompareExchangeWeak(&tileBuffer[index], 0u, key);
		if(cmp.exchanged){
			atomicAdd(&tileCount, 1u);
		}
		if(cmp.exchanged || cmp.old_value == key){
			markUsed(index, stamp);
			return index;
		}
		if(cmp.old_value != 0u){
			index = (index + 1u) % TILE_COUNT;
		}
	}
	return TILE_NONE;
}

fn find(surface: u32, level: u32, tile: vec2<u32>) -> u32 {
	let keySlot = getTileKey(surface, level, tile);
	let key = keySlot.x;
	var index = keySlot.y;
	for(var i = 0u; i < uniforms.searchCount; i++){
		let existing = atomicLoad(&tileBuffer[index]);
		if(existing == key){
			return index;
		}
		if(existing == 0u){
			return TILE_NONE;
		}
		index = (index + 1u) % TILE_COUNT;
	}
	return TILE_NONE;
}

//weight: the paths the sample is the average of. That of a cell is sampleCount paths, so it weighs sampleCount: the count
//is in paths, and a higher sampleCount resolves a texel in fewer frames. That of a ray is 1 path, see getIndirectPathSum.
//The count is capped at countMax, so the average keeps following changes in the scene
fn addSample(accumulated: vec4<f32>, lighting: vec3<f32>, weight: f32, countMax: f32) -> vec4<f32> {
	return vec4(mix(accumulated.rgb, lighting, weight / (accumulated.a + weight)), min(accumulated.a + weight, countMax));
}

/*
the most paths the texel of a cell holds while the light changes: the ones it gets in changeFrames frames.
With accumulationMax paths at every level the levels are behind the light by different times: a texel gets a sample in
the frames a cell picks a pixel in it, a texel of the level above gets those of the 4 texels below it. So a coarser
level followed the light 4 times faster per level, and within a level the texels that are small on screen were slower
than the large ones. The resolve blends 2 levels, so a moving shadow showed as copies of itself, the fresh one in the
blocks of the coarser texels.
Now every texel holds changeFrames frames of light: the fine texel the paths below, a coarser one 4 times that per
level (see trace_main), so they all are as far behind
 */
fn getChangingCountMax(point: CellPoint) -> f32 {
	let level = u32(getLevel(point.worldPosition, point.n));
	let toCamera = uniforms.cameraPosition - point.worldPosition;
	let d = max(length(toCamera), 1e-6);
	//the screen pixels the texel covers: less along the view direction at an angle
	let texelPixels = getTexelSize(level) * uniforms.pixelScale / (uniforms.pixelSize * d);
	let area = texelPixels * texelPixels * abs(dot(point.n, toCamera)) / d;
	//the chance per frame that a cell picks a pixel in the texel: 1 of the pixels of a cell per frame
	let rate = min(area / f32(CELL_SIZE * CELL_SIZE), 1.);
	return f32(uniforms.sampleCount) * max(uniforms.changeFrames * rate, CHANGING_SAMPLES_MIN);
}

/*
how much the dragged model can change the lighting of the point of a cell, 0 to 1. The model is a capsule: a sphere
around it, from where it was at the start of the drag to where it is now, so what it left counts too.
-the model itself: 1, its texels move with it
-in its sun shadow, however far: 1, for every surface there and not only the ones that face the sun. The distance to
 the capsule as seen from the sun, with a margin of its radius for the bounce light around the shadow
-else the part of the hemisphere of the point the sphere covers, times dragInfluence: what it can occlude or reflect.
 That is the square of its angular size times the cosine to it. The size is added to the cosine: a floor right next to
 the model sees its center near the horizon, but not the model
 */
fn getDragInfluence(point: CellPoint) -> f32 {
	let model = unpackModelTriangle(textureLoad(modelTexture, point.coord, 0).x).x;
	return getCapsuleInfluence(point.worldPosition, point.n, model, uniforms.dragFrom, uniforms.dragTo, uniforms.dragRadius, uniforms.dragModel);
}

//the influence of a model that went from start to end on the point p with normal n of the model model, see getDragInfluence
fn getCapsuleInfluence(p: vec3<f32>, n: vec3<f32>, model: u32, start: vec3<f32>, end: vec3<f32>, r: f32, dragModel: u32) -> f32 {
	//not local: the whole scene counts as changed. Every texel in view gets the short history during the drag, and every
	//texel is reset after it, when it is next used
	if(uniforms.dragLocal == 0u || model == dragModel){
		return 1.;
	}
	let ab = end - start;
	let ap = p - start;

	let t = clamp(dot(ap, ab) / max(dot(ab, ab), 1e-8), 0., 1.);
	let toModel = ab * t - ap;
	let d = max(length(toModel), 1e-6);
	let size = min(r / d, 1.);
	//not 0 for a surface that faces away from the model: it gets the light the surfaces around it bounce, which do face it
	let facing = clamp(dot(n, toModel) / d + size, DRAG_BOUNCE_FACING, 1.);
	let occlusion = size * size * facing * uniforms.dragInfluence;

	//the same without the part along the sun direction. Whatever the normal: a surface in the shadow that faces away
	//from the sun is lit by the bounce of the ones that face it, a floor that was in the sun lights what stands on it.
	//For the same reason up to a radius outside the shadow
	let l = uniforms.lightDirection;
	var shadow = 0.;
	let flatAB = ab - l * dot(ab, l);
	let flatAP = ap - l * dot(ap, l);
	let ts = clamp(dot(flatAP, flatAB) / max(dot(flatAB, flatAB), 1e-8), 0., 1.);
	let toCaster = ab * ts - ap;
	let ahead = dot(toCaster, l);
	//only what is between the point and the sun
	if(ahead > 0.){
		let side = length(toCaster - l * ahead);
		shadow = 1. - smoothstep(r, r * 2. + ahead * uniforms.sunRadius, side);
	}
	return min(max(occlusion, shadow), 1.);
}

/*
the most influence on a point of the strokes whose reset frame is after lastSampled: the drags the lighting that was
sampled then knows nothing of, or not all. A stroke is the whole drag, and it stays after the drag
 */
fn getStrokeInfluence(p: vec3<f32>, n: vec3<f32>, model: u32, lastSampled: u32) -> f32 {
	var influence = 0.;
	for(var i = 0u; i < DRAG_STROKE_COUNT; i++){
		var a = uniforms.dragStrokeA0;
		var b = uniforms.dragStrokeB0;
		switch i {
			case 1u: { a = uniforms.dragStrokeA1; b = uniforms.dragStrokeB1; }
			case 2u: { a = uniforms.dragStrokeA2; b = uniforms.dragStrokeB2; }
			case 3u: { a = uniforms.dragStrokeA3; b = uniforms.dragStrokeB3; }
			default: {}
		}
		if(u32(b.w) <= lastSampled){
			continue;
		}
		influence = max(influence, getCapsuleInfluence(p, n, model, a.xyz, b.xyz, a.w, u32(uniforms.dragStrokeModels[i])));
	}
	return influence;
}

//the most paths the texel of a cell holds. levels: a coarser texel holds 4 times that per level, see trace_main
struct CountMax {
	count : f32,
	levels : bool,
}

/*
-accumulationMax, a texel that holds them is resolved and takes no more
-while the sun moves the paths of changeFrames frames, see getChangingCountMax
-while a model is dragged those too, but only where it has influence: divided by the influence, so less of it is a
 longer history, up to accumulationMax where the texel is as without the drag. The rest of the scene stays resolved
 and traces nothing. After the drag the texels there are reset, see resetStaleTexel
 */
fn getCountMax(point: CellPoint) -> CountMax {
	if(uniforms.lightChanging != 0u){
		return CountMax(getChangingCountMax(point), true);
	}
	var result = CountMax(uniforms.accumulationMax, false);
	if(uniforms.dragState == 0u){
		return result;
	}
	let influence = getDragInfluence(point);
	if(influence <= 0.){
		return result;
	}
	let count = getChangingCountMax(point) / influence;
	if(count < uniforms.accumulationMax){
		result = CountMax(count, true);
	}
	return result;
}

//the sign bit of the sample count of a texel, which is never negative: set while a thread adds a sample to the texel
const TEXEL_LOCK = 0x80000000u;
//the times a thread tries to get the lock of a texel before its sample is dropped
const TEXEL_LOCK_TRIES = 64u;

//the texel at index, a texel index of getTexelIndex. Without the lock bit: a locked texel would read as a negative count
fn loadLighting(index: u32) -> vec4<f32> {
	return unpackLighting(vec2(atomicLoad(&lightingBuffer[index * 2u]), atomicLoad(&lightingBuffer[index * 2u + 1u]) & ~TEXEL_LOCK));
}

/*
adds a sample to a texel and returns the texel. A texel of a coarser level gets the samples of all the finer texels below it
in the same frame, 4 per level: as a plain read and write most of them got lost, the threads read the same count and the
last write won. So the texel is locked for the update, with the sign bit of its count.
The update is inside the loop, in the branch of the thread that got the lock: the threads of a workgroup run in step, so
one that leaves the loop with the lock waits there until the others leave it too, which they don't without the lock.
A new texel starts from prior, the texel of the level above, as at most seedCount paths.
When the lock is not got the sample is dropped and prior is returned.
weight: the paths of the sample, countMax: the paths the texel holds at most, see addSample
 */
fn addTexelSample(index: u32, lighting: vec3<f32>, weight: f32, prior: vec4<f32>, countMax: f32) -> vec4<f32> {
	let rg = &lightingBuffer[index * 2u];
	let ba = &lightingBuffer[index * 2u + 1u];
	var result = prior;
	for(var i = 0u; i < TEXEL_LOCK_TRIES; i++){
		let unlocked = atomicLoad(ba) & ~TEXEL_LOCK;
		if(atomicCompareExchangeWeak(ba, unlocked, unlocked | TEXEL_LOCK).exchanged){
			var texel = unpackLighting(vec2(atomicLoad(rg), unlocked));
			if(texel.a == 0. && prior.a > 0.){
				texel = vec4(prior.rgb, min(prior.a, uniforms.seedCount));
			}
			texel = addSample(texel, lighting, weight, countMax);
			let packed = packLighting(texel);
			atomicStore(rg, packed.x);
			//the count is positive, so this releases the lock
			atomicStore(ba, packed.y & ~TEXEL_LOCK);
			result = texel;
			break;
		}
	}
	return result;
}

//the feed texel of the hit point of the last getCachedLighting that found no lighting in the tiles of the cells.
//feedOn: there is a feed (rayFeedPaths), the rest is only set then
var<private> feedOn: bool;
//what the feed texel holds, alpha 0 when it has no paths or no tile yet
var<private> feedLighting: vec4<f32>;
//the lighting changes at the hit point now: the sun moves, or a dragged model is near
var<private> feedChanging: bool;
//its face and its tile (TILE_NONE when it has none yet), see getFeedTexel
var<private> feedSurface: Surface;
var<private> feedTile: u32;

//the part of the paths that go on from a feed texel that is full, and add their sample to it. So it keeps following
//the lighting of the texels its paths end in: the first samples of a texel are from when the feed around it was
//still empty, with less bounce light
const FEED_REFRESH = 1. / 16.;

//the frame a feed tile is marked as used in, see markUsed: the previous one, so it is not replaced by another tile for
//evictFrames, like a tile that is in view
fn getFeedStamp() -> u32 {
	return max(u32(uniforms.frame), 1u) - 1u;
}

/*
the lighting the lightmap has at a hit point: the texel at the level the camera sees it at, else the coarser ones.
The lightmap is a radiance cache for the paths of getIndirectPathSum, like the hash map of updateSpatialHash.wgsl.
The model of the hit is hitModelIndex of rayTraceScene.wgsl.
A tile that is out of view gets no samples of the cells, so after the light changed it holds the light of when it left:
the bounce light of what is behind the camera would stay that of an earlier time of the day. Such a tile is no hit.

The feed (rayFeedPaths): the cells only fill the tiles of what is in view, so a path that hits something else found
no lighting and went on for up to bounceCount bounces. The feed is a second set of tiles that the rays fill
themselves, for the hit points the tiles of the cells have nothing for. See getIndirectPathSum for how.
-in tiles of their own (FEED_LEVEL): a texel of a tile of the cells that the camera doesn't see (behind an edge, next
 to one that is seen) would hold the few noisy paths of the rays, and the resolve blends it with its neighbours
-at 1 level for the whole scene, see getFeedLevel: the level of the cells follows the distance to the camera, so a
 surface would get other feed tiles whenever the camera moves, each starting empty
When the tiles of the cells have no lighting this returns none, and sets the feed texel of the hit point: feedOn and
the vars with it. A feed tile is kept while rays look at it, and a stale one is reset
 */
fn getCachedLighting(p: vec3<f32>, n: vec3<f32>) -> vec4<f32> {
	feedOn = false;
	if(uniforms.cacheBounces == 0u){
		return vec4(0.);
	}
	let surface = getSurface(hitModelIndex, hitTriangle, p, n);
	let level = u32(getLevel(p, n));
	//only read when the light changed or a model was dragged: the times of the tiles of the cells are written by
	//claim_main, not in this pass
	let checkTime = uniforms.lightChangeEnd != 0u || uniforms.dragStrokeEnd != 0u;
	for(var i = 0u; i <= min(uniforms.ancestorLevels, ANCESTOR_LEVELS_MAX); i++){
		let texel = getTexel(surface, level + i);
		let tileIndex = find(surface.id, level + i, texel / TILE_SIZE);
		if(tileIndex != TILE_NONE){
			if(checkTime){
				let used = atomicLoad(&tileTimeBuffer[tileIndex]);
				if(uniforms.lightChangeEnd != 0u && getStaleFrames(used) > uniforms.changeFrames * STALE_BOUNCE){
					continue;
				}
				//out of view since a drag that changed the lighting here, see markUsed
				if(used < uniforms.dragStrokeEnd && getStrokeInfluence(p, n, hitModelIndex, used) > DRAG_BOUNCE_INFLUENCE){
					continue;
				}
			}
			let lighting = loadLighting(getTexelIndex(tileIndex, texel));
			if(lighting.a > 0.){
				return lighting;
			}
		}
	}
	if(uniforms.rayFeedPaths <= 0.){
		return vec4(0.);
	}

	feedOn = true;
	feedSurface = surface;
	feedLighting = vec4(0.);
	feedChanging = uniforms.lightChanging != 0u;
	if(!feedChanging && uniforms.dragState != 0u){
		feedChanging = getCapsuleInfluence(p, n, hitModelIndex, uniforms.dragFrom, uniforms.dragTo, uniforms.dragRadius, uniforms.dragModel) > DRAG_BOUNCE_INFLUENCE;
	}
	let feedLevel = getFeedLevel();
	let texel = getTexel(surface, feedLevel);
	feedTile = find(surface.id, feedLevel | FEED_LEVEL, texel / TILE_SIZE);
	if(feedTile == TILE_NONE){
		return vec4(0.);
	}
	if(checkTime){
		let used = atomicLoad(&tileTimeBuffer[feedTile]);
		var stale = uniforms.lightChangeEnd != 0u && getStaleFrames(used) > uniforms.changeFrames * STALE_BOUNCE;
		if(!stale && used < uniforms.dragStrokeEnd){
			stale = getStrokeInfluence(p, n, hitModelIndex, used) > DRAG_BOUNCE_INFLUENCE;
		}
		//its texels keep their lighting as a start, as 1 path. Only by the thread that replaces the old time
		if(stale && atomicExchange(&tileTimeBuffer[feedTile], getFeedStamp()) == used){
			ageTile(feedTile, 0.);
		}
	}
	//the tile stays while rays look at it
	markUsed(feedTile, getFeedStamp());
	feedLighting = loadLighting(getTexelIndex(feedTile, texel));
	return vec4(0.);
}

//the feed texel of the hit point of the last getCachedLighting, which set feedOn, to add a sample to. Its tile is added
//when it is not in the lightmap yet. TILE_NONE when the lightmap is full there
fn getFeedTexel() -> u32 {
	let feedLevel = getFeedLevel();
	let texel = getTexel(feedSurface, feedLevel);
	var tileIndex = feedTile;
	if(tileIndex == TILE_NONE){
		tileIndex = findOrInsert(feedSurface.id, feedLevel | FEED_LEVEL, texel / TILE_SIZE, getFeedStamp());
	}
	return select(getTexelIndex(tileIndex, texel), TILE_NONE, tileIndex == TILE_NONE);
}

/*
the sum of the paths first, first + stride, first + 2 * stride.. of the sampleCount paths of getIndirectLighting: the paths
of getIndirectPath of pathTrace.wgsl, but as 1 loop over the rays of all of them instead of a loop over the bounces in a
loop over the paths.
The threads of a wave wait at the end of a loop on the ones that are still in it. Most paths end at their first hit, in
the cache, so in the loop over the bounces all of them waited on the few that go on. Here a thread whose path ended
starts on its next path in the same step, while the others bounce.
The sun ray of a hit is a step of the loop too, a traceRay to the sky instead of isOccluded: that takes more steps
through the BVH as it doesn't stop at the first hit, but nothing waits on it.
This is the only traceRay call site, like sampleRay: use the one or the other in a shader.

A path ends at the first hit that the tiles of the cells have lighting for (getCachedLighting). Where they have none:
-without the feed (rayFeedPaths 0) it goes on like sampleRay, for up to bounceCount bounces
-with the feed it is at most 2 rays and a sun ray, whatever bounceCount:
 1 the first hit ends in its feed texel, with the chance of the part of rayFeedPaths paths that texel holds. Else the
   path feeds it: it goes on for 1 more ray (the sun sample with directionalProbability, else a bounce), and the light
   that brings to the hit, without its albedo, is added to the texel as a sample of 1 path. Like the paths of a cell are
   for the texel of the cell. A full texel still feeds FEED_REFRESH of its paths, and all of them while the light
   changes there. The tile of the texel is added here
 2 the second hit ends in its feed texel when that holds anything, however few paths: the noise averages out in the
   texel of the first hit. Else in its sun light only, the sun ray times directionalProbability, without a bounce.
   No tile is added here: the feed is what the surfaces in view see, not the whole scene
 So the light of a surface that neither the camera nor a surface in view sees is its sun light only, the rest has all
 its bounces: a feed texel is fed by paths that end in the other texels, over the frames
 */
fn getIndirectPathSum(p: vec3<f32>, n: vec3<f32>, seed: vec2<u32>, first: u32, stride: u32, sampleCount: u32) -> vec3<f32>{
	var sum = vec3(0.);
	if(first >= sampleCount || uniforms.bounceCount <= 0){
		return sum;
	}
	let sunColor = uniforms.directionalColor * uniforms.directionalBrightness;
	var i = first;
	var ro = p + n * MIN_DIST;
	var rd = getIndirectDirection(n, seed, i, sampleCount);
	var col = vec3(1.);
	var bounce = 0;
	//the ray is the sun ray of the last hit of the path
	var sunRay = false;
	//the texel the path feeds, and what the path lets through from that hit on: col without what is before it
	var feedTexel = TILE_NONE;
	var feedCol = vec3(1.);

	loop {
		let hit = traceRay(ro + rd * MIN_DIST, rd);
		let hitPoint = ro + rd * hit.distance;
		let sky = isSky(hitPoint);
		var ended = true;
		//the light the path ends in
		var light = vec3(0.);
		if(sunRay){
			//col has the cosine at the hit
			if(sky){
				light = sunColor;
			}
		}else if(sky){
			light = hit.color;
		}else{
			let hitNormal = hit.normal;
			col *= hit.color;
			feedCol *= hit.color;
			let cached = getCachedLighting(hitPoint, hitNormal);
			if(cached.a > 0.){
				light = cached.rgb;
			}else{
				let r = hash44ff(vec4(hitPoint, f32(i)));
				//the path goes on with a sun ray that counts for this much, or with a bounce
				var sunWeight = 0.;
				var bounces = false;
				if(!feedOn){
					if(r.z < uniforms.directionalProbability){
						sunWeight = 1.;
					}else{
						bounce++;
						bounces = bounce < uniforms.bounceCount;
					}
				}else if(bounce == 0){
					var endChance = 0.;
					if(!feedChanging){
						endChance = min(feedLighting.a / uniforms.rayFeedPaths, 1. - FEED_REFRESH);
					}
					if(r.w < endChance){
						light = feedLighting.rgb;
					}else{
						feedTexel = getFeedTexel();
						feedCol = vec3(1.);
						if(r.z < uniforms.directionalProbability){
							sunWeight = 1.;
						}else{
							bounce = 1;
							bounces = true;
						}
					}
				}else if(feedLighting.a > 0.){
					light = feedLighting.rgb;
				}else{
					sunWeight = uniforms.directionalProbability;
				}

				if(sunWeight > 0.){
					let l = sampleSunDirection(r.xy);
					let cosine = dot(hitNormal, l);
					//a hit that faces away from the sun gets no sun ray
					if(cosine > 0.){
						col *= cosine * sunWeight;
						feedCol *= cosine * sunWeight;
						ro = hitPoint + hitNormal * MIN_DIST;
						rd = l;
						sunRay = true;
						ended = false;
					}
				}else if(bounces){
					ro = hitPoint;
					rd = cosWeightedRandomHemisphereDirection(hitNormal, r.xy);
					ended = false;
				}
			}
		}

		if(ended){
			sum += col * light;
			if(feedTexel != TILE_NONE){
				addTexelSample(feedTexel, feedCol * light, 1., vec4(0.), uniforms.rayFeedPaths);
				feedTexel = TILE_NONE;
			}
			i += stride;
			if(i >= sampleCount){
				break;
			}
			ro = p + n * MIN_DIST;
			rd = getIndirectDirection(n, seed, i, sampleCount);
			col = vec3(1.);
			bounce = 0;
			sunRay = false;
		}
	}
	return sum;
}


//the pixel a cell samples this frame and the surface point it shows. valid is false for the sky and outside the screen
struct CellPoint {
	valid : bool,
	coord : vec2<u32>,
	worldPosition : vec3<f32>,
	n : vec3<f32>,
}

fn getCellPoint(cellID: vec2<u32>) -> CellPoint {
	var point: CellPoint;
	point.valid = false;
	//not the first 2 numbers of the hash, which were the white noise of the paths when the cell was their seed
	let rand = hash34uf(vec3<u32>(cellID, u32(uniforms.frame))).zw;
	//random pixel in the cell, the same one in every pass of the frame
	point.coord = cellID * CELL_SIZE + vec2<u32>(rand * f32(CELL_SIZE));
	if(point.coord.x >= u32(uniforms.resolution.x) || point.coord.y >= u32(uniforms.resolution.y)){
		return point;
	}

	let normalDepth = loadNormalDepth(normalDepthTexture, point.coord);
	point.n = normalDepth.xyz;
	let uv = (vec2<f32>(point.coord) + .5) / uniforms.resolution;
	point.worldPosition = depthToPosition(normalDepth.w, uv * 2. - 1.);
	//the sky gets no texels, it is drawn directly (see resolveVirtualLightmap.wgsl)
	point.valid = !isSky(point.worldPosition);
	return point;
}

//the texels of the point of a cell at its level and the coarser ones, as texel indices of getTexelIndex. TILE_NONE where
//the lightmap is full. insert: claim_main adds the tiles that don't exist yet and marks them used, trace_main only finds them
fn getCellTexels(point: CellPoint, insert: bool) -> array<u32, ANCESTOR_LEVELS_MAX + 1u> {
	//only tiles with a pixel in view get a slot, so a large face (a floor) takes no more of the lightmap than what is seen of it
	let model = unpackModelTriangle(textureLoad(modelTexture, point.coord, 0).x);
	let surface = getSurface(model.x, model.y, point.worldPosition, point.n);
	let level = u32(getLevel(point.worldPosition, point.n));
	cellPosition = point.worldPosition;
	cellNormal = point.n;
	cellModel = model.x;
	var indices: array<u32, ANCESTOR_LEVELS_MAX + 1u>;
	for(var i = 0u; i <= ANCESTOR_LEVELS_MAX; i++){
		indices[i] = TILE_NONE;
	}
	for(var i = 0u; i <= min(uniforms.ancestorLevels, ANCESTOR_LEVELS_MAX); i++){
		let texel = getTexel(surface, level + i);
		var tileIndex: u32;
		if(insert){
			tileIndex = findOrInsert(surface.id, level + i, texel / TILE_SIZE, u32(uniforms.frame));
		}else{
			tileIndex = find(surface.id, level + i, texel / TILE_SIZE);
		}
		indices[i] = select(getTexelIndex(tileIndex, texel), TILE_NONE, tileIndex == TILE_NONE);
	}
	return indices;
}

/*
the seed of the random numbers of the paths of a cell (see getBlueNoise in pathTrace.wgsl): the texel it traces for, not
the cell. Blue noise spreads the error over neighbouring seeds, and neighbouring texels are what the resolve blends. With
the cell as seed that was lost: which cell samples a texel changes every frame, so neither the pattern over the texels nor
the R2 sequence of a texel held. Now a texel walks 1 R2 sequence, by the samples it took, see getTexelSampleNumber.
texelIndex: the texel claim_main found, still in the slot of the cell (only the trace slots before them are written
after it). Its position in the lightmap, with the tile slots in rows of 32: getBlueNoise wraps the seed at the noise
size, 32 tiles of 8 texels for the 256 of the texture, so each tile gets its own window of it. The tiles are at hashed
slots, so the pattern is continuous within a tile and white across the tile borders. That instead of the texel on its
surface, which takes getSurface again in every lane
 */
fn getTexelSeed(texelIndex: u32) -> vec2<u32> {
	let t = vec2(texelIndex % TILE_SIZE, (texelIndex / TILE_SIZE) % TILE_SIZE);
	let tile = texelIndex / (TILE_SIZE * TILE_SIZE);
	return t + vec2(tile % 32u, tile / 32u) * TILE_SIZE;
}

/*
the sampleNumber of pathTrace.wgsl for a texel: the samples it holds, as addSample adds sampleCount to its count per
sample. A texel only gets a sample in the frames a cell picks a pixel in it (about 40% of them at texelsPerPixel .4 and
cells of 4 x 4 pixels), so with the frame as the number it jumped through its R2 sequence at random: random points, the
blue noise was about as good as white noise.
The frame when the count is held at its cap (countMax of getCountMax: the light changes, a model is dragged) or would
reach it with this sample: the count then stops going up, and the paths would be the same every frame
 */
fn getTexelSampleNumber(texelIndex: u32, countMax: f32) -> i32 {
	let count = loadLighting(texelIndex).a;
	let weight = f32(uniforms.sampleCount);
	if(count + weight > countMax){
		return -1;
	}
	return i32(count / weight);
}

//the slot in traceCells of the texel a cell bids on: after the slots of the cells that trace, the cells of the dispatch
//of claim_main in rows. See VirtualLightmap.update
fn getCellTexelSlot(cellID: vec2<u32>) -> u32 {
	let groupSize = vec2(WORK_GROUP_SIZE_X, WORK_GROUP_SIZE_Y) * CELL_SIZE;
	let cellCount = (vec2<u32>(uniforms.resolution) + groupSize - 1u) / groupSize * vec2(WORK_GROUP_SIZE_X, WORK_GROUP_SIZE_Y);
	return cellCount.x * cellCount.y + cellID.y * cellCount.x + cellID.x;
}

//the bits of a claim below the frame, and the mask of the frame: 20 bits, so it wraps after 4.8 hours at 60 fps
const CLAIM_FRAME_SHIFT = 12u;
const CLAIM_FRAME_MASK = 0xFFFFFu;

//the frame in the high bits, so a claim of an earlier frame never wins. The low bits are random per cell and frame, so
//every cell of a texel is as likely to win: the order the GPU runs the cells in doesn't favour a corner of the texel
fn getClaimKey(cellID: vec2<u32>) -> u32 {
	let frame = u32(uniforms.frame);
	return (frame << CLAIM_FRAME_SHIFT) | (tileHash(cellID.x ^ tileHash(cellID.y ^ tileHash(frame))) & ((1u << CLAIM_FRAME_SHIFT) - 1u));
}

//the path count of a texel is at most countMax, which is not 0 so it keeps its lighting as a start. Like ageTile
fn capTexel(index: u32, countMax: f32) {
	let ba = &lightingBuffer[index * 2u + 1u];
	let v = unpack2x16float(atomicLoad(ba));
	if(v.y > countMax){
		atomicStore(ba, pack2x16float(vec2(v.x, countMax)));
	}
}

/*
a model that was dragged changed the lighting of the texels in its influence, also of the ones that got no samples
then: out of view, behind something, or in view but small on screen or at a grazing angle, so a cell rarely picks a
pixel in them. Such a texel was resolved with the lighting of before the drag, and stayed that way.
The frame part of claimBuffer is the frame a texel was last sampled in: a cell that traces bids on its texel, and
marks the coarser ones above it (see claim_main). A texel that was last sampled before the reset frame of a stroke
that has influence on it is reset here, when a cell uses it again:
-after a drag that is every texel in its influence, once: also the ones that followed the drag, their lighting is
 that of its last frames and not of where the model ended
-during a drag only the texels that had no sample for a while. The others follow it, see getCountMax
It keeps DRAG_RETURN_PATHS paths where the influence is 1, and more by the third power where it is less: the texels
around a small model keep most of what they have, a texel in a shadow that came or went starts over.
The cell then bids on its texel, which is no longer resolved, so the frames of these texels are after the reset frame
and this happens once. Called for the texels of the point of getCellTexels, by claim_main
 */
fn resetStaleTexel(index: u32) {
	let frame = u32(uniforms.frame);
	let claim = atomicLoad(&claimBuffer[index]);
	//0: never sampled
	var lastSampled = 0u;
	if(claim != 0u){
		lastSampled = frame - ((frame - (claim >> CLAIM_FRAME_SHIFT)) & CLAIM_FRAME_MASK);
	}
	if(lastSampled >= uniforms.dragStrokeEnd){
		return;
	}
	let influence = getStrokeInfluence(cellPosition, cellNormal, cellModel, lastSampled);
	if(influence > 0.){
		capTexel(index, DRAG_RETURN_PATHS / (influence * influence * influence));
	}
}

/*
the update is 3 passes, so a texel gets its paths from 1 cell per frame instead of from every cell that sees it. Where a
texel covers many pixels (low texelsPerPixel, or close to a surface at texelSizeMin) the cells used to all trace, and
most of their samples were lost in the race on the texel anyway.
1 claim_main: every cell inserts its tiles and bids on its texel, which it keeps for select_main. A texel that holds
  accumulationMax paths is resolved and gets no bid, unless the light changes there (the day cycle, or a dragged model
  near it: see getCountMax)
2 select_main: the cell with the highest bid of a texel goes in traceCells
3 trace_main: only those cells trace, packed together, so no thread of a workgroup waits on the paths of another
 */
@compute @workgroup_size(WORK_GROUP_SIZE_X, WORK_GROUP_SIZE_Y, 1)
fn claim_main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
	let cellID = GlobalInvocationID.xy;
	let texelSlot = getCellTexelSlot(cellID);
	//until the cell bids
	traceCells[texelSlot] = TILE_NONE;
	let point = getCellPoint(cellID);
	if(!point.valid){
		return;
	}
	let indices = getCellTexels(point, true);
	let levels = min(uniforms.ancestorLevels, ANCESTOR_LEVELS_MAX);
	//before the count of the texel is read below: a texel that is reset is not resolved
	if(uniforms.dragStrokeEnd != 0u){
		for(var i = 0u; i <= levels; i++){
			if(indices[i] != TILE_NONE){
				resetStaleTexel(indices[i]);
			}
		}
	}
	let texelIndex = indices[0];
	//the lightmap is full here
	if(texelIndex == TILE_NONE){
		return;
	}
	//resolved, and not held below accumulationMax by the sun or a dragged model. The tiles were still marked as used above
	if(uniforms.lightChanging == 0u && loadLighting(texelIndex).a >= uniforms.accumulationMax
		&& getCountMax(point).count >= uniforms.accumulationMax){
		return;
	}
	traceCells[texelSlot] = texelIndex;

	//the texel gets a sample this frame, from this cell or from the one that wins the bid, and the coarser texels above
	//it get that sample too: the frame of their claim is the frame they were last sampled in, see resetStaleTexel. Not
	//over a claim of this frame, which can be the bid of a cell that has the texel as its own
	let stamp = (u32(uniforms.frame) & CLAIM_FRAME_MASK) << CLAIM_FRAME_SHIFT;
	for(var i = 1u; i <= levels; i++){
		if(indices[i] != TILE_NONE){
			let ancestorClaim = &claimBuffer[indices[i]];
			let ancestorOld = atomicLoad(ancestorClaim);
			if((ancestorOld >> CLAIM_FRAME_SHIFT) != (stamp >> CLAIM_FRAME_SHIFT)){
				atomicCompareExchangeWeak(ancestorClaim, ancestorOld, stamp);
			}
		}
	}

	//atomicMax within the frame: a bid of an earlier frame is replaced whatever its value. A spurious failure retries
	let key = getClaimKey(cellID);
	let claim = &claimBuffer[texelIndex];
	var old = atomicLoad(claim);
	for(var i = 0u; i < 16u; i++){
		if((old >> CLAIM_FRAME_SHIFT) == (key >> CLAIM_FRAME_SHIFT) && old >= key){
			return;
		}
		let cmp = atomicCompareExchangeWeak(claim, old, key);
		if(cmp.exchanged){
			return;
		}
		old = cmp.old_value;
	}
}

@compute @workgroup_size(WORK_GROUP_SIZE_X, WORK_GROUP_SIZE_Y, 1)
fn select_main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
	let cellID = GlobalInvocationID.xy;
	//the texel claim_main found: the cells that lose or didn't bid stop here, without a look at the gBuffer or the lightmap
	let texelIndex = traceCells[getCellTexelSlot(cellID)];
	let key = getClaimKey(cellID);
	if(texelIndex == TILE_NONE || atomicLoad(&claimBuffer[texelIndex]) != key){
		return;
	}
	//at most traceCellsMax cells trace: a view of texels that are all new (the camera turns fast) would trace every cell
	//in 1 frame. A cell that is left out traced nothing and its texel is as it was, so it bids again next frame.
	//The cells over the limit are the last ones to get here, which is the same part of the screen every frame. So when
	//the previous frame had more winners than the limit, a random part of them is left out first, as many as were too
	//much then: the limit itself only cuts off what is left, the frame the count jumps and the noise of the random part
	let parity = u32(uniforms.frame) & 1u;
	atomicAdd(&traceArgs[4u + parity], 1u);
	let previous = atomicLoad(&traceArgs[5u - parity]);
	//TODO: consider writing out uniforms.traceCellsMax
	if(previous > uniforms.traceCellsMax){
		let random = f32(tileHash(key ^ 0x9E3779B9u) >> 8u) / 16777216.;
		if(random * f32(previous) >= f32(uniforms.traceCellsMax)){
			return;
		}
	}
	let index = atomicAdd(&traceArgs[3], 1u);
	if(index >= uniforms.traceCellsMax){
		return;
	}
	traceCells[index] = cellID.x | (cellID.y << 16u);
	//the workgroups of trace_main, see VirtualLightmap.update
	if(index % (TRACE_GROUP_SIZE / uniforms.traceLanes) == 0u){
		atomicAdd(&traceArgs[0], 1u);
	}
	if(index == 0u){
		atomicStore(&traceArgs[1], 1u);
		atomicStore(&traceArgs[2], 1u);
	}
}

//the paths of a cell summed per thread, see trace_main
var<workgroup> pathSums: array<vec3<f32>, TRACE_GROUP_SIZE>;
//the sun sample of a thread, see trace_main
var<workgroup> sunSamples: array<vec3<f32>, TRACE_GROUP_SIZE>;

//traceLanes threads per cell, which each trace every traceLanes-th of its sampleCount paths: the paths are sequential
//in a thread, so 1 thread per cell makes a frame with few cells to trace (most texels resolved) wait on the latency of
//all the paths of a cell. The same paths as getLighting, the sum is the same. See getIndirectPathSum for how a lane
//traces its paths.
//Every lane also traces a sun sample, traceLanes per cell instead of the 1 of getLighting: the threads of a wave wait
//on a shadow ray of 1 of them, so the rays of the others take no extra time and the penumbra gets less noise
@compute @workgroup_size(TRACE_GROUP_SIZE, 1, 1)
fn trace_main(
	@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>,
	@builtin(local_invocation_index) localIndex : u32
) {
	let lanes = uniforms.traceLanes;
	let cell = GlobalInvocationID.x / lanes;
	let lane = GlobalInvocationID.x % lanes;
	//the count went on past the limit, see select_main
	var hasCell = cell < min(atomicLoad(&traceArgs[3]), uniforms.traceCellsMax);
	var cellID = vec2(0u);
	var point: CellPoint;
	if(hasCell){
		let packed = traceCells[cell];
		cellID = vec2(packed & 0xFFFFu, packed >> 16u);
		//the surface point only: the texels are looked up below, by the lane that writes them
		point = getCellPoint(cellID);
		hasCell = point.valid;
	}
	var sum = vec3(0.);
	var sun = vec3(0.);
	//while the light changes the same frames of light at every level instead of the same paths, see getChangingCountMax
	//or where a dragged model changes it. Before the paths, for getTexelSampleNumber
	var cellCountMax: CountMax;
	if(hasCell){
		cellCountMax = getCountMax(point);
		let texelIndex = traceCells[getCellTexelSlot(cellID)];
		let seed = getTexelSeed(texelIndex);
		sampleNumber = getTexelSampleNumber(texelIndex, cellCountMax.count);
		sum = getIndirectPathSum(point.worldPosition, point.n, seed, lane, lanes, uniforms.sampleCount);
		sun = getSeededDirectSample(point.worldPosition, point.n, seed, lane, lanes);
	}
	pathSums[localIndex] = sum;
	sunSamples[localIndex] = sun;
	workgroupBarrier();
	if(!hasCell || lane != 0u){
		return;
	}
	var indirect = vec3(0.);
	var direct = vec3(0.);
	for(var i = 0u; i < lanes; i++){
		indirect += pathSums[localIndex + i];
		direct += sunSamples[localIndex + i];
	}
	let lighting = combineLighting(indirect / f32(uniforms.sampleCount), direct / f32(lanes));
	let indices = getCellTexels(point, false);

	//1 cell per texel of the finest level, but a coarser texel gets the samples of the finer texels below it, which all
	//count: see addTexelSample.
	//From coarse to fine: the coarser texels cover more pixels, so they converge faster. A new texel starts from the lighting of the
	//coarser one, as seedCount paths, instead of from its first noisy sample. The resolve uses them as the prior of the finer ones.
	//The coarser levels are also what is shown when the surface gets further away, so nothing has to be copied to half the resolution
	let ancestorLevels = min(uniforms.ancestorLevels, ANCESTOR_LEVELS_MAX);
	let changing = cellCountMax.levels;
	let countMax = cellCountMax.count;
	var prior = vec4(0.);
	for(var i = i32(ancestorLevels); i >= 0; i--){
		let index = indices[i];
		if(index == TILE_NONE){
			continue;
		}
		//a coarser texel gets 4 times the samples per level
		let levelCountMax = select(countMax, min(countMax * f32(1u << (2u * u32(i))), CHANGING_COUNT_MAX), changing);
		prior = addTexelSample(index, lighting, f32(uniforms.sampleCount), prior, levelCountMax);
	}
}

/*
the light stopped changing: the resolved texels hold the lighting of the last frames of the change, the average lags
behind it. They go back to refreshCount paths, so they keep that lighting as a start but take samples again until
they hold accumulationMax of the light as it is now. Per texel of the whole lightmap, also the ones out of view.
Once resolved again a texel keeps refreshCount / accumulationMax of the lagging lighting for good, a faint ghost of where
the shadows were. So this runs a second time when the texels in view are resolved again (see VirtualLightmap.update),
which leaves the square of that part. A texel that then still holds refreshCount paths or less took no samples in
between, it is out of view: it goes back to refreshCountIdle, which is lower, as its noise is not seen. The first time
refreshCountIdle is refreshCount, which leaves those texels as they are
 */
@compute @workgroup_size(WORK_GROUP_SIZE_X, WORK_GROUP_SIZE_Y, 1)
fn refresh_main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
	let side = LIGHTMAP_SIZE;
	if(GlobalInvocationID.x >= side || GlobalInvocationID.y >= side){
		return;
	}
	let index = GlobalInvocationID.y * side + GlobalInvocationID.x;
	var lighting = loadLighting(index);
	let count = select(uniforms.refreshCount, uniforms.refreshCountIdle, lighting.a <= uniforms.refreshCount);
	if(lighting.a > count){
		lighting.a = count;
		//the count is in the second word
		atomicStore(&lightingBuffer[index * 2u + 1u], packLighting(lighting).y);
	}
}
