
struct Model {
	worldMatrix : mat4x4<f32>,
	inverseModel : mat4x4<f32>,
	quaternion : vec4<f32>,
	color: vec3<f32>,
	//sphere = 0
	//cube = 1
	//cylinder = 2
	//mesh = 6, the triangles of meshBuffer
	primitiveType : f32,
	//the lengths of the axes of worldMatrix, see Scene.writeModel
	scale : vec3<f32>,
}

//the boxes of the 2 children of a node. A child is a node index, or a model index | BVH_LEAF. See ModelBVH.ts
struct BVHNode {
	aMin : vec3<f32>,
	a : u32,
	aMax : vec3<f32>,
	b : u32,
	bMin : vec3<f32>,
	bMax : vec3<f32>,
}

struct HitResult{
	color : vec3<f32>,
	normal : vec3<f32>,
	distance: f32
}

const MIN_DIST: f32 = 0.0001;
const MAX_DIST: f32 = 1e10;
//enough for the building (BVH depth 16, so at most 15 entries), see Scene.init. The stack length matters even in compute:
//32 instead of 16 made updateOctree 1.09x slower (dawn.node, D3D12)
const BVH_STACK_SIZE = 16u;
const BVH_LEAF = 0x80000000u;
//a stack entry is the distance to the node with the low bits replaced by the node index, see traceBVH
const BVH_STACK_INDEX_MASK = 0xffffu;
//the sky sphere is the last model and encloses the scene, so every ray hits something.
//Points beyond this fraction of its radius are on the sky. The farthest corner of the building is .5 m inside the sky,
//about .984 of its radius (see Scene.skyRadius): at .95 the far corners of the building counted as sky
const SKY_RADIUS = .99;

//true: traceRay keeps the model of its last hit in hitModelIndex, for a cache that is keyed on the model: see getCachedLighting
//in updateVirtualLightmap.wgsl. A pipeline constant, so the shaders that leave it false compile without the stores
override TRACK_HIT_MODEL: bool = false;
var<private> hitModelIndex: u32;
//the triangle of the last hit when that is a mesh, with TRACK_HIT_MODEL
var<private> hitTriangle: u32;

const PRIMITIVE_MESH = 3u;
//the triangle mesh of the mesh models, see LightmapMesh.ts. Declared here instead of in the including shaders, at a
//binding they leave free. Bind it with Scene.bind
@group(0) @binding(16) var<storage, read> meshBuffer : array<vec4<f32>>;
//vec4s per triangle, see getMeshTriangle
const MESH_TRIANGLE_SIZE = 5u;
//MESH_STACK_SIZE of LightmapMesh.ts
const MESH_STACK_SIZE = 32u;
const MESH_NONE = 0xFFFFFFFFu;
//the closest triangle of the last traceMesh
var<private> meshHitTriangle: u32;

struct MeshTriangle {
	v0 : vec3<f32>,
	v1 : vec3<f32>,
	v2 : vec3<f32>,
	//the UV island, see LightmapMesh.createCharts
	chart : u32,
	//the corners in the chart, in local units
	c0 : vec2<f32>,
	c1 : vec2<f32>,
	c2 : vec2<f32>,
}

fn getMeshTriangleOffset(triangle: u32) -> u32 {
	return bitcast<u32>(meshBuffer[0].y) + triangle * MESH_TRIANGLE_SIZE;
}

//a triangle: v0 + chart, v1 + n0, v2 + n1, c0 + c1, c2 + n2. The normals octahedral, see LightmapMesh.packNormal
fn getMeshTriangle(triangle: u32) -> MeshTriangle {
	let o = getMeshTriangleOffset(triangle);
	let a = meshBuffer[o];
	let c = meshBuffer[o + 3u];
	var t: MeshTriangle;
	t.v0 = a.xyz;
	t.v1 = meshBuffer[o + 1u].xyz;
	t.v2 = meshBuffer[o + 2u].xyz;
	t.chart = bitcast<u32>(a.w);
	t.c0 = c.xy;
	t.c1 = c.zw;
	t.c2 = meshBuffer[o + 4u].xy;
	return t;
}

fn meshDecodeNormal(packed: f32) -> vec3<f32> {
	let e = unpack2x16snorm(bitcast<u32>(packed));
	var n = vec3(e, 1. - abs(e.x) - abs(e.y));
	if(n.z < 0.){
		n = vec3((1. - abs(n.yx)) * select(vec2(-1.), vec2(1.), n.xy >= vec2(0.)), n.z);
	}
	return normalize(n);
}

//the local position (xyz) and the local normal (w, packed) of a corner of a triangle, for drawing the mesh
fn getMeshVertex(triangle: u32, corner: u32) -> vec4<f32> {
	let o = getMeshTriangleOffset(triangle);
	//the normal of v0 is next to v1, of v1 next to v2
	let normal = select(meshBuffer[o + corner + 1u].w, meshBuffer[o + 4u].z, corner == 2u);
	return vec4(meshBuffer[o + corner].xyz, normal);
}

//the size of a chart in local units
fn getMeshChartSize(chart: u32) -> vec2<f32> {
	return meshBuffer[bitcast<u32>(meshBuffer[0].z) + chart].xy;
}

//the barycentrics (v1, v2) of the point on the triangle closest to p, near enough when p is on its plane
fn getMeshBarycentrics(t: MeshTriangle, p: vec3<f32>) -> vec2<f32> {
	let e1 = t.v1 - t.v0;
	let e2 = t.v2 - t.v0;
	let w = p - t.v0;
	let d00 = dot(e1, e1);
	let d01 = dot(e1, e2);
	let d11 = dot(e2, e2);
	let d20 = dot(w, e1);
	let d21 = dot(w, e2);
	let den = d00 * d11 - d01 * d01;
	if(den <= 0.){
		return vec2(0.);
	}
	var b = max(vec2(d11 * d20 - d01 * d21, d00 * d21 - d01 * d20) / den, vec2(0.));
	let sum = b.x + b.y;
	if(sum > 1.){
		b /= sum;
	}
	return b;
}

//the distance along the ray (x) and the barycentrics (yz) of the hit, x is MAX_DIST on a miss. Moeller-Trumbore, both sides
fn intersectTriangle(ro: vec3<f32>, rd: vec3<f32>, triangle: u32) -> vec3<f32> {
	let o = getMeshTriangleOffset(triangle);
	let v0 = meshBuffer[o].xyz;
	let e1 = meshBuffer[o + 1u].xyz - v0;
	let e2 = meshBuffer[o + 2u].xyz - v0;
	let pv = cross(rd, e2);
	let det = dot(e1, pv);
	if(abs(det) < 1e-12){
		return vec3(MAX_DIST, 0., 0.);
	}
	let invDet = 1. / det;
	let tv = ro - v0;
	let u = dot(tv, pv) * invDet;
	let qv = cross(tv, e1);
	let v = dot(rd, qv) * invDet;
	if(u < 0. || v < 0. || u + v > 1.){
		return vec3(MAX_DIST, 0., 0.);
	}
	return vec3(dot(e2, qv) * invDet, u, v);
}

//the local normal (xyz) and the local distance (w) of the closest triangle, w is MAX_DIST on a miss. Keeps the triangle in
//meshHitTriangle. The normal is the smooth one, on the side the ray comes from: a mesh doesn't have to be closed.
//The BVH is in local space, see LightmapMesh.buildBVH: the near child is visited next and the far one pushed
fn traceMesh(ro: vec3<f32>, rd: vec3<f32>, distBound: vec2<f32>) -> vec4<f32> {
	let nodeOffset = bitcast<u32>(meshBuffer[0].x);
	let invRd = getInverseDirection(rd);
	let roInvRd = ro * invRd;

	var stack: array<u32, MESH_STACK_SIZE>;
	var stackSize = 0u;
	var node = 0u;
	var best = vec3(distBound.y, 0., 0.);
	var bestTriangle = MESH_NONE;
	loop {
		let a = meshBuffer[nodeOffset + node * 2u];
		let b = meshBuffer[nodeOffset + node * 2u + 1u];
		let leftFirst = bitcast<u32>(a.w);
		let count = bitcast<u32>(b.w);
		var pop = true;
		if(count > 0u){
			for(var i = 0u; i < count; i++){
				let hit = intersectTriangle(ro, rd, leftFirst + i);
				if(hit.x >= distBound.x && hit.x < best.x){
					best = hit;
					bestTriangle = leftFirst + i;
				}
			}
		}else{
			let l = meshBuffer[nodeOffset + leftFirst * 2u];
			let lMax = meshBuffer[nodeOffset + leftFirst * 2u + 1u];
			let r = meshBuffer[nodeOffset + leftFirst * 2u + 2u];
			let rMax = meshBuffer[nodeOffset + leftFirst * 2u + 3u];
			let tL = intersectAABB(roInvRd, invRd, l.xyz, lMax.xyz);
			let tR = intersectAABB(roInvRd, invRd, r.xyz, rMax.xyz);
			let dL = select(MAX_DIST, max(tL.x, 0.), tL.y >= max(tL.x, 0.) && tL.x < best.x);
			let dR = select(MAX_DIST, max(tR.x, 0.), tR.y >= max(tR.x, 0.) && tR.x < best.x);
			if(min(dL, dR) < MAX_DIST){
				let leftNear = dL <= dR;
				node = select(leftFirst + 1u, leftFirst, leftNear);
				pop = false;
				if(max(dL, dR) < MAX_DIST && stackSize < MESH_STACK_SIZE){
					stack[stackSize] = select(leftFirst, leftFirst + 1u, leftNear);
					stackSize++;
				}
			}
		}
		if(pop){
			if(stackSize == 0u){
				break;
			}
			stackSize--;
			node = stack[stackSize];
		}
	}

	meshHitTriangle = bestTriangle;
	if(bestTriangle == MESH_NONE){
		return vec4(0., 0., 0., MAX_DIST);
	}
	let o = getMeshTriangleOffset(bestTriangle);
	let v0 = meshBuffer[o];
	let v1 = meshBuffer[o + 1u];
	let v2 = meshBuffer[o + 2u];
	let w = vec3(1. - best.y - best.z, best.y, best.z);
	var n = meshDecodeNormal(v1.w) * w.x + meshDecodeNormal(v2.w) * w.y + meshDecodeNormal(meshBuffer[o + 4u].z) * w.z;
	let geometric = cross(v1.xyz - v0.xyz, v2.xyz - v0.xyz);
	n = select(n, -n, dot(geometric, rd) > 0.);
	return vec4(normalize(n), best.x);
}

fn isSky(p: vec3<f32>) -> bool {
	let sky = sceneBuffer[u32(uniforms.modelCount) - 1u];
	return length((sky.inverseModel * vec4(p, 1.)).xyz) > SKY_RADIUS;
}

fn iSphere(ro: vec3<f32>, rd: vec3<f32>, distBound: vec2<f32>, sphereRadius: f32) -> vec4<f32> {
    let b = dot(ro, rd);
    let c = dot(ro, ro) - sphereRadius * sphereRadius;
    let h = b * b - c;

    if (h < 0.0) {
        return vec4(0,0,0, MAX_DIST);
    } else {
        let h_sqrt = sqrt(h);
        let d1 = -b - h_sqrt;
        let d2 = -b + h_sqrt;

        if (d1 >= distBound.x && d1 <= distBound.y) {
            let normal = normalize(ro + rd * d1);
            return vec4(normal, d1);
        } else if (d2 >= distBound.x && d2 <= distBound.y) {
            let normal = normalize(ro + rd * d2);
            return vec4(normal, d2);
        } else {
            return vec4(0,0,0, MAX_DIST);
        }
    }
}

fn iBox(ro: vec3<f32>, rd: vec3<f32>, distBound: vec2<f32>, boxSize: vec3<f32>) -> vec4<f32> {
    let m = sign(rd) / max(abs(rd), vec3<f32>(1e-8));
    let n = m * ro;
    let k = abs(m) * boxSize;

    let t1 = -n - k;
    let t2 = -n + k;

    let tN = max(max(t1.x, t1.y), t1.z);
    let tF = min(min(t2.x, t2.y), t2.z);

    if (tN > tF || tF <= 0.0) {
        return vec4<f32>(vec3<f32>(0.0), MAX_DIST);
    } else {
        let normal = -sign(rd) * step(t1.yzx, t1.xyz) * step(t1.zxy, t1.xyz);

        if (tN >= distBound.x && tN <= distBound.y) {
            return vec4(normal, tN);
        } else if (tF >= distBound.x && tF <= distBound.y) {
            return vec4(normal, tF);
        } else {
            return vec4(0,0,0, MAX_DIST);
        }
    }
}

fn iCylinder(ro: vec3<f32>, rd: vec3<f32>, distBound: vec2<f32>, pa: vec3<f32>, pb: vec3<f32>, ra: f32) -> vec4<f32> {
    let ca = pb - pa;
    let oc = ro - pa;

    let caca = dot(ca, ca);
    let card = dot(ca, rd);
    let caoc = dot(ca, oc);

    let a = caca - card * card;
    let b = caca * dot(oc, rd) - caoc * card;
    let c = caca * dot(oc, oc) - caoc * caoc - ra * ra * caca;
    let h = b * b - a * c;

    if (h < 0.0) {
        return vec4(0,0,0, MAX_DIST);
    }

    let h_sqrt = sqrt(h);
    let d = (-b - h_sqrt) / a;

    let y = caoc + d * card;
    if (y > 0.0 && y < caca && d >= distBound.x && d <= distBound.y) {
        let normal = (oc + d * rd - ca * y / caca) / ra;
        return vec4(normal, d);
    }

	let d2 = (select(caca, 0., y < 0.0) - caoc) / card;

    if (abs(b + a * d2) < h_sqrt && d2 >= distBound.x && d2 <= distBound.y) {
        let normal = normalize(ca * sign(y) / caca);
        return vec4(normal, d2);
    } else {
        return vec4(0,0,0, MAX_DIST);
    }
}

fn rotateQuaternion(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let temp = cross(q.xyz, v) + q.w * v;
    return v + 2.0 * cross(q.xyz, temp);
}

//the local normal (xyz) and the world distance (w) of the hit, w is MAX_DIST on a miss.
//Reads only the inverse matrix and the type of the model, not the whole Model.
//Together with isOccluded for the shadow rays, that made updateOctree 1.25x faster (dawn.node, D3D12). Each alone was slower.
//maxDist: the world distance of the closest hit so far, MAX_DIST when none. Only a mesh uses it, see below
fn getModelHit(ro: vec3<f32>, rd: vec3<f32>, modelIndex: u32, maxDist: f32) -> vec4<f32> {
	let distBound = vec2(MIN_DIST, MAX_DIST);

	let invModel = sceneBuffer[modelIndex].inverseModel;
	let localRo = (invModel * vec4(ro, 1.0)).xyz;
	let localRdScaled = (invModel * vec4(rd, 0.0)).xyz;
	let rdScale = length(localRdScaled);
	let localRd = localRdScaled / rdScale;

	var hit = vec4<f32>(vec3<f32>(0.0), MAX_DIST);

	// Check intersection based on primitive type
	switch(u32(sceneBuffer[modelIndex].primitiveType)) {
		case 0u: {
			hit = iSphere(localRo, localRd, distBound, 1.0);
			break;
		}
		case 1u: {
			hit = iBox(localRo, localRd, distBound, vec3(1.));
			break;
		}
		case 2u: {
			hit = iCylinder(localRo, localRd, distBound, vec3(0., -1., 0.), vec3(0., 1., 0.), 1.);
			break;
		}
		//the mesh fits the -1 to 1 cube, like the primitives: only the rays that hit the cube go through its BVH.
		//And only up to the closest hit so far: a ray that is stopped in front of the cube skips the mesh, and the
		//others skip the nodes behind that hit. The threads of a wave wait on the one that is in the mesh, so every
		//ray that stays out of it counts
		case PRIMITIVE_MESH: {
			//in local units, like the distances of the local ray
			let meshBound = vec2(MIN_DIST, min(maxDist * rdScale, MAX_DIST));
			//from outside the cube its distance is where the ray enters it, from inside where it leaves it: then the
			//mesh can be closer than the hit so far, whatever that distance is
			let outside = any(abs(localRo) > vec3(1.));
			if(iBox(localRo, localRd, distBound, vec3(1.)).w < select(MAX_DIST, meshBound.y, outside)){
				hit = traceMesh(localRo, localRd, meshBound);
			}
			break;
		}
		default: { break; }
	}

	if(hit.w == MAX_DIST) {
		return hit;
	}
/*	if(dot(localRd, hit.xyz) < 0.0) {
		//reject back faces
		return hit;
	}*/

	//to the world distance: the local ray was normalized, which divided its length by rdScale
	hit.w /= rdScale;

	return hit;
}

fn intersectModel(ro: vec3<f32>, rd: vec3<f32>, modelIndex: u32, result: ptr<function, HitResult>) {
	let hit = getModelHit(ro, rd, modelIndex, (*result).distance);

	// Keep track of closest hit
	if (hit.w < (*result).distance) {
		(*result).normal = normalize(rotateQuaternion(sceneBuffer[modelIndex].quaternion, hit.xyz));
		(*result).color = sceneBuffer[modelIndex].color;
		(*result).distance = hit.w;
		if(TRACK_HIT_MODEL){
			hitModelIndex = modelIndex;
			hitTriangle = meshHitTriangle;
		}
	}
}

//large instead of infinite for axis aligned rays, so a ray in the plane of a box side gives no NaN
fn getInverseDirection(rd: vec3<f32>) -> vec3<f32> {
	return select(vec3(-1.), vec3(1.), rd >= vec3(0.)) / max(abs(rd), vec3(1e-8));
}

//the distances to where the ray enters (x) and leaves (y) the box, x > y when it misses the line. x is negative when ro is inside
//roInvRd: ro * invRd, so a side is a single fma
fn intersectAABB(roInvRd: vec3<f32>, invRd: vec3<f32>, boxMin: vec3<f32>, boxMax: vec3<f32>) -> vec2<f32> {
	let t0 = fma(boxMin, invRd, -roInvRd);
	let t1 = fma(boxMax, invRd, -roInvRd);
	let tNear = min(t0, t1);
	let tFar = max(t0, t1);
	return vec2(max(max(tNear.x, tNear.y), tNear.z), min(min(tFar.x, tFar.y), tFar.z));
}

//the distance to an inner node, MAX_DIST when the ray misses it or the child is a leaf
fn getNodeDistance(t: vec2<f32>, child: u32) -> f32 {
	let tNear = max(t.x, 0.);
	return select(MAX_DIST, tNear, t.y >= tNear && child < BVH_LEAF);
}

//the distance to a leaf box, like iBox: where the ray enters it, or leaves it when ro is inside. MAX_DIST when it misses the box
//or the child is a node
fn getLeafDistance(t: vec2<f32>, child: u32) -> f32 {
	let d = select(t.y, t.x, t.x >= MIN_DIST);
	return select(MAX_DIST, d, t.x <= t.y && d >= MIN_DIST && child >= BVH_LEAF);
}

//the normal of the side where the ray enters the box, like iBox (also when ro is inside)
fn getAABBNormal(ro: vec3<f32>, invRd: vec3<f32>, boxMin: vec3<f32>, boxMax: vec3<f32>) -> vec3<f32> {
	let tNear = min((boxMin - ro) * invRd, (boxMax - ro) * invRd);
	return normalize(-sign(invRd) * step(tNear.yzx, tNear.xyz) * step(tNear.zxy, tNear.xyz));
}

//the BVH models, see ModelBVH.ts. Their boxes are the leaves, so a leaf is hit when the box test of its parent is.
//Visits the near child next and pushes the far one. A stack entry is the distance to the node, truncated, with the
//node index in the low 16 bits: that is never more than the distance, so a node is only skipped when it is behind the hit
fn traceBVH(ro: vec3<f32>, rd: vec3<f32>, result: ptr<function, HitResult>) {
	let invRd = getInverseDirection(rd);
	let roInvRd = ro * invRd;

	var stack: array<u32, BVH_STACK_SIZE>;
	var stackSize = 0u;
	var nodeIndex = 0u;
	//the leaf that was hit, a node index (0) when none
	var hitModel = 0u;

	loop {
		let node = bvhBuffer[nodeIndex];
		let tA = intersectAABB(roInvRd, invRd, node.aMin, node.aMax);
		let tB = intersectAABB(roInvRd, invRd, node.bMin, node.bMax);

		let leafA = getLeafDistance(tA, node.a);
		if(leafA < (*result).distance) {
			(*result).distance = leafA;
			(*result).normal = getAABBNormal(ro, invRd, node.aMin, node.aMax);
			hitModel = node.a;
		}
		let leafB = getLeafDistance(tB, node.b);
		if(leafB < (*result).distance) {
			(*result).distance = leafB;
			(*result).normal = getAABBNormal(ro, invRd, node.bMin, node.bMax);
			hitModel = node.b;
		}

		let dA = getNodeDistance(tA, node.a);
		let dB = getNodeDistance(tB, node.b);
		let visitA = dA < (*result).distance;
		let visitB = dB < (*result).distance;
		if(visitA && visitB) {
			let aIsNear = dA <= dB;
			nodeIndex = select(node.b, node.a, aIsNear);
			stack[stackSize] = (bitcast<u32>(max(dA, dB)) & ~BVH_STACK_INDEX_MASK) | select(node.a, node.b, aIsNear);
			stackSize++;
			continue;
		}
		if(visitA || visitB) {
			nodeIndex = select(node.b, node.a, visitA);
			continue;
		}

		//pop the nearest node that is not behind a hit that was found after it was pushed
		var found = false;
		while(stackSize > 0u) {
			stackSize--;
			let entry = stack[stackSize];
			if(bitcast<f32>(entry & ~BVH_STACK_INDEX_MASK) < (*result).distance) {
				nodeIndex = entry & BVH_STACK_INDEX_MASK;
				found = true;
				break;
			}
		}
		if(!found) {
			break;
		}
	}

	if(hitModel >= BVH_LEAF) {
		(*result).color = sceneBuffer[hitModel & ~BVH_LEAF].color;
		if(TRACK_HIT_MODEL){
			hitModelIndex = hitModel & ~BVH_LEAF;
		}
	}
}

//the static models through the BVH, then the dynamic models one by one, then the sky (the last model) when nothing was hit.
//The models are in this order in sceneBuffer, see Scene.ts
fn traceRay(ro: vec3<f32>, rd: vec3<f32>) -> HitResult {
	var result: HitResult;
	result.distance = MAX_DIST;
	let bvhModelCount = u32(uniforms.bvhModelCount);
	let skyIndex = u32(uniforms.modelCount) - 1u;

	if(bvhModelCount > 0u) {
		traceBVH(ro, rd, &result);
	}

	for(var i = bvhModelCount; i < skyIndex; i++) {
		intersectModel(ro, rd, i, &result);
	}

	if(result.distance == MAX_DIST) {
		intersectModel(ro, rd, skyIndex, &result);
	}

	return result;
}

//true when a static or dynamic model is in the way, so the ray doesn't reach the sky. Stops at the first hit, in any order
fn isOccluded(ro: vec3<f32>, rd: vec3<f32>) -> bool {
	let bvhModelCount = u32(uniforms.bvhModelCount);
	if(bvhModelCount > 0u) {
		let invRd = getInverseDirection(rd);
		let roInvRd = ro * invRd;
		var stack: array<u32, BVH_STACK_SIZE>;
		var stackSize = 0u;
		var nodeIndex = 0u;

		loop {
			let node = bvhBuffer[nodeIndex];
			let tA = intersectAABB(roInvRd, invRd, node.aMin, node.aMax);
			let tB = intersectAABB(roInvRd, invRd, node.bMin, node.bMax);
			if(getLeafDistance(tA, node.a) < MAX_DIST || getLeafDistance(tB, node.b) < MAX_DIST) {
				return true;
			}
			let visitA = getNodeDistance(tA, node.a) < MAX_DIST;
			let visitB = getNodeDistance(tB, node.b) < MAX_DIST;
			if(visitA && visitB) {
				nodeIndex = node.a;
				stack[stackSize] = node.b;
				stackSize++;
				continue;
			}
			if(visitA || visitB) {
				nodeIndex = select(node.b, node.a, visitA);
				continue;
			}
			if(stackSize == 0u) {
				break;
			}
			stackSize--;
			nodeIndex = stack[stackSize];
		}
	}

	let skyIndex = u32(uniforms.modelCount) - 1u;
	for(var i = bvhModelCount; i < skyIndex; i++) {
		if(getModelHit(ro, rd, i, MAX_DIST).w < MAX_DIST) {
			return true;
		}
	}
	return false;
}
