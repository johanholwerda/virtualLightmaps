import Ray from './Ray';
import Vector3 from './Vector3';
import Quaternion from './Quaternion';
import type Matrix4x4 from './Matrix4x4';

// Structural subsets of Transform, so this module stays independent of the
// scene graph. Transform satisfies both, so call sites pass one unchanged.

export interface WorldTransform {
  readonly worldPosition: Vector3;
  readonly worldRotation: Quaternion;
  readonly worldScaleScalar: number;
}

export interface WorldMatrix {
  readonly worldMatrix: Matrix4x4;
}

// Scratch objects, so the hit tests allocate nothing. Each is only used within one call that doesn't
// call back into a function using the same one (intersectRayQuad has its own set for its corners).
const localRay = new Ray();
const inverseRotation = new Quaternion();
const tempA = new Vector3();
const tempB = new Vector3();
const tempC = new Vector3();
const tempD = new Vector3();
const tempE = new Vector3();
const quadCorners = [new Vector3(), new Vector3(), new Vector3(), new Vector3()];
const negSize = new Vector3();
const unitCube = new Vector3(1, 1, 1);

/**
 * Distance along `rd` (unit length) to a sphere, given the ray origin relative to its centre: where the ray enters it,
 * or where it leaves it when the origin is inside (like the slab test), or -1.
 */
function intersectSphere(oc: Vector3, rd: Vector3, radius: number): number {
  const b = Vector3.dot(oc, rd);
  const c = Vector3.dot(oc, oc) - radius * radius;
  const h = b * b - c;

  if (h < 0.0) return -1;

  const sqrtH = Math.sqrt(h);
  const near = -b - sqrtH;
  if (near >= 0) return near;

  const far = -b + sqrtH;
  return far >= 0 ? far : -1;
}

export function intersectRaySphereNoScale(ray: Ray, center: Vector3, radius: number): number {
  return intersectSphere(tempA.copy(ray.origin).subtract(center), ray.direction, radius);
}

export function intersectRaySphere(ray: Ray, radius: number, transform: WorldTransform): number {
  const rayLocal = transformRayInto(ray, transform, localRay);
  const dist = intersectSphere(rayLocal.origin, rayLocal.direction, radius);
  return dist < 0 ? -1 : dist * rayLocal.scaleFactor;
}

export function intersectRayPlane(ray: Ray, normal: Vector3, distance: number): number {
  const denom = Vector3.dot(normal, ray.direction);
  if (Math.abs(denom) < 1e-6) return -1; // parallel; the division would yield NaN

  const dist = -(Vector3.dot(normal, ray.origin) + distance) / denom;
  if (dist <= 0) return -1;
  return dist;
}

export function intersectRayAABB(ray: Ray, min: Vector3, max: Vector3): number {
  return intersectRaySlabs(ray.origin, ray.direction, min, max);
}

export function transformRay(ray: Ray, transform: WorldTransform): Ray {
  return transformRayInto(ray, transform, new Ray());
}

// the ray in the transform's local space (uniform scale only), written into `out`
function transformRayInto(ray: Ray, transform: WorldTransform, out: Ray): Ray {
  const inverse = inverseRotation.copy(transform.worldRotation).conjugate();
  const scale = transform.worldScaleScalar;

  const ro = out.origin.copy(ray.origin).subtract(transform.worldPosition);
  Quaternion.rotateVector3(inverse, ro, ro).multiplyScalar(1 / scale);
  Quaternion.rotateVector3(inverse, ray.direction, out.direction);
  out.scaleFactor = scale;
  return out;
}

// the range of t the slab tests narrow down; module level so intersectRaySlabs allocates nothing
const slabRange = { near: 0, far: 0 };

// narrows slabRange to where the ray is between this pair of planes; false when it never is
function clipSlab(origin: number, direction: number, slabMin: number, slabMax: number): boolean {
  if (direction === 0) {
    // parallel to this pair of planes: the ray is either always within the
    // slab or never enters it, and imposes no bound on t either way
    return origin >= slabMin && origin <= slabMax;
  }
  const inv = 1 / direction;
  let t1 = (slabMin - origin) * inv;
  let t2 = (slabMax - origin) * inv;
  if (t1 > t2) {
    const swap = t1;
    t1 = t2;
    t2 = swap;
  }
  if (t1 > slabRange.near) slabRange.near = t1;
  if (t2 < slabRange.far) slabRange.far = t2;
  return true;
}

/**
 * Slab test against the box from min to max: the distance to where the ray enters it, or where it leaves it when
 * the origin is inside, or -1. Branches on a zero direction component: dividing by it evaluates `0/0` when the
 * origin lies on a slab plane (and `-Infinity + Infinity` in the compact `-n +- k` form), and the resulting NaN
 * slips past both rejection tests.
 */
function intersectRaySlabs(o: Vector3, d: Vector3, min: Vector3, max: Vector3): number {
  slabRange.near = -Infinity;
  slabRange.far = Infinity;

  if (!clipSlab(o.x, d.x, min.x, max.x)) return -1;
  if (!clipSlab(o.y, d.y, min.y, max.y)) return -1;
  if (!clipSlab(o.z, d.z, min.z, max.z)) return -1;

  const tN = slabRange.near,
        tF = slabRange.far;
  if (tN > tF || tF < 0.0) return -1;

  return tN < 0 ? tF : tN;
}

// against an origin-centred box with half-extents `size`, in the ray's local space
function intersectRayBoxLocal(rayLocal: Ray, size: Vector3): number {
  const dist = intersectRaySlabs(rayLocal.origin, rayLocal.direction, negSize.copy(size).negate(), size);
  return dist < 0 ? -1 : dist * rayLocal.scaleFactor;
}

export function intersectRayBox(ray: Ray, size: Vector3, transform: WorldTransform): number {
  return intersectRayBoxLocal(transformRayInto(ray, transform, localRay), size);
}

export function intersectRayCube(ray: Ray, transform: WorldTransform): number {
  return intersectRayBoxLocal(transformRayInto(ray, transform, localRay), unitCube);
}

export function intersectRayTriangle(ray: Ray, v0: Vector3, v1: Vector3, v2: Vector3): number {
  const v0v1 = tempA.copy(v1).subtract(v0);
  const v0v2 = tempB.copy(v2).subtract(v0);
  const pvec = tempC.crossVectors(ray.direction, v0v2);
  const det = Vector3.dot(v0v1, pvec);

  // relative to the edge lengths: det scales with the triangle, so a fixed epsilon rejects small triangles outright
  const epsilon = 1e-7 * Math.sqrt(Vector3.dot(v0v1, v0v1) * Vector3.dot(v0v2, v0v2));
  if (det < epsilon) return -1;

  const invDet = 1 / det;
  const tvec = tempD.copy(ray.origin).subtract(v0);

  const u = Vector3.dot(tvec, pvec) * invDet;
  if (u < 0 || u > 1) return -1;

  const qvec = tempE.crossVectors(tvec, v0v1);
  const v = Vector3.dot(ray.direction, qvec) * invDet;
  if (v < 0 || u + v > 1) return -1;

  const t = Vector3.dot(v0v2, qvec) * invDet;
  return t > 0 ? t : -1;
}

export function intersectRayQuad(ray: Ray, t: WorldMatrix): number {
  const world = t.worldMatrix;
  const p0 = quadCorners[0].setValues(-1, -1, 0).transform(world);
  const p1 = quadCorners[1].setValues(1, -1, 0).transform(world);
  const p2 = quadCorners[2].setValues(-1, 1, 0).transform(world);
  const p3 = quadCorners[3].setValues(1, 1, 0).transform(world);

  const dist1 = intersectRayTriangle(ray, p0, p1, p2);
  if (dist1 >= 0) return dist1;

  return intersectRayTriangle(ray, p2, p1, p3);
}

export function intersectRayTorus(
  ray: Ray,
  torusR: number,        // major radius (R)
  torusr: number,        // minor radius (r)
  transform: WorldTransform, // brings ray into torus local space with axis = +Y
): number {
  const rayLocal = transformRayInto(ray, transform, localRay);

  // Normalize by major radius to improve numerical stability in the quartic solver
  const invR = 1.0 / torusR;
  const ro = tempA.copy(rayLocal.origin).multiplyScalar(invR);
  const rd = rayLocal.direction;

  // iq's quartic with the major radius normalised to 1, so his Ra2 factors drop out
  let po = 1.0;
  const ra2 = (torusr * invR) * (torusr * invR);
  const m = Vector3.dot(ro, ro);
  const n = Vector3.dot(ro, rd);

  // bounding sphere radius = R + r
  {
    const sum = 1.0 + (torusr * invR);
    const h = n * n - m + sum * sum;
    if (h < 0.0) return -1.0;
  }

  // Quartic setup (swap z->y for Y-axis torus)
  const k = (m - ra2 - 1.0) * 0.5;
  let k3 = n;
  let k2 = n * n + rd.y * rd.y + k;
  let k1 = k * n + ro.y * rd.y;
  let k0 = k * k + ro.y * ro.y - ra2;

  // stabilization
  if (Math.abs(k3 * (k3 * k3 - k2) + k1) < 0.01) {
    po = -1.0;
    const tmp = k1; k1 = k3; k3 = tmp;
    k0 = 1.0 / k0;
    k1 = k1 * k0;
    k2 = k2 * k0;
    k3 = k3 * k0;
  }

  const c2 = (2.0 * k2 - 3.0 * k3 * k3) / 3.0;
  const c1 = (k3 * (k3 * k3 - k2) + k1) * 2.0;
  const c0 = (k3 * (k3 * (-3.0 * k3 * k3 + 4.0 * k2) - 8.0 * k1) + 4.0 * k0) / 3.0;

  const Q = c2 * c2 + c0;
  const Rv = 3.0 * c0 * c2 - c2 * c2 * c2 - c1 * c1;
  let h = Rv * Rv - Q * Q * Q;
  let z = 0.0;

  if (h < 0.0) {
    const sQ = Math.sqrt(Math.max(0, Q));
    const denom = sQ * Q;
    const arg = denom !== 0 ? Math.min(1, Math.max(-1, Rv / denom)) : 1;
    z = 2.0 * sQ * Math.cos(Math.acos(arg) / 3.0);
  } else {
    const s = Math.sqrt(h);
    const sQ = Math.cbrt(s + Math.abs(Rv));
    const term = sQ !== 0 ? Q / sQ : 0;
    z = Math.sign(Rv) * Math.abs(sQ + term);
  }

  z = c2 - z;
  let d1 = z - 3.0 * c2;
  let d2 = z * z - 3.0 * c0;

  if (Math.abs(d1) < 1.0e-4) {
    if (d2 < 0.0) return -1.0;
    d2 = Math.sqrt(d2);
  } else {
    if (d1 < 0.0) return -1.0;
    d1 = Math.sqrt(d1 / 2.0);
    d2 = c1 / d1;
  }

  let result = Number.POSITIVE_INFINITY;
  h = d1 * d1 - z + d2;
  if (h > 0.0) {
    h = Math.sqrt(h);
    let t1 = -d1 - h - k3; t1 = (po < 0.0) ? 2.0 / t1 : t1;
    let t2 = -d1 + h - k3; t2 = (po < 0.0) ? 2.0 / t2 : t2;
    if (t1 > 0.0) result = t1;
    if (t2 > 0.0) result = Math.min(result, t2);
  }

  h = d1 * d1 - z - d2;
  if (h > 0.0) {
    h = Math.sqrt(h);
    let t1 = d1 - h - k3; t1 = (po < 0.0) ? 2.0 / t1 : t1;
    let t2 = d1 + h - k3; t2 = (po < 0.0) ? 2.0 / t2 : t2;
    if (t1 > 0.0) result = Math.min(result, t1);
    if (t2 > 0.0) result = Math.min(result, t2);
  }

  return isFinite(result) ? result * torusR * rayLocal.scaleFactor : -1.0;
}

export function intersectRayCapsule(
  ray: Ray,
  pa: Vector3,
  pb: Vector3,
  radius: number,
  transform: WorldTransform,
): number {
  const rayLocal = transformRayInto(ray, transform, localRay);

  const ba = tempA.copy(pb).subtract(pa);
  const oa = tempB.copy(rayLocal.origin).subtract(pa);

  const baba = Vector3.dot(ba, ba);
  const bard = Vector3.dot(ba, rayLocal.direction);
  const baoa = Vector3.dot(ba, oa);
  const rdoa = Vector3.dot(rayLocal.direction, oa);
  const oaoa = Vector3.dot(oa, oa);

  const a = baba - bard * bard;
  const b = baba * rdoa - baoa * bard;
  const c = baba * oaoa - baoa * baoa - radius * radius * baba;
  let h = b * b - a * c;

  if (h >= 0.0) {
    // parallel to the axis: a is 0 and t would be NaN, so go straight to the cap the ray reaches first
    const parallel = a <= 1e-9 * baba;
    const t = parallel ? 0 : (-b - Math.sqrt(h)) / a;
    let d = -1;

    const y = parallel ? (bard > 0 ? 0 : baba) : baoa + t * bard;

    // body
    if (y > 0.0 && y < baba) {
      d = t;
    } else {
      // caps
      const oc = (y <= 0.0) ? oa : tempC.copy(rayLocal.origin).subtract(pb);
      const capB = Vector3.dot(rayLocal.direction, oc);
      const capC = Vector3.dot(oc, oc) - radius * radius;
      h = capB * capB - capC;
      if (h > 0.0) {
        d = -capB - Math.sqrt(h);
      }
    }

    if (d >= 0) {
      return d * rayLocal.scaleFactor;
    }
  }

  return -1;
}

export function intersectRayCylinder(
  ray: Ray,
  pa: Vector3,
  pb: Vector3,
  radius: number,
  transform: WorldTransform,
): number {
  const rayLocal = transformRayInto(ray, transform, localRay);

  const ca = tempA.copy(pb).subtract(pa);
  const oc = tempB.copy(rayLocal.origin).subtract(pa);

  const caca = Vector3.dot(ca, ca);
  const card = Vector3.dot(ca, rayLocal.direction);
  const caoc = Vector3.dot(ca, oc);

  const a = caca - card * card;
  const b = caca * Vector3.dot(oc, rayLocal.direction) - caoc * card;
  const c = caca * Vector3.dot(oc, oc) - caoc * caoc - radius * radius * caca;

  // parallel to the axis: a is 0 and the body root is NaN, so only the cap the ray reaches first can be hit
  if (a <= 1e-9 * caca) {
    if (c > 0) return -1; // outside the radius
    const d = ((card > 0 ? 0 : caca) - caoc) / card;
    return d >= 0 ? d * rayLocal.scaleFactor : -1;
  }

  let h = b * b - a * c;

  if (h < 0.0) return -1;

  h = Math.sqrt(h);
  let d = (-b - h) / a;

  const y = caoc + d * card;
  if (y > 0.0 && y < caca && d >= 0) {
    return d * rayLocal.scaleFactor;
  }

  d = ((y < 0.0 ? 0.0 : caca) - caoc) / card;

  if (Math.abs(b + a * d) < h && d >= 0) {
    return d * rayLocal.scaleFactor;
  }

  return -1;
}

export function intersectRayCone(
  ray: Ray,
  pa: Vector3,
  pb: Vector3,
  ra: number,
  rb: number,
  transform: WorldTransform,
): number {
  const rayLocal = transformRayInto(ray, transform, localRay);

  const ba = tempA.copy(pb).subtract(pa);
  const oa = tempB.copy(rayLocal.origin).subtract(pa);
  const ob = tempC.copy(rayLocal.origin).subtract(pb);

  const m0 = Vector3.dot(ba, ba);
  const m1 = Vector3.dot(oa, ba);
  const m2 = Vector3.dot(ob, ba);
  const m3 = Vector3.dot(rayLocal.direction, ba);

  // caps
  if (m1 < 0.0) {
    // oa * m3 - rd * m1: delayed division
    const diff = tempD.copy(oa).multiplyScalar(m3).subtract(tempE.copy(rayLocal.direction).multiplyScalar(m1));
    if (Vector3.dot(diff, diff) < (ra * ra * m3 * m3)) {
      const d = -m1 / m3;
      if (d >= 0) {
        return d * rayLocal.scaleFactor;
      }
    }
  } else if (m2 > 0.0) {
    const diff = tempD.copy(ob).multiplyScalar(m3).subtract(tempE.copy(rayLocal.direction).multiplyScalar(m2));
    if (Vector3.dot(diff, diff) < (rb * rb * m3 * m3)) {
      const d = -m2 / m3;
      if (d >= 0) {
        return d * rayLocal.scaleFactor;
      }
    }
  }

  // body
  const m4 = Vector3.dot(rayLocal.direction, oa);
  const m5 = Vector3.dot(oa, oa);
  const rr = ra - rb;
  const hy = m0 + rr * rr;

  const k2 = m0 * m0 - m3 * m3 * hy;
  const k1 = m0 * m0 * m4 - m1 * m3 * hy + m0 * ra * rr * m3;
  const k0 = m0 * m0 * m5 - m1 * m1 * hy + m0 * ra * (rr * m1 * 2.0 - m0 * ra);

  const h = k1 * k1 - k2 * k0;
  if (h < 0.0) return -1;

  const t = (-k1 - Math.sqrt(h)) / k2;

  const y = m1 + t * m3;
  if (y > 0.0 && y < m0 && t >= 0) {
    return t * rayLocal.scaleFactor;
  }

  return -1;
}