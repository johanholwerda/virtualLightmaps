import Vector3 from './Vector3';

export default class Ray {
  public origin = new Vector3();
  public direction = new Vector3();
  public scaleFactor: number = 1;

  constructor(ro: Vector3 | null = null, rd: Vector3 | null = null, scaleFactor: number = 1) {
    this.scaleFactor = scaleFactor;
    if (ro) this.origin.copy(ro);
    if (rd) this.direction.copy(rd);
  }

  public setVectors(ro: Vector3, rd: Vector3) {
    this.origin.copy(ro);
    this.direction.copy(rd);
  }
}

export function rayGetClosestPointOnRayToPoint(ray: Ray, p: Vector3): Vector3 {
  // on the line through the ray: t is not clamped to the ray's start
  const d = ray.direction;
  const t = Vector3.dot(Vector3.subtract(p, ray.origin), d) / Vector3.dot(d, d);
  return Vector3.multiplyScalar(d, t).add(ray.origin);
}