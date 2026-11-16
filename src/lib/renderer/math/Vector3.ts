import type Matrix4x4 from './Matrix4x4';
import type Matrix3x3 from './Matrix3x3';
import Vector2 from './Vector2';
import { EPSILON } from '../utils/MathUtils';

export default class Vector3 {
  public x: number;
  public y: number;
  public z: number;

  constructor(x: number = 0, y: number = 0, z: number = 0) {
    this.x = x;
    this.y = y;
    this.z = z;
  }

  public static get ZERO(): Vector3 {
    return new Vector3(0, 0, 0);
  }

  public static get RIGHT(): Vector3 {
    return new Vector3(1, 0, 0);
  }

  public static get UP(): Vector3 {
    return new Vector3(0, 1, 0);
  }

  public static get FORWARD(): Vector3 {
    return new Vector3(0, 0, 1);
  }

  public static get ONE(): Vector3 {
    return new Vector3(1, 1, 1);
  }

  public get xy(): Vector2 {
    return new Vector2(this.x, this.y);
  }

  public get yz(): Vector2 {
    return new Vector2(this.y, this.z);
  }

  public get xz(): Vector2 {
    return new Vector2(this.x, this.z);
  }

  public static dot(a: Vector3, b: Vector3): number {
    return a.x * b.x + a.y * b.y + a.z * b.z;
  }

  public static distance(a: Vector3, b: Vector3): number {
    const dx = a.x - b.x,
          dy = a.y - b.y,
          dz = a.z - b.z;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  public static distanceSquared(a: Vector3, b: Vector3): number {
    const dx = a.x - b.x,
          dy = a.y - b.y,
          dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
  }

  public static add(a: Vector3, b: Vector3) {
    return a.clone().add(b);
  }

  public static subtract(a: Vector3, b: Vector3) {
    return a.clone().subtract(b);
  }

  public static multiply(a: Vector3, b: Vector3) {
    return a.clone().multiply(b);
  }

  public static divide(a: Vector3, b: Vector3) {
    return a.clone().divide(b);
  }

  public static multiplyScalar(a: Vector3, b: number) {
    return a.clone().multiplyScalar(b);
  }

  public static powScalar(a: Vector3, b: number) {
    return a.clone().powScalar(b);
  }

  public static normalize(a: Vector3) {
    return a.clone().normalize();
  }

  public static max(a: Vector3, b: Vector3) {
    return a.clone().max(b);
  }

  public static min(a: Vector3, b: Vector3) {
    return a.clone().min(b);
  }

  public static lerp(a: Vector3, b: Vector3, t: number) {
    return a.clone().lerp(b, t);
  }

  public static cross(a: Vector3, b: Vector3): Vector3 {
    return new Vector3().crossVectors(a, b);
  }

  //the normalized up × dir: the x axis of a rotation that looks along dir (normalized). when up is parallel to dir any
  //perpendicular axis gives a valid rotation, then the world axis that is least aligned with dir is used as up
  public static lookAtRight(up: Vector3, dir: Vector3, out: Vector3): Vector3 {
    out.setValues(up.y * dir.z - up.z * dir.y, up.z * dir.x - up.x * dir.z, up.x * dir.y - up.y * dir.x);
    let length = out.length();
    if (length < EPSILON) {
      const ax = Math.abs(dir.x), ay = Math.abs(dir.y), az = Math.abs(dir.z);
      if (ax <= ay && ax <= az) {
        out.setValues(0, -dir.z, dir.y); // up (1, 0, 0)
      } else if (ay <= az) {
        out.setValues(dir.z, 0, -dir.x); // up (0, 1, 0)
      } else {
        out.setValues(-dir.y, dir.x, 0); // up (0, 0, 1)
      }
      length = out.length();
    }
    return out.multiplyScalar(1 / length);
  }

  public static fract(p: Vector3): Vector3 {
    return new Vector3(p.x - Math.floor(p.x), p.y - Math.floor(p.y), p.z - Math.floor(p.z));
  }

  public static floor(p: Vector3): Vector3 {
    return new Vector3(Math.floor(p.x), Math.floor(p.y), Math.floor(p.z));
  }

  public static transform(a: Vector3, transformMatrix: Matrix4x4): Vector3 {
    return a.clone().transform(transformMatrix);
  }

  public static transformMat3(a: Vector3, transformMatrix: Matrix3x3): Vector3 {
    return a.clone().transformMat3(transformMatrix);
  }

  public static reflect(incident: Vector3, normal: Vector3): Vector3 {
    const d = 2 * Vector3.dot(incident, normal);
    return new Vector3(incident.x - normal.x * d, incident.y - normal.y * d, incident.z - normal.z * d);
  }

  public clone() {
    return new Vector3(this.x, this.y, this.z);
  }

  public copy(v: Vector3): Vector3 {
    this.x = v.x;
    this.y = v.y;
    this.z = v.z;
    return this;
  }

  public setValues(x: number = 0, y: number = 0, z: number = 0): Vector3 {
    this.x = x;
    this.y = y;
    this.z = z;
    return this;
  }

  public length(): number {
    return Math.sqrt(this.lengthSquared());
  }

  public lengthSquared(): number {
    return Vector3.dot(this, this);
  }

  public equals(b: Vector3): boolean {
    return b.x === this.x && b.y === this.y && b.z === this.z;
  }

  public normalize(): Vector3 {
    let l = this.x * this.x + this.y * this.y + this.z * this.z;
    if (l > 0) {
      l = 1 / Math.sqrt(l);
    }
    this.x *= l;
    this.y *= l;
    this.z *= l;
    return this;
  }

  public get normalized(): Vector3 {
    return this.clone().normalize();
  }

  public negate(): Vector3 {
    this.x = -this.x;
    this.y = -this.y;
    this.z = -this.z;
    return this;
  }

  public multiplyScalar(v: number): Vector3 {
    this.x *= v;
    this.y *= v;
    this.z *= v;
    return this;
  }

  public addScalar(v: number): Vector3 {
    this.x += v;
    this.y += v;
    this.z += v;
    return this;
  }

  public multiply(v: Vector3): Vector3 {
    this.x *= v.x;
    this.y *= v.y;
    this.z *= v.z;
    return this;
  }

  public divide(v: Vector3): Vector3 {
    this.x /= v.x;
    this.y /= v.y;
    this.z /= v.z;
    return this;
  }

  public add(v: Vector3): Vector3 {
    this.x += v.x;
    this.y += v.y;
    this.z += v.z;
    return this;
  }

  public subtract(v: Vector3): Vector3 {
    this.x -= v.x;
    this.y -= v.y;
    this.z -= v.z;
    return this;
  }

  // this = a × b; a or b may be this
  public crossVectors(a: Vector3, b: Vector3): Vector3 {
    const ax = a.x,
          ay = a.y,
          az = a.z;
    const bx = b.x,
          by = b.y,
          bz = b.z;
    return this.setValues(ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx);
  }

  public min(a: Vector3) {
    this.x = Math.min(this.x, a.x);
    this.y = Math.min(this.y, a.y);
    this.z = Math.min(this.z, a.z);
    return this;
  }

  public max(a: Vector3) {
    this.x = Math.max(this.x, a.x);
    this.y = Math.max(this.y, a.y);
    this.z = Math.max(this.z, a.z);
    return this;
  }

  public powScalar(a: number) {
    this.x = Math.pow(this.x, a);
    this.y = Math.pow(this.y, a);
    this.z = Math.pow(this.z, a);
    return this;
  }

  public lerp(a: Vector3, b: number) {
    this.x = this.x + (a.x - this.x) * b;
    this.y = this.y + (a.y - this.y) * b;
    this.z = this.z + (a.z - this.z) * b;
    return this;
  }

  public rotateX(origin: Vector3, angle: number): Vector3 {
    const y = this.y - origin.y;
    const z = this.z - origin.z;

    this.y = y * Math.cos(angle) - z * Math.sin(angle) + origin.y;
    this.z = y * Math.sin(angle) + z * Math.cos(angle) + origin.z;
    return this;
  }

  public rotateY(origin: Vector3, angle: number): Vector3 {
    const x = this.x - origin.x;
    const z = this.z - origin.z;

    this.x = x * Math.cos(angle) + z * Math.sin(angle) + origin.x;
    this.z = z * Math.cos(angle) - x * Math.sin(angle) + origin.z;
    return this;
  }

  public rotateZ(origin: Vector3, angle: number): Vector3 {
    const x = this.x - origin.x;
    const y = this.y - origin.y;

    this.x = x * Math.cos(angle) - y * Math.sin(angle) + origin.x;
    this.y = x * Math.sin(angle) + y * Math.cos(angle) + origin.y;
    return this;
  }

  public transform(transformMatrix: Matrix4x4): Vector3 {
    const x = this.x,
          y = this.y,
          z = this.z,
          m = transformMatrix.m;
    let w = m[3] * x + m[7] * y + m[11] * z + m[15];
    w = w || 1.0;
    this.x = (m[0] * x + m[4] * y + m[8] * z + m[12]) / w;
    this.y = (m[1] * x + m[5] * y + m[9] * z + m[13]) / w;
    this.z = (m[2] * x + m[6] * y + m[10] * z + m[14]) / w;
    return this;
  }

  public transformMat3(transformMatrix: Matrix3x3): Vector3 {
    const x = this.x,
          y = this.y,
          z = this.z,
          m = transformMatrix.m;
    this.x = x * m[0] + y * m[3] + z * m[6];
    this.y = x * m[1] + y * m[4] + z * m[7];
    this.z = x * m[2] + y * m[5] + z * m[8];
    return this;
  }

  public randomize(): Vector3 {
    this.x = Math.random() * 2 - 1;
    this.y = Math.random() * 2 - 1;
    this.z = Math.random() * 2 - 1;
    return this;
  }

  public randomize01(): Vector3 {
    this.x = Math.random();
    this.y = Math.random();
    this.z = Math.random();
    return this;
  }

  public randomizeSphere(): Vector3 {
    do {
      this.randomize();
    } while (this.lengthSquared() > 1);

    return this;
  }

  public fract(): Vector3 {
    this.x -= Math.floor(this.x);
    this.y -= Math.floor(this.y);
    this.z -= Math.floor(this.z);
    return this;
  }

  public get array(): number[] {
    return [this.x, this.y, this.z];
  }
}
