import type Matrix2x2 from './Matrix2x2';
import type Matrix4x4 from './Matrix4x4';
import type Matrix3x3 from './Matrix3x3';

export default class Vector2 {
  public x: number;
  public y: number;

  constructor(x: number = 0, y: number = 0) {
    this.x = x;
    this.y = y;
  }

  public static get ZERO(): Vector2 {
    return new Vector2(0, 0);
  }

  public static get ONE(): Vector2 {
    return new Vector2(1, 1);
  }

  public static dot(a: Vector2, b: Vector2): number {
    return a.x * b.x + a.y * b.y;
  }

  public static distance(a: Vector2, b: Vector2): number {
    return Math.sqrt(Vector2.distanceSquared(a, b));
  }

  public static distanceSquared(a: Vector2, b: Vector2): number {
    const dx = a.x - b.x, dy = a.y - b.y;
    return dx * dx + dy * dy;
  }

  public static transformMat2(a: Vector2, transformMatrix: Matrix2x2) {
    return a.clone().multiplyMatrix(transformMatrix);
  }

  public static transformMat3(a: Vector2, transformMatrix: Matrix3x3) {
    const x = a.x, y = a.y, m = transformMatrix.m;
    return new Vector2(m[0] * x + m[3] * y + m[6], m[1] * x + m[4] * y + m[7]);
  }

  public static transformMat4(a: Vector2, transformMatrix: Matrix4x4) {
    const x = a.x, y = a.y, m = transformMatrix.m;
    return new Vector2(m[0] * x + m[4] * y + m[12], m[1] * x + m[5] * y + m[13]);
  }

  public static multiplyScalar(a: Vector2, s: number): Vector2 {
    return new Vector2(a.x * s, a.y * s);
  }

  public static add(a: Vector2, b: Vector2): Vector2 {
    return new Vector2(a.x + b.x, a.y + b.y);
  }

  public static subtract(a: Vector2, b: Vector2): Vector2 {
    return new Vector2(a.x - b.x, a.y - b.y);
  }

  public static lerp(a: Vector2, b: Vector2, t: number): Vector2 {
    return a.clone().lerp(b, t);
  }

  public static fract(p: Vector2): Vector2 {
    return new Vector2(p.x - Math.floor(p.x), p.y - Math.floor(p.y));
  }

  public static floor(p: Vector2): Vector2 {
    return new Vector2(Math.floor(p.x), Math.floor(p.y));
  }

  public static normalize(v: Vector2): Vector2 {
    return v.clone().normalize();
  }

  public clone() {
    return new Vector2(this.x, this.y);
  }

  public copy(v: Vector2): Vector2 {
    this.x = v.x;
    this.y = v.y;
    return this;
  }

  public setValues(x: number, y: number): Vector2 {
    this.x = x;
    this.y = y;
    return this;
  }

  public length(): number {
    return Math.sqrt(Vector2.dot(this, this));
  }

  public equals(b: Vector2): boolean {
    return b.x === this.x && b.y === this.y;
  }

  public normalize(): Vector2 {
    let l = this.x * this.x + this.y * this.y;
    if (l > 0) {
      l = 1 / Math.sqrt(l);
    }
    this.x *= l;
    this.y *= l;
    return this;
  }

  public negate(): Vector2 {
    this.x = -this.x;
    this.y = -this.y;
    return this;
  }

  public multiplyScalar(v: number): Vector2 {
    this.x *= v;
    this.y *= v;
    return this;
  }

  public addScalar(v: number): Vector2 {
    this.x += v;
    this.y += v;
    return this;
  }

  public multiplyMatrix(matrix: Matrix2x2): Vector2 {
    const m = matrix.m;
    const x = m[0] * this.x + m[2] * this.y;
    const y = m[1] * this.x + m[3] * this.y;

    return this.setValues(x, y);
  }

  public multiply(v: Vector2): Vector2 {
    this.x *= v.x;
    this.y *= v.y;
    return this;
  }

  public add(v: Vector2): Vector2 {
    this.x += v.x;
    this.y += v.y;
    return this;
  }

  public subtract(v: Vector2): Vector2 {
    this.x -= v.x;
    this.y -= v.y;
    return this;
  }

  public subtractScalar(v: number): Vector2 {
    this.x -= v;
    this.y -= v;
    return this;
  }

  public rotate(origin: Vector2, angle: number): Vector2 {
    const s = Math.sin(angle);
    const c = Math.cos(angle);

    this.x -= origin.x;
    this.y -= origin.y;

    const xNew = this.x * c - this.y * s;
    const yNew = this.x * s + this.y * c;

    this.x = xNew + origin.x;
    this.y = yNew + origin.y;
    return this;
  }

  public randomize(): Vector2 {
    this.x = Math.random() * 2 - 1;
    this.y = Math.random() * 2 - 1;
    return this;
  }

  public randomize01(): Vector2 {
    this.x = Math.random();
    this.y = Math.random();
    return this;
  }

  public randomizeDisc(): Vector2 {
    do {
      this.randomize();
    } while (Vector2.dot(this, this) > 1);
    return this;
  }

  //needed for tween class; in place, like Vector3.lerp and Vector4.lerp
  public lerp(b: Vector2, t: number): Vector2 {
    this.x += (b.x - this.x) * t;
    this.y += (b.y - this.y) * t;
    return this;
  }
}
