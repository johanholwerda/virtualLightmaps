import Vector3 from './Vector3';
import Matrix3x3 from './Matrix3x3';
import { EPSILON } from '../utils/MathUtils';

const lookAtForward = new Vector3();
const lookAtRight = new Vector3();
const lookAtUp = new Vector3();
const lookAtMatrix = new Matrix3x3();

export default class Quaternion {
  public x: number;
  public y: number;
  public z: number;
  public w: number;

  constructor(x: number = 0, y: number = 0, z: number = 0, w: number = 1) {
    this.x = x;
    this.y = y;
    this.z = z;
    this.w = w;
  }

  // getters returning a fresh instance, like Vector3.ZERO: a shared static would be corrupted by in-place methods
  public static get IDENTITY(): Quaternion {
    return new Quaternion(0, 0, 0, 1);
  }

  public static get ZERO(): Quaternion {
    return new Quaternion(0, 0, 0, 0);
  }

  public static dot(a: Quaternion, b: Quaternion): number {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
  }

  //normalized lerp along the shortest path: a cheaper slerp that doesn't rotate at a constant speed
  public static lerp(a: Quaternion, b: Quaternion, t: number): Quaternion {
    //q and -q are the same rotation, take the one closest to a
    const sign = Quaternion.dot(a, b) < 0 ? -1 : 1;
    return new Quaternion(
      a.x + t * (sign * b.x - a.x),
      a.y + t * (sign * b.y - a.y),
      a.z + t * (sign * b.z - a.z),
      a.w + t * (sign * b.w - a.w),
    ).normalize();
  }

  public static lookAt(sourcePoint: Vector3, destPoint: Vector3, up: Vector3): Quaternion {
    return new Quaternion().lookAt(sourcePoint, destPoint, up);
  }

  public static createFromAxisAngle(axis: Vector3, angle: number): Quaternion {
    return new Quaternion().setAxisAngle(axis, angle);
  }

  public static getAxisAngle(q: Quaternion): { axis: Vector3; angle: number } {
    //clamped: rounding can put w of a unit quaternion just past 1, where acos is NaN
    const angle = Math.acos(Math.min(1, Math.max(-1, q.w))) * 2.0;
    const s = Math.sin(angle / 2.0);
    const axis = new Vector3();
    if (s > EPSILON) {
      axis.x = q.x / s;
      axis.y = q.y / s;
      axis.z = q.z / s;
    } else {
      // If s is zero, return any axis (no rotation - axis does not matter)
      axis.x = 1;
      axis.y = 0;
      axis.z = 0;
    }
    return { axis, angle };
  }

  public static getAngle(a: Quaternion, b: Quaternion) {
    const dotproduct = Quaternion.dot(a, b);

    //clamped like in getAxisAngle: equal quaternions can give just over 1
    return Math.acos(Math.min(1, Math.max(-1, 2 * dotproduct * dotproduct - 1)));
  }

  public static multiply(a: Quaternion, b: Quaternion): Quaternion {
    return a.clone().multiply(b);
  }

  public static calculateW(a: Quaternion): Quaternion {
    const x = a.x,
          y = a.y,
          z = a.z;
    return new Quaternion(x, y, z, Math.sqrt(Math.abs(1.0 - x * x - y * y - z * z)));
  }

  public static exp(a: Quaternion): Quaternion {
    const x = a.x,
          y = a.y,
          z = a.z,
          w = a.w;

    const r = Math.sqrt(x * x + y * y + z * z);
    const et = Math.exp(w);
    const s = r > 0 ? (et * Math.sin(r)) / r : 0;
    return new Quaternion(x * s, y * s, z * s, et * Math.cos(r));
  }

  public static ln(a: Quaternion): Quaternion {
    const x = a.x,
          y = a.y,
          z = a.z,
          w = a.w;

    const r = Math.sqrt(x * x + y * y + z * z);
    const t = r > 0 ? Math.atan2(r, w) / r : 0;
    return new Quaternion(x * t, y * t, z * t, 0.5 * Math.log(x * x + y * y + z * z + w * w));
  }

  public static scale(a: Quaternion, b: number): Quaternion {
    return new Quaternion(a.x * b, a.y * b, a.z * b, a.w * b);
  }

  public static pow(a: Quaternion, b: number): Quaternion {
    return Quaternion.exp(Quaternion.scale(Quaternion.ln(a), b));
  }

  public static slerp(a: Quaternion, b: Quaternion, t: number): Quaternion {
    const ax = a.x,
          ay = a.y,
          az = a.z,
          aw = a.w;
    let bx = b.x,
        by = b.y,
        bz = b.z,
        bw = b.w;

    let scale0, scale1;

    // calc cosine
    let cosom = ax * bx + ay * by + az * bz + aw * bw;
    // adjust signs (if necessary)
    if (cosom < 0.0) {
      cosom = -cosom;
      bx = -bx;
      by = -by;
      bz = -bz;
      bw = -bw;
    }
    // calculate coefficients
    if (1.0 - cosom > EPSILON) {
      // standard case (slerp)
      const omega = Math.acos(cosom);
      const sinom = Math.sin(omega);
      scale0 = Math.sin((1.0 - t) * omega) / sinom;
      scale1 = Math.sin(t * omega) / sinom;
    } else {
      // "from" and "to" quaternions are very close
      //  ... so we can do a linear interpolation
      scale0 = 1.0 - t;
      scale1 = t;
    }

    // calculate final values
    return new Quaternion(
      scale0 * ax + scale1 * bx,
      scale0 * ay + scale1 * by,
      scale0 * az + scale1 * bz,
      scale0 * aw + scale1 * bw,
    );
  }

  public static random(): Quaternion {
    // Implementation of http://planning.cs.uiuc.edu/node198.html
    const u1 = Math.random();
    const u2 = Math.random();
    const u3 = Math.random();

    const sqrt1MinusU1 = Math.sqrt(1 - u1);
    const sqrtU1 = Math.sqrt(u1);

    return new Quaternion(
      sqrt1MinusU1 * Math.sin(2.0 * Math.PI * u2),
      sqrt1MinusU1 * Math.cos(2.0 * Math.PI * u2),
      sqrtU1 * Math.sin(2.0 * Math.PI * u3),
      sqrtU1 * Math.cos(2.0 * Math.PI * u3),
    );
  }

  public static invert(a: Quaternion): Quaternion {
    return a.clone().invert();
  }

  public static conjugate(a: Quaternion): Quaternion {
    return a.clone().conjugate();
  }

  public static fromMat3(mat: Matrix3x3) {
    return new Quaternion().fromMat3(mat);
  }

  public static fromEuler(x: number, y: number, z: number) {
    return new Quaternion().rotateEulerValues(x, y, z);
  }

  /**
   * Compute the unit quaternion that rotates unit vector `a` onto unit vector `b`
   * along the shortest arc. Handles the antiparallel case by picking an arbitrary
   * perpendicular axis.
   */
  public static rotationBetween(a: Vector3, b: Vector3): Quaternion {
    const d = Math.max(-1, Math.min(1, Vector3.dot(a, b)));
    if (d > 0.999999) return new Quaternion(0, 0, 0, 1);
    if (d < -0.999999) {
      let axis = Vector3.cross(a, new Vector3(1, 0, 0));
      if (axis.length() < 1e-6) axis = Vector3.cross(a, new Vector3(0, 1, 0));
      return Quaternion.createFromAxisAngle(axis.normalize(), Math.PI);
    }
    const axis = Vector3.cross(a, b).normalize();
    return Quaternion.createFromAxisAngle(axis, Math.acos(d));
  }

  /**
   * Apply a unit quaternion to a Vector3. `out` may be `v`.
   */
  public static rotateVector3(q: Quaternion, v: Vector3, out: Vector3 = new Vector3()): Vector3 {
    const tx = 2 * (q.y * v.z - q.z * v.y);
    const ty = 2 * (q.z * v.x - q.x * v.z);
    const tz = 2 * (q.x * v.y - q.y * v.x);
    return out.setValues(
      v.x + q.w * tx + (q.y * tz - q.z * ty),
      v.y + q.w * ty + (q.z * tx - q.x * tz),
      v.z + q.w * tz + (q.x * ty - q.y * tx),
    );
  }

  public clone(): Quaternion {
    return new Quaternion(this.x, this.y, this.z, this.w);
  }

  public copy(b: Quaternion): Quaternion {
    return this.setValues(b.x, b.y, b.z, b.w);
  }

  public setValues(x: number, y: number, z: number, w: number): Quaternion {
    this.x = x;
    this.y = y;
    this.z = z;
    this.w = w;
    return this;
  }

  public dot(b: Quaternion): number {
    return Quaternion.dot(this, b);
  }

  public identity(): Quaternion {
    return this.setValues(0, 0, 0, 1);
  }

  public multiply(a: Quaternion): Quaternion {
    const ax = this.x,
          ay = this.y,
          az = this.z,
          aw = this.w;
    const bx = a.x,
          by = a.y,
          bz = a.z,
          bw = a.w;

    this.x = ax * bw + aw * bx + ay * bz - az * by;
    this.y = ay * bw + aw * by + az * bx - ax * bz;
    this.z = az * bw + aw * bz + ax * by - ay * bx;
    this.w = aw * bw - ax * bx - ay * by - az * bz;
    return this;
  }

  // the inverse of a unit quaternion, and cheaper than invert
  public conjugate(): Quaternion {
    this.x = -this.x;
    this.y = -this.y;
    this.z = -this.z;
    return this;
  }

  public invert(): Quaternion {
    const dot = Quaternion.dot(this, this);
    const invDot = dot ? 1.0 / dot : 0;
    return this.setValues(-this.x * invDot, -this.y * invDot, -this.z * invDot, this.w * invDot);
  }

  public rotateEulerValues(x: number, y: number, z: number): Quaternion {
    // note the yxz order for compatibility with older examples
    this.rotateY(y);
    this.rotateX(x);
    this.rotateZ(z);
    return this;
  }

  public rotateX(rad: number): Quaternion {
    rad *= 0.5;

    const ax = this.x,
          ay = this.y,
          az = this.z,
          aw = this.w;
    const bx = Math.sin(rad),
          bw = Math.cos(rad);

    this.x = ax * bw + aw * bx;
    this.y = ay * bw + az * bx;
    this.z = az * bw - ay * bx;
    this.w = aw * bw - ax * bx;
    return this;
  }

  public rotateY(rad: number): Quaternion {
    rad *= 0.5;

    const ax = this.x,
          ay = this.y,
          az = this.z,
          aw = this.w;
    const by = Math.sin(rad),
          bw = Math.cos(rad);

    this.x = ax * bw - az * by;
    this.y = ay * bw + aw * by;
    this.z = az * bw + ax * by;
    this.w = aw * bw - ay * by;
    return this;
  }

  public rotateZ(rad: number): Quaternion {
    rad *= 0.5;

    const ax = this.x,
          ay = this.y,
          az = this.z,
          aw = this.w;
    const bz = Math.sin(rad),
          bw = Math.cos(rad);
    this.x = ax * bw + ay * bz;
    this.y = ay * bw - ax * bz;
    this.z = az * bw + aw * bz;
    this.w = aw * bw - az * bz;
    return this;
  }

  public setAxisAngle(axis: Vector3, angle: number): Quaternion {
    angle = angle * 0.5;
    const s = Math.sin(angle);
    this.x = s * axis.x;
    this.y = s * axis.y;
    this.z = s * axis.z;
    this.w = Math.cos(angle);
    return this;
  }

  //rotates by the axis angle rotation after this one: this = rotation * this
  public rotateAxisAngle(axis: Vector3, angle: number): Quaternion {
    return this.copy(Quaternion.createFromAxisAngle(axis, angle).multiply(this));
  }

  //the rotation whose z axis points from destPoint to sourcePoint. identity when the points are the same
  public lookAt(sourcePoint: Vector3, destPoint: Vector3, up: Vector3): Quaternion {
    const forward = lookAtForward.copy(sourcePoint).subtract(destPoint);
    const length = forward.length();
    if (length < EPSILON) return this.identity();
    forward.multiplyScalar(1 / length);
    const right = Vector3.lookAtRight(up, forward, lookAtRight);
    //forward and right are perpendicular unit vectors, so up is as well
    const up2 = lookAtUp.crossVectors(forward, right);

    Matrix3x3.setValues(lookAtMatrix, right.x, right.y, right.z, up2.x, up2.y, up2.z, forward.x, forward.y, forward.z);
    return this.fromMat3(lookAtMatrix);
  }

  public fromMat3(mat: Matrix3x3): Quaternion {
    // Algorithm in Ken Shoemake's article in 1987 SIGGRAPH course notes
    // article "Quaternion Calculus and Fast Animation".
    const m = mat.m;
    const fTrace = m[0] + m[4] + m[8];

    if (fTrace > 0.0) {
      // |w| > 1/2, may as well choose w > 1/2
      const fRoot = Math.sqrt(fTrace + 1.0); // 2w
      this.w = 0.5 * fRoot;
      const ifRoot = 0.5 / fRoot; // 1/(4w)
      this.x = (m[5] - m[7]) * ifRoot;
      this.y = (m[6] - m[2]) * ifRoot;
      this.z = (m[1] - m[3]) * ifRoot;
    } else {
      // |w| <= 1/2
      let i = 0;
      if (m[4] > m[0]) i = 1;
      if (m[8] > m[i * 3 + i]!) i = 2;
      const j = (i + 1) % 3;
      const k = (i + 2) % 3;

      const fRoot = Math.sqrt(m[i * 3 + i]! - m[j * 3 + j]! - m[k * 3 + k]! + 1.0);
      const iv = 0.5 * fRoot;
      const ifRoot = 0.5 / fRoot;
      this.w = (m[j * 3 + k]! - m[k * 3 + j]!) * ifRoot;
      const jv = (m[j * 3 + i]! + m[i * 3 + j]!) * ifRoot;
      const kv = (m[k * 3 + i]! + m[i * 3 + k]!) * ifRoot;

      this.x = i === 0 ? iv : j === 0 ? jv : kv;
      this.y = i === 1 ? iv : j === 1 ? jv : kv;
      this.z = i === 2 ? iv : j === 2 ? jv : kv;
    }

    return this;
  }

  public normalize() {
    const l = this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w;
    if (l === 0) {
      return this.identity();
    }
    const inv = 1 / Math.sqrt(l);
    this.x *= inv;
    this.y *= inv;
    this.z *= inv;
    this.w *= inv;
    return this;
  }

  public get array(): number[] {
    return [this.x, this.y, this.z, this.w];
  }
}
