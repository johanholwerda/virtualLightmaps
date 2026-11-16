import Quaternion from '../math/Quaternion';
import Vector3 from '../math/Vector3';
import Matrix3x3 from '../math/Matrix3x3';
import Matrix4x4 from '../math/Matrix4x4';
import type Vector2 from '../math/Vector2';
import LogGL from '../core/LogGL';

const SINGULARITY_RANGE = 0.0001; // don't decrease further, or singularities will not be found

export default class Transform {
  public name: string = '';
  public id: number = -1;
  public static id: number = 0;

  private _rotation = new Quaternion();
  private _position = new Vector3();
  private _scale = new Vector3(1, 1, 1);

  private _matrix = new Matrix4x4();
  private _localMatrix = new Matrix4x4();

  private _parent: Transform | null = null;
  private _children: Transform[] = [];

  private _signalDirtyDown: boolean = true;
  private _dirty: boolean = true;

  constructor() {
    this.id = Transform.id++;
  }

  public get dirty(): boolean {
    return this._dirty;
  }

  public get rotation(): Quaternion {
    return this._rotation.clone();
  }

  public set rotation(rotation: Quaternion) {
    this._rotation.copy(rotation);
    this.setDirty();
  }

  public get position(): Vector3 {
    return this._position.clone();
  }

  public set position(p: Vector3) {
    this._position.copy(p);
    this.setDirty();
  }

  public get scale(): Vector3 {
    return this._scale.clone();
  }

  public set scale(value: Vector3) {
    this._scale.copy(value);
    this.setDirty();
  }

  public get localMatrix(): Matrix4x4 {
    this.updateMatrix();
    return this._localMatrix;
  }

  public get parent(): Transform | null {
    return this._parent;
  }

  //keeps the children of both parents up to date, addChild and removeChild set this
  public set parent(parent: Transform | null) {
    if (this._parent === parent) return;
    for (let ancestor = parent; ancestor; ancestor = ancestor._parent) {
      if (ancestor === this) {
        LogGL.warn('Transform: a transform can not be its own ancestor', this.name, parent!.name);
        return;
      }
    }
    if (this._parent) {
      const siblings = this._parent._children;
      const index = siblings.indexOf(this);
      if (index >= 0) siblings.splice(index, 1);
    }
    this._parent = parent;
    if (parent) {
      parent._children.push(this);
    }
    this.setDirty();
  }

  public get children(): Transform[] {
    return this._children;
  }

  public getChild(name:string): Transform | null {
    for (const child of this._children) {
      if (child.name === name) {
        return child;
      }
    }
    return null;
  }


  public get flipped(): number {
    let flipped = Math.sign(this.scale.x) * Math.sign(this.scale.y) * Math.sign(this.scale.z);
    return this.parent ? this.parent.flipped * flipped : flipped;
  }

  public get worldMatrix(): Matrix4x4 {
    this.updateMatrix();
    return this._matrix;
  }

  public set localMatrix(value: Matrix4x4) {
    const prs = Matrix4x4.toTranslationRotationScale(value);
    this.setPositionRotationScale(prs.position, prs.rotation, prs.scale);
  }

  public get worldPosition(): Vector3 {
    this.updateMatrix();
    return new Vector3().transform(this._matrix);
  }

  public set worldPosition(p: Vector3) {
    if (this.parent) {
      this.position = p.clone().transform(Matrix4x4.invert(this.parent.worldMatrix));
    } else {
      this.position = p;
    }
  }

  public set scaleScalar(value: number) {
    this._scale.setValues(value, value, value);
    this.setDirty();
  }

  public get worldScaleScalar(): number {
    const m = this.worldMatrix.m;
    return Math.sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]);
  }

  //a copy, like rotation: changing it does not change the transform
  public get worldRotation(): Quaternion {
    if (this._parent) {
      return Quaternion.multiply(this._parent.worldRotation, this._rotation);
    }
    return this._rotation.clone();
  }

  public set worldRotation(q: Quaternion) {
    if (this.parent) {
      this.rotation = Quaternion.invert(this.parent.worldRotation).multiply(q.clone());
    } else {
      this.rotation = q;
    }
  }

  public get rotationMatrixMat4(): Matrix4x4 {
    return Matrix4x4.fromQuat(this._rotation);
  }

  public get worldRotationMatrixMat4(): Matrix4x4 {
    return Matrix4x4.fromQuat(this.worldRotation);
  }

  public get rotationMatrixMat3(): Matrix3x3 {
    return Matrix3x3.fromQuat(this._rotation);
  }

  public set rotationMatrixMat3(mat: Matrix3x3) {
    this._rotation.fromMat3(mat);
    this.setDirty();
  }

  public get worldRotationMatrixMat3(): Matrix3x3 {
    return Matrix3x3.fromQuat(this.worldRotation);
  }

  public set eulerX(value: number) {
    this._rotation.identity();
    this.rotateX(value);
  }

  public set eulerY(value: number) {
    this._rotation.identity();
    this.rotateY(value);
  }

  public set eulerZ(value: number) {
    this._rotation.identity();
    this.rotateZ(value);
  }

  public set eulerXY(value: Vector2) {
    this.setEulerValues(value.x, value.y, 0);
  }

  public get euler(): Vector3 {
    return Transform.getEulerYXZ(Matrix3x3.fromQuat(this._rotation));
  }

  public set euler(r: Vector3) {
    this._rotation.identity();
    this.rotateEulerValues(r.x, r.y, r.z);
  }

  public get forward(): Vector3 {
    const mat3 = this.rotationMatrixMat3.m;
    return new Vector3(mat3[6], mat3[7], mat3[8]);
  }

  public get worldForward(): Vector3 {
    const mat3 = this.worldRotationMatrixMat3.m;
    return new Vector3(mat3[6], mat3[7], mat3[8]);
  }

  public get right(): Vector3 {
    const mat3 = this.rotationMatrixMat3.m;
    return new Vector3(mat3[0], mat3[1], mat3[2]);
  }

  public get worldRight(): Vector3 {
    const mat3 = this.worldRotationMatrixMat3.m;
    return new Vector3(mat3[0], mat3[1], mat3[2]);
  }

  public get up(): Vector3 {
    const mat3 = this.rotationMatrixMat3.m;
    return new Vector3(mat3[3], mat3[4], mat3[5]);
  }

  public get worldUp(): Vector3 {
    const mat3 = this.worldRotationMatrixMat3.m;
    return new Vector3(mat3[3], mat3[4], mat3[5]);
  }

  public get direction(): Vector3 {
    return this.worldForward;
  }

  private static getEulerYXZ(mat: Matrix3x3): Vector3 {
    const m = mat.m;
    let x;
    let y;
    let z;

    // Note, x = asin(m[7]) so m[7] = sin(x), so m[7] can never be larger than 1 or smaller than -1
    if (m[7] < -1 + SINGULARITY_RANGE) {
      // singularity at x = +90 degrees
      x = Math.PI / 2;
      y = Math.atan2(m[3], m[0]);
      z = 0;
    } else if (m[7] > 1 - SINGULARITY_RANGE) {
      // singularity at x = -90 degrees
      x = -Math.PI / 2;
      y = -Math.atan2(m[3], m[0]);
      z = 0;
    } else {
      x = Math.asin(-m[7]);
      y = Math.atan2(m[6], m[8]);
      z = Math.atan2(m[1], m[4]);
    }

    return new Vector3(x, y, z);
  }

  public identity() {
    this._position.setValues(0, 0, 0);
    this._rotation.identity();
    this._scale.setValues(1, 1, 1);
    this.setDirty();
  }

  public addChild(transform: Transform) {
    transform.parent = this;
  }

  public removeChild(transform: Transform) {
    if (transform._parent === this) {
      transform.parent = null;
    }
  }

  public getChildrenRecursive() {
    const children: Transform[] = [...this._children];
    for(let i = 0; i < this._children.length; i++){
      children.push(...this._children[i].getChildrenRecursive());
    }
    return children;
  }

  public setDirty(): void {
    if (!this._dirty) {
      this._dirty = true;
      this.signalDirtyDown();
    }
  }

  public clone(): Transform {
    return new Transform().copy(this);
  }

  public copy(t: Transform): Transform {
    this.name = t.name;
    this._scale.copy(t._scale);
    this._rotation.copy(t._rotation);
    this._position.copy(t._position);
    //also tells the children
    this.setDirty();
    return this;
  }

  public lookAt(pos: Vector3, target: Vector3, up: Vector3 = Vector3.UP): void {
    this.position = pos;
    this.lookAtFromCurrentPos(target, up);
  }

  public lookAtFromCurrentPos(target: Vector3, up: Vector3 = Vector3.UP) {
    this._rotation = Quaternion.lookAt(this._position, target, up);
    this.setDirty();
  }

  public setPositionRotationScale(position: Vector3, rotation: Quaternion, scale: Vector3) {
    this._position.copy(position);
    this._rotation.copy(rotation);
    this._scale.copy(scale);
    this.setDirty();
  }

  public setPositionValues(x: number, y: number, z: number): void {
    this._position.setValues(x, y, z);
    this.setDirty();
  }

  public setScaleValues(x: number, y: number, z: number): void {
    this._scale.setValues(x, y, z);
    this.setDirty();
  }

  public translateX(value: number): void {
    this._position.x += value;
    this.setDirty();
  }

  public translateY(value: number): void {
    this._position.y += value;
    this.setDirty();
  }

  public translateZ(value: number): void {
    this._position.z += value;
    this.setDirty();
  }

  public translate(value: Vector3): void {
    this._position.add(value);
    this.setDirty();
  }

  public translateValues(x: number, y: number, z: number): void {
    this._position.x += x;
    this._position.y += y;
    this._position.z += z;
    this.setDirty();
  }

  public rotateEulerValues(x: number, y: number, z: number): Quaternion {
    this._rotation.rotateEulerValues(x, y, z);
    this.setDirty();
    return this._rotation;
  }

  public rotateEuler(euler: Vector3): Quaternion {
    return this.rotateEulerValues(euler.x, euler.y, euler.z);
  }

  public setEulerValues(x: number, y: number, z: number): Quaternion {
    this._rotation.identity();
    return this.rotateEulerValues(x, y, z);
  }

  // NOTE: axis must be normalized;
  public rotateAxisAngle(axis: Vector3, angle: number): Quaternion {
    this._rotation.rotateAxisAngle(axis, angle);
    this.setDirty();
    return this._rotation;
  }

  public rotateX(value: number): void {
    this._rotation.rotateX(value);
    this.setDirty();
  }

  public rotateY(value: number): void {
    this._rotation.rotateY(value);
    this.setDirty();
  }

  public rotateZ(value: number): void {
    this._rotation.rotateZ(value);
    this.setDirty();
  }

  public setDirection(forward: Vector3, up: Vector3) {
    const right = Vector3.cross(up, forward);
    right.normalize();

    const up2 = Vector3.cross(forward, right);
    up2.normalize();

    this.rotationMatrixMat3 = new Matrix3x3(right.x, right.y, right.z, up2.x, up2.y, up2.z, forward.x, forward.y, forward.z);
  }

  protected updateMatrix(): boolean {
    if (!this._dirty && !this._signalDirtyDown) {
      return false;
    }
    if (this._dirty) {
      Matrix4x4.fromRotationTranslationScale(this._rotation, this._position, this._scale, this._localMatrix);
    }
    if (this._parent) {
      Matrix4x4.multiply(this._parent.worldMatrix, this._localMatrix, this._matrix);
    } else if (this._dirty) {
      Matrix4x4.copy(this._localMatrix, this._matrix);
    }

    this._dirty = false;
    this._signalDirtyDown = false;

    return true;
  }

  private signalDirtyDown(): void {
    if (!this._signalDirtyDown) {
      this._signalDirtyDown = true;
      for (const child of this._children) child.signalDirtyDown();
    }
  }

}
