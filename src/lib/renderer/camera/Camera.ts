import Vector2 from '../math/Vector2';
import Vector3 from '../math/Vector3';
import Vector4 from '../math/Vector4';
import Projection from './Projection';
import Matrix4x4 from '../math/Matrix4x4';
import Transform from '../scene/Transform';
import ProjectionOrthographic from './ProjectionOrthographic';
import Ray from '../math/Ray';

export type CameraOptions = {
  fov: number;
  nearPlane: number;
  farPlane: number;
  orthographic: boolean;
};

export const defaultCameraOptions: CameraOptions = {
  fov: 1,
  nearPlane: 0.01,
  farPlane: 10,
  orthographic: false,
};

export default class Camera extends Transform {
  public projection: Projection;

  protected _viewProjection = new Matrix4x4();
  protected _viewMatrix = new Matrix4x4();
  private viewProjectionOld: Matrix4x4 = new Matrix4x4();
  private _changed: boolean = true;
  //the Projection.version the view projection was made with
  private projectionVersion: number = -1;
  //pass its signal to the dom event listeners of a camera controller, so destruct() removes them
  protected readonly listenerController = new AbortController();

  constructor(opts: Partial<CameraOptions> = {}) {
    const options = {...defaultCameraOptions, ...opts};
    super();
    if (options.orthographic) {
      this.projection = new ProjectionOrthographic(new Vector2(-1, -1), new Vector2(1, 1), options.nearPlane, options.farPlane);
    } else {
      this.projection = new Projection(options.fov, options.nearPlane, options.farPlane);
    }
  }

  public checkChanged(){
    const viewProjection = this.viewProjection;
    const m = viewProjection.m;
    const old = this.viewProjectionOld.m;
    this._changed = false;
    for(let i = 0; i < 16; i++) {
      if(m[i] !== old[i]) {
        this._changed = true;
        this.viewProjectionOld.copy(viewProjection);
        break;
      }
    }
  }

  public get changed(): boolean {
    return this._changed;
  }

  public get view() {
    return this;
  }

  public get viewProjection(): Matrix4x4 {
    this.updateMatrix();
    return this._viewProjection;
  }

  public get projectionMatrix(): Matrix4x4 {
    return this.projection.matrix;
  }

  public get viewMatrix(): Matrix4x4 {
    this.updateMatrix();
    return this._viewMatrix;
  }

  public get matrix(): Matrix4x4 {
    return this.viewMatrix;
  }

  public get worldRotationMatrix(): Matrix4x4 {
    return Matrix4x4.invert(this.worldRotationMatrixMat4);
  }

  public worldToScreenNormalized(worldPos: Vector3): Vector4 {
    const out = new Vector4(worldPos.x, worldPos.y, worldPos.z, 1);
    out.transform(this.viewProjection);
    out.x /= out.w;
    out.y /= out.w;
    out.x = out.x * 0.5 + 0.5;
    out.y = out.y * -0.5 + 0.5;

    return out;
  }

  public worldToScreen(worldPos: Vector3): Vector4 {
    const out = new Vector4(worldPos.x, worldPos.y, worldPos.z, 1);
    out.transform(this.viewProjection);
    out.x /= out.w;
    out.y /= out.w;

    return out;
  }

  // accepts normalized screen pos [0->1]
  public screenToWorldPoint(screenPos: Vector2): Vector3 {
    // from 0 - 1 to opengl coords
    const x = 2.0 * screenPos.x - 1;
    const y = -2.0 * screenPos.y + 1;
    return new Vector3(x, y, 1).transform(Matrix4x4.invert(this.viewProjection));
  }

  public getDepth(worldPos: Vector3): number {
    return new Vector4(worldPos.x, worldPos.y, worldPos.z, 1).transform(this.viewProjection).z;
  }

  /** Picking ray through a screen position in NDC (x and y in [-1,1]). */
  public getRay(screenPosition: Vector2, out: Ray = new Ray()): Ray {
    if (this.projection instanceof ProjectionOrthographic) {
      // rays are parallel; the origin sits on the near plane under the cursor
      const invViewProj = Matrix4x4.invert(this.viewProjection);

      const nearPoint = new Vector4(screenPosition.x, screenPosition.y, 0, 1).transform(invViewProj);
      nearPoint.x /= nearPoint.w;
      nearPoint.y /= nearPoint.w;
      nearPoint.z /= nearPoint.w;

      const farPoint = new Vector4(screenPosition.x, screenPosition.y, 1, 1).transform(invViewProj);
      farPoint.x /= farPoint.w;
      farPoint.y /= farPoint.w;
      farPoint.z /= farPoint.w;

      out.origin.setValues(nearPoint.x, nearPoint.y, nearPoint.z);
      out.direction
        .setValues(farPoint.x - nearPoint.x, farPoint.y - nearPoint.y, farPoint.z - nearPoint.z)
        .normalize();
    } else {
      // all rays originate from the camera position
      out.origin.copy(this.worldPosition);

      const dir = new Vector4(screenPosition.x, screenPosition.y, 1, 1).transform(Matrix4x4.invert(this.projectionMatrix));
      dir.z = -1;
      dir.w = 0;
      //the world matrix is the inverse of the view matrix
      dir.transform(this.worldMatrix);

      out.direction.setValues(dir.x, dir.y, dir.z).normalize();
    }
    return out;
  }

  //removes the dom event listeners
  public destruct() {
    this.listenerController.abort();
  }

  protected updateMatrix(): boolean {
    const transformChanged = super.updateMatrix();
    //updates the projection when it is dirty. by version: reading projectionMatrix or frustumCorner elsewhere first
    //clears the dirty flag, but not the change
    const projectionMatrix = this.projection.matrix;
    if (transformChanged || this.projection.version !== this.projectionVersion) {
      //before worldMatrix, which calls this again
      this.projectionVersion = this.projection.version;
      Matrix4x4.invert(this.worldMatrix, this._viewMatrix);
      Matrix4x4.multiply(projectionMatrix, this._viewMatrix, this._viewProjection);
      return true;
    } else {
      return false;
    }
  }
}
