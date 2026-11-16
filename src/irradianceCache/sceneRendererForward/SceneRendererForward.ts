import Renderer from "monks-webgpu/renderer/render/Renderer";
import Mesh from "monks-webgpu/renderer/mesh/Mesh";
import GeoSphere from "monks-webgpu/utils/mesh/geometries/GeoSphere";
import Cube from "monks-webgpu/utils/mesh/geometries/Cube";
import Cylinder from "monks-webgpu/utils/mesh/geometries/Cylinder";
import Camera from "monks-webgpu/renderer/camera/Camera";
import VertexFragmentShader from "monks-webgpu/renderer/material/VertexFragmentShader";
import Scene, { PrimitiveType } from "../scene/Scene";
import forward from "./forward.wgsl?raw";
import forwardMesh from "./forwardMesh.wgsl?raw";
import rayTraceScene from "../includes/rayTraceScene.wgsl?raw";
import hash from 'monks-webgpu/shaders/hash.wgsl?raw';
import normalPacking from 'monks-webgpu/shaders/normalPacking.wgsl?raw';
import gBufferPacking from '../includes/gBuffer.wgsl?raw';

export default class SceneRendererForward {
  private renderer:Renderer;
  private meshes: Mesh[] = [];
  public material:VertexFragmentShader;
  //the MESH models, which have no vertex buffer: see forwardMesh.wgsl
  private meshMaterial:VertexFragmentShader;

  constructor(renderer:Renderer){
    this.renderer = renderer;

    this.meshes.push(new GeoSphere(renderer, 1, 4));
    this.meshes.push(new Cube(renderer));
    this.meshes.push(new Cylinder(renderer, 1, 1, 2, 64));

    this.material = new VertexFragmentShader(renderer, 'forward', forward, {includes: [rayTraceScene, hash, normalPacking, gBufferPacking]});
    this.meshMaterial = new VertexFragmentShader(renderer, 'forwardMesh', forwardMesh, {includes: [rayTraceScene, normalPacking, gBufferPacking]});
    this.meshMaterial.cullMode = 'back';
  }

  public init(scene:Scene){
    scene.bind(this.material);
    this.material.setBuffer('modelIndicesByType', scene.modelIndicesByType);
    scene.bind(this.meshMaterial);
    this.meshMaterial.setBuffer('modelIndicesByType', scene.modelIndicesByType);
  }

  //one instanced draw per primitive type, each sorted front to back, then the sky.
  //the sky covers the whole screen, drawn first every pixel would be shaded twice
  public draw(camera:Camera, scene:Scene){
    scene.sortFrontToBack(camera.worldPosition);

    this.material.setMatrix('viewProjection', camera.viewProjection);
    this.material.setVector3('cameraPosition', camera.worldPosition);

    //the cubes first, they are most of the scene and occlude the most
    this.material.cullMode = 'back';
    for(const type of SceneRendererForward.drawOrder){
      const instanceCount = scene.instanceCounts[type];
      if(instanceCount == 0) continue;
      this.material.setFloat('instanceOffset', scene.instanceOffsets[type]);
      this.renderer.draw(this.meshes[type], this.material, {instanceCount});
    }

    const meshCount = scene.instanceCounts[PrimitiveType.MESH];
    if(meshCount > 0){
      this.meshMaterial.setMatrix('viewProjection', camera.viewProjection);
      this.meshMaterial.setVector3('cameraPosition', camera.worldPosition);
      this.meshMaterial.setFloat('instanceOffset', scene.instanceOffsets[PrimitiveType.MESH]);
      this.renderer.drawVertices(scene.mesh.triangleCount * 3, this.meshMaterial, null, {instanceCount: meshCount});
    }

    //the camera is inside the sky sphere
    this.material.cullMode = 'front';
    this.material.setFloat('instanceOffset', scene.skyInstanceOffset);
    this.renderer.draw(this.meshes[PrimitiveType.SPHERE], this.material, {instanceCount: 1});
  }

  private static readonly drawOrder = [PrimitiveType.CUBE, PrimitiveType.CYLINDER, PrimitiveType.SPHERE];

}