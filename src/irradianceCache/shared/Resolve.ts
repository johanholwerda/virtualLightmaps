import Renderer from "monks-webgpu/renderer/render/Renderer";
import VertexFragmentShader from "monks-webgpu/renderer/material/VertexFragmentShader";
import Texture from "monks-webgpu/renderer/texture/Texture";
import resolve from "./resolve.wgsl?raw";
import RenderTexture from "monks-webgpu/renderer/texture/RenderTexture";
import ParamGroup from "monks-webgpu/utils/params/ParamGroup";
import toneMap from 'monks-webgpu/shaders/toneMap.wgsl?raw';
import Vector3 from "monks-webgpu/renderer/math/Vector3";
import normalPacking from 'monks-webgpu/shaders/normalPacking.wgsl?raw';
import gBufferPacking from '../includes/gBuffer.wgsl?raw';
import cameraRay from '../includes/cameraRay.wgsl?raw';
import Camera from "monks-webgpu/renderer/camera/Camera";
import Matrix4x4 from "monks-webgpu/renderer/math/Matrix4x4";
import Scene from "../scene/Scene";

export default class Resolve {
  private renderer:Renderer;
  private material:VertexFragmentShader;
  private inverseView: Matrix4x4 = new Matrix4x4();
  private inverseProjection: Matrix4x4 = new Matrix4x4();

  constructor(renderer:Renderer){
    this.renderer = renderer;

    this.material = new VertexFragmentShader(renderer, 'resolve', resolve, {includes: [toneMap, normalPacking, gBufferPacking, cameraRay]});
  }

  public get exposureMaterial() {
    return this.material;
  }

  //denoiseSamples 1: no blur
  public init(paramGroup:ParamGroup, scene:Scene, denoiseSamples = 16, sampleCountMax = denoiseSamples){
    scene.bindSkyColor(this.material);

    const params = { materials: [this.material] };
    const pg = paramGroup.addGroup('resolve');
    //the sample count at which the blur of the pixels with few samples has faded out
    pg.addParamFloat('denoiseSamples', denoiseSamples, 1, 256, params);
    //the distance of a neighbour to the plane of the pixel, relative to its depth, up to which it counts as the same surface
    pg.addParamFloat('depthTolerance', .05, 0, .2, params);
    pg.addParamBool('showSampleCount', false, params);
    //the sample count that shows white
    pg.addParamFloat('sampleCountMax', sampleCountMax, 1, 256, params);
  }

  public draw(source:Texture, gBuffer:RenderTexture, camera:Camera, destination:RenderTexture | null = null){
    Matrix4x4.invert(camera.viewMatrix, this.inverseView);
    Matrix4x4.invert(camera.projectionMatrix, this.inverseProjection);
    this.material.setMatrix('inverseView', this.inverseView);
    this.material.setMatrix('inverseProjection', this.inverseProjection);
    this.material.setVector3('cameraPosition', camera.worldPosition);
    this.material.setTexture('albedoTexture', gBuffer.views[0]);
    this.material.setTexture('normalDepthTexture', gBuffer.views[1]);
    this.renderer.blit(source, destination, this.material);
  }

}