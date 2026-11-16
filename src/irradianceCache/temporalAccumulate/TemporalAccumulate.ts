import ParamGroup from "webGPU-framework/utils/params/ParamGroup";
import type { ParamMaterial } from "webGPU-framework/utils/params/ParamBase";
import Camera from "webGPU-framework/renderer/camera/Camera";
import ReprojectFill from "../shared/ReprojectFill";

//the baseline to compare ReprojectAndFill with: every pixel gets 1 sample per frame, like groundTruth, and the
//reprojected history is an exponential average over about accumulationMax frames, like temporal AA without clamping.
//With a static camera the pixels at accumulationMax stop getting samples
export default class TemporalAccumulate extends ReprojectFill {

  protected initFill(pgFill:ParamGroup, pgFillMaterials:ParamMaterial[]){
    this.fillShader.setFloat('newSampleCount', 1);
    this.reprojectShader.setFloat('newPixelStride', 1);
    this.fillShader.setFloat('sampleCount', 1);
    //the frames the exponential average is about long while the camera moves, and the samples at which a pixel
    //stops getting them while it is static. See beforeReproject
    pgFill.addParamFloat('accumulationMax', 64, 1, 256, { materials: pgFillMaterials });
  }

  //all pixels are refined
  protected getRefineBudget(pixelCount:number){
    return pixelCount;
  }

  //moving: every pixel with history is refined, at accumulationMax too. The fill keeps the count there, which makes the
  //running average an exponential one. Static: only the pixels below accumulationMax are refined, the others stop
  //getting samples. The refine budget covers all pixels, see getRefineBudget
  protected beforeReproject(camera:Camera){
    //not the view projection, that has the TAA jitter. A moving light changes the lighting like a moving camera
    const isStatic = !camera.changed && !this.scene.lightChanging;
    //the pixels below prioritySamples are refined, see reproject.wgsl
    this.reprojectShader.setFloat('prioritySamples', isStatic ? 0 : 1e30);
  }
}
