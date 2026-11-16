import ParamGroup from "webGPU-framework/utils/params/ParamGroup";
import type { ParamMaterial } from "webGPU-framework/utils/params/ParamBase";
import ParamInt from "webGPU-framework/utils/params/ParamInt";
import ParamFloat from "webGPU-framework/utils/params/ParamFloat";
import ReprojectFill from "../shared/ReprojectFill";

//reprojects the lighting of the previous frame and path traces the pixels that have no history or too few samples.
//New pixels get a cheap path trace (1 spp by default) in the frame they appear, then samples accumulate within a fixed
//refine budget: a fast camera rotation disoccludes a lot of pixels, path tracing them with many samples at once made the
//frame time spike. Resolve blurs the pixels that have few samples yet. TemporalAccumulate is the baseline to compare with
export default class ReprojectAndFill extends ReprojectFill {
  private refineSamples!: ParamFloat;
  private sampleCount!: ParamInt;

  protected initFill(pgFill:ParamGroup, pgFillMaterials:ParamMaterial[]){
    //the samples a new pixel gets, in the frame it appears. Every new pixel gets them
    const params = { materials: pgFillMaterials };
    pgFill.addParamInt('newSampleCount', 8, 1, 16, params);
    //1 in newPixelStride new pixels is path traced per frame, the resolve fills in the others: 4 makes a fast camera
    //rotation cost at most 1/4 spp. See reproject.wgsl
    pgFill.addParamInt('newPixelStride', 1, 1, 16, params);
    //the samples a refined pixel gets per frame. 1 with a 4x budget traces the same number of indirect paths as 4, but
    //with 4x the threads and sun shadow rays. It spreads the samples over 4x the pixels, which gives the resolve blur
    //more neighbours with samples
    this.sampleCount = pgFill.addParamInt('sampleCount', 1, 1, 64, params);
    //the sample count at which a pixel stops getting samples. The reprojection leaves those out of the refine pixels
    pgFill.addParamFloat('accumulationMax', 256, 1, 2048, params);
    //the pixels with fewer samples get the refine budget first. Around the resolve's denoiseSamples: it blurs them
    pgFill.addParamFloat('prioritySamples', 16, 1, 256, params);
    //the refine budget in samples per screen pixel per frame, so it scales with the resolution: .125 is about 1M
    //samples per frame at 4K, a 1/8 spp path trace of the screen. The fixed budget it replaced (256K pixels) was .125
    //at 1080p but only .03 at 4K, which resolved slowly there. Divided by sampleCount into refined pixels, see
    //getRefineBudget
    this.refineSamples = pgFill.addParamFloat('refineSamples', .125, .01, 1);
  }

  protected getRefineBudget(pixelCount:number){
    return Math.max(1, Math.round(pixelCount * this.refineSamples.value / this.sampleCount.value));
  }
}
