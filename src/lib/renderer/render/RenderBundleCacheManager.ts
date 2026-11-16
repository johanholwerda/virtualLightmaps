type CachedBundle = {
  //the render target state, see Renderer.getAttachmentKey
  attachmentKey: string;
  version: number;
  bindGroupIds: number[];
  uniformOffset: number;
  bundle: GPURenderBundle;
  lastUse: number;
}

/*
caches the render bundles of Renderer.drawCachedBundle and removes the ones that haven't been used for a while.
by material and key, with a few bundles each (like for bind groups that are switched between), which are told apart
by comparing what they were recorded with, so a lookup builds no strings
 */
export class RenderBundleCacheManager {
  private static readonly MAX_UNUSED_FRAMES = 10;

  private bundlesByMaterial: Map<number, Map<string | number, CachedBundle[]>> = new Map();
  private frameNumber: number = 0;

  public get(materialId: number, key: string | number, attachmentKey: string, version: number, bindGroupIds: number[], uniformOffset: number): GPURenderBundle | undefined {
    const bundles = this.bundlesByMaterial.get(materialId)?.get(key);
    if (!bundles) return undefined;
    for (const cached of bundles) {
      if (cached.attachmentKey === attachmentKey && cached.version === version && cached.uniformOffset === uniformOffset && RenderBundleCacheManager.equals(cached.bindGroupIds, bindGroupIds)) {
        cached.lastUse = this.frameNumber;
        return cached.bundle;
      }
    }
    return undefined;
  }

  //bindGroupIds is copied, so the caller can reuse its array
  public set(materialId: number, key: string | number, attachmentKey: string, version: number, bindGroupIds: number[], uniformOffset: number, bundle: GPURenderBundle): void {
    let bundlesByKey = this.bundlesByMaterial.get(materialId);
    if (!bundlesByKey) {
      bundlesByKey = new Map();
      this.bundlesByMaterial.set(materialId, bundlesByKey);
    }
    let bundles = bundlesByKey.get(key);
    if (!bundles) {
      bundles = [];
      bundlesByKey.set(key, bundles);
    }
    bundles.push({ attachmentKey, version, bindGroupIds: bindGroupIds.slice(), uniformOffset, bundle, lastUse: this.frameNumber });
  }

  public get size(): number {
    let size = 0;
    for (const bundlesByKey of this.bundlesByMaterial.values()) {
      for (const bundles of bundlesByKey.values()) {
        size += bundles.length;
      }
    }
    return size;
  }

  public update(): void {
    for (const [materialId, bundlesByKey] of this.bundlesByMaterial) {
      for (const [key, bundles] of bundlesByKey) {
        let kept = 0;
        for (const cached of bundles) {
          if (this.frameNumber - cached.lastUse <= RenderBundleCacheManager.MAX_UNUSED_FRAMES) bundles[kept++] = cached;
        }
        bundles.length = kept;
        if (kept === 0) bundlesByKey.delete(key);
      }
      if (bundlesByKey.size === 0) this.bundlesByMaterial.delete(materialId);
    }
    this.frameNumber++;
  }

  private static equals(a: number[], b: number[]): boolean {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
      if (a[i] !== b[i]) return false;
    }
    return true;
  }
}
