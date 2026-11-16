# virtualLightmap

The irradiance cache example of the MediaMonks WebGPU framework (`src/sandbox/examples/irradianceCache`), isolated, with
the virtual lightmap as the default draw mode. The surfels draw mode and the performance test are removed. `src/lib` holds
only the framework files the example uses, imported as `monks-webgpu/...`.

## Running

```
npm i
npm run dev
```

Other commands: `npm run build` (to `dist`), `npm run preview` and `npm run typecheck`.

## Layout

- `src/main.ts`: starts the example
- `src/irradianceCache/IrradianceCacheDraw.ts`: the draw modes and the GUI
- `src/irradianceCache/virtualLightmap`: the virtual lightmap
- `src/irradianceCache/data/BattleHelmet.json`: the helmet model of the scene
- `src/lib`: the trimmed framework
