// src/compiler/index.js — CSS-C / CSSML compiler stub
// Full CSSML compiler lives in the CSS-C gate; this stub handles
// manifest parsing for the field optimizer compile driver.

export class CSSMLCompiler {
  compile(source) {
    const tensors = [];
    const operations = [];
    const kernels = new Map();

    // Extract --tensor declarations
    for (const m of source.matchAll(/--(\w+):\s*tensor\(([^)]+)\)/g)) {
      tensors.push({ name: m[1], spec: m[2] });
    }

    // Extract compute-shader blocks
    for (const m of source.matchAll(/\.([\w-]+)\s*\{[^}]*entry_point:\s*"(\w+)"/gs)) {
      kernels.set(m[1], { selector: m[1], entryPoint: m[2] });
      operations.push({ selector: m[1], entryPoint: m[2] });
    }

    // Emit minimal manifest JSON for gate registration
    const manifest = JSON.stringify({
      tensors: tensors.map(t => t.name),
      kernels: Object.fromEntries(kernels),
      operations: operations.map(o => o.selector)
    }, null, 2);

    return { graph: { tensors, operations }, kernels, manifest };
  }
}
