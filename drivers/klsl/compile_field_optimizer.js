// compile_field_optimizer.js
// Compiles field_optimizer.klsl → HLSL / WGSL / GLSL and registers with CSS-C gate
//
// Run from drivers/klsl/:
//   node compile_field_optimizer.js

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { KLSLCompiler } from './src/compiler/klsl-compiler.js';
import { CSSMLCompiler } from './src/compiler/index.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const KLSL_PATH     = path.join(__dirname, 'field_optimizer.klsl');
const MANIFEST_PATH = path.join(__dirname, 'field_optimizer.manifest.cssc');
const OUTPUT_DIR    = path.join(__dirname, 'build', 'shaders');

console.log('╔══════════════════════════════════════════════════════════════╗');
console.log('║      FIELD OPTIMIZER — KLSL → HLSL → CSS-C                  ║');
console.log('╚══════════════════════════════════════════════════════════════╝');
console.log('');

// ============================================================
// STEP 1: Read KLSL source
// ============================================================

const klslSource = fs.readFileSync(KLSL_PATH, 'utf-8');
console.log(`[1] KLSL: ${KLSL_PATH} (${klslSource.length} bytes)`);

// ============================================================
// STEP 2: Compile KLSL → HLSL (sm5 / D3D11)
// ============================================================

console.log('\n[2] Compiling KLSL → HLSL...');
const klslCompiler = new KLSLCompiler();
const hlslOutput = klslCompiler.compile(klslSource, {
  target: 'hlsl',
  shaderModel: '5.0',
  profiles: { CSMain: 'cs_5_0', GradAccumCS: 'cs_5_0' }
});

fs.mkdirSync(OUTPUT_DIR, { recursive: true });
fs.writeFileSync(path.join(OUTPUT_DIR, 'field_optimizer.hlsl'), hlslOutput);
console.log(`   Generated ${hlslOutput.length} bytes`);
console.log(`   Wrote: ${OUTPUT_DIR}/field_optimizer.hlsl`);

// ============================================================
// STEP 3: Compile KLSL → WGSL (WebGPU)
// ============================================================

console.log('\n[3] Compiling KLSL → WGSL...');
const wgslOutput = klslCompiler.compile(klslSource, {
  target: 'wgsl',
  workgroupSize: [256, 1, 1]
});
fs.writeFileSync(path.join(OUTPUT_DIR, 'field_optimizer.wgsl'), wgslOutput);
console.log(`   Wrote: ${OUTPUT_DIR}/field_optimizer.wgsl`);

// ============================================================
// STEP 4: Compile KLSL → GLSL (WebGL2 / ES 3.1)
// ============================================================

console.log('\n[4] Compiling KLSL → GLSL (WebGL2)...');
const glslOutput = klslCompiler.compile(klslSource, {
  target: 'glsl',
  version: 'es_310',
  compute: true
});
fs.writeFileSync(path.join(OUTPUT_DIR, 'field_optimizer.glsl'), glslOutput);
console.log(`   Wrote: ${OUTPUT_DIR}/field_optimizer.glsl`);

// ============================================================
// STEP 5: Parse CSS-C manifest
// ============================================================

console.log('\n[5] Parsing CSS-C manifest...');
const manifestSource = fs.readFileSync(MANIFEST_PATH, 'utf-8');
const cssmlCompiler = new CSSMLCompiler();
const compiledManifest = cssmlCompiler.compile(manifestSource);

console.log(`   Tensors:    ${compiledManifest.graph.tensors.length}`);
console.log(`   Operations: ${compiledManifest.graph.operations.length}`);
console.log(`   Kernels:    ${compiledManifest.kernels.size}`);

fs.writeFileSync(path.join(OUTPUT_DIR, 'field_optimizer.cssc'), compiledManifest.manifest);
console.log(`   Wrote: ${OUTPUT_DIR}/field_optimizer.cssc`);

// ============================================================
// STEP 6: Write gate registration JSON
// ============================================================

console.log('\n[6] Writing gate registration...');

const gateRegistration = {
  name: 'field_optimizer',
  version: '1.0.0',
  shader: {
    klsl: KLSL_PATH,
    hlsl: path.join(OUTPUT_DIR, 'field_optimizer.hlsl'),
    wgsl: path.join(OUTPUT_DIR, 'field_optimizer.wgsl'),
    glsl: path.join(OUTPUT_DIR, 'field_optimizer.glsl')
  },
  entryPoints: { CSMain: 'cs_5_0', GradAccumCS: 'cs_5_0' },
  workgroup: [256, 1, 1],
  phase: "Ch'en",
  tierSupport: ['webgpu', 'sm5', 'webgl2', 'cpu'],
  fields: {
    gravity_scale: { antigrav: 0.0, normal: 1.0, heavygrav: 2.0 }
  },
  tensors: {
    params: { type: 'float', size: 'N' },
    grads:  { type: 'float', size: 'N' },
    m1:     { type: 'float', size: 'N' },
    m2:     { type: 'float', size: 'N' }
  },
  uniforms: [
    'lr', 'beta1', 'beta2', 'epsilon', 'weight_decay',
    'grad_clip', 'target_loss', 'current_loss',
    'gravity_scale', 'step', 'wind_direction', 'inertia_friction', 'N'
  ]
};

fs.writeFileSync(
  path.join(OUTPUT_DIR, 'field_optimizer.gate.json'),
  JSON.stringify(gateRegistration, null, 2)
);
console.log(`   Wrote: ${OUTPUT_DIR}/field_optimizer.gate.json`);

// ============================================================
// STEP 7: Emit runtime binding
// ============================================================

console.log('\n[7] Emitting CSS-C runtime binding...');

const runtimeBinding = `// Auto-generated — do not edit
// field_optimizer runtime binding for CSS-C gate
import { registerKernel } from '../../cssc_gate_run/cssc_gate.js';

registerKernel({
  name: 'field_optimizer',
  entryPoint: 'CSMain',
  workgroup: [256, 1, 1],
  phase: "Ch'en",
  shaderSources: {
    hlsl: './shaders/field_optimizer.hlsl',
    wgsl: './shaders/field_optimizer.wgsl',
    glsl: './shaders/field_optimizer.glsl'
  },
  tensors: {
    params: { binding: 0, access: 'read_write', type: 'float' },
    grads:  { binding: 1, access: 'read_write', type: 'float' },
    m1:     { binding: 2, access: 'read_write', type: 'float' },
    m2:     { binding: 3, access: 'read_write', type: 'float' }
  },
  uniforms: {
    lr: 0.001, beta1: 0.9, beta2: 0.999, epsilon: 1e-8,
    weight_decay: 0.01, grad_clip: 1.0, target_loss: 0.1,
    current_loss: 2.5, gravity_scale: 1.0, step: 1,
    wind_direction: 1.0, inertia_friction: 0.5, N: 0
  },
  variants: {
    antigrav:  { gravity_scale: 0.0 },
    normal:    { gravity_scale: 1.0 },
    heavygrav: { gravity_scale: 2.0 }
  }
});

console.log('[field_optimizer] Registered with CSS-C gate');
`;

fs.writeFileSync(path.join(OUTPUT_DIR, 'field_optimizer.register.js'), runtimeBinding);
console.log(`   Wrote: ${OUTPUT_DIR}/field_optimizer.register.js`);

// ============================================================
// SUMMARY
// ============================================================

console.log('\n╔══════════════════════════════════════════════════════════════╗');
console.log('║                      COMPILATION COMPLETE                    ║');
console.log('╚══════════════════════════════════════════════════════════════╝\n');
console.log('Outputs:');
console.log(`  ${OUTPUT_DIR}/field_optimizer.hlsl          (sm5 / d3d11)`);
console.log(`  ${OUTPUT_DIR}/field_optimizer.wgsl          (webgpu)`);
console.log(`  ${OUTPUT_DIR}/field_optimizer.glsl          (webgl2 / es 3.1)`);
console.log(`  ${OUTPUT_DIR}/field_optimizer.cssc          (manifest)`);
console.log(`  ${OUTPUT_DIR}/field_optimizer.gate.json     (gate registration)`);
console.log(`  ${OUTPUT_DIR}/field_optimizer.register.js   (runtime binding)`);
console.log('\nGravity variants:');
console.log('  .field-optimizer.antigrav   → gravity_scale = 0.0  (bypass)');
console.log('  .field-optimizer.normal     → gravity_scale = 1.0  (standard)');
console.log('  .field-optimizer.heavygrav  → gravity_scale = 2.0  (loss head)');
console.log('\nReady to dispatch through the gate.');
