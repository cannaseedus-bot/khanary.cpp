// compile_cssc_gemm.js
// Compiles cssc_gemm.kuhul (declarative format) → HLSL / WGSL / GLSL / CPU K'UHUL / XVM descriptor
//
// Run from drivers/klsl/:
//   node compile_cssc_gemm.js

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { KLSLCompiler } from './src/compiler/klsl-compiler.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const SRC_PATH   = path.join(__dirname, 'cssc_gemm.kuhul');
const OUTPUT_DIR = path.join(__dirname, 'build', 'shaders');

console.log('╔══════════════════════════════════════════════════════════════╗');
console.log('║        CSSC_GEMM — .kuhul → ALL TARGETS + CPU REDIRECT      ║');
console.log('╚══════════════════════════════════════════════════════════════╝');
console.log('');

const source = fs.readFileSync(SRC_PATH, 'utf-8');
console.log(`[0] Source: ${SRC_PATH} (${source.length} bytes)`);

const compiler = new KLSLCompiler();
const fmt = compiler.detectFormat(source);
console.log(`    Format detected: ${fmt}`);

fs.mkdirSync(OUTPUT_DIR, { recursive: true });

// ============================================================
// 1  WGSL  — WebGPU tier (native kernel, no translation)
// ============================================================

console.log('\n[1] Compiling → WGSL (WebGPU)...');
const wgsl = compiler.compile(source, { target: 'wgsl' });
fs.writeFileSync(path.join(OUTPUT_DIR, 'cssc_gemm.wgsl'), wgsl);
console.log(`    ${wgsl.length} bytes → build/shaders/cssc_gemm.wgsl`);

// ============================================================
// 2  HLSL  — sm5 / D3D11 tier
// ============================================================

console.log('\n[2] Compiling → HLSL (sm5)...');
const hlsl = compiler.compile(source, { target: 'hlsl' });
fs.writeFileSync(path.join(OUTPUT_DIR, 'cssc_gemm.hlsl'), hlsl);
console.log(`    ${hlsl.length} bytes → build/shaders/cssc_gemm.hlsl`);

// ============================================================
// 3  GLSL  — WebGL2 / ES 3.1 tier
// ============================================================

console.log('\n[3] Compiling → GLSL (WebGL2 ES 3.1)...');
const glsl = compiler.compile(source, { target: 'glsl' });
fs.writeFileSync(path.join(OUTPUT_DIR, 'cssc_gemm.glsl'), glsl);
console.log(`    ${glsl.length} bytes → build/shaders/cssc_gemm.glsl`);

// ============================================================
// 4  K'UHUL CPU descriptor  — CPU tier redirect
//    When webgpu/sm5/webgl2 all fail (e.g. Intel HD 4600 on D3D12)
//    the gate falls to CPU opcodes.  The opcodes are the same as
//    GPU; they run through the XVM interpreter from this .kuhul.
// ============================================================

console.log("\n[4] Emitting K'UHUL CPU descriptor (CPU tier redirect)...");
const kuhulCpu = compiler.compile(source, { target: 'kuhul' });
fs.writeFileSync(path.join(OUTPUT_DIR, 'cssc_gemm.cpu.kuhul'), kuhulCpu);
console.log(`    ${kuhulCpu.length} bytes → build/shaders/cssc_gemm.cpu.kuhul`);

// ============================================================
// 5  XVM dispatch descriptor  — JSON pointer the gate reads
// ============================================================

console.log('\n[5] Emitting XVM dispatch descriptor...');
const xvmDesc = compiler.compile(source, {
  target: 'xvm',
  name: 'cssc_gemm',
  sourceKuhul: './cssc_gemm.cpu.kuhul'
});
fs.writeFileSync(path.join(OUTPUT_DIR, 'cssc_gemm.xvm.json'), xvmDesc);
console.log(`    ${xvmDesc.length} bytes → build/shaders/cssc_gemm.xvm.json`);

// ============================================================
// 6  Gate registration JSON
//    The fallback chain: webgpu → sm5 → webgl2 → cpu
//    cpu.source_kuhul points to the .cpu.kuhul file above;
//    klslc.exe --xvm compiles it to XVM bytecode at dispatch time.
// ============================================================

console.log('\n[6] Writing gate registration...');

const gateReg = {
  name: 'cssc_gemm',
  version: '1.0.0',
  source: {
    kuhul: SRC_PATH,
    format: fmt
  },
  shader: {
    wgsl:  path.join(OUTPUT_DIR, 'cssc_gemm.wgsl'),
    hlsl:  path.join(OUTPUT_DIR, 'cssc_gemm.hlsl'),
    glsl:  path.join(OUTPUT_DIR, 'cssc_gemm.glsl')
  },
  cpu: {
    source_kuhul: path.join(OUTPUT_DIR, 'cssc_gemm.cpu.kuhul'),
    xvm_desc:     path.join(OUTPUT_DIR, 'cssc_gemm.xvm.json'),
    compile_cmd:  'klslc.exe --xvm cssc_gemm.cpu.kuhul -o cssc_gemm.xvm'
  },
  entryPoint: 'CSMain',
  workgroup: [16, 16, 1],
  tierSupport: ['webgpu', 'sm5', 'webgl2', 'cpu'],
  fallbackChain: 'webgpu -> sm5 -> webgl2 -> cpu',
  uniforms: { M: 'u32', N: 'u32', K: 'u32' },
  inputs:  { A: { type: 'f32' }, B: { type: 'f32' } },
  outputs: { C: { type: 'f32' } }
};

fs.writeFileSync(
  path.join(OUTPUT_DIR, 'cssc_gemm.gate.json'),
  JSON.stringify(gateReg, null, 2)
);
console.log(`    Wrote: build/shaders/cssc_gemm.gate.json`);

// ============================================================
// SUMMARY
// ============================================================

console.log('\n╔══════════════════════════════════════════════════════════════╗');
console.log('║                   COMPILATION COMPLETE                       ║');
console.log('╚══════════════════════════════════════════════════════════════╝\n');
console.log('Outputs:');
console.log('  build/shaders/cssc_gemm.wgsl          WebGPU (native WGSL kernel)');
console.log('  build/shaders/cssc_gemm.hlsl          sm5 / D3D11');
console.log('  build/shaders/cssc_gemm.glsl          WebGL2 ES 3.1');
console.log("  build/shaders/cssc_gemm.cpu.kuhul     CPU tier — K'UHUL XVM descriptor");
console.log('  build/shaders/cssc_gemm.xvm.json      XVM dispatch descriptor');
console.log('  build/shaders/cssc_gemm.gate.json     CSS-C gate registration\n');
console.log('Fallback chain: webgpu → sm5 → webgl2 → cpu');
console.log('CPU redirect:   cssc_gemm.cpu.kuhul → klslc.exe --xvm → XVM bytecode');
console.log('Opcodes:        same for CPU and GPU paths (XVM = unified ISA)\n');
