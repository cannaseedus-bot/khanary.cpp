// klsl-compiler.js — KLSL → HLSL / WGSL / GLSL / K'UHUL compiler
// Handles two source formats:
//   glyph:       ⟁ shader ... [Xul]  (.klsl files)
//   declarative: @shader_language / @shaders JSON block  (.kuhul files)

export class KLSLCompiler {

  // ============================================================
  // FORMAT DETECTION
  // ============================================================

  detectFormat(source) {
    const trimmed = source.trimStart();
    if (trimmed.startsWith('@shader_language') || trimmed.startsWith('@shaders')) return 'declarative';
    if (trimmed.includes('⟁ shader')) return 'glyph';
    // second pass: any @-directive line before a JSON block
    if (/@\w+\s*:/.test(trimmed)) return 'declarative';
    return 'glyph';
  }

  compile(source, options = {}) {
    const target = options.target || 'hlsl';
    const fmt = this.detectFormat(source);

    if (fmt === 'declarative') {
      const decl = this.parseDeclarative(source);
      switch (target) {
        case 'hlsl':  return this.emitHLSLDeclarative(decl, options);
        case 'wgsl':  return this.emitWGSLDeclarative(decl, options);
        case 'glsl':  return this.emitGLSLDeclarative(decl, options);
        case 'kuhul': return this.emitKUHULDeclarative(decl, options);
        case 'xvm':   return this.emitXVMDescDeclarative(decl, options);
        default: throw new Error(`Unknown target: ${target}`);
      }
    }

    const ast = this.parse(source);
    switch (target) {
      case 'hlsl':  return this.emitHLSL(ast, options);
      case 'wgsl':  return this.emitWGSL(ast, options);
      case 'glsl':  return this.emitGLSL(ast, options);
      case 'kuhul': return this.emitKUHULGlyph(ast, options);
      default: throw new Error(`Unknown target: ${target}`);
    }
  }

  // ============================================================
  // DECLARATIVE PARSER  (@shader_language / @shaders JSON)
  // ============================================================

  parseDeclarative(source) {
    const decl = { shaderLanguage: 'wgsl', targetBackends: [], shaders: {} };

    // @shader_language: wgsl
    const langM = source.match(/@shader_language\s*:\s*(\w+)/);
    if (langM) decl.shaderLanguage = langM[1].trim();

    // @target_backends: ["wgpu","d3d11"]
    const beM = source.match(/@target_backends\s*:\s*(\[.*?\])/s);
    if (beM) {
      try { decl.targetBackends = JSON.parse(beM[1]); } catch (_) {}
    }

    // @shaders: { ... }  — find the opening brace after "@shaders:"
    const shadersIdx = source.indexOf('@shaders:');
    if (shadersIdx !== -1) {
      const afterDir = source.slice(shadersIdx + '@shaders:'.length).trimStart();
      // balanced-brace extraction
      if (afterDir.startsWith('{')) {
        let depth = 0, i = 0;
        for (; i < afterDir.length; i++) {
          if (afterDir[i] === '{') depth++;
          else if (afterDir[i] === '}') { depth--; if (depth === 0) { i++; break; } }
        }
        try { decl.shaders = JSON.parse(afterDir.slice(0, i)); } catch (_) {}
      }
    }

    return decl;
  }

  // ============================================================
  // PARSER — KLSL → AST
  // ============================================================

  parse(source) {
    const lines = source.split('\n');
    const ast = { shaders: [], constants: [], buffers: [], functions: [] };
    let currentShader = null;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const trimmed = line.trim();

      if (trimmed.startsWith('#') || trimmed === '') continue;

      if (trimmed.startsWith('⟁ shader')) {
        currentShader = {
          name: trimmed.replace('⟁ shader', '').trim(),
          stage: null, threads: null, shaderModel: null,
          constants: [], buffers: [], functions: [], entries: []
        };
        continue;
      }

      if (trimmed.includes('⟁Wo⟁ stage')) {
        const m = trimmed.match(/stage\s+"([^"]+)"/);
        if (m && currentShader) currentShader.stage = m[1];
        continue;
      }

      if (trimmed.includes('⟁Wo⟁ threads')) {
        const m = trimmed.match(/threads\s+\[([^\]]+)\]/);
        if (m && currentShader) currentShader.threads = m[1].split(',').map(s => s.trim());
        continue;
      }

      if (trimmed.includes('⟁Wo⟁ shader_model')) {
        const m = trimmed.match(/shader_model\s+"([^"]+)"/);
        if (m && currentShader) currentShader.shaderModel = m[1];
        continue;
      }

      if (trimmed.includes('⟁Wo⟁ ConstantBuffer')) {
        const cb = this.parseConstantBuffer(lines, i, currentShader);
        if (cb) { currentShader.constants.push(cb.buffer); i = cb.endIndex; }
        continue;
      }

      if (trimmed.includes('⟁Wo⟁ RWStructuredBuffer') || trimmed.includes('⟁Wo⟁ StructuredBuffer')) {
        const buf = this.parseBuffer(trimmed);
        if (buf && currentShader) currentShader.buffers.push(buf);
        continue;
      }

      if (trimmed.includes('⟁Wo⟁ fn')) {
        const fn = this.parseFunction(lines, i, currentShader);
        if (fn) { currentShader.functions.push(fn.func); i = fn.endIndex; }
        continue;
      }

      if (trimmed.includes('⟁Pop')) {
        const entry = this.parseEntry(lines, i, currentShader);
        if (entry) { currentShader.entries.push(entry.entry); i = entry.endIndex; }
        continue;
      }

      if (trimmed === '[Xul]' && currentShader) {
        ast.shaders.push(currentShader);
        currentShader = null;
      }
    }

    return ast;
  }

  parseConstantBuffer(lines, startIndex, shader) {
    const buffer = { name: '', fields: [] };
    const m = lines[startIndex].match(/ConstantBuffer\s+(\w+)\s*:\s*register\((\w+)\)/);
    if (!m) return null;
    buffer.name = m[1];
    buffer.register = m[2];

    let i = startIndex + 1;
    while (i < lines.length && !lines[i].trim().startsWith('}')) {
      const fm = lines[i].trim().match(/^(\w+)\s+(\w+)\s*;/);
      if (fm) buffer.fields.push({ type: fm[1], name: fm[2] });
      i++;
    }
    return { buffer, endIndex: i };
  }

  parseBuffer(line) {
    const m = line.match(/(?:RW)?StructuredBuffer<(\w+)>\s+(\w+)\s*:\s*register\((\w+)\)/);
    if (!m) return null;
    return { type: m[1], name: m[2], register: m[3], readWrite: line.includes('RWStructuredBuffer') };
  }

  parseFunction(lines, startIndex, shader) {
    const m = lines[startIndex].trim().match(/⟁Wo⟁ fn\s+(\w+)\s*\(([^)]*)\)\s*->\s*(\w+)/);
    if (!m) return null;
    const func = { name: m[1], params: this.parseParams(m[2]), returnType: m[3], body: [] };

    let i = startIndex + 1;
    while (i < lines.length) {
      const line = lines[i].trim();
      if (line === '[Xul]') break;
      if (line.startsWith('⟁Sek⟁')) func.body.push({ type: 'statement', code: line.replace('⟁Sek⟁', '').trim() });
      else if (line.startsWith("⟁Kumk'u⟁")) func.body.push({ type: 'return', code: line.replace("⟁Kumk'u⟁", '').trim() });
      else if (line) func.body.push({ type: 'raw', code: line });
      i++;
    }
    return { func, endIndex: i };
  }

  parseEntry(lines, startIndex, shader) {
    const m = lines[startIndex].trim().match(/⟁Pop\s+(\w+)⟁\s*\(([^)]*)\)/);
    if (!m) return null;
    const entry = { name: m[1], params: m[2].trim(), body: [] };

    let i = startIndex + 1;
    while (i < lines.length) {
      const line = lines[i].trim();
      if (line === '[Xul]') break;
      if (line.startsWith('⟁Sek⟁')) {
        let code = line.replace('⟁Sek⟁', '').trim();
        if (code.endsWith('{')) entry.body.push({ type: 'if-open', code: code.slice(0, -1).trim() });
        else if (code === '}') entry.body.push({ type: 'block-close' });
        else entry.body.push({ type: 'statement', code });
      } else if (line.startsWith("⟁Kumk'u⟁")) {
        entry.body.push({ type: 'return', code: line.replace("⟁Kumk'u⟁", '').trim() });
      } else if (line === '}') {
        entry.body.push({ type: 'block-close' });
      } else if (line && !line.startsWith('#')) {
        entry.body.push({ type: 'raw', code: line });
      }
      i++;
    }
    return { entry, endIndex: i };
  }

  parseParams(paramString) {
    if (!paramString.trim()) return [];
    return paramString.split(',').map(p => {
      const [name, type] = p.split(':').map(s => s.trim());
      return { name, type };
    });
  }

  // ============================================================
  // HLSL EMITTER
  // ============================================================

  emitHLSL(ast, options) {
    const lines = ['// Generated from KLSL by klsl-compiler', '// Target: HLSL cs_5_0', ''];
    const shader = ast.shaders[0];

    for (const cb of shader.constants) {
      lines.push(`cbuffer ${cb.name} : register(${cb.register}) {`);
      for (const f of cb.fields) lines.push(`  ${this.hlslType(f.type)} ${f.name};`);
      lines.push('};', '');
    }

    for (const buf of shader.buffers) {
      const q = buf.readWrite ? 'RWStructuredBuffer' : 'StructuredBuffer';
      lines.push(`${q}<${this.hlslType(buf.type)}> ${buf.name} : register(${buf.register});`);
    }
    lines.push('');

    for (const fn of shader.functions) {
      const params = fn.params.map(p => `${this.hlslType(p.type)} ${p.name}`).join(', ');
      lines.push(`${this.hlslType(fn.returnType)} ${fn.name}(${params}) {`);
      for (const s of fn.body) lines.push('  ' + this.hlslStatement(s));
      lines.push('}', '');
    }

    for (const entry of shader.entries) {
      lines.push(`[numthreads(${shader.threads.join(', ')})]`);
      lines.push(`void ${entry.name}(${this.hlslEntryParams(entry.params)}) {`);
      for (const s of entry.body) lines.push('  ' + this.hlslStatement(s));
      lines.push('}', '');
    }

    return lines.join('\n');
  }

  hlslType(t) {
    return { float: 'float', float2: 'float2', float3: 'float3', float4: 'float4', uint: 'uint', int: 'int', bool: 'bool' }[t] || t;
  }

  hlslStatement(stmt) {
    switch (stmt.type) {
      case 'statement': return this.hlslSyntax(stmt.code) + ';';
      case 'return':    return this.hlslSyntax(stmt.code) + ';';
      case 'if-open':   return this.hlslSyntax(stmt.code) + ' {';
      case 'block-close': return '}';
      default: return stmt.code || '';
    }
  }

  hlslSyntax(code) {
    return code.replace(/⟁Sek⟁/g, '').replace(/⟁Kumk'u⟁/g, 'return').trim();
  }

  hlslEntryParams(params) {
    return params.includes('SV_DispatchThreadID') ? 'uint3 id : SV_DispatchThreadID' : params;
  }

  // ============================================================
  // WGSL EMITTER
  // ============================================================

  emitWGSL(ast, options) {
    const lines = ['// Generated from KLSL by klsl-compiler', '// Target: WGSL (WebGPU)', ''];
    const shader = ast.shaders[0];
    const wgSize = options.workgroupSize || [256, 1, 1];

    for (const cb of shader.constants) {
      lines.push(`struct ${cb.name} {`);
      for (const f of cb.fields) lines.push(`  ${f.name}: ${this.wgslType(f.type)},`);
      lines.push('};', '', `@group(0) @binding(0) var<uniform> params: ${cb.name};`, '');
    }

    let binding = 1;
    for (const buf of shader.buffers) {
      const access = buf.readWrite ? 'read_write' : 'read';
      lines.push(`@group(0) @binding(${binding}) var<storage, ${access}> ${buf.name}: array<${this.wgslType(buf.type)}>;`);
      binding++;
    }
    lines.push('');

    for (const fn of shader.functions) {
      const params = fn.params.map(p => `${p.name}: ${this.wgslType(p.type)}`).join(', ');
      lines.push(`fn ${fn.name}(${params}) -> ${this.wgslType(fn.returnType)} {`);
      for (const s of fn.body) lines.push('  ' + this.wgslStatement(s));
      lines.push('}', '');
    }

    for (const entry of shader.entries) {
      lines.push(`@compute @workgroup_size(${wgSize.join(', ')})`);
      lines.push(`fn ${entry.name}(@builtin(global_invocation_id) id: vec3<u32>) {`);
      for (const s of entry.body) lines.push('  ' + this.wgslStatement(s));
      lines.push('}', '');
    }

    return lines.join('\n');
  }

  wgslType(t) {
    return { float: 'f32', float2: 'vec2<f32>', float3: 'vec3<f32>', float4: 'vec4<f32>', uint: 'u32', int: 'i32', bool: 'bool' }[t] || t;
  }

  wgslStatement(stmt) {
    switch (stmt.type) {
      case 'statement': return this.wgslSyntax(stmt.code) + ';';
      case 'return':    return this.wgslSyntax(stmt.code) + ';';
      case 'if-open':   return this.wgslSyntax(stmt.code) + ' {';
      case 'block-close': return '}';
      default: return stmt.code || '';
    }
  }

  wgslSyntax(code) {
    return code
      .replace(/⟁Sek⟁/g, '').replace(/⟁Kumk'u⟁/g, 'return')
      .replace(/\bfloat2\b/g, 'vec2<f32>').replace(/\bfloat3\b/g, 'vec3<f32>')
      .replace(/\bfloat4\b/g, 'vec4<f32>').replace(/\bfloat\b/g, 'f32')
      .replace(/\buint\b/g, 'u32').replace(/\bint\b/g, 'i32')
      .trim();
  }

  // ============================================================
  // GLSL EMITTER (WebGL2 / ES 3.1)
  // ============================================================

  emitGLSL(ast, options) {
    const lines = ['#version 310 es', 'precision highp float;', 'precision highp int;', '',
      '// Generated from KLSL by klsl-compiler', '// Target: GLSL ES 3.1 (WebGL2 compute)', ''];
    const shader = ast.shaders[0];

    for (const cb of shader.constants) {
      lines.push(`layout(std140, binding = 0) uniform ${cb.name} {`);
      for (const f of cb.fields) lines.push(`  ${this.glslType(f.type)} ${f.name};`);
      lines.push('};', '');
    }

    let binding = 1;
    for (const buf of shader.buffers) {
      const q = buf.readWrite ? 'buffer' : 'readonly buffer';
      lines.push(`layout(std430, binding = ${binding}) ${q} ${buf.name}_block {`);
      lines.push(`  ${this.glslType(buf.type)} ${buf.name}_data[];`);
      lines.push('};');
      binding++;
    }
    lines.push('');

    for (const fn of shader.functions) {
      const params = fn.params.map(p => `${this.glslType(p.type)} ${p.name}`).join(', ');
      lines.push(`${this.glslType(fn.returnType)} ${fn.name}(${params}) {`);
      for (const s of fn.body) lines.push('  ' + this.glslStatement(s));
      lines.push('}', '');
    }

    for (const entry of shader.entries) {
      lines.push('layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;');
      lines.push(`void ${entry.name}() {`);
      lines.push('  uint i = gl_GlobalInvocationID.x;');
      for (const s of entry.body) {
        if (s.code?.includes('id.x') && s.code?.includes('i =')) continue;
        lines.push('  ' + this.glslStatement(s));
      }
      lines.push('}');
    }

    return lines.join('\n');
  }

  glslType(t) {
    return { float: 'float', float2: 'vec2', float3: 'vec3', float4: 'vec4', uint: 'uint', int: 'int', bool: 'bool' }[t] || t;
  }

  glslStatement(stmt) {
    switch (stmt.type) {
      case 'statement': return this.glslSyntax(stmt.code) + ';';
      case 'return':    return this.glslSyntax(stmt.code) + ';';
      case 'if-open':   return this.glslSyntax(stmt.code) + ' {';
      case 'block-close': return '}';
      default: return stmt.code || '';
    }
  }

  glslSyntax(code) {
    return code
      .replace(/⟁Sek⟁/g, '').replace(/⟁Kumk'u⟁/g, 'return')
      .replace(/\bparams_buf\[/g, 'params_buf_data[')
      .replace(/\bgrads_buf\[/g, 'grads_buf_data[')
      .replace(/\bm1_buf\[/g, 'm1_buf_data[')
      .replace(/\bm2_buf\[/g, 'm2_buf_data[')
      .trim();
  }

  // ============================================================
  // K'UHUL CPU GLYPH EMITTER (glyph format, from glyph AST)
  // ============================================================

  emitKUHULGlyph(ast, options) {
    const lines = [
      '// K\'UHUL CPU descriptor — generated by klsl-compiler',
      '// Target: XVM CPU dispatch (.kuhul glyph format)',
      '',
      `⟁ shader ${ast.shaders[0]?.name || 'kernel'}_cpu`,
      `  ⟁Wo⟁ stage "compute"`,
      `  ⟁Wo⟁ threads [256, 1, 1]`,
      `  ⟁Wo⟁ shader_model "cpu_xvm"`,
      ''
    ];
    const shader = ast.shaders[0];
    if (!shader) return lines.join('\n');

    for (const cb of shader.constants) {
      lines.push(`  ⟁Wo⟁ ConstantBuffer ${cb.name} : register(${cb.register})`);
      for (const f of cb.fields) lines.push(`    ${f.type} ${f.name};`);
      lines.push('  }', '');
    }
    for (const buf of shader.buffers) {
      const tag = buf.readWrite ? '⟁Wo⟁ RWStructuredBuffer' : '⟁Wo⟁ StructuredBuffer';
      lines.push(`  ${tag}<${buf.type}> ${buf.name} : register(${buf.register})`);
    }
    lines.push('');
    for (const entry of shader.entries) {
      lines.push(`  ⟁Pop ${entry.name}⟁ (uint3 id : SV_DispatchThreadID)`);
      for (const s of entry.body) lines.push('    ⟁Sek⟁ ' + (s.code || ''));
      lines.push('  [Xul]');
    }
    lines.push('[Xul]');
    return lines.join('\n');
  }

  // ============================================================
  // DECLARATIVE → HLSL  (WGSL kernel body → HLSL)
  // ============================================================

  emitHLSLDeclarative(decl, options) {
    const out = [
      '// Generated by klsl-compiler from declarative .kuhul',
      '// Target: HLSL cs_5_0',
      ''
    ];

    for (const [name, shader] of Object.entries(decl.shaders)) {
      const uniforms = shader['@uniforms'] || {};
      const inputs   = shader['@inputs']   || {};
      const outputs  = shader['@outputs']  || {};
      const wg       = shader['@workgroup'] || [16, 16, 1];
      const kernel   = shader['@kernel']   || '';

      // cbuffer
      const ukeys = Object.keys(uniforms);
      if (ukeys.length) {
        out.push('cbuffer Params : register(b0) {');
        for (const [k, t] of Object.entries(uniforms)) out.push(`  ${this.hlslTypeFromWGSL(t)} ${k};`);
        out.push('};', '');
      }

      // input buffers
      let reg = 0;
      for (const [k, spec] of Object.entries(inputs)) {
        out.push(`StructuredBuffer<${this.hlslTypeFromWGSL(spec['@type'])}> ${k} : register(t${reg++});`);
      }
      // output buffers
      let ureg = 0;
      for (const [k, spec] of Object.entries(outputs)) {
        out.push(`RWStructuredBuffer<${this.hlslTypeFromWGSL(spec['@type'])}> ${k} : register(u${ureg++});`);
      }
      out.push('');

      // entry point
      out.push(`[numthreads(${wg.join(', ')})]`);
      out.push('void CSMain(uint3 global_id : SV_DispatchThreadID) {');
      for (const line of this.translateKernelWGSLtoHLSL(kernel)) out.push('  ' + line);
      out.push('}', '');
    }

    return out.join('\n');
  }

  // ============================================================
  // DECLARATIVE → WGSL  (kernel body is already WGSL)
  // ============================================================

  emitWGSLDeclarative(decl, options) {
    const out = [
      '// Generated by klsl-compiler from declarative .kuhul',
      '// Target: WGSL (WebGPU)',
      ''
    ];

    for (const [name, shader] of Object.entries(decl.shaders)) {
      const uniforms = shader['@uniforms'] || {};
      const inputs   = shader['@inputs']   || {};
      const outputs  = shader['@outputs']  || {};
      const wg       = shader['@workgroup'] || [16, 16, 1];
      const kernel   = shader['@kernel']   || '';

      // params struct
      const ukeys = Object.keys(uniforms);
      if (ukeys.length) {
        out.push('struct Params {');
        for (const [k, t] of Object.entries(uniforms)) out.push(`  ${k}: ${t},`);
        out.push('};', '', '@group(0) @binding(0) var<uniform> params: Params;', '');
      }

      let binding = 1;
      for (const [k, spec] of Object.entries(inputs)) {
        out.push(`@group(0) @binding(${binding}) var<storage, read> ${k}: array<${spec['@type']}>;`);
        binding++;
      }
      for (const [k, spec] of Object.entries(outputs)) {
        out.push(`@group(0) @binding(${binding}) var<storage, read_write> ${k}: array<${spec['@type']}>;`);
        binding++;
      }
      out.push('');

      out.push(`@compute @workgroup_size(${wg.join(', ')})`);
      out.push('fn CSMain(@builtin(global_invocation_id) global_id: vec3<u32>) {');
      for (const line of kernel.split('\n')) out.push('  ' + line.trim());
      out.push('}', '');
    }

    return out.join('\n');
  }

  // ============================================================
  // DECLARATIVE → GLSL (ES 3.1)
  // ============================================================

  emitGLSLDeclarative(decl, options) {
    const out = [
      '#version 310 es',
      'precision highp float;',
      'precision highp int;',
      '',
      '// Generated by klsl-compiler from declarative .kuhul',
      '// Target: GLSL ES 3.1',
      ''
    ];

    for (const [name, shader] of Object.entries(decl.shaders)) {
      const uniforms = shader['@uniforms'] || {};
      const inputs   = shader['@inputs']   || {};
      const outputs  = shader['@outputs']  || {};
      const wg       = shader['@workgroup'] || [16, 16, 1];
      const kernel   = shader['@kernel']   || '';

      const ukeys = Object.keys(uniforms);
      if (ukeys.length) {
        out.push('layout(std140, binding = 0) uniform Params {');
        for (const [k, t] of Object.entries(uniforms)) out.push(`  ${this.glslTypeFromWGSL(t)} ${k};`);
        out.push('};', '');
      }

      let binding = 1;
      for (const [k, spec] of Object.entries(inputs)) {
        out.push(`layout(std430, binding = ${binding}) readonly buffer ${k}_block { ${this.glslTypeFromWGSL(spec['@type'])} ${k}_data[]; };`);
        binding++;
      }
      for (const [k, spec] of Object.entries(outputs)) {
        out.push(`layout(std430, binding = ${binding}) buffer ${k}_block { ${this.glslTypeFromWGSL(spec['@type'])} ${k}_data[]; };`);
        binding++;
      }
      out.push('');

      out.push(`layout(local_size_x = ${wg[0]}, local_size_y = ${wg[1]}, local_size_z = ${wg[2]}) in;`);
      out.push('void main() {');
      out.push('  uvec3 global_id = gl_GlobalInvocationID;');
      for (const line of this.translateKernelWGSLtoGLSL(kernel, inputs, outputs)) out.push('  ' + line);
      out.push('}', '');
    }

    return out.join('\n');
  }

  // ============================================================
  // DECLARATIVE → K'UHUL CPU DESCRIPTOR  (CPU redirect target)
  //   This IS the CPU tier: when webgpu/sm5/webgl2 all fail,
  //   the gate dispatches this .kuhul file to klslc.exe --xvm
  // ============================================================

  emitKUHULDeclarative(decl, options) {
    const out = [
      '// K\'UHUL CPU descriptor — generated by klsl-compiler',
      '// Target: XVM CPU dispatch via klslc.exe --xvm',
      '// CSS-C gate redirects here when GPU tiers unavailable',
      ''
    ];

    for (const [name, shader] of Object.entries(decl.shaders)) {
      const uniforms = shader['@uniforms'] || {};
      const inputs   = shader['@inputs']   || {};
      const outputs  = shader['@outputs']  || {};
      const wg       = shader['@workgroup'] || [16, 16, 1];
      const kernel   = shader['@kernel']   || '';

      const cpuName = `${name}_cpu`;
      out.push(`⟁ shader ${cpuName}`);
      out.push(`  ⟁Wo⟁ stage "compute"`);
      out.push(`  ⟁Wo⟁ threads [${wg.join(', ')}]`);
      out.push(`  ⟁Wo⟁ shader_model "cpu_xvm"`);
      out.push('');

      // uniforms → ConstantBuffer
      const ukeys = Object.keys(uniforms);
      if (ukeys.length) {
        out.push('  ⟁Wo⟁ ConstantBuffer Params : register(b0) {');
        for (const [k, t] of Object.entries(uniforms)) out.push(`    ${this.hlslTypeFromWGSL(t)} ${k};`);
        out.push('  }', '');
      }

      // inputs → StructuredBuffer
      let reg = 0;
      for (const [k, spec] of Object.entries(inputs)) {
        out.push(`  ⟁Wo⟁ StructuredBuffer<${this.hlslTypeFromWGSL(spec['@type'])}> ${k} : register(t${reg++})`);
      }
      // outputs → RWStructuredBuffer
      let ureg = 0;
      for (const [k, spec] of Object.entries(outputs)) {
        out.push(`  ⟁Wo⟁ RWStructuredBuffer<${this.hlslTypeFromWGSL(spec['@type'])}> ${k} : register(u${ureg++})`);
      }
      out.push('');

      // entry point with translated kernel
      out.push(`  ⟁Pop CSMain⟁ (uint3 id : SV_DispatchThreadID)`);
      for (const line of this.translateKernelToKUHUL(kernel)) {
        out.push('    ⟁Sek⟁ ' + line);
      }
      out.push('  [Xul]');
      out.push('[Xul]', '');
    }

    return out.join('\n');
  }

  // ============================================================
  // DECLARATIVE → XVM DISPATCH DESCRIPTOR  (JSON gate pointer)
  // ============================================================

  emitXVMDescDeclarative(decl, options) {
    const shaderNames = Object.keys(decl.shaders);
    const name = options.name || shaderNames[0] || 'kernel';
    const sourceKuhul = options.sourceKuhul || `./${name}.cpu.kuhul`;

    const desc = {
      '@kind': 'xvm.cpu-dispatch.v1',
      name,
      source_kuhul: sourceKuhul,
      shader_language: decl.shaderLanguage,
      shaders: {}
    };

    for (const [sname, shader] of Object.entries(decl.shaders)) {
      desc.shaders[sname] = {
        entry: 'CSMain',
        workgroup: shader['@workgroup'] || [16, 16, 1],
        uniforms: Object.keys(shader['@uniforms'] || {}),
        inputs:   Object.keys(shader['@inputs']   || {}),
        outputs:  Object.keys(shader['@outputs']  || {})
      };
    }

    return JSON.stringify(desc, null, 2);
  }

  // ============================================================
  // KERNEL TRANSLATORS
  // ============================================================

  // WGSL kernel body → HLSL statements
  translateKernelWGSLtoHLSL(kernel) {
    return kernel.split('\n').map(line => {
      let l = line.trim();
      if (!l) return '';
      l = l
        .replace(/\blet\b/g, 'uint')          // let col = ... → uint col = ...
        .replace(/\bvar\s+(\w+):\s*f32\b/g, 'float $1')
        .replace(/\bvar\s+(\w+):\s*u32\b/g, 'uint $1')
        .replace(/\bparams\.(\w+)/g, '$1')     // params.M → M
        .replace(/\bf32\b/g, 'float')
        .replace(/\bu32\b/g, 'uint')
        .replace(/\bvec3<u32>/g, 'uint3')
        .replace(/\bglobal_id\.x\b/g, 'global_id.x')
        .replace(/\bglobal_id\.y\b/g, 'global_id.y');
      // ensure semicolon
      if (l && !l.endsWith(';') && !l.endsWith('{') && !l.endsWith('}') && !l.startsWith('//')) l += ';';
      return l;
    }).filter(l => l !== '');
  }

  // WGSL kernel body → GLSL statements  (buffer name _data[] suffix)
  translateKernelWGSLtoGLSL(kernel, inputs, outputs) {
    const bufNames = [
      ...Object.keys(inputs  || {}),
      ...Object.keys(outputs || {})
    ];
    return kernel.split('\n').map(line => {
      let l = line.trim();
      if (!l) return '';
      l = l
        .replace(/\blet\b/g, 'uint')
        .replace(/\bvar\s+(\w+):\s*f32\b/g, 'float $1')
        .replace(/\bvar\s+(\w+):\s*u32\b/g, 'uint $1')
        .replace(/\bparams\.(\w+)/g, '$1')
        .replace(/\bf32\b/g, 'float')
        .replace(/\bu32\b/g, 'uint')
        .replace(/\bglobal_id\.x\b/g, 'global_id.x')
        .replace(/\bglobal_id\.y\b/g, 'global_id.y');
      // rewrite buffer[idx] → buffer_data[idx]
      for (const n of bufNames) {
        l = l.replace(new RegExp(`\\b${n}\\[`, 'g'), `${n}_data[`);
      }
      if (l && !l.endsWith(';') && !l.endsWith('{') && !l.endsWith('}') && !l.startsWith('//')) l += ';';
      return l;
    }).filter(l => l !== '');
  }

  // WGSL kernel body → K'UHUL glyph statements
  translateKernelToKUHUL(kernel) {
    return kernel.split('\n').map(line => {
      let l = line.trim();
      if (!l) return null;
      // preserve WGSL-ish syntax; XVM will interpret
      l = l.replace(/\bparams\.(\w+)/g, '$1');
      return l;
    }).filter(l => l !== null && l !== '');
  }

  // ============================================================
  // TYPE HELPERS (WGSL → HLSL / GLSL)
  // ============================================================

  hlslTypeFromWGSL(t) {
    return { f32: 'float', u32: 'uint', i32: 'int', f16: 'half',
             'vec2<f32>': 'float2', 'vec3<f32>': 'float3', 'vec4<f32>': 'float4' }[t] || t;
  }

  glslTypeFromWGSL(t) {
    return { f32: 'float', u32: 'uint', i32: 'int',
             'vec2<f32>': 'vec2', 'vec3<f32>': 'vec3', 'vec4<f32>': 'vec4' }[t] || t;
  }
}
