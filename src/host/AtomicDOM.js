/*
 * AtomicDOM.js
 * Apps-host helper for building Atomic DOM records from ASX Blocks and the
 * Atomic CSS key grammar declared in data/manifest/css.manifest.json.
 *
 * This is a projection helper. It does not execute K'UHUL, own semantic
 * authority, store secret values, mutate topology, or authorize commits.
 */
(function attachAtomicDOM(root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.AtomicDOM = factory();
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function buildAtomicDOM() {
  'use strict';

  const VERSION = '0.1.0';
  const CSS_KEY_GRAMMAR_MANIFEST = 'data/manifest/css.manifest.json';

  const KEY_GRAMMAR_TOKENS = Object.freeze([
    '⟁KEYGRAMMAR',
    'key:address(book)',
    'book:declared-neighborhood',
    'grammar:constrained-book-traversal'
  ]);

  const ASX_BLOCK_TOKENS = Object.freeze([
    '⟁SYNTAXUNITKINDS',
    'Block',
    'Component',
    '⟁TYPE:WIDGET',
    '⟁TYPE:WIDGETCARD'
  ]);

  const ATOMIC_CSS_TOKENS = Object.freeze([
    '⟁HTMLBIND',
    '⟁WIDGET',
    '⟁WIDGETCARD',
    '⟁WIDGETHEADER',
    '⟁WIDGETBODY',
    '⟁WIDGETFOOTER',
    '⟁WIDGETBTN',
    '⟁BTNPRIMARY',
    '⟁Pprimary',
    '⟁Psecondary'
  ]);

  const DEFAULT_PALETTES = Object.freeze({
    khanary_dark: Object.freeze({
      name: 'khanary_dark',
      background: '#01040a',
      panel: '#020b12',
      text: '#e5f2ff',
      primary: '#22f2b3',
      secondary: '#38bdf8',
      accent: '#a855f7'
    }),
    terminal_blue: Object.freeze({
      name: 'terminal_blue',
      background: '#020617',
      panel: '#07111f',
      text: '#dbeafe',
      primary: '#38bdf8',
      secondary: '#22f2b3',
      accent: '#818cf8'
    }),
    high_contrast: Object.freeze({
      name: 'high_contrast',
      background: '#000000',
      panel: '#111827',
      text: '#ffffff',
      primary: '#facc15',
      secondary: '#4ade80',
      accent: '#fb7185'
    })
  });

  const USER_OPTION_SLOTS = Object.freeze([
    'branding',
    'api_token_secret_ref',
    'runtime_config',
    'color_switcher',
    'palette_choice'
  ]);

  function cloneJson(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function arrayOf(value) {
    if (value === undefined || value === null) return [];
    return Array.isArray(value) ? value.slice() : [value];
  }

  function token(name) {
    const all = new Set(KEY_GRAMMAR_TOKENS.concat(ASX_BLOCK_TOKENS, ATOMIC_CSS_TOKENS));
    if (!all.has(name)) {
      throw new Error('Unknown Atomic DOM token: ' + String(name));
    }
    return name;
  }

  function tokens() {
    return Object.freeze({
      keyGrammar: KEY_GRAMMAR_TOKENS.slice(),
      asxBlocks: ASX_BLOCK_TOKENS.slice(),
      atomicCss: ATOMIC_CSS_TOKENS.slice()
    });
  }

  function palette(name) {
    const key = name || 'khanary_dark';
    if (!Object.prototype.hasOwnProperty.call(DEFAULT_PALETTES, key)) {
      throw new Error('Unknown Atomic DOM palette: ' + String(key));
    }
    return cloneJson(DEFAULT_PALETTES[key]);
  }

  function block(kind, props, children) {
    const blockKind = kind || 'Component';
    if (ASX_BLOCK_TOKENS.indexOf(blockKind) === -1 && blockKind !== 'Atom') {
      throw new Error('Unsupported ASX block kind: ' + String(blockKind));
    }
    return {
      '@type': 'atomic_dom.block',
      '@style': 'react_like_create',
      '@kind': blockKind,
      '@manifest': CSS_KEY_GRAMMAR_MANIFEST,
      '@tokens': ['⟁KEYGRAMMAR', '⟁SYNTAXUNITKINDS'],
      props: cloneJson(props || {}),
      children: arrayOf(children)
    };
  }

  function widget(props, children) {
    const record = block('Component', props, children);
    record['@type'] = 'atomic_dom.widget';
    record['@tokens'] = record['@tokens'].concat(['⟁WIDGET', '⟁WIDGETCARD', '⟁HTMLBIND']);
    return record;
  }

  function theme(options) {
    const opts = options || {};
    const chosen = palette(opts.palette || opts.palette_choice || 'khanary_dark');
    return {
      '@type': 'atomic_dom.theme',
      '@manifest': CSS_KEY_GRAMMAR_MANIFEST,
      '@tokens': ['⟁KEYGRAMMAR', '⟁HTMLBIND'],
      purpose: 'user_option_projection',
      branding: cloneJson(opts.branding || {}),
      palette: chosen,
      color_switcher: {
        selected: chosen.name,
        options: Object.keys(DEFAULT_PALETTES)
      },
      runtime_config: cloneJson(opts.runtime_config || {}),
      api_token_secret_ref: opts.api_token_secret_ref || null,
      stores_secret_values: false
    };
  }

  function request(endpoint, payload) {
    return {
      '@type': 'atomic_dom.request',
      '@style': 'ajax_like_call',
      endpoint: String(endpoint || ''),
      payload: cloneJson(payload || {}),
      authority: {
        semantic_authority: false,
        topology_mutation: false,
        commit_authorized: false,
        next_gate: "Ch'en/XCFE/Xul"
      }
    };
  }

  return Object.freeze({
    version: VERSION,
    manifest: CSS_KEY_GRAMMAR_MANIFEST,
    optionSlots: USER_OPTION_SLOTS.slice(),
    keyGrammarTokens: tokens,
    token,
    h: block,
    block,
    widget,
    theme,
    palette,
    request
  });
});
