#!/usr/bin/env node
// Detect the written key of every song chart and emit a review report.
// Runs FULLY OFFLINE from inside the songs repo (the one holding songs.json + the .txt charts).
// It writes NO changes to songs.json — only two reports you verify manually:
//   - key-review.csv   : one row per song, editable `final_key` column (source of truth for the
//                        later "glue" script). Sorted worst-confidence-first for efficient review.
//   - key-review.html  : read-only view — color-coded by confidence, chords shown inline.
//
// The detected key is ALWAYS one of a closed set of 24 values (12 major + 12 minor):
//   Major: C Db D Eb E F F# G Ab A Bb B
//   Minor: Cm C#m Dm Ebm Em Fm F#m Gm Abm Am Bbm Bm
// Guardrail: values come only from these arrays by construction — nothing else can appear.
//
// Method: Krumhansl-Schmuckler key finding over a chord-tone pitch-class histogram, nudged
// (not overridden) by the first/last chord as tonic hints.
//
// Usage (run from the songs repo root):
//   node detect-keys.mjs
//   node detect-keys.mjs --songs ./songs.json --songs-dir . --out-csv ./key-review.csv --out-html ./key-review.html
//   node detect-keys.mjs --limit 25          # quick sanity check; prints a table, writes nothing
//
// Flags:
//   --songs <path>     songs.json (default ./songs.json)
//   --songs-dir <dir>  repo root the chart paths are relative to (default: dir of --songs)
//   --base-url <pre>   URL prefix to strip from markupUrl to get the local path
//                      (default https://raw.githubusercontent.com/KidCrippler/songs/master/)
//   --out-csv <path>   (default ./key-review.csv)
//   --out-html <path>  (default ./key-review.html)
//   -n, --num N        randomly sample N songs, then write the reports for that subset
//   --limit N          only process first N songs; print table, write nothing (quick debug)

import { readFileSync, writeFileSync, existsSync } from 'fs';
import { dirname, join } from 'path';

const DEFAULT_BASE_URL = 'https://raw.githubusercontent.com/KidCrippler/songs/master/';
const CONFIDENCE_THRESHOLD = 0.55; // below this → flagged as low confidence

// ---- Closed 24-key vocabulary (the guardrail) ----
const MAJOR_NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'];
const MINOR_NAMES = ['Cm', 'C#m', 'Dm', 'Ebm', 'Em', 'Fm', 'F#m', 'Gm', 'Abm', 'Am', 'Bbm', 'Bm'];
const keyName = (pc, isMinor) => (isMinor ? MINOR_NAMES : MAJOR_NAMES)[pc];

// ---- Chord parsing (mirrors backend/src/services/songParser.ts rules) ----
const CHORD_REGEX = /^[A-G][#b]?(m|M|[Mm]aj|[Mm]in|dim|aug|add|o|°|º|\+)?[0-9]*\+?(sus[24]?)?(b[0-9]+)?(\/[A-G][#b]?)?!?$/;
const BRACKETED_CHORD_REGEX = /^\[[A-G][#b]?(m|M|[Mm]aj|[Mm]in|dim|aug|add|o|°|º|\+)?[0-9]*\+?(sus[24]?)?(b[0-9]+)?(\/[A-G][#b]?)?\]!?$/;
const NOTE_PC = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };

function stripDirectional(s) {
  return s.replace(/[‎‏؜⁦-⁩‪-‮﻿]/g, '');
}

// Parse a chord token → { root, minor, third, fifth } pitch classes, or null.
function parseChord(tokenRaw) {
  let t = tokenRaw.trim();
  if (t.startsWith('[') && t.endsWith(']')) t = t.slice(1, -1);
  t = t.replace(/!$/, '');
  if (!t || t[0] < 'A' || t[0] > 'G') return null;
  let pc = NOTE_PC[t[0]];
  let i = 1;
  if (t[i] === '#') { pc = (pc + 1) % 12; i++; }
  else if (t[i] === 'b') { pc = (pc + 11) % 12; i++; }
  const rest = t.slice(i);
  const isDim = /^(dim|o|°|º)/.test(rest);
  const isAug = /^(aug|\+)/.test(rest);
  const isMinor = isDim || /^(m(?!aj)|min)/.test(rest); // lowercase m/min, not maj/M
  const third = (pc + (isMinor ? 3 : 4)) % 12;
  const fifth = (pc + (isDim ? 6 : isAug ? 8 : 7)) % 12;
  return { root: pc, minor: isMinor, third, fifth };
}

// If every token on the line is a chord (or ignorable marker), return [{ token, chord }].
function chordsInLine(line) {
  const cleaned = stripDirectional(line).trim();
  if (!cleaned) return null;
  const tokens = cleaned.split(/\s+/);
  const out = [];
  for (const tok of tokens) {
    if (tok === '-' || tok === '[]' || /^x?\d+$/i.test(tok) || tok.startsWith('(') || tok.endsWith(')')) continue;
    if (/^(-{2,3}>|<-{2,3})$/.test(tok)) continue;
    if (/^\{[^}]+\}$/.test(tok)) continue;
    if (/^\[?\/[A-G][#b]?\]?$/.test(tok)) continue; // bass-only
    if (!(CHORD_REGEX.test(tok) || BRACKETED_CHORD_REGEX.test(tok))) return null; // not a pure chord line
    const chord = parseChord(tok);
    if (chord) out.push({ token: tok, chord });
  }
  return out.length ? out : null;
}

function extractChords(text) {
  const out = [];
  for (const line of text.split('\n')) {
    const cs = chordsInLine(line);
    if (cs) out.push(...cs);
  }
  return out; // [{ token, chord }]
}

// ---- Krumhansl-Schmuckler key finding ----
const KS_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
const KS_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];

function correlate(hist, profile, tonic) {
  const n = 12;
  const rotated = Array.from({ length: n }, (_, i) => profile[(i - tonic + n) % n]);
  const mH = hist.reduce((a, b) => a + b, 0) / n;
  const mP = rotated.reduce((a, b) => a + b, 0) / n;
  let num = 0, dH = 0, dP = 0;
  for (let i = 0; i < n; i++) {
    const a = hist[i] - mH, b = rotated[i] - mP;
    num += a * b; dH += a * a; dP += b * b;
  }
  return dH && dP ? num / Math.sqrt(dH * dP) : 0;
}

function detectKey(entries) {
  const chords = entries.map(e => e.chord);
  if (chords.length < 2) return null;
  const hist = new Array(12).fill(0);
  for (const c of chords) { hist[c.root] += 3; hist[c.third] += 1.5; hist[c.fifth] += 1; }

  const first = chords[0], last = chords[chords.length - 1];
  // Endpoint hint: first/last chord is often (not always) the tonic. Nudge, don't override —
  // K-S still decides when accidentals point elsewhere (e.g. a Bb ⇒ F, not C).
  const FIRST_BONUS = 0.12, LAST_BONUS = 0.12;
  const endpointBonus = (tonic, minor) =>
    (first.root === tonic && first.minor === minor ? FIRST_BONUS : 0) +
    (last.root === tonic && last.minor === minor ? LAST_BONUS : 0);

  const scored = [];
  for (let tonic = 0; tonic < 12; tonic++) {
    for (const minor of [false, true]) {
      const r = correlate(hist, minor ? KS_MINOR : KS_MAJOR, tonic);
      scored.push({ tonic, minor, r, score: r + endpointBonus(tonic, minor) });
    }
  }
  scored.sort((a, b) => b.score - a.score);
  const best = scored[0], second = scored[1];
  const confidence = Math.min(1, Math.max(0, best.score - second.score) * 4 + Math.max(0, best.r) * 0.4);

  return {
    key: keyName(best.tonic, best.minor),
    confidence: Number(confidence.toFixed(2)),
    firstChord: keyName(first.root, first.minor),
    lastChord: keyName(last.root, last.minor),
    numChords: chords.length,
    tokens: entries.map(e => e.token),
  };
}

// ---- Offline path resolution ----
function relPath(url, baseUrl) {
  if (url.startsWith(baseUrl)) return url.slice(baseUrl.length);
  const m = url.match(/\/(?:master|main)\/(.+)$/); // fallback: strip up to branch
  return m ? m[1] : url.replace(/^https?:\/\/[^/]+\//, '');
}
function resolveLocalPath(url, baseUrl, songsDir) {
  return join(songsDir, relPath(url, baseUrl));
}
function topDir(url, baseUrl) {
  const seg = relPath(url, baseUrl).split('/');
  return seg.length > 1 ? seg[0] : '';
}

// ---- Runner ----
function parseArgs() {
  const a = process.argv.slice(2);
  const get = (flag, def) => { const i = a.indexOf(flag); return i >= 0 ? a[i + 1] : def; };
  const songs = get('--songs', './songs.json');
  return {
    songs,
    songsDir: get('--songs-dir', dirname(songs)),
    baseUrl: get('--base-url', DEFAULT_BASE_URL),
    outCsv: get('--out-csv', './key-review.csv'),
    outHtml: get('--out-html', './key-review.html'),
    sample: (get('-n', null) ?? get('--num', null)) != null ? parseInt(get('-n', null) ?? get('--num', null), 10) : null,
    limit: get('--limit', null) ? parseInt(get('--limit'), 10) : null,
  };
}

// Fisher-Yates in place.
function shuffle(a) {
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

const esc = (s) => `"${String(s ?? '').replace(/"/g, '""')}"`;
const escHtml = (s) => String(s ?? '').replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

function buildCsv(rows) {
  const out = ['id,name,singer,detected_key,confidence,first_chord,last_chord,num_chords,status,final_key'];
  for (const r of rows) {
    out.push([
      r.id, esc(r.name), esc(r.singer), r.key, r.confidence,
      r.firstChord, r.lastChord, r.numChords, r.status, r.key, // final_key pre-filled with the guess
    ].join(','));
  }
  return out.join('\n') + '\n';
}

function buildHtml(rows, stats) {
  // 'err' = no key at all (no chords / missing file). Otherwise bucket by numeric confidence.
  const cls = (r) => r.confidence === '' ? 'err' : r.confidence >= 0.8 ? 'hi' : r.confidence >= CONFIDENCE_THRESHOLD ? 'mid' : 'lo';
  const body = rows.map(r => `<tr class="${cls(r)}" data-dir="${escHtml(r.dir || '')}">
    <td>${r.id}</td><td class="name">${escHtml(r.name)}</td><td>${escHtml(r.singer)}</td>
    <td>${escHtml(r.dir || '')}</td>
    <td class="key">${escHtml(r.key)}</td><td>${r.confidence !== '' ? r.confidence : ''}</td>
    <td class="chd">${escHtml(r.firstChord)}→${escHtml(r.lastChord)}</td><td>${r.numChords || ''}</td>
    <td>${escHtml(r.status)}</td><td class="seq">${escHtml((r.tokens || []).slice(0, 40).join(' '))}${(r.tokens || []).length > 40 ? ' …' : ''}</td>
  </tr>`).join('\n');

  const dirs = [...new Set(rows.map(r => r.dir).filter(Boolean))].sort();
  const dirOptions = ['<option value="">all directories</option>', ...dirs.map(d => `<option>${escHtml(d)}</option>`)].join('');

  return `<!doctype html><html><head><meta charset="utf-8"><title>Key review</title><style>
  body{font:14px/1.4 system-ui,sans-serif;margin:1rem;color:#222}
  h1{font-size:1.1rem} .stats{margin:.5rem 0;color:#555}
  .controls{display:flex;gap:.5rem;align-items:center;margin-bottom:.5rem;flex-wrap:wrap}
  input{padding:.4rem;width:20rem} select{padding:.4rem} #count{color:#555}
  table{border-collapse:collapse;width:100%} th,td{padding:.25rem .5rem;border-bottom:1px solid #eee;text-align:left;vertical-align:top}
  th{cursor:pointer;position:sticky;top:0;background:#fff;border-bottom:2px solid #ccc}
  .name{font-weight:600} .key{font-weight:700;font-size:1.05rem} .chd,.seq{font-family:ui-monospace,monospace;direction:ltr;unicode-bidi:isolate}
  .seq{color:#555;max-width:32rem} tr.hi .key{color:#0a7d33} tr.mid .key{color:#b26a00} tr.lo .key{color:#c00}
  tr.err{background:#fff4f4} tr.err td{color:#900} tr:hover{background:#f6f9ff}
  </style></head><body>
  <h1>Song key review — ${stats.total} songs</h1>
  <div class="stats">✅ ok ${stats.ok} · 🟡 low-confidence ${stats.low} · ⚠️ no chords ${stats.noChords} · ❌ file missing ${stats.missing}
  &nbsp;|&nbsp; original song order. Green ≥0.8, amber ≥${CONFIDENCE_THRESHOLD}, red below. Edit corrections in the CSV, not here.</div>
  <div class="controls">
    <input id="f" placeholder="filter by name / singer / key…" oninput="filt()">
    <select id="fd" onchange="filt()">${dirOptions}</select>
    <select id="fc" onchange="filt()">
      <option value="">all confidence</option>
      <option value="hi">high (≥0.8)</option>
      <option value="mid">medium (≥${CONFIDENCE_THRESHOLD})</option>
      <option value="lo">low (&lt;${CONFIDENCE_THRESHOLD})</option>
      <option value="err">no key (missing / no chords)</option>
    </select>
    <span id="count"></span>
  </div>
  <table id="t"><thead><tr>
    <th onclick="sortBy(0,1)">id</th><th onclick="sortBy(1)">name</th><th onclick="sortBy(2)">singer</th>
    <th onclick="sortBy(3)">dir</th><th onclick="sortBy(4)">key</th><th onclick="sortBy(5,1)">conf</th>
    <th onclick="sortBy(6)">first→last</th><th onclick="sortBy(7,1)">#</th><th onclick="sortBy(8)">status</th><th>chords</th>
  </tr></thead><tbody>
${body}
  </tbody></table>
  <script>
  const tb=document.querySelector('#t tbody'),f=document.getElementById('f'),fd=document.getElementById('fd'),fc=document.getElementById('fc');
  function filt(){const q=f.value.toLowerCase(),d=fd.value,c=fc.value;let shown=0;
    for(const tr of tb.rows){
      const ok=tr.textContent.toLowerCase().includes(q)&&(!d||tr.dataset.dir===d)&&(!c||tr.classList.contains(c));
      tr.style.display=ok?'':'none';if(ok)shown++;}
    document.getElementById('count').textContent=shown+' shown';}
  function sortBy(i,num){const rows=[...tb.rows];const asc=tb.dataset.s!==i+'';tb.dataset.s=asc?i:'';
    rows.sort((a,b)=>{let x=a.cells[i].textContent,y=b.cells[i].textContent;
      if(num){x=parseFloat(x)||0;y=parseFloat(y)||0;return asc?x-y:y-x;}
      return asc?x.localeCompare(y):y.localeCompare(x);});
    rows.forEach(r=>tb.appendChild(r));}
  filt();
  </script></body></html>`;
}

function main() {
  const args = parseArgs();
  if (!existsSync(args.songs)) throw new Error(`songs.json not found: ${args.songs}`);
  const data = JSON.parse(readFileSync(args.songs, 'utf-8'));
  let songs = data.songs || [];
  if (args.sample && args.sample < songs.length) {
    // Pick a random subset but keep them in original songs.json order for output.
    const idx = shuffle([...songs.keys()]).slice(0, args.sample).sort((a, b) => a - b);
    songs = idx.map(i => songs[i]);
    console.log(`Randomly sampled ${songs.length} of ${data.songs.length} songs (kept in original order).`);
  }
  if (args.limit) songs = songs.slice(0, args.limit);
  console.log(`Processing ${songs.length} songs offline from ${args.songsDir} ...`);

  const rows = [];
  const stats = { total: songs.length, ok: 0, low: 0, noChords: 0, missing: 0 };
  for (const song of songs) {
    const anyUrl = song.lyrics?.markupUrl || song.lyrics?.imageUrl || '';
    const dir = anyUrl ? topDir(anyUrl, args.baseUrl) : '';
    const base = { id: song.id, name: song.name, singer: song.singer, dir, key: '', confidence: '', firstChord: '', lastChord: '', numChords: 0, tokens: [] };
    const markupUrl = song.lyrics?.markupUrl;
    if (!markupUrl) { stats.missing++; rows.push({ ...base, status: 'no-markup-url' }); continue; }
    const path = resolveLocalPath(markupUrl, args.baseUrl, args.songsDir);
    if (!existsSync(path)) { stats.missing++; rows.push({ ...base, status: 'file-missing' }); continue; }
    const detected = detectKey(extractChords(readFileSync(path, 'utf-8')));
    if (!detected) { stats.noChords++; rows.push({ ...base, status: 'no-chords' }); continue; }
    const status = detected.confidence < CONFIDENCE_THRESHOLD ? 'low-confidence' : 'ok';
    if (status === 'ok') stats.ok++; else stats.low++;
    rows.push({ ...base, ...detected, status });
  }

  // Rows stay in original songs.json order; use the HTML filters to focus on low-confidence.
  console.log(`ok ${stats.ok} | low-confidence ${stats.low} | no-chords ${stats.noChords} | file-missing ${stats.missing}`);

  if (args.limit) {
    console.log('\nid       key   conf  first→last        #  status          name');
    for (const r of rows) {
      console.log(`${r.id}  ${(r.key || '-').padEnd(4)} ${String(r.confidence || '').padEnd(5)} ` +
        `${((r.firstChord || '') + '→' + (r.lastChord || '')).padEnd(16)} ${String(r.numChords || '').padStart(3)}  ` +
        `${r.status.padEnd(15)} ${r.name}`);
    }
    console.log('\n(--limit set: no files written)');
    return;
  }

  writeFileSync(args.outCsv, buildCsv(rows));
  writeFileSync(args.outHtml, buildHtml(rows, stats));
  console.log(`\nWrote ${args.outCsv}`);
  console.log(`Wrote ${args.outHtml}`);
}

main();
