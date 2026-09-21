#!/usr/bin/env node
// Apply reviewed keys from key-review.csv into songs.json — WITHOUT re-serializing the JSON,
// so the file's exact formatting (tabs, spacing, field order) is preserved byte-for-byte
// except for the lines we touch.
//
// For each song that has a valid `final_key` in the CSV, a line
//     "key" : "<KEY>",
// is inserted directly ABOVE the song's `keyShiftToOriginal` field, or — if that field is
// absent — directly above `dateCreated` (which always exists). Re-running is idempotent:
// an existing `key` line is replaced in place rather than duplicated.
//
// The key is validated against the closed 24-value vocabulary; anything else is skipped
// with a warning (so a typo in the CSV surfaces instead of corrupting the index).
//
// Usage (run from the songs repo root):
//   node apply-keys.mjs                 # writes songs.json.new (review, then rename)
//   node apply-keys.mjs --in-place      # backs up to songs.json.bak, then overwrites songs.json
//
// Flags:
//   --songs <path>   (default ./songs.json)
//   --csv <path>     (default ./key-review.csv)
//   --out <path>     (default ./songs.json.new; ignored with --in-place)
//   --in-place       overwrite --songs after writing a .bak

import { readFileSync, writeFileSync, existsSync } from 'fs';

const VALID_KEYS = new Set([
  'C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B',
  'Cm', 'C#m', 'Dm', 'Ebm', 'Em', 'Fm', 'F#m', 'Gm', 'Abm', 'Am', 'Bbm', 'Bm',
]);

// Enharmonic preference: replace the un-preferred spelling with the preferred one
// (applies to both major and minor, e.g. Gb→F#, Gbm→F#m, G#m→Abm).
const ENHARMONIC = { Gb: 'F#', 'A#': 'Bb', 'D#': 'Eb', 'G#': 'Ab' };
function normalizeKey(k) {
  const m = k?.match(/^([A-G][#b]?)(m?)$/);
  if (!m) return k;
  return (ENHARMONIC[m[1]] ?? m[1]) + m[2];
}

function parseArgs() {
  const a = process.argv.slice(2);
  const get = (flag, def) => { const i = a.indexOf(flag); return i >= 0 ? a[i + 1] : def; };
  const songs = get('--songs', './songs.json');
  return {
    songs,
    csv: get('--csv', './key-review.csv'),
    out: get('--out', './songs.json.new'),
    inPlace: a.includes('--in-place'),
  };
}

// Minimal RFC-4180 CSV parser (handles quotes, "" escapes, commas/newlines in fields, CRLF).
function parseCsv(text) {
  const rows = [];
  let row = [], field = '', inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inQuotes) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; }
        else inQuotes = false;
      } else field += c;
    } else if (c === '"') inQuotes = true;
    else if (c === ',') { row.push(field); field = ''; }
    else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++;
      row.push(field); field = '';
      if (row.length > 1 || row[0] !== '') rows.push(row);
      row = [];
    } else field += c;
  }
  if (field !== '' || row.length) { row.push(field); if (row.length > 1 || row[0] !== '') rows.push(row); }
  return rows;
}

function loadKeyMap(csvPath) {
  const rows = parseCsv(readFileSync(csvPath, 'utf-8'));
  if (!rows.length) throw new Error('CSV is empty');
  const header = rows[0].map(h => h.trim());
  const idCol = header.indexOf('id');
  const keyCol = header.indexOf('final_key');
  if (idCol < 0 || keyCol < 0) throw new Error("CSV must have 'id' and 'final_key' columns");

  const map = new Map();
  const invalid = [];
  let blank = 0;
  for (let r = 1; r < rows.length; r++) {
    const id = rows[r][idCol]?.trim();
    const raw = rows[r][keyCol]?.trim();
    if (!id) continue;
    if (!raw) { blank++; continue; }
    const key = normalizeKey(raw);
    if (!VALID_KEYS.has(key)) { invalid.push({ id, key: raw }); continue; }
    map.set(id, key);
  }
  return { map, invalid, blank };
}

function apply(text, map) {
  const eol = text.includes('\r\n') ? '\r\n' : '\n';
  const lines = text.split(/\r\n|\n/);

  const ID_RE = /^(\s*)"id"\s*:\s*(\d+)\s*,?\s*$/;      // numeric song id (skips quoted category ids)
  const KEY_RE = /^(\s*)"key"\s*:\s*"[^"]*"\s*,?\s*$/;   // existing key line → replace
  const ANCHOR_RE = /^(\s*)"(?:keyShiftToOriginal|dateCreated)"\s*:/; // insert above the first of these

  const out = [];
  let curId = null, pendingKey = undefined, applied = false;
  const appliedIds = new Set();

  for (const line of lines) {
    const idM = line.match(ID_RE);
    if (idM) {
      curId = idM[2];
      pendingKey = map.get(curId);
      applied = false;
      out.push(line);
      continue;
    }

    if (pendingKey !== undefined && !applied) {
      const keyM = line.match(KEY_RE);
      if (keyM) { // replace existing key value in place, preserving indentation
        out.push(`${keyM[1]}"key" : "${pendingKey}",`);
        applied = true;
        appliedIds.add(curId);
        continue;
      }
      const anchorM = line.match(ANCHOR_RE);
      if (anchorM) { // insert new key line above the anchor, matching its indentation
        out.push(`${anchorM[1]}"key" : "${pendingKey}",`);
        out.push(line);
        applied = true;
        appliedIds.add(curId);
        continue;
      }
    }
    out.push(line);
  }

  return { text: out.join(eol), appliedIds };
}

function main() {
  const args = parseArgs();
  if (!existsSync(args.songs)) throw new Error(`songs.json not found: ${args.songs}`);
  if (!existsSync(args.csv)) throw new Error(`CSV not found: ${args.csv}`);

  const { map, invalid, blank } = loadKeyMap(args.csv);
  const original = readFileSync(args.songs, 'utf-8');
  const { text, appliedIds } = apply(original, map);

  const notFound = [...map.keys()].filter(id => !appliedIds.has(id));

  if (invalid.length) {
    console.warn(`⚠️  ${invalid.length} rows had an invalid key (skipped): ` +
      invalid.slice(0, 10).map(x => `${x.id}="${x.key}"`).join(', ') + (invalid.length > 10 ? ' …' : ''));
  }
  if (notFound.length) {
    console.warn(`⚠️  ${notFound.length} CSV ids not found in songs.json (skipped): ` +
      notFound.slice(0, 10).join(', ') + (notFound.length > 10 ? ' …' : ''));
  }

  const target = args.inPlace ? args.songs : args.out;
  if (args.inPlace) {
    writeFileSync(`${args.songs}.bak`, original);
    console.log(`Backed up original → ${args.songs}.bak`);
  }
  writeFileSync(target, text);

  console.log(`Applied ${appliedIds.size} keys · ${blank} blank · ${invalid.length} invalid · ${notFound.length} not found`);
  console.log(`Wrote ${target}`);
  if (!args.inPlace) console.log(`Review it, then: mv ${target} ${args.songs}`);
}

main();
