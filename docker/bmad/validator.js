/**
 * BMAD Validator Service — validates Cypress/Playwright test scripts
 * Endpoint: POST /validate  { script, framework }
 * Returns:  { valid, syntax_ok, errors, assertions_found, warnings }
 */
const express = require('express');
const acorn   = require('acorn');
const app     = express();
app.use(express.json({ limit: '1mb' }));

// ── Cypress assertion patterns ───────────────────────────────────────────────
const CYPRESS_REQUIRED = [/cy\.(visit|get|contains|should|url)\s*\(/];
const CYPRESS_STRUCTURE = /describe\s*\(.*?,\s*\(\s*\)\s*=>\s*\{/s;
const CYPRESS_IT_BLOCK  = /\bit\s*\(.*?,\s*(async\s*)?\(\s*\)\s*=>/s;

// ── Playwright assertion patterns ────────────────────────────────────────────
const PW_REQUIRED  = [/page\.(goto|fill|click|locator)\s*\(/, /expect\s*\(/];
const PW_STRUCTURE = /test\s*\(/;
const PW_ASYNC     = /async\s*\(\s*\{\s*page\s*\}/;

function validateSyntax(script) {
  try {
    acorn.parse(script, { ecmaVersion: 2022, sourceType: 'module' });
    return { ok: true, error: null };
  } catch (e) {
    return { ok: false, error: e.message };
  }
}

function validateCypress(script) {
  const errors = [], warnings = [];
  const syntaxResult = validateSyntax(script);
  if (!syntaxResult.ok) errors.push(`Syntax error: ${syntaxResult.error}`);
  if (!CYPRESS_STRUCTURE.test(script)) errors.push('Missing describe() block structure');
  if (!CYPRESS_IT_BLOCK.test(script))  errors.push('Missing it() test block');

  let assertionsFound = 0;
  for (const p of CYPRESS_REQUIRED) {
    if (p.test(script)) assertionsFound++;
    else warnings.push(`Missing expected pattern: ${p}`);
  }
  if ((script.match(/cy\./g) || []).length < 2)
    errors.push('Too few cy.* calls — script appears incomplete');
  if (/await\s+page\./.test(script))
    errors.push('Wrong framework: Playwright syntax detected in Cypress script');

  return { valid: errors.length === 0, syntax_ok: syntaxResult.ok,
           errors, warnings, assertions_found: assertionsFound };
}

function validatePlaywright(script) {
  const errors = [], warnings = [];
  const syntaxResult = validateSyntax(script);
  if (!syntaxResult.ok) errors.push(`Syntax error: ${syntaxResult.error}`);
  if (!PW_STRUCTURE.test(script))  errors.push('Missing test() block');
  if (!PW_ASYNC.test(script))      errors.push('Missing async ({ page }) parameter');

  let assertionsFound = 0;
  for (const p of PW_REQUIRED) {
    if (p.test(script)) assertionsFound++;
    else errors.push(`Missing required Playwright pattern: ${p}`);
  }
  if (/cy\./.test(script))
    errors.push('Wrong framework: Cypress syntax detected in Playwright script');
  if (!/await\s+/.test(script))
    warnings.push('No await keywords found — async calls may be missing');

  return { valid: errors.length === 0, syntax_ok: syntaxResult.ok,
           errors, warnings, assertions_found: assertionsFound };
}

// ── Routes ───────────────────────────────────────────────────────────────────
app.get('/health', (req, res) => res.json({ status: 'ok', service: 'bmad-validator' }));

app.post('/validate', (req, res) => {
  const { script, framework } = req.body;
  if (!script || !framework)
    return res.status(400).json({ error: 'script and framework required' });
  const result = framework === 'cypress' ? validateCypress(script) : validatePlaywright(script);
  res.json(result);
});

app.post('/batch-validate', (req, res) => {
  const { scripts, framework } = req.body;
  if (!Array.isArray(scripts) || !framework)
    return res.status(400).json({ error: 'scripts array and framework required' });
  const results = scripts.map(script =>
    framework === 'cypress' ? validateCypress(script) : validatePlaywright(script));
  res.json({ results, valid_count: results.filter(r => r.valid).length });
});

const PORT = process.env.PORT || 8001;
app.listen(PORT, '0.0.0.0', () => console.log(`BMAD validator listening on :${PORT}`));
