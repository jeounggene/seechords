# Report a Bug Page — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `/report-bug` page to the SeeChords promotion site that lets users open a pre-filled GitHub Issue via a client-side form.

**Architecture:** New Flask route `GET /report-bug` renders a new `report_bug.html` template. The template reuses the same CSS variables and layout structure as `site_home.html`. A small inline `<script>` builds a GitHub Issues URL from the form fields and calls `window.open()` — no server-side form handling.

**Tech Stack:** Flask (Jinja2 templates), vanilla HTML/CSS/JS, Fly.io (`fly deploy --config server/fly.toml`)

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `server/templates/report_bug.html` | The full report-a-bug page (form + JS logic) |
| Modify | `server/app.py` | Add `@app.route('/report-bug')` |
| Modify | `server/templates/site_home.html` | Add "Report a bug" link in footer `<nav>` |

---

### Task 1: Add the Flask route

**Files:**
- Modify: `server/app.py` (near line 1661, after the `privacy_policy` route)

- [ ] **Step 1: Add the route**

  In `server/app.py`, after the `privacy_policy` function (around line 1664), add:

  ```python
  @app.route('/report-bug')
  def report_bug():
      return render_template('report_bug.html')
  ```

- [ ] **Step 2: Smoke-test the route locally**

  From the repo root:
  ```bash
  cd server && python -c "
  from app import app
  with app.test_client() as c:
      r = c.get('/report-bug')
      assert r.status_code == 200, f'Expected 200, got {r.status_code}'
      print('OK: /report-bug returns 200')
  "
  ```
  Expected output: `OK: /report-bug returns 200`

  (This will fail with a TemplateNotFound error until Task 2 is done — that's expected.)

- [ ] **Step 3: Commit**

  ```bash
  git add server/app.py
  git commit -m "feat: add /report-bug Flask route"
  ```

---

### Task 2: Create the report_bug.html template

**Files:**
- Create: `server/templates/report_bug.html`

- [ ] **Step 1: Create the template**

  Create `server/templates/report_bug.html` with this content:

  ```html
  <!DOCTYPE html>
  <html lang="en">
  <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="Report a bug or request a feature for SeeChords.">
    <title>Report a Bug — SeeChords</title>
    <style>
      :root {
        --bg: #080812;
        --bg-card: #12122a;
        --border: #2a2a50;
        --accent: #6c47ff;
        --accent-hover: #8264ff;
        --text: #f0f0f8;
        --muted: #8888aa;
        --radius: 12px;
      }
      * { box-sizing: border-box; margin: 0; padding: 0; }
      body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: var(--bg);
        color: var(--text);
        line-height: 1.6;
        min-height: 100vh;
      }
      a { color: var(--accent-hover); text-decoration: none; }
      a:hover { text-decoration: underline; }
      .wrap { max-width: 920px; margin: 0 auto; padding: 0 24px; }
      header {
        padding: 28px 0 48px;
        background: linear-gradient(135deg, #110d2e 0%, #1a0938 45%, #080812 100%);
        border-bottom: 1px solid var(--border);
      }
      .brand {
        display: flex;
        align-items: center;
        gap: 0.43em;
        margin-bottom: 12px;
        font-size: clamp(1.75rem, 4vw, 2.25rem);
      }
      .brand-logo {
        width: 1.5556em;
        height: 1.5556em;
        flex-shrink: 0;
        border-radius: 0.214em;
        object-fit: contain;
        display: block;
      }
      .logo {
        font-size: 1em;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #fff;
        margin-bottom: 0;
      }
      .tagline {
        font-size: 1.15rem;
        color: var(--muted);
        max-width: 36em;
      }
      main { padding: 48px 0 64px; }
      .card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 28px 32px;
        max-width: 600px;
      }
      label {
        display: block;
        font-size: 0.9rem;
        color: var(--muted);
        margin-bottom: 6px;
        margin-top: 20px;
      }
      label:first-child { margin-top: 0; }
      input, select, textarea {
        width: 100%;
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: 8px;
        color: var(--text);
        font-family: inherit;
        font-size: 0.95rem;
        padding: 10px 14px;
        outline: none;
        transition: border-color 0.15s;
      }
      input:focus, select:focus, textarea:focus {
        border-color: var(--accent);
      }
      textarea { resize: vertical; min-height: 90px; }
      select option { background: #12122a; }
      .btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        padding: 12px 22px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.95rem;
        border: 1px solid transparent;
        cursor: pointer;
        transition: background 0.15s, border-color 0.15s;
        font-family: inherit;
      }
      .btn-primary {
        background: var(--accent);
        color: #fff;
        margin-top: 24px;
      }
      .btn-primary:hover { background: var(--accent-hover); text-decoration: none; }
      #steps-row { display: block; }
      footer {
        border-top: 1px solid var(--border);
        padding: 28px 0 40px;
        color: var(--muted);
        font-size: 0.9rem;
      }
      footer .wrap { display: flex; flex-wrap: wrap; gap: 16px; align-items: center; justify-content: space-between; }
      footer nav { display: flex; gap: 20px; }
    </style>
  </head>
  <body>
    <header>
      <div class="wrap">
        <div class="brand">
          <img class="brand-logo" src="{{ url_for('static', filename='icon128.png') }}" alt="SeeChords" width="128" height="128" />
          <p class="logo">SeeChords</p>
        </div>
        <p class="tagline">Found a bug or have an idea? Let us know.</p>
      </div>
    </header>
    <main>
      <div class="wrap">
        <div class="card">
          <form id="bug-form">
            <label for="type">Type</label>
            <select id="type" name="type">
              <option value="bug">Bug report</option>
              <option value="feature">Feature request</option>
            </select>

            <label for="description">Description <span style="color:var(--accent)">*</span></label>
            <textarea id="description" name="description" placeholder="What happened? What did you expect?" required></textarea>

            <div id="steps-row">
              <label for="steps">Steps to reproduce</label>
              <textarea id="steps" name="steps" placeholder="1. Open YouTube&#10;2. ..."></textarea>
            </div>

            <label for="yt-url">YouTube URL (if applicable)</label>
            <input id="yt-url" name="yt_url" type="url" placeholder="https://www.youtube.com/watch?v=..." />

            <label for="browser">Browser / OS</label>
            <input id="browser" name="browser" type="text" />

            <label for="version">Extension version (if known)</label>
            <input id="version" name="version" type="text" placeholder="e.g. 1.1" />

            <button class="btn btn-primary" type="submit">Open GitHub Issue</button>
          </form>
        </div>
      </div>
    </main>
    <footer>
      <div class="wrap">
        <p>SeeChords extension · API at <a href="https://seechords.fly.dev/api/health">seechords.fly.dev</a></p>
        <nav>
          <a href="/privacy">Privacy</a>
          <a href="/report-bug">Report a bug</a>
        </nav>
      </div>
    </footer>
    <script>
      // Pre-fill browser/OS from user agent
      document.getElementById('browser').value = navigator.userAgent;

      // Toggle steps field visibility based on type
      const typeEl = document.getElementById('type');
      const stepsRow = document.getElementById('steps-row');
      typeEl.addEventListener('change', function () {
        stepsRow.style.display = this.value === 'bug' ? 'block' : 'none';
      });

      document.getElementById('bug-form').addEventListener('submit', function (e) {
        e.preventDefault();

        const type = document.getElementById('type').value;
        const description = document.getElementById('description').value.trim();
        const steps = document.getElementById('steps').value.trim();
        const ytUrl = document.getElementById('yt-url').value.trim();
        const browser = document.getElementById('browser').value.trim();
        const version = document.getElementById('version').value.trim();

        if (!description) return;

        const prefix = type === 'bug' ? '[Bug]' : '[Feature]';
        const firstLine = description.split('\n')[0].slice(0, 80);
        const title = prefix + ' ' + firstLine;

        let body = '';
        if (type === 'bug') {
          body += '## Bug Report\n\n';
        } else {
          body += '## Feature Request\n\n';
        }

        body += '**Description**\n' + description + '\n\n';

        if (type === 'bug' && steps) {
          body += '**Steps to reproduce**\n' + steps + '\n\n';
        }

        if (ytUrl) {
          body += '**YouTube URL**\n' + ytUrl + '\n\n';
        }

        if (browser) {
          body += '**Browser / OS**\n' + browser + '\n\n';
        }

        if (version) {
          body += '**Extension version**\n' + version + '\n\n';
        }

        const url = 'https://github.com/jeounggene/seechords/issues/new'
          + '?title=' + encodeURIComponent(title)
          + '&body=' + encodeURIComponent(body);

        window.open(url, '_blank');
      });
    </script>
  </body>
  </html>
  ```

- [ ] **Step 2: Smoke-test the route now returns 200 with content**

  ```bash
  cd server && python -c "
  from app import app
  with app.test_client() as c:
      r = c.get('/report-bug')
      assert r.status_code == 200, f'Expected 200, got {r.status_code}'
      assert b'report-bug' in r.data, 'Expected report-bug in response'
      print('OK: /report-bug returns 200 with content')
  "
  ```
  Expected output: `OK: /report-bug returns 200 with content`

- [ ] **Step 3: Commit**

  ```bash
  git add server/templates/report_bug.html
  git commit -m "feat: add report-a-bug page template"
  ```

---

### Task 3: Add footer link on home page

**Files:**
- Modify: `server/templates/site_home.html` (footer `<nav>`, around line 209)

- [ ] **Step 1: Add the link**

  In `server/templates/site_home.html`, find the footer `<nav>` block:

  ```html
        <nav>
          <a href="/privacy">Privacy</a>
          {% if donation_url %}
          <a href="{{ donation_url }}" rel="noopener noreferrer" target="_blank">Support the project</a>
          {% endif %}
        </nav>
  ```

  Replace it with:

  ```html
        <nav>
          <a href="/privacy">Privacy</a>
          <a href="/report-bug">Report a bug</a>
          {% if donation_url %}
          <a href="{{ donation_url }}" rel="noopener noreferrer" target="_blank">Support the project</a>
          {% endif %}
        </nav>
  ```

- [ ] **Step 2: Smoke-test the home page contains the link**

  ```bash
  cd server && python -c "
  from app import app
  with app.test_client() as c:
      r = c.get('/')
      assert r.status_code == 200, f'Expected 200, got {r.status_code}'
      assert b'/report-bug' in r.data, 'Expected /report-bug link in home page'
      print('OK: home page contains /report-bug link')
  "
  ```
  Expected output: `OK: home page contains /report-bug link`

- [ ] **Step 3: Commit**

  ```bash
  git add server/templates/site_home.html
  git commit -m "feat: add report-a-bug link to home page footer"
  ```

---

### Task 4: Deploy to Fly.io

**Files:** None (deploy only)

- [ ] **Step 1: Deploy the API app**

  Run from the repo root (not from `server/`):

  ```bash
  cd /Users/genej/projects/seechords
  fly deploy --config server/fly.toml
  ```

  Expected: deployment completes with `v<N> deployed successfully`

- [ ] **Step 2: Verify the live route**

  ```bash
  curl -s -o /dev/null -w "%{http_code}" https://seechords.fly.dev/report-bug
  ```

  Expected output: `200`

- [ ] **Step 3: Manual smoke test**

  Open `https://seechords.fly.dev/report-bug` in a browser. Verify:
  - Form renders with all fields
  - "Bug report" selected by default shows the "Steps to reproduce" field
  - Switching to "Feature request" hides the "Steps to reproduce" field
  - Browser/OS field is auto-filled
  - Submitting opens a pre-filled GitHub Issue in a new tab at `github.com/jeounggene/seechords/issues/new`

  Also verify `https://seechords.fly.dev/` footer shows "Report a bug" link.
