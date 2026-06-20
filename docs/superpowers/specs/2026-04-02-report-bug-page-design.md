# Report a Bug Page — Design Spec

**Date:** 2026-04-02  
**Status:** Approved

## Overview

Add a `/report-bug` page to the SeeChords promotion site. The page lets users submit bug reports or feature requests by filling out a form that opens a pre-filled GitHub Issue in a new tab. No backend changes required.

## Architecture

- New Flask route: `GET /report-bug` → renders `server/templates/report_bug.html`
- New template: `server/templates/report_bug.html` — same dark purple theme, same CSS variables as `site_home.html`
- `site_home.html` footer gets a "Report a bug" link alongside the existing Privacy link
- No new Python dependencies, no server-side form handling

## Form Fields

| Field | Type | Required | Notes |
|---|---|---|---|
| Bug type | Dropdown | Yes | Options: Bug, Feature Request |
| Description | Textarea | Yes | What happened / what you want |
| Steps to reproduce | Textarea | No | Only shown for Bug type |
| YouTube URL | Text input | No | Video where bug occurred |
| Browser / OS | Text input | No | Pre-filled from `navigator.userAgent`, editable |
| Extension version | Text input | No | User enters manually |

## GitHub Integration

On submit, JavaScript constructs a GitHub Issue URL:

```
https://github.com/jeounggene/seechords/issues/new?title=<title>&body=<body>
```

- `title` = `[Bug]` or `[Feature]` prefix + first line of description
- `body` = markdown-formatted content from all filled fields
- `window.open(url, '_blank')` opens the issue in a new tab
- No form submission to the server

## Page Layout

- **Header**: SeeChords brand logo + tagline "Found a bug or have an idea? Let us know."
- **Main**: single `.card` containing the form
- **Footer**: identical to home page (Privacy, Support the project links)

## Changes to Existing Files

- `server/app.py`: add `@app.route('/report-bug')` route
- `server/templates/site_home.html`: add "Report a bug" link in footer `<nav>`

## Out of Scope

- Email notifications
- Issue tracking database
- User authentication
