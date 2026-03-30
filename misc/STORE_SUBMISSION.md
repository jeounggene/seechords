# SeeChords — Chrome Web Store Submission Checklist

## Store Listing Tab

**Language:** English

**Category:** Education

**Detailed Description (min 25 chars):**
SeeChords helps you learn guitar by displaying real-time chord diagrams overlaid on YouTube videos. When you visit a YouTube video that has chord data available, the extension shows a synchronized chord strip at the bottom of the video with chord names and fingering diagrams that advance automatically with playback. Practice along with any song — see exactly which chords to play and when. You can also submit audio for analysis to generate chord data for any YouTube video. Features include transpose controls, timing offset adjustment, and a scrolling beat-synced chord display.

---

## Privacy Practices Tab

**Single Purpose Description:**
Help users learn guitar chords by displaying synchronized chord diagrams on YouTube videos.

### Permission Justifications

**activeTab:**
Used to query the active tab's URL to extract the YouTube video ID, and to display chord analysis status in the popup.

**scripting:**
Used to inject a small script into the YouTube page to read the audio stream URL from the player's existing data. This enables optional chord analysis when the user explicitly requests it. No audio is downloaded, stored, or retained by the extension itself — audio is streamed to the SeeChords backend for real-time analysis only, then immediately deleted. No copyrighted content is saved, redistributed, or made available to other users; only the derived chord names and timing data are stored.

**webRequest:**
Used to passively observe network requests to googlevideo.com to identify audio stream URLs. These URLs are used only when the user explicitly initiates chord analysis. No requests are modified, blocked, or redirected. No audio data is stored by the extension — it is forwarded to the backend for transient analysis and immediately deleted afterward.

**Host permissions (youtube.com, googlevideo.com, seechords.fly.dev):**
- youtube.com: Required to run the content script that injects the chord overlay UI on YouTube video pages.
- googlevideo.com: Required to observe audio stream network requests for optional chord analysis.
- seechords.fly.dev: Required to communicate with the SeeChords API to fetch and submit chord data.

**Remote code justification:**
SeeChords does not use any remote code. All JavaScript is bundled within the extension package. The extension only exchanges JSON data with the SeeChords API server (seechords.fly.dev). No code is fetched, evaluated, or executed from any remote source.

### Data Use Certification
- The extension does not collect or transmit any personal data.
- The extension does not use cookies or tracking.
- The extension does not sell or share data with third parties.
- Audio submitted for analysis is processed server-side and immediately deleted — no copyrighted audio is stored, cached, or redistributed.
- Only derived, non-copyrightable data (chord names and beat timestamps) is retained.

**Privacy Policy URL:** (host privacy-policy.html somewhere — e.g. GitHub Pages, or add a /privacy route to your Fly server)

---

## Account Tab
- [ ] Enter contact email
- [ ] Verify contact email

## Assets Still Needed
- [ ] At least one screenshot (1280x800 or 640x400)
- [ ] Store icon (128x128 PNG) — you already have icons/icon128.png, upload that
- [ ] Optional: promotional images (440x280 small tile, 920x680 large tile)

## Privacy Policy Hosting
The file `extension/privacy-policy.html` has been created. You need to host it at a public URL. Options:
1. Add a route to your Fly server: `@app.route('/privacy')` that serves it
2. Push to GitHub and use GitHub Pages
3. Host on any static site
