# Changelog

All notable changes to SeeChords are documented here.

## [1.1.2] — 2026-04-03
### Added
- Popup toggle to hide SeeChords entirely on YouTube pages
- Overlay removes immediately when hidden; re-injects immediately when un-hidden
- `"storage"` permission added to manifest to support hide preference

## [1.1.1] — 2026-03-31
### Fixed
- Timeline scroll no longer steals page focus during playback — replaced `scrollIntoView` with a direct scroll on the timeline container so users can browse comments or other parts of the page while a song plays

## [1.1.0] — 2026-03-31
### Added
- Overlay, below-video panel, and fullscreen display modes
- Beat-synced timeline with click-to-seek
- Chord cards with transpose support
- Timeline edge padding so first/last chords can scroll to center
- API/worker split deployment (Dockerfile.api, Dockerfile.worker)
- Marketing home page hero screenshot and site promo assets

### Fixed
- Uniform beat_times option for consistent beat spacing
- Hidden sync UI polish

## [1.0.0] — initial release
### Added
- Chrome extension with chord overlay on YouTube videos
- Flask server with BTC chord model and Beat This! beat tracker
- Upload zone and analysis progress polling
- Web player with audio controls, seek bar, and beat-sync loop
- Report-a-bug page
