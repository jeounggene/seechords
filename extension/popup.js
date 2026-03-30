/* SeeChords popup – shows current video status */

'use strict';

document.addEventListener('DOMContentLoaded', async () => {
  const titleEl = document.getElementById('statusTitle');
  const msgEl   = document.getElementById('statusMsg');

  // Get the active tab
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab || !tab.url) {
    titleEl.textContent = 'Not on YouTube';
    msgEl.textContent = 'Open a YouTube video to see chords.';
    return;
  }

  const url = new URL(tab.url);
  if (!url.hostname.includes('youtube.com')) {
    titleEl.textContent = 'Not on YouTube';
    msgEl.textContent = 'Navigate to a YouTube video to use SeeChords.';
    return;
  }

  const videoId = url.searchParams.get('v');
  if (!videoId) {
    titleEl.textContent = 'No Video Detected';
    msgEl.textContent = 'Open a YouTube video page (not the homepage).';
    return;
  }

  titleEl.innerHTML = `Video: <span class="vid-id">${videoId}</span>`;
  msgEl.textContent = 'Checking for chords…';

  chrome.runtime.sendMessage({ type: 'CHECK_CHORDS', videoId }, (response) => {
    if (response && response.found) {
      msgEl.textContent = `✓ Chords available! Key: ${response.data.key || '?'}, BPM: ${response.data.bpm || '?'}`;
      msgEl.classList.add('success');
    } else {
      msgEl.textContent = 'No chords yet. Use the overlay on the video page to upload matching audio.';
      msgEl.classList.add('error');
    }
  });
});
