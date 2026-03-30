/* SeeChords – background service worker
 *
 * Handles messages from content script and popup:
 *  - CHECK_CHORDS: check if chords exist for a videoId
 *  - UPLOAD_AND_ANALYZE: upload MP3 + videoId to backend
 *  - POLL_STATUS: poll analysis job status
 */

'use strict';

const API_BASE = 'https://seechords.fly.dev';

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === 'CHECK_CHORDS') {
    fetchChords(msg.videoId).then(sendResponse);
    return true; // async
  }

  if (msg.type === 'UPLOAD_AND_ANALYZE') {
    uploadAndAnalyze(msg.videoId, msg.fileData, msg.fileName, msg.title).then(sendResponse);
    return true;
  }

  if (msg.type === 'POLL_STATUS') {
    pollStatus(msg.jobId).then(sendResponse);
    return true;
  }

  if (msg.type === 'COMPARE_CHORDS') {
    compareChords(msg.videoId, msg.referenceText).then(sendResponse);
    return true;
  }

  if (msg.type === 'ALIGN_CHORDS') {
    alignChords(msg.videoId, msg.referenceText).then(sendResponse);
    return true;
  }

  if (msg.type === 'GET_API_BASE') {
    sendResponse({ apiBase: API_BASE });
    return false;
  }

  if (msg.type === 'CHECK_AUDIO_STREAM') {
    sendResponse({ available: true }); // always available now — we use backend yt-dlp fallback
    return false;
  }

  if (msg.type === 'EXTRACT_AND_ANALYZE') {
    extractAndAnalyze(msg.videoId, msg.title, sender.tab.id).then(sendResponse).catch(err => {
      sendResponse({ error: err.message });
    });
    return true;
  }

  if (msg.type === 'LIST_VERSIONS') {
    listVersions(msg.videoId).then(sendResponse);
    return true;
  }

  if (msg.type === 'LOAD_VERSION') {
    loadVersion(msg.versionId).then(sendResponse);
    return true;
  }

  if (msg.type === 'RATE_CHORDS') {
    rateChords(msg.versionId, msg.videoId, msg.stars).then(sendResponse);
    return true;
  }

  if (msg.type === 'GET_RATING') {
    getRating(msg.versionId).then(sendResponse);
    return true;
  }
});

async function rateChords(versionId, videoId, stars) {
  try {
    const res = await fetch(`${API_BASE}/api/rate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ versionId, videoId, stars }),
    });
    if (res.ok) return await res.json();
    return { error: 'Failed to save rating' };
  } catch (err) {
    return { error: err.message };
  }
}

async function getRating(versionId) {
  try {
    const res = await fetch(`${API_BASE}/api/rating/${versionId}`);
    if (res.ok) return await res.json();
    return { stars: 0 };
  } catch (err) {
    return { stars: 0 };
  }
}

async function listVersions(videoId) {
  try {
    const res = await fetch(`${API_BASE}/api/versions/${encodeURIComponent(videoId)}`);
    if (res.ok) return await res.json();
    return { versions: [] };
  } catch (err) {
    return { versions: [], error: err.message };
  }
}

async function loadVersion(versionId) {
  try {
    const res = await fetch(`${API_BASE}/api/version/${encodeURIComponent(versionId)}`);
    if (res.ok) return { found: true, data: await res.json() };
    return { found: false };
  } catch (err) {
    return { found: false, error: err.message };
  }
}

async function fetchChords(videoId) {
  try {
    const res = await fetch(`${API_BASE}/api/chords/${encodeURIComponent(videoId)}`);
    if (res.ok) {
      const data = await res.json();
      return { found: true, data };
    }
    return { found: false };
  } catch (err) {
    return { found: false, error: err.message };
  }
}

async function uploadAndAnalyze(videoId, fileDataUrl, fileName, title) {
  try {
    // Convert data URL to blob
    const res = await fetch(fileDataUrl);
    const blob = await res.blob();

    const form = new FormData();
    form.append('videoId', videoId);
    form.append('file', blob, fileName);
    if (title) form.append('title', title);

    const apiRes = await fetch(`${API_BASE}/api/analyze`, {
      method: 'POST',
      body: form,
    });
    const data = await apiRes.json();
    return data;
  } catch (err) {
    return { error: err.message };
  }
}

async function extractAndAnalyze(videoId, title, tabId) {
  try {
    // Strategy 1: Try to get audio stream URL from page and fetch directly
    const streamInfo = await getStreamInfo(tabId);
    if (streamInfo?.directUrl) {
      console.log('[SeeChords] Trying direct stream fetch...');
      try {
        const res = await fetch(streamInfo.directUrl);
        if (res.ok) {
          const blob = await res.blob();
          if (blob.size > 10000) { // sanity check — real audio is >10KB
            console.log('[SeeChords] Direct fetch succeeded, size:', blob.size);
            const ext = streamInfo.mime?.includes('webm') ? '.webm' : '.mp4';
            const form = new FormData();
            form.append('videoId', videoId);
            form.append('file', blob, `audio${ext}`);
            if (title) form.append('title', title);
            const apiRes = await fetch(`${API_BASE}/api/analyze`, { method: 'POST', body: form });
            return await apiRes.json();
          }
        }
        console.log('[SeeChords] Direct fetch failed or too small, falling back to backend');
      } catch (e) {
        console.log('[SeeChords] Direct fetch error:', e.message);
      }
    }

    // Strategy 2: Let the backend download audio via yt-dlp
    console.log('[SeeChords] Using backend yt-dlp for', videoId);
    const apiRes = await fetch(`${API_BASE}/api/analyze-youtube`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ videoId, title }),
    });
    const data = await apiRes.json();
    return data;
  } catch (err) {
    return { error: err.message };
  }
}

// Try to extract a direct audio stream URL from YouTube's player data
async function getStreamInfo(tabId) {
  try {
    const results = await chrome.scripting.executeScript({
      target: { tabId },
      world: 'MAIN',
      func: () => {
        try {
          const player = document.getElementById('movie_player');
          const pr = player?.getPlayerResponse?.()
                  || window.ytInitialPlayerResponse
                  || document.querySelector('ytd-watch-flexy')?.playerData_;
          if (!pr?.streamingData) return null;

          const sd = pr.streamingData;
          const af = sd.adaptiveFormats || [];
          const preferred = [140, 251, 250, 249];

          // Check if any format has a direct URL
          for (const itag of preferred) {
            const fmt = af.find(f => f.itag === itag);
            if (fmt?.url) return { directUrl: fmt.url, mime: fmt.mimeType };
          }

          // Try constructing URL from serverAbrStreamingUrl
          if (sd.serverAbrStreamingUrl) {
            const base = sd.serverAbrStreamingUrl;
            const audioFmt = af.find(f => f.itag === 140) || af.find(f => f.mimeType?.startsWith('audio/'));
            if (audioFmt) {
              // Construct direct download URL
              const url = base.replace(/&sabr=[^&]*/, '') + '&itag=' + audioFmt.itag;
              return {
                directUrl: url,
                mime: audioFmt.mimeType,
                contentLength: audioFmt.contentLength
              };
            }
          }
        } catch (e) {
          console.log('[SeeChords page] getStreamInfo error:', e.message);
        }
        return null;
      }
    });
    return results?.[0]?.result;
  } catch (e) {
    console.log('[SeeChords] getStreamInfo injection failed:', e.message);
    return null;
  }
}

async function pollStatus(jobId) {
  try {
    const res = await fetch(`${API_BASE}/api/status/${encodeURIComponent(jobId)}`);
    const data = await res.json();
    return data;
  } catch (err) {
    return { status: 'error', message: err.message };
  }
}

async function compareChords(videoId, referenceText) {
  try {
    const res = await fetch(`${API_BASE}/api/compare`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ videoId, referenceText }),
    });
    const data = await res.json();
    return data;
  } catch (err) {
    return { error: err.message };
  }
}

async function alignChords(videoId, referenceText) {
  try {
    const res = await fetch(`${API_BASE}/api/align`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ videoId, referenceText }),
    });
    const data = await res.json();
    return data;
  } catch (err) {
    return { error: err.message };
  }
}



// DASH Audio Interception
const audioStreamUrls = {};

// Primary: extract audio URL from YouTube's own player data
async function findAudioStreamUrl(tabId) {
  console.log('[SeeChords] findAudioStreamUrl called for tab', tabId);

  // 1. Check cached URL from webRequest listener
  if (audioStreamUrls[tabId]) {
    console.log('[SeeChords] Found cached webRequest URL for tab', tabId);
    return audioStreamUrls[tabId];
  }

  // 2. Inject into the page to extract audio URL
  try {
    console.log('[SeeChords] Injecting script into tab', tabId);
    const results = await chrome.scripting.executeScript({
      target: { tabId },
      world: 'MAIN',
      func: () => {
        const log = [];

        // Method A: YouTube player API  (movie_player has internal APIs)
        try {
          const player = document.getElementById('movie_player');
          if (player) {
            // getPlayerResponse() returns the live player response with resolved URLs
            if (typeof player.getPlayerResponse === 'function') {
              const pr = player.getPlayerResponse();
              log.push('getPlayerResponse exists: ' + !!pr);
              log.push('streamingData: ' + !!pr?.streamingData);
              const af = pr?.streamingData?.adaptiveFormats;
              if (af) {
                const audioFmts = af.filter(f => f.mimeType?.startsWith('audio/'));
                log.push('Player API audio formats: ' + audioFmts.length);
                // Log first audio format keys for debugging
                if (audioFmts.length > 0) {
                  log.push('First audio format keys: ' + Object.keys(audioFmts[0]).join(', '));
                }
                const preferred = [140, 251, 250, 249];
                for (const itag of preferred) {
                  const fmt = af.find(f => f.itag === itag);
                  if (fmt?.url) {
                    log.push('Player API FOUND url for itag ' + itag);
                    return { url: fmt.url, log };
                  }
                }
                const audio = audioFmts.find(f => f.url);
                if (audio?.url) {
                  log.push('Player API FOUND url for itag ' + audio.itag);
                  return { url: audio.url, log };
                }
              }
            }

            // Try getVideoData or config internals
            if (typeof player.getVideoData === 'function') {
              const vd = player.getVideoData();
              log.push('getVideoData keys: ' + Object.keys(vd || {}).join(', '));
            }

            // Try accessing internal config
            const cfg = player.getUpdatedConfigurationData?.() || player.config_;
            if (cfg) {
              log.push('config keys: ' + Object.keys(cfg).join(', '));
            }
          } else {
            log.push('movie_player not found');
          }
        } catch (e) {
          log.push('Method A (Player API) error: ' + e.message);
        }

        // Method B: ytInitialPlayerResponse with deep key inspection
        try {
          const pr = window.ytInitialPlayerResponse
                  || document.querySelector('ytd-watch-flexy')?.playerData_;
          if (pr?.streamingData) {
            const sd = pr.streamingData;
            log.push('streamingData keys: ' + Object.keys(sd).join(', '));
            
            // Check for serverAbrStreamingUrl (newer YT uses server-side ABR)
            if (sd.serverAbrStreamingUrl) {
              log.push('serverAbrStreamingUrl found: ' + sd.serverAbrStreamingUrl.substring(0, 100));
            }

            // Log the first audio format's full property names  
            if (sd.adaptiveFormats) {
              const audioFmt = sd.adaptiveFormats.find(f => f.mimeType?.startsWith('audio/'));
              if (audioFmt) {
                log.push('Sample audio format ALL keys: ' + Object.keys(audioFmt).join(', '));
                // Show values of non-url keys that might contain URL info
                for (const [k, v] of Object.entries(audioFmt)) {
                  if (typeof v === 'string' && v.length > 20) {
                    log.push(`  ${k}: ${v.substring(0, 120)}...`);
                  } else if (typeof v === 'object' && v !== null) {
                    log.push(`  ${k}: ${JSON.stringify(v).substring(0, 120)}`);
                  }
                }
              }
            }
          }
        } catch (e) {
          log.push('Method B (ytInitialPlayerResponse) error: ' + e.message);
        }

        // Method C: Performance Resource Timing API
        try {
          const entries = performance.getEntriesByType('resource');
          let found = [];
          for (let i = entries.length - 1; i >= 0; i--) {
            const name = entries[i].name;
            if (name.includes('googlevideo.com/videoplayback')) {
              found.push(name.substring(0, 150));
              if (name.includes('mime=audio') ||
                  name.includes('itag=140') || name.includes('itag=249') ||
                  name.includes('itag=250') || name.includes('itag=251')) {
                log.push('Method C FOUND audio in resource timing');
                return { url: name, log };
              }
            }
          }
          log.push('Method C googlevideo URLs found: ' + found.length);
          if (found.length > 0) {
            log.push('Method C first URL: ' + found[0]);
          }
        } catch (e) {
          log.push('Method C error: ' + e.message);
        }

        console.log('[SeeChords page] ALL METHODS FAILED:\n' + log.join('\n'));
        return { url: null, log };
      }
    });
    const result = results?.[0]?.result;
    if (result?.log) {
      for (const line of result.log) {
        console.log('[SeeChords]', line);
      }
    }
    if (result?.url) {
      audioStreamUrls[tabId] = result.url;
      return result.url;
    }
  } catch (e) {
    console.error('[SeeChords] Script injection failed:', e.message);
  }
  console.log('[SeeChords] No audio stream found for tab', tabId);
  return null;
}

chrome.webRequest.onBeforeRequest.addListener(
  (details) => {
    const url = details.url;
    if (url.includes('/videoplayback')) {
      if (url.includes('mime=audio') || 
          url.includes('itag=140') || 
          url.includes('itag=249') || 
          url.includes('itag=250') || 
          url.includes('itag=251') ||
          url.includes('itag=139') ||
          url.includes('itag=256')) {
        if (details.tabId >= 0) {
          audioStreamUrls[details.tabId] = url;
          console.log("Captured audio stream for tab", details.tabId);
        }
      }
    }
  },
  { urls: ["*://*.googlevideo.com/*"] }
);

chrome.tabs.onRemoved.addListener((tabId) => {
  delete audioStreamUrls[tabId];
});
