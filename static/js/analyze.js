
const mode = document.getElementById('mode');
const exercise = document.getElementById('exercise');
const uploadForm = document.getElementById('uploadForm');
const videoInput = document.getElementById('videoInput');
const uploadZone = document.getElementById('uploadZone');
const fileName = document.getElementById('fileName');

if (mode && exercise) {
  mode.addEventListener('change', async () => {
    const r = await fetch(`/api/exercises?mode=${encodeURIComponent(mode.value)}`);
    const items = await r.json();
    exercise.innerHTML = items.map(x => `<option>${x}</option>`).join('');
  });
}

const updateFileName = file => {
  if (!fileName) return;
  fileName.textContent = file ? file.name : 'No file selected';
};

videoInput?.addEventListener('change', () => updateFileName(videoInput.files?.[0]));

['dragenter', 'dragover'].forEach(evt => {
  uploadZone?.addEventListener(evt, e => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
  });
});
['dragleave', 'drop'].forEach(evt => {
  uploadZone?.addEventListener(evt, e => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
  });
});
uploadZone?.addEventListener('drop', e => {
  const files = e.dataTransfer?.files;
  if (files?.length && videoInput) {
    const dt = new DataTransfer();
    dt.items.add(files[0]);
    videoInput.files = dt.files;
    updateFileName(files[0]);
  }
});

uploadForm?.addEventListener('submit', () => {
  window.showLoading?.(
    'Analyzing biomechanical motion',
    'Pose detection → joint kinematics → form analysis → PINN fatigue estimation'
  );
});

let recorder, chunks = [], stream;
const preview = document.getElementById('preview');
const start = document.getElementById('startRec');
const stop = document.getElementById('stopRec');
const statusEl = document.getElementById('recordStatus');
const recordIndicator = document.getElementById('recordIndicator');

if (start) {
  start.onclick = async () => {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      preview.srcObject = stream;
      chunks = [];
      recorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
      recorder.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
      recorder.start();
      start.disabled = true;
      stop.disabled = false;
      recordIndicator?.classList.add('show');
      statusEl.textContent = 'Recording… perform the exercise, then press Stop & analyze.';
    } catch (e) {
      statusEl.textContent = 'Camera error: ' + e.message;
    }
  };

  stop.onclick = () => {
    stop.disabled = true;
    statusEl.textContent = 'Preparing recording…';

    recorder.onstop = async () => {
      stream.getTracks().forEach(t => t.stop());
      recordIndicator?.classList.remove('show');

      const blob = new Blob(chunks, { type: 'video/webm' });
      const fd = new FormData();
      fd.append('video', blob, 'browser_recording.webm');
      fd.append('exercise', exercise.value);
      fd.append('mode', mode.value);
      const rpe = document.querySelector('[name=rpe]').value;
      fd.append('rpe', rpe);
      fd.append('use_pinn', document.querySelector('[name=use_pinn]').checked ? 'true' : 'false');

      statusEl.textContent = 'Analyzing recording…';
      window.showLoading?.(
        'Analyzing camera recording',
        'Processing pose landmarks, biomechanics, form quality and fatigue prediction.'
      );

      try {
        const r = await fetch('/api/analyze-recording', { method: 'POST', body: fd });
        const data = await r.json();
        if (data.redirect) {
          location = data.redirect;
        } else {
          window.hideLoading?.();
          statusEl.textContent = data.error || 'Analysis failed';
          start.disabled = false;
        }
      } catch (err) {
        window.hideLoading?.();
        statusEl.textContent = 'Analysis failed: ' + err.message;
        start.disabled = false;
      }
    };

    recorder.stop();
  };
}
