import { useRef, useState } from 'react';

const ACCEPTED = '.jpg,.jpeg,.png,.webp,.bmp,.tif,.tiff';

export default function UploadPanel({ onAnalyze, disabled }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  function pick(file) {
    if (file) onAnalyze(file);
  }

  function onDrop(event) {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files?.[0];
    pick(file);
  }

  return (
    <div
      className={`upload-box${dragging ? ' dragging' : ''}${disabled ? ' disabled' : ''}`}
      onClick={() => !disabled && inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
    >
      <div className="upload-icon" aria-hidden="true">
        ⬆
      </div>
      <h3>Upload an image for analysis</h3>
      <p>Drag and drop, or click to browse</p>
      <p className="muted">JPEG, PNG, WebP, BMP &middot; up to 16 MB</p>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED}
        hidden
        onChange={(e) => {
          pick(e.target.files?.[0]);
          e.target.value = '';
        }}
      />
    </div>
  );
}