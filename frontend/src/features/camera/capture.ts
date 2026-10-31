/** Encode the current video frame (unmirrored) as a downscaled JPEG. */
export function captureFrame(video: HTMLVideoElement, maxWidth: number, quality: number,
  canvas: HTMLCanvasElement = document.createElement('canvas')): Promise<Blob | null> {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh || video.readyState < 2) return Promise.resolve(null);
  const scale = Math.min(1, maxWidth / vw);
  canvas.width = Math.round(vw * scale);
  canvas.height = Math.round(vh * scale);
  const ctx = canvas.getContext('2d');
  if (!ctx) return Promise.resolve(null);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve) => canvas.toBlob((b) => resolve(b), 'image/jpeg', quality));
}
