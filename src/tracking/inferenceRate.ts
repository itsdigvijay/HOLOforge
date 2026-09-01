/** Selects the highest rate with comfortable inference and render headroom. */
export function chooseAutoInferenceRate(
  cameraFps: number,
  inferenceTimeMs: number,
  renderFps: number,
): 30 | 45 | 60 {
  if (renderFps > 0 && renderFps < 52) return 30;
  const renderHealthy = renderFps === 0 || renderFps >= 54;
  if (renderHealthy && cameraFps >= 55 && inferenceTimeMs <= 12.5) return 60;
  if (renderHealthy && cameraFps >= 40 && inferenceTimeMs <= 18) return 45;
  return 30;
}
