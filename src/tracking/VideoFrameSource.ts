/** Captures the current camera frame for transfer to a tracking worker. */
export type VideoFrameCapture = () => VideoFrame | Promise<ImageBitmap>;

export type VideoFrameListener = (
  now: number,
  mediaTime: number,
  capture: VideoFrameCapture,
) => void;

export interface VideoFrameSubscription {
  readonly usingVideoFrameCallback: boolean;
  unsubscribe(): void;
}

const sources = new WeakMap<HTMLVideoElement, SharedVideoFrameSource>();

/**
 * One camera-frame clock per video element. Tracking engines subscribe to this
 * clock instead of creating competing requestVideoFrameCallback/rAF loops.
 */
class SharedVideoFrameSource {
  private readonly listeners = new Set<VideoFrameListener>();
  private videoFrameCallbackId: number | null = null;
  private animationFrameId: number | null = null;
  private lastVideoTime = -1;

  readonly usingVideoFrameCallback: boolean;

  constructor(private readonly video: HTMLVideoElement) {
    this.usingVideoFrameCallback =
      typeof video.requestVideoFrameCallback === 'function';
  }

  subscribe(listener: VideoFrameListener): VideoFrameSubscription {
    this.listeners.add(listener);
    if (this.listeners.size === 1) this.start();

    let active = true;
    return {
      usingVideoFrameCallback: this.usingVideoFrameCallback,
      unsubscribe: () => {
        if (!active) return;
        active = false;
        this.listeners.delete(listener);
        if (this.listeners.size === 0) {
          this.stop();
          sources.delete(this.video);
        }
      },
    };
  }

  private start(): void {
    this.lastVideoTime = -1;
    if (this.usingVideoFrameCallback) {
      this.videoFrameCallbackId = this.video.requestVideoFrameCallback(
        this.processVideoFrame,
      );
    } else {
      this.animationFrameId = requestAnimationFrame(this.processFallbackFrame);
    }
  }

  private readonly processVideoFrame: VideoFrameRequestCallback = (
    now,
    metadata,
  ) => {
    this.videoFrameCallbackId = null;
    if (this.listeners.size === 0) return;

    this.lastVideoTime = metadata.mediaTime;
    this.notifyListeners(now, metadata.mediaTime);
    this.videoFrameCallbackId = this.video.requestVideoFrameCallback(
      this.processVideoFrame,
    );
  };

  private readonly processFallbackFrame = (now: number) => {
    this.animationFrameId = null;
    if (this.listeners.size === 0) return;

    const mediaTime = this.video.currentTime;
    if (mediaTime !== this.lastVideoTime) {
      this.lastVideoTime = mediaTime;
      this.notifyListeners(now, mediaTime);
    }
    this.animationFrameId = requestAnimationFrame(this.processFallbackFrame);
  };

  private notifyListeners(now: number, mediaTime: number): void {
    let sharedFrame: VideoFrame | null = null;
    const capture: VideoFrameCapture =
      typeof VideoFrame === 'function'
        ? () => {
            sharedFrame ??= new VideoFrame(this.video, {
              timestamp: Math.max(0, Math.round(now * 1000)),
            });
            // Transferring a VideoFrame relinquishes ownership. Give each due
            // tracker a lightweight clone backed by one camera-frame resource.
            return new VideoFrame(sharedFrame);
          }
        : () => createImageBitmap(this.video);

    try {
      this.listeners.forEach((listener) =>
        listener(now, mediaTime, capture),
      );
    } finally {
      const frameToClose = sharedFrame as VideoFrame | null;
      frameToClose?.close();
    }
  }

  private stop(): void {
    if (this.videoFrameCallbackId !== null) {
      this.video.cancelVideoFrameCallback(this.videoFrameCallbackId);
      this.videoFrameCallbackId = null;
    }
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
  }
}

export function subscribeToVideoFrames(
  video: HTMLVideoElement,
  listener: VideoFrameListener,
): VideoFrameSubscription {
  let source = sources.get(video);
  if (!source) {
    source = new SharedVideoFrameSource(video);
    sources.set(video, source);
  }
  return source.subscribe(listener);
}
