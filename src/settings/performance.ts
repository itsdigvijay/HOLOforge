export type PerformanceMode = 'performance' | 'balanced' | 'quality';
export type OverlayDetail = 'basic' | 'full';

export interface CameraCaptureProfile {
  width: number;
  height: number;
  frameRate: number;
}

export interface PerformanceProfile {
  camera: CameraCaptureProfile;
  handInferenceFps: number;
  overlayPixelRatioCap: 1 | 1.5 | 2;
  overlayDetail: OverlayDetail;
}

export const PERFORMANCE_PROFILES: Record<
  PerformanceMode,
  PerformanceProfile
> = {
  performance: {
    camera: { width: 640, height: 480, frameRate: 60 },
    handInferenceFps: 30,
    overlayPixelRatioCap: 1,
    overlayDetail: 'basic',
  },
  balanced: {
    camera: { width: 1280, height: 720, frameRate: 30 },
    handInferenceFps: 30,
    overlayPixelRatioCap: 1.5,
    overlayDetail: 'full',
  },
  quality: {
    camera: { width: 1280, height: 720, frameRate: 60 },
    handInferenceFps: 45,
    overlayPixelRatioCap: 2,
    overlayDetail: 'full',
  },
};

export const DEFAULT_PERFORMANCE_MODE: PerformanceMode = 'performance';
