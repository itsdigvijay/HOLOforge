import { useEffect, useRef, useState, type RefObject } from 'react';
import {
  HandTrackingManager,
  type HandTrackingStatus,
} from '../tracking';

export function useHandTracking(
  videoRef: RefObject<HTMLVideoElement | null>,
  cameraReady: boolean,
) {
  const managerRef = useRef<HandTrackingManager | null>(null);
  const lifecycleRef = useRef(0);
  const [status, setStatus] = useState<HandTrackingStatus>('idle');

  if (!managerRef.current) {
    managerRef.current = new HandTrackingManager();
  }

  useEffect(() => {
    const manager = managerRef.current;
    if (!manager) return;
    return manager.subscribeStatus(setStatus);
  }, []);

  useEffect(() => {
    const manager = managerRef.current;
    if (!manager) return;

    if (!cameraReady || !videoRef.current) {
      manager.stop();
      return;
    }

    void manager
      .start(videoRef.current)
      .then(() => undefined)
      .catch(() => undefined);
  }, [cameraReady, videoRef]);

  useEffect(() => {
    const manager = managerRef.current;
    if (!manager) return;
    const lifecycle = ++lifecycleRef.current;

    return () => {
      manager.stop();
      queueMicrotask(() => {
        // React Strict Mode immediately remounts effects in development.
        if (lifecycleRef.current === lifecycle) manager.dispose();
      });
    };
  }, []);

  return { manager: managerRef.current, status };
}
