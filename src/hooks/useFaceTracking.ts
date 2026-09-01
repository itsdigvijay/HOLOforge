import { useEffect, useRef, useState, type RefObject } from 'react';
import {
  FaceTrackingManager,
  type FaceTrackingStatus,
} from '../tracking';

export function useFaceTracking(
  videoRef: RefObject<HTMLVideoElement | null>,
  cameraReady: boolean,
) {
  const managerRef = useRef<FaceTrackingManager | null>(null);
  const lifecycleRef = useRef(0);
  const [status, setStatus] = useState<FaceTrackingStatus>('idle');

  if (!managerRef.current) managerRef.current = new FaceTrackingManager();

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
        if (lifecycleRef.current === lifecycle) manager.dispose();
      });
    };
  }, []);

  return { manager: managerRef.current, status };
}
