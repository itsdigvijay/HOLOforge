import type { ManagerStatus } from '../types';

/** Webcam lifecycle boundary. Camera access will be implemented in a later step. */
export class CameraManager {
  private currentStatus: ManagerStatus = 'idle';

  get status(): ManagerStatus {
    return this.currentStatus;
  }

  initialize(): void {
    this.currentStatus = 'ready';
  }

  dispose(): void {
    this.currentStatus = 'idle';
  }
}
