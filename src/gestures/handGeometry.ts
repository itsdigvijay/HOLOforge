/** A structural point type compatible with MediaPipe and HOLOFORGE landmarks. */
export interface HandPoint {
  x: number;
  y: number;
  z: number;
}

export interface HandVector {
  x: number;
  y: number;
  z: number;
}

export interface HandVelocity extends HandVector {
  /** Magnitude of the velocity vector in normalized image units per second. */
  speed: number;
}

export interface HandRotation {
  /** Palm-normal rotation around the vertical camera axis, in degrees. */
  yaw: number;
  /** Palm-normal tilt around the horizontal camera axis, in degrees. */
  pitch: number;
  /** Image-plane rotation of the wrist-to-middle-knuckle axis, in degrees. */
  roll: number;
}

export type FingerName = 'thumb' | 'index' | 'middle' | 'ring' | 'pinky';
export type GeometryHandedness = 'Left' | 'Right';

export interface FingerStateOptions {
  /** Minimum joint angle considered straight. Defaults to 150 degrees. */
  extendedAngleDegrees?: number;
  /** Maximum joint angle considered bent. Defaults to 125 degrees. */
  curledAngleDegrees?: number;
}

export interface PinchOptions {
  /** Maximum tip separation as a fraction of palm width. Defaults to 0.28. */
  maximumPalmWidthRatio?: number;
}

const WRIST = 0;
const INDEX_MCP = 5;
const MIDDLE_MCP = 9;
const RING_MCP = 13;
const PINKY_MCP = 17;
const LANDMARK_COUNT = 21;
const EPSILON = 1e-6;
const DEFAULT_EXTENDED_ANGLE = 150;
const DEFAULT_CURLED_ANGLE = 125;

const FINGER_JOINTS: Readonly<Record<FingerName, readonly [number, number, number, number]>> = {
  thumb: [1, 2, 3, 4],
  index: [5, 6, 7, 8],
  middle: [9, 10, 11, 12],
  ring: [13, 14, 15, 16],
  pinky: [17, 18, 19, 20],
};

const PALM_INDICES = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP] as const;

function assertHandLandmarks(
  landmarks: readonly HandPoint[],
): asserts landmarks is readonly HandPoint[] {
  if (landmarks.length < LANDMARK_COUNT) {
    throw new RangeError(
      `Expected ${LANDMARK_COUNT} hand landmarks, received ${landmarks.length}.`,
    );
  }
}

function subtract(first: HandPoint, second: HandPoint): HandVector {
  return {
    x: first.x - second.x,
    y: first.y - second.y,
    z: first.z - second.z,
  };
}

function dot(first: HandVector, second: HandVector): number {
  return first.x * second.x + first.y * second.y + first.z * second.z;
}

function cross(first: HandVector, second: HandVector): HandVector {
  return {
    x: first.y * second.z - first.z * second.y,
    y: first.z * second.x - first.x * second.z,
    z: first.x * second.y - first.y * second.x,
  };
}

function magnitude(vector: HandVector): number {
  return Math.hypot(vector.x, vector.y, vector.z);
}

function normalize(vector: HandVector): HandVector {
  const length = magnitude(vector);
  if (length <= EPSILON) return { x: 0, y: 0, z: 0 };
  return {
    x: vector.x / length,
    y: vector.y / length,
    z: vector.z / length,
  };
}

function normalizeDegrees(angle: number): number {
  return ((angle + 180) % 360 + 360) % 360 - 180;
}

function angleAt(
  first: HandPoint,
  vertex: HandPoint,
  third: HandPoint,
): number {
  const incoming = subtract(first, vertex);
  const outgoing = subtract(third, vertex);
  const denominator = magnitude(incoming) * magnitude(outgoing);
  if (denominator <= EPSILON) return 0;
  // Floating point error can make an otherwise valid cosine slightly exceed
  // [-1, 1], which would turn acos() into NaN.
  const cosine = Math.min(1, Math.max(-1, dot(incoming, outgoing) / denominator));
  return (Math.acos(cosine) * 180) / Math.PI;
}

function getPalmWidth(landmarks: readonly HandPoint[]): number {
  return Math.max(
    EPSILON,
    distanceBetweenPoints(landmarks[INDEX_MCP], landmarks[PINKY_MCP]),
  );
}

function getPalmLength(landmarks: readonly HandPoint[]): number {
  return Math.max(
    EPSILON,
    distanceBetweenPoints(landmarks[WRIST], landmarks[MIDDLE_MCP]),
  );
}

/** Euclidean distance in the same 3D coordinate space as the input points. */
export function distanceBetweenPoints(
  first: HandPoint,
  second: HandPoint,
): number {
  return Math.hypot(
    first.x - second.x,
    first.y - second.y,
    first.z - second.z,
  );
}

/**
 * Returns the primary flexion-joint angle in degrees. A straight finger is
 * close to 180 degrees; the value falls toward 90 degrees as it curls.
 */
export function getFingerAngle(
  landmarks: readonly HandPoint[],
  finger: FingerName,
): number {
  assertHandLandmarks(landmarks);
  const [base, proximal, intermediate, tip] = FINGER_JOINTS[finger];
  return finger === 'thumb'
    ? angleAt(landmarks[proximal], landmarks[intermediate], landmarks[tip])
    : angleAt(landmarks[base], landmarks[proximal], landmarks[intermediate]);
}

/**
 * Tests extension using both articulated joints, then verifies that the tip is
 * farther from the palm than the intermediate joint. That last comparison is
 * rotation invariant and rejects a straight finger folded back over the palm.
 */
export function isFingerExtended(
  landmarks: readonly HandPoint[],
  finger: FingerName,
  options: FingerStateOptions = {},
): boolean {
  assertHandLandmarks(landmarks);
  const [base, proximal, intermediate, tip] = FINGER_JOINTS[finger];
  const threshold = options.extendedAngleDegrees ?? DEFAULT_EXTENDED_ANGLE;
  const distalAngle = angleAt(
    landmarks[proximal],
    landmarks[intermediate],
    landmarks[tip],
  );
  const proximalAngle = angleAt(
    landmarks[base],
    landmarks[proximal],
    landmarks[intermediate],
  );
  const palmCenter = getPalmCenter(landmarks);
  const tipDistance = distanceBetweenPoints(landmarks[tip], palmCenter);
  const jointDistance = distanceBetweenPoints(
    landmarks[intermediate],
    palmCenter,
  );
  const proximalThreshold = finger === 'thumb' ? threshold - 15 : threshold;
  return (
    distalAngle >= threshold &&
    proximalAngle >= proximalThreshold &&
    tipDistance >= jointDistance * 1.04
  );
}

/**
 * Detects curl from joint flexion rather than screen-space tip direction. The
 * palm-relative fallback catches thumbs tucked across the palm, whose MCP joint
 * naturally bends less than the other fingers.
 */
export function isFingerCurled(
  landmarks: readonly HandPoint[],
  finger: FingerName,
  options: FingerStateOptions = {},
): boolean {
  assertHandLandmarks(landmarks);
  const [, proximal, intermediate, tip] = FINGER_JOINTS[finger];
  const threshold = options.curledAngleDegrees ?? DEFAULT_CURLED_ANGLE;
  const primaryAngle = getFingerAngle(landmarks, finger);
  const distalAngle = angleAt(
    landmarks[proximal],
    landmarks[intermediate],
    landmarks[tip],
  );
  if (primaryAngle <= threshold || distalAngle <= threshold) return true;

  const palmCenter = getPalmCenter(landmarks);
  const palmWidth = getPalmWidth(landmarks);
  const tipToPalm = distanceBetweenPoints(landmarks[tip], palmCenter);
  return finger === 'thumb'
    ? tipToPalm <= palmWidth * 0.55
    : tipToPalm <= palmWidth * 0.7;
}

/** Centroid of the wrist and four finger MCP joints. */
export function getPalmCenter(
  landmarks: readonly HandPoint[],
): HandPoint {
  assertHandLandmarks(landmarks);
  let x = 0;
  let y = 0;
  let z = 0;
  for (const index of PALM_INDICES) {
    x += landmarks[index].x;
    y += landmarks[index].y;
    z += landmarks[index].z;
  }
  const divisor = PALM_INDICES.length;
  return { x: x / divisor, y: y / divisor, z: z / divisor };
}

/**
 * Unit vector perpendicular to the palm plane. Landmark winding reverses for
 * left and right hands, so the left-hand cross product is flipped to keep the
 * normal's meaning consistent across hands.
 */
export function getPalmNormal(
  landmarks: readonly HandPoint[],
  handedness: GeometryHandedness,
): HandVector {
  assertHandLandmarks(landmarks);
  const towardIndex = subtract(landmarks[INDEX_MCP], landmarks[WRIST]);
  const towardPinky = subtract(landmarks[PINKY_MCP], landmarks[WRIST]);
  const normal = normalize(cross(towardIndex, towardPinky));
  const sign = handedness === 'Left' ? -1 : 1;
  return { x: normal.x * sign, y: normal.y * sign, z: normal.z * sign };
}

/** Unit vector from the wrist toward the middle-finger knuckle. */
export function getPalmDirection(
  landmarks: readonly HandPoint[],
): HandVector {
  assertHandLandmarks(landmarks);
  return normalize(subtract(landmarks[MIDDLE_MCP], landmarks[WRIST]));
}

/** Palm-center displacement between two hand frames. */
export function getHandMovementDistance(
  currentLandmarks: readonly HandPoint[],
  previousLandmarks: readonly HandPoint[],
): number {
  return distanceBetweenPoints(
    getPalmCenter(currentLandmarks),
    getPalmCenter(previousLandmarks),
  );
}

/**
 * Palm-center velocity in normalized units per second. Supplying elapsed time
 * explicitly keeps this function deterministic and straightforward to test.
 */
export function getHandVelocity(
  currentLandmarks: readonly HandPoint[],
  previousLandmarks: readonly HandPoint[],
  elapsedMilliseconds: number,
): HandVelocity {
  if (!Number.isFinite(elapsedMilliseconds) || elapsedMilliseconds <= 0) {
    throw new RangeError('elapsedMilliseconds must be greater than zero.');
  }
  const currentCenter = getPalmCenter(currentLandmarks);
  const previousCenter = getPalmCenter(previousLandmarks);
  const inverseSeconds = 1000 / elapsedMilliseconds;
  const x = (currentCenter.x - previousCenter.x) * inverseSeconds;
  const y = (currentCenter.y - previousCenter.y) * inverseSeconds;
  const z = (currentCenter.z - previousCenter.z) * inverseSeconds;
  return { x, y, z, speed: Math.hypot(x, y, z) };
}

/**
 * Coarse Euler orientation derived from the palm normal and longitudinal axis.
 * These angles describe camera-relative hand orientation, not skeletal joint
 * rotation, which makes them appropriate for effect alignment.
 */
export function getHandRotation(
  landmarks: readonly HandPoint[],
  handedness: GeometryHandedness,
): HandRotation {
  assertHandLandmarks(landmarks);
  const normal = getPalmNormal(landmarks, handedness);
  const yaw = normalizeDegrees(
    (Math.atan2(normal.x, normal.z) * 180) / Math.PI,
  );
  const pitch = normalizeDegrees(
    (Math.atan2(normal.y, Math.hypot(normal.x, normal.z)) * 180) / Math.PI,
  );
  const palmDirection = getPalmDirection(landmarks);
  // Screen y increases downward, so an upright hand points toward -y and has
  // zero roll. atan2(x, -y) then rotates naturally with the image.
  const roll = normalizeDegrees(
    (Math.atan2(palmDirection.x, -palmDirection.y) * 180) / Math.PI,
  );
  return { yaw, pitch, roll };
}

/** Distance between the two palm centroids. */
export function distanceBetweenHands(
  firstHand: readonly HandPoint[],
  secondHand: readonly HandPoint[],
): number {
  return distanceBetweenPoints(
    getPalmCenter(firstHand),
    getPalmCenter(secondHand),
  );
}

/** Thumb/index pinch using a palm-width-normalized separation threshold. */
export function isPinching(
  landmarks: readonly HandPoint[],
  options: PinchOptions = {},
): boolean {
  assertHandLandmarks(landmarks);
  const maximumRatio = options.maximumPalmWidthRatio ?? 0.28;
  const thumbTip = landmarks[FINGER_JOINTS.thumb[3]];
  const indexTip = landmarks[FINGER_JOINTS.index[3]];
  return (
    distanceBetweenPoints(thumbTip, indexTip) / getPalmWidth(landmarks) <=
    maximumRatio
  );
}

/** True when every finger is geometrically extended. */
export function isOpenPalm(landmarks: readonly HandPoint[]): boolean {
  return (
    isFingerExtended(landmarks, 'thumb') &&
    isFingerExtended(landmarks, 'index') &&
    isFingerExtended(landmarks, 'middle') &&
    isFingerExtended(landmarks, 'ring') &&
    isFingerExtended(landmarks, 'pinky')
  );
}

/**
 * Four curled fingers plus a thumb that is curled or held near the palm. All
 * thresholds scale with the hand itself, so this works at different distances.
 */
export function isFist(landmarks: readonly HandPoint[]): boolean {
  assertHandLandmarks(landmarks);
  if (
    !isFingerCurled(landmarks, 'index') ||
    !isFingerCurled(landmarks, 'middle') ||
    !isFingerCurled(landmarks, 'ring') ||
    !isFingerCurled(landmarks, 'pinky')
  ) {
    return false;
  }
  const thumbTip = landmarks[FINGER_JOINTS.thumb[3]];
  return (
    isFingerCurled(landmarks, 'thumb') ||
    distanceBetweenPoints(thumbTip, getPalmCenter(landmarks)) <=
      getPalmLength(landmarks) * 0.9
  );
}

/** Extended index with the remaining non-thumb fingers curled. */
export function isPointing(landmarks: readonly HandPoint[]): boolean {
  return (
    isFingerExtended(landmarks, 'index') &&
    isFingerCurled(landmarks, 'middle') &&
    isFingerCurled(landmarks, 'ring') &&
    isFingerCurled(landmarks, 'pinky') &&
    !isPinching(landmarks)
  );
}
