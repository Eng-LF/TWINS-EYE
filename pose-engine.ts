import { FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";
import type {
  ComparisonMetric,
  DeviationAssessment,
  DeviationThresholds,
  IntervalComparison,
  IntervalSelection,
  MovementSample,
  PlayerTrack,
  PoseAnalyzer,
  PoseFrameResult,
  PoseLandmark,
} from "./pose-types";

const TASKS_VISION_VERSION = "1.2.2";
const WASM_PATH = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}/wasm`;
const MODEL_PATH =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

const MIN_VISIBILITY = 0.45;
const MEASUREMENT_VISIBILITY = 0.58;
const MIN_SAMPLES_PER_INTERVAL = 3;

const LANDMARK = {
  leftShoulder: 11,
  rightShoulder: 12,
  leftHip: 23,
  rightHip: 24,
  leftKnee: 25,
  rightKnee: 26,
  leftAnkle: 27,
  rightAnkle: 28,
} as const;

export const POSE_CONNECTIONS: ReadonlyArray<readonly [number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 7],
  [0, 4], [4, 5], [5, 6], [6, 8],
  [9, 10],
  [11, 12],
  [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
  [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
  [11, 23], [12, 24], [23, 24],
  [23, 25], [25, 27], [27, 29], [29, 31], [27, 31],
  [24, 26], [26, 28], [28, 30], [30, 32], [28, 32],
];

type Center = { x: number; y: number };
type TrackState = {
  id: string;
  center: Center;
  lastSeenSec: number;
  previousCenter?: Center;
  previousTimeSec?: number;
};
type Candidate = {
  landmarks: PoseLandmark[];
  bounds: PlayerTrack["bounds"];
  center: Center;
  confidence: number;
};

class PlayerTracker {
  private tracks = new Map<string, TrackState>();
  private nextId = 1;

  reset() {
    this.tracks.clear();
  }

  update(candidates: Candidate[], timeSec: number, width: number, height: number) {
    for (const [id, track] of this.tracks) {
      if (timeSec - track.lastSeenSec > 2) this.tracks.delete(id);
    }

    const pairs: Array<{ candidateIndex: number; id: string; distance: number }> = [];
    candidates.forEach((candidate, candidateIndex) => {
      for (const track of this.tracks.values()) {
        const elapsed = timeSec - track.lastSeenSec;
        if (elapsed < 0 || elapsed > 1.25) continue;
        const distance = Math.hypot(
          candidate.center.x - track.center.x,
          candidate.center.y - track.center.y,
        );
        const matchLimit = Math.min(0.22, 0.055 + elapsed * 0.9);
        if (distance <= matchLimit) pairs.push({ candidateIndex, id: track.id, distance });
      }
    });

    pairs.sort((a, b) => a.distance - b.distance);
    const assignedCandidates = new Set<number>();
    const assignedTracks = new Set<string>();
    const assignment = new Map<number, TrackState>();

    for (const pair of pairs) {
      if (assignedCandidates.has(pair.candidateIndex) || assignedTracks.has(pair.id)) continue;
      const track = this.tracks.get(pair.id);
      if (!track) continue;
      assignedCandidates.add(pair.candidateIndex);
      assignedTracks.add(pair.id);
      assignment.set(pair.candidateIndex, track);
    }

    const playerTracks: PlayerTrack[] = [];
    const samples: MovementSample[] = [];

    candidates.forEach((candidate, index) => {
      let state = assignment.get(index);
      if (!state) {
        state = {
          id: `P${String(this.nextId++).padStart(2, "0")}`,
          center: candidate.center,
          lastSeenSec: timeSec,
        };
        this.tracks.set(state.id, state);
      }

      const deltaTime =
        state.previousTimeSec === undefined ? undefined : timeSec - state.previousTimeSec;
      const speedPxPerSec =
        state.previousCenter &&
        deltaTime !== undefined &&
        deltaTime >= 0.05 &&
        deltaTime <= 1.25
          ? (Math.hypot(
              (candidate.center.x - state.previousCenter.x) * width,
              (candidate.center.y - state.previousCenter.y) * height,
            ) /
            deltaTime)
          : undefined;

      const updatedTrack: TrackState = {
        id: state.id,
        center: candidate.center,
        lastSeenSec: timeSec,
        previousCenter: candidate.center,
        previousTimeSec: timeSec,
      };
      this.tracks.set(updatedTrack.id, updatedTrack);

      playerTracks.push({
        id: updatedTrack.id,
        bounds: candidate.bounds,
        landmarks: candidate.landmarks,
        confidence: candidate.confidence,
      });
      samples.push(
        createMovementSample(
          updatedTrack.id,
          candidate.landmarks,
          timeSec,
          candidate.confidence,
          speedPxPerSec,
        ),
      );
    });

    return { tracks: playerTracks, samples };
  }
}

function visibility(point: PoseLandmark | undefined) {
  return point?.visibility ?? 0;
}

function visible(point: PoseLandmark | undefined, threshold = MEASUREMENT_VISIBILITY) {
  return point && visibility(point) >= threshold ? point : undefined;
}

function centerOf(landmarks: PoseLandmark[]): Center | undefined {
  const hips = [landmarks[LANDMARK.leftHip], landmarks[LANDMARK.rightHip]]
    .map((point) => visible(point, MIN_VISIBILITY))
    .filter((point): point is PoseLandmark => Boolean(point));
  if (hips.length === 2) {
    return {
      x: (hips[0].x + hips[1].x) / 2,
      y: (hips[0].y + hips[1].y) / 2,
    };
  }

  const shoulders = [landmarks[LANDMARK.leftShoulder], landmarks[LANDMARK.rightShoulder]]
    .map((point) => visible(point, MIN_VISIBILITY))
    .filter((point): point is PoseLandmark => Boolean(point));
  if (shoulders.length === 2) {
    return {
      x: (shoulders[0].x + shoulders[1].x) / 2,
      y: (shoulders[0].y + shoulders[1].y) / 2,
    };
  }

  return undefined;
}

function makeCandidate(points: PoseLandmark[]): Candidate | undefined {
  const center = centerOf(points);
  if (!center) return undefined;

  const reliablePoints = points.filter((point) => visibility(point) >= MIN_VISIBILITY);
  if (reliablePoints.length < 6) return undefined;

  const xs = reliablePoints.map((point) => point.x);
  const ys = reliablePoints.map((point) => point.y);
  const minX = Math.max(0, Math.min(...xs));
  const minY = Math.max(0, Math.min(...ys));
  const maxX = Math.min(1, Math.max(...xs));
  const maxY = Math.min(1, Math.max(...ys));
  if (maxX - minX < 0.008 || maxY - minY < 0.015) return undefined;

  const averageConfidence =
    reliablePoints.reduce((sum, point) => sum + visibility(point), 0) /
    reliablePoints.length;

  return {
    landmarks: points,
    center,
    confidence: averageConfidence,
    bounds: {
      x: minX,
      y: minY,
      width: maxX - minX,
      height: maxY - minY,
    },
  };
}

function angleDegrees(a: PoseLandmark | undefined, b: PoseLandmark | undefined, c: PoseLandmark | undefined) {
  const pointA = visible(a);
  const pointB = visible(b);
  const pointC = visible(c);
  if (!pointA || !pointB || !pointC) return undefined;

  const abx = pointA.x - pointB.x;
  const aby = pointA.y - pointB.y;
  const cbx = pointC.x - pointB.x;
  const cby = pointC.y - pointB.y;
  const abLength = Math.hypot(abx, aby);
  const cbLength = Math.hypot(cbx, cby);
  if (abLength < 0.002 || cbLength < 0.002) return undefined;

  const cosine = Math.max(-1, Math.min(1, (abx * cbx + aby * cby) / (abLength * cbLength)));
  return (Math.acos(cosine) * 180) / Math.PI;
}

function createMovementSample(
  trackId: string,
  landmarks: PoseLandmark[],
  timestampSec: number,
  confidence: number,
  speedPxPerSec?: number,
): MovementSample {
  const leftShoulder = landmarks[LANDMARK.leftShoulder];
  const rightShoulder = landmarks[LANDMARK.rightShoulder];
  const leftHip = landmarks[LANDMARK.leftHip];
  const rightHip = landmarks[LANDMARK.rightHip];
  const leftKnee = landmarks[LANDMARK.leftKnee];
  const rightKnee = landmarks[LANDMARK.rightKnee];
  const leftAnkle = landmarks[LANDMARK.leftAnkle];
  const rightAnkle = landmarks[LANDMARK.rightAnkle];

  const kneeLeftDeg = angleDegrees(leftHip, leftKnee, leftAnkle);
  const kneeRightDeg = angleDegrees(rightHip, rightKnee, rightAnkle);
  const hipLeftDeg = angleDegrees(leftShoulder, leftHip, leftKnee);
  const hipRightDeg = angleDegrees(rightShoulder, rightHip, rightKnee);

  return {
    trackId,
    timestampSec,
    kneeLeftDeg,
    kneeRightDeg,
    hipLeftDeg,
    hipRightDeg,
    symmetryDeg:
      kneeLeftDeg !== undefined && kneeRightDeg !== undefined
        ? Math.abs(kneeLeftDeg - kneeRightDeg)
        : undefined,
    speedPxPerSec:
      speedPxPerSec !== undefined && Number.isFinite(speedPxPerSec)
        ? speedPxPerSec
        : undefined,
    confidence,
  };
}

export async function createPoseAnalyzer(): Promise<PoseAnalyzer> {
  if (typeof window === "undefined" || typeof HTMLVideoElement === "undefined") {
    throw new Error("Pose analysis requires a browser video element.");
  }

  const fileset = await FilesetResolver.forVisionTasks(WASM_PATH);
  const landmarker = await PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_PATH, delegate: "CPU" },
    runningMode: "VIDEO",
    numPoses: 10,
    minPoseDetectionConfidence: 0.35,
    minPosePresenceConfidence: 0.35,
    minTrackingConfidence: 0.35,
    outputSegmentationMasks: false,
  });

  const tracker = new PlayerTracker();
  let lastVideoTimeSec = -1;
  let lastModelTimeMs = 0;
  let closed = false;

  return {
    processFrame(video, timestampSec): PoseFrameResult {
      if (closed) throw new Error("Pose analyzer has been closed.");
      if (!video.videoWidth || !video.videoHeight || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        return { tracks: [], samples: [] };
      }

      if (timestampSec < lastVideoTimeSec - 0.5) tracker.reset();
      lastVideoTimeSec = timestampSec;
      const modelTimestampMs = Math.max(timestampSec * 1000, lastModelTimeMs + 1);
      lastModelTimeMs = modelTimestampMs;

      const result = landmarker.detectForVideo(video, modelTimestampMs);
      const candidates = result.landmarks
        .map((pose) => makeCandidate(pose))
        .filter((pose): pose is Candidate => Boolean(pose));
      return tracker.update(candidates, timestampSec, video.videoWidth, video.videoHeight);
    },
    close() {
      if (closed) return;
      closed = true;
      tracker.reset();
      landmarker.close();
    },
  };
}

function median(values: number[]) {
  if (values.length === 0) return undefined;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle];
}

function valuesInWindow(
  samples: MovementSample[],
  trackId: string,
  start: number,
  end: number,
  key: keyof Pick<
    MovementSample,
    | "kneeLeftDeg"
    | "kneeRightDeg"
    | "hipLeftDeg"
    | "hipRightDeg"
    | "symmetryDeg"
    | "speedPxPerSec"
  >,
) {
  return samples
    .filter(
      (sample) =>
        sample.trackId === trackId &&
        sample.timestampSec >= start &&
        sample.timestampSec <= end &&
        sample.confidence >= 0.5,
    )
    .map((sample) => sample[key])
    .filter((value): value is number => value !== undefined && Number.isFinite(value));
}

export function compareIntervals(
  samples: MovementSample[],
  selection: IntervalSelection,
  trackId: string,
): IntervalComparison {
  const metricSpecs = [
    { key: "kneeLeftDeg", label: "Left knee angle", unit: "°" },
    { key: "kneeRightDeg", label: "Right knee angle", unit: "°" },
    { key: "hipLeftDeg", label: "Left hip angle", unit: "°" },
    { key: "hipRightDeg", label: "Right hip angle", unit: "°" },
    { key: "symmetryDeg", label: "Knee angle asymmetry", unit: "°" },
    { key: "speedPxPerSec", label: "Relative movement speed", unit: "px/s" },
  ] as const;

  const metrics: ComparisonMetric[] = metricSpecs.map((spec) => {
    const beforeValues = valuesInWindow(
      samples,
      trackId,
      selection.before.start,
      selection.before.end,
      spec.key,
    );
    const afterValues = valuesInWindow(
      samples,
      trackId,
      selection.after.start,
      selection.after.end,
      spec.key,
    );
    const before =
      beforeValues.length >= MIN_SAMPLES_PER_INTERVAL ? median(beforeValues) : undefined;
    const after =
      afterValues.length >= MIN_SAMPLES_PER_INTERVAL ? median(afterValues) : undefined;
    return {
      key: spec.key,
      label: spec.label,
      unit: spec.unit,
      before,
      after,
      delta: before !== undefined && after !== undefined ? after - before : undefined,
      sampleCountBefore: beforeValues.length,
      sampleCountAfter: afterValues.length,
    };
  });

  const samplesBefore = samples.filter(
    (sample) =>
      sample.trackId === trackId &&
      sample.timestampSec >= selection.before.start &&
      sample.timestampSec <= selection.before.end &&
      sample.confidence >= 0.5,
  ).length;
  const samplesAfter = samples.filter(
    (sample) =>
      sample.trackId === trackId &&
      sample.timestampSec >= selection.after.start &&
      sample.timestampSec <= selection.after.end &&
      sample.confidence >= 0.5,
  ).length;

  return { metrics, samplesBefore, samplesAfter };
}

export function evaluateDeviation(
  comparison: IntervalComparison,
  thresholds: DeviationThresholds,
): DeviationAssessment {
  const reasons: DeviationAssessment["reasons"] = [];
  const assessable = comparison.metrics.filter(
    (metric) => metric.before !== undefined && metric.after !== undefined,
  );
  if (assessable.length === 0) return { canAssess: false, level: "low", reasons };

  let level: DeviationAssessment["level"] = "low";
  const addReason = (
    metric: ComparisonMetric,
    magnitude: number,
    moderateThreshold: number,
    description: string,
    thresholdUnit = metric.unit,
  ) => {
    if (magnitude < moderateThreshold) return;
    const severe = magnitude >= moderateThreshold * 1.75;
    if (severe) level = "high";
    else if (level === "low") level = "moderate";
    reasons.push({
      metric: metric.label,
      change: description,
      threshold: `prototype threshold ${moderateThreshold.toFixed(0)}${thresholdUnit}`,
    });
  };

  for (const metric of assessable) {
    if (
      (metric.key === "kneeLeftDeg" || metric.key === "kneeRightDeg" ||
        metric.key === "hipLeftDeg" || metric.key === "hipRightDeg") &&
      metric.delta !== undefined
    ) {
      addReason(
        metric,
        Math.abs(metric.delta),
        thresholds.angleDeltaDeg,
        `${metric.delta > 0 ? "+" : ""}${metric.delta.toFixed(1)}${metric.unit}`,
      );
    }

    if (metric.key === "symmetryDeg" && metric.delta !== undefined && metric.delta > 0) {
      addReason(
        metric,
        metric.delta,
        thresholds.symmetryIncreaseDeg,
        `asymmetry increased ${metric.delta.toFixed(1)}${metric.unit}`,
      );
    }

    if (
      metric.key === "speedPxPerSec" &&
      metric.before !== undefined &&
      metric.after !== undefined &&
      metric.before > 0
    ) {
      const percentChange = ((metric.after - metric.before) / metric.before) * 100;
      addReason(
        metric,
        Math.abs(percentChange),
        thresholds.speedChangePercent,
        `${percentChange > 0 ? "+" : ""}${percentChange.toFixed(0)}% (pixel-based)`,
        "%",
      );
    }
  }

  return { canAssess: true, level, reasons };
}
