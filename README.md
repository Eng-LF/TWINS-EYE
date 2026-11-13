# TWINS EYE

TWINS EYE is a browser-based football movement-analysis prototype for reviewing a selected player's visible pose and movement measurements around an event.

## Run

This app uses the workspace's managed web workflow. To check the app's TypeScript:

```sh
pnpm --filter @workspace/twins-eye run typecheck
```

The app loads a pretrained MediaPipe Pose Landmarker Lite model and its WebAssembly runtime from their public hosting URLs when analysis is started. A network connection is needed for the first model load.

## Use with a match video

1. Open the TWINS EYE app and choose an MP4 file.
2. Start analysis and allow the pose model to load.
3. Play the video. Inference runs on sampled frames while playback is active.
4. Select a detected player by choosing their visible tracking box.
5. Set the before-event and after-event intervals, then review the measured comparisons and any prototype-rule alerts.

The video is processed in the browser and is not uploaded to an application server. The model and WebAssembly files are downloaded from the URLs configured in `src/lib/pose-engine.ts`.

## Measurements and limitations

- Pose landmarks and joint angles are derived from the uploaded frames. A landmark below the visibility threshold is omitted rather than estimated.
- Player identifiers are short-lived visual tracks, not verified identities. Occlusion, overlap, camera cuts, and missed detections can interrupt or change a track.
- Speed is reported in pixels per second. It is not calibrated to real-world distance or speed.
- Before/after values use the median of at least three sufficiently confident measurements per metric in each interval. Joint-angle rules use an absolute median shift; symmetry rules flag an increase in the left/right knee-angle gap; speed rules use an absolute percent change in pixel-based speed.
- Deviation thresholds are configurable prototype rules, not validated medical standards. A threshold crossing is moderate; a change at least 1.75 times the configured threshold is high. An alert indicates a movement change for review; it does not confirm an injury or replace medical assessment.
- Small, distant, occluded, or poorly lit players may not yield enough reliable landmarks. The app should report unavailable measurements rather than fill gaps with estimates.

## Manual validation

Use an MP4 with at least one clearly visible player. Confirm that the model reports a load or inference error if it cannot initialize; that detected boxes and skeleton landmarks appear only when the model returns them; that selecting a player limits the measurements to that track; and that interval comparisons remain unavailable until both windows contain enough valid samples. Replay a clip with clear movement changes to exercise the rule-based alert. The result is footage-dependent and is not a clinical test.

## Python requirements

The current prototype performs inference in the browser and has no Python runtime dependencies. `requirements.txt` is included to make that explicit; it does not install unused Python packages.
