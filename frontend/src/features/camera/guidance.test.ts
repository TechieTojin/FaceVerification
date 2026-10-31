import { describe, expect, it } from 'vitest';
import { computeGuidance, shouldSendImmediately, yawFromLandmarks, type TrackedFace } from './guidance';

const cfg = { minWidth: 0.2, maxWidth: 0.75, yawThreshold: 18, blinkThreshold: 0.5 };
const stats = { brightness: 0.5, sharpness: 0.8 };
const face = (o: Partial<TrackedFace> = {}): TrackedFace => ({
  box: { x: 0.32, y: 0.25, w: 0.36, h: 0.47 }, yaw: 0, pitch: 0, closureBoth: 0, closureAny: 0, points: [], ...o,
});

describe('computeGuidance', () => {
  it('blocks when no face or several faces are visible', () => {
    expect(computeGuidance([], stats, 'LOOK_STRAIGHT', cfg)).toMatchObject({ sendable: false, issues: ['NO_FACE'] });
    const g = computeGuidance([face(), face()], stats, 'LOOK_STRAIGHT', cfg);
    expect(g.sendable).toBe(false);
    expect(g.headline).toBe('Only one person should be visible');
  });

  it('asks the user to move closer / back', () => {
    expect(computeGuidance([face({ box: { x: 0.45, y: 0.4, w: 0.1, h: 0.13 } })], stats, 'LOOK_STRAIGHT', cfg).headline).toBe('Move closer');
    expect(computeGuidance([face({ box: { x: 0.05, y: 0.05, w: 0.85, h: 0.9 } })], stats, 'LOOK_STRAIGHT', cfg).headline)
      .toBe('Move back slightly');
  });

  it('flags lighting and blur', () => {
    expect(computeGuidance([face()], { brightness: 0.1, sharpness: 0.8 }, 'LOOK_STRAIGHT', cfg).issues).toContain('TOO_DARK');
    expect(computeGuidance([face()], { brightness: 0.95, sharpness: 0.8 }, 'LOOK_STRAIGHT', cfg).issues).toContain('TOO_BRIGHT');
    const blurry = computeGuidance([face()], { brightness: 0.5, sharpness: 0.05 }, 'LOOK_STRAIGHT', cfg);
    expect(blurry.issues).toContain('BLURRY');
    expect(blurry.sendable).toBe(true);   // blur is advisory; the server makes the quality decision
  });

  it('only requires a frontal pose on straight-ahead steps', () => {
    expect(computeGuidance([face({ yaw: 25 })], stats, 'LOOK_STRAIGHT', cfg)).toMatchObject({ sendable: false, headline: 'Look directly at the camera' });
    const turn = computeGuidance([face({ yaw: 25 })], stats, 'TURN_LEFT', cfg);
    expect(turn.sendable).toBe(true);
    expect(turn.stepProgress).toBe(1);
  });

  it('reports progress toward turn and blink actions', () => {
    expect(computeGuidance([face({ yaw: 9 })], stats, 'TURN_LEFT', cfg).stepProgress).toBeCloseTo(0.5);
    expect(computeGuidance([face({ yaw: 9 })], stats, 'TURN_RIGHT', cfg).stepProgress).toBe(0);
    expect(computeGuidance([face({ closureBoth: 0.25 })], stats, 'BLINK', cfg).stepProgress).toBeCloseTo(0.5);
  });
});

describe('frame timing helpers', () => {
  it('sends immediately only for closed eyes during a blink step', () => {
    expect(shouldSendImmediately(face({ closureBoth: 0.8 }), 'BLINK', cfg)).toBe(true);
    expect(shouldSendImmediately(face({ closureBoth: 0.8 }), 'TURN_LEFT', cfg)).toBe(false);
    expect(shouldSendImmediately(face({ closureBoth: 0.2 }), 'BLINK', cfg)).toBe(false);
    expect(shouldSendImmediately(null, 'BLINK', cfg)).toBe(false);
  });

  it('yaw from landmarks follows the server convention (+ = subject turned to their left)', () => {
    // unmirrored camera: subject turning to their left moves the nose toward image-right
    expect(yawFromLandmarks(0.62, 0.3, 0.7)).toBeGreaterThan(0);
    expect(yawFromLandmarks(0.38, 0.3, 0.7)).toBeLessThan(0);
    expect(yawFromLandmarks(0.5, 0.3, 0.7)).toBe(0);
  });
});
