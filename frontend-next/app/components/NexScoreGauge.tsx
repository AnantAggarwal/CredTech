'use client';

// app/components/NexScoreGauge.tsx
import { useEffect, useRef, useState } from 'react';
import type { Grade } from '@/app/types';
import { gradeColor } from './GradeBadge';
import styles from './NexScoreGauge.module.css';

interface NexScoreGaugeProps {
  score: number;
  grade: Grade;
}

const RADIUS = 80;
const STROKE_WIDTH = 12;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;
// We use 270° arc (from 135° to 405°) for the gauge
const ARC_FRACTION = 0.75;
const ARC_LENGTH = CIRCUMFERENCE * ARC_FRACTION;

function useCountUp(target: number, duration = 1400) {
  const [value, setValue] = useState(0);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    let startTime: number | null = null;
    const step = (timestamp: number) => {
      if (!startTime) startTime = timestamp;
      const elapsed = timestamp - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 4);
      setValue(Math.round(target * eased));
      if (progress < 1) {
        rafRef.current = requestAnimationFrame(step);
      }
    };
    rafRef.current = requestAnimationFrame(step);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [target, duration]);

  return value;
}

export default function NexScoreGauge({ score, grade }: NexScoreGaugeProps) {
  const displayScore = useCountUp(Math.round(score));
  const color = gradeColor(grade);

  // Calculate stroke-dashoffset for the filled arc
  const fillFraction = (score / 100) * ARC_FRACTION;
  const fillLength = CIRCUMFERENCE * fillFraction;
  const offset = ARC_LENGTH - fillLength;

  // SVG viewBox size
  const size = (RADIUS + STROKE_WIDTH) * 2 + 4;
  const cx = size / 2;
  const cy = size / 2;

  // Rotation: start at 135° (bottom-left), sweep 270°
  const rotation = 135;

  return (
    <div className={styles.gaugeWrapper}>
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        aria-label={`NexScore: ${Math.round(score)} out of 100`}
        role="img"
        className={styles.svg}
      >
        {/* Background track */}
        <circle
          cx={cx}
          cy={cy}
          r={RADIUS}
          fill="none"
          stroke="rgba(255,255,255,0.05)"
          strokeWidth={STROKE_WIDTH}
          strokeLinecap="round"
          strokeDasharray={`${ARC_LENGTH} ${CIRCUMFERENCE - ARC_LENGTH}`}
          strokeDashoffset={0}
          transform={`rotate(${rotation} ${cx} ${cy})`}
        />

        {/* Glow circle (slightly larger, blurred) */}
        <circle
          cx={cx}
          cy={cy}
          r={RADIUS}
          fill="none"
          stroke={color}
          strokeWidth={STROKE_WIDTH + 6}
          strokeLinecap="round"
          strokeDasharray={`${fillLength} ${CIRCUMFERENCE - fillLength}`}
          strokeDashoffset={0}
          transform={`rotate(${rotation} ${cx} ${cy})`}
          opacity={0.15}
          style={{
            filter: 'blur(4px)',
            transition: 'stroke-dasharray 1.4s cubic-bezier(0.4, 0, 0.2, 1)',
          }}
        />

        {/* Active arc */}
        <circle
          cx={cx}
          cy={cy}
          r={RADIUS}
          fill="none"
          stroke={color}
          strokeWidth={STROKE_WIDTH}
          strokeLinecap="round"
          strokeDasharray={`${ARC_LENGTH} ${CIRCUMFERENCE - ARC_LENGTH}`}
          strokeDashoffset={offset}
          transform={`rotate(${rotation} ${cx} ${cy})`}
          style={{
            transition: 'stroke-dashoffset 1.4s cubic-bezier(0.4, 0, 0.2, 1)',
          }}
        />

        {/* Score text */}
        <text
          x={cx}
          y={cy - 8}
          textAnchor="middle"
          dominantBaseline="middle"
          fill={color}
          fontSize="36"
          fontFamily="JetBrains Mono, monospace"
          fontWeight="700"
          letterSpacing="-2"
        >
          {displayScore}
        </text>

        {/* /100 label */}
        <text
          x={cx}
          y={cy + 24}
          textAnchor="middle"
          dominantBaseline="middle"
          fill="rgba(148, 163, 184, 0.6)"
          fontSize="12"
          fontFamily="JetBrains Mono, monospace"
          fontWeight="400"
        >
          / 100
        </text>

        {/* Min label */}
        <text
          x={cx - RADIUS * 0.76}
          y={cy + RADIUS * 0.76 + 2}
          textAnchor="middle"
          fill="rgba(100, 116, 139, 0.7)"
          fontSize="9"
          fontFamily="JetBrains Mono, monospace"
        >
          0
        </text>

        {/* Max label */}
        <text
          x={cx + RADIUS * 0.76}
          y={cy + RADIUS * 0.76 + 2}
          textAnchor="middle"
          fill="rgba(100, 116, 139, 0.7)"
          fontSize="9"
          fontFamily="JetBrains Mono, monospace"
        >
          100
        </text>
      </svg>

      <div className={styles.gaugeLabel}>
        <span className={styles.gaugeTitle}>NEXSCORE™</span>
      </div>
    </div>
  );
}
