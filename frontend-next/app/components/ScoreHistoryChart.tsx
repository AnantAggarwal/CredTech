'use client';

// app/components/ScoreHistoryChart.tsx
import { useEffect, useRef } from 'react';
import type { HistoryPoint } from '@/app/types';
import styles from './ScoreHistoryChart.module.css';

interface ScoreHistoryChartProps {
  history: HistoryPoint[];
}

export default function ScoreHistoryChart({ history }: ScoreHistoryChartProps) {
  const pathRef = useRef<SVGPathElement | null>(null);

  if (!history || history.length < 2) {
    return (
      <div className={styles.empty}>
        <span>Insufficient history data</span>
      </div>
    );
  }

  const W = 600;
  const H = 160;
  const PAD = { top: 16, right: 16, bottom: 32, left: 40 };
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;

  const scores = history.map((h) => h.nexscore);
  const minScore = Math.max(0, Math.min(...scores) - 5);
  const maxScore = Math.min(100, Math.max(...scores) + 5);
  const range = maxScore - minScore || 1;

  const toX = (i: number) => PAD.left + (i / (history.length - 1)) * innerW;
  const toY = (s: number) => PAD.top + innerH - ((s - minScore) / range) * innerH;

  // Build SVG path
  const points = history.map((h, i) => ({ x: toX(i), y: toY(h.nexscore) }));
  const d = points
    .map((p, i) => (i === 0 ? `M ${p.x} ${p.y}` : `L ${p.x} ${p.y}`))
    .join(' ');

  // Area fill path
  const areaD = `${d} L ${points[points.length - 1].x} ${H - PAD.bottom} L ${points[0].x} ${H - PAD.bottom} Z`;

  // Y-axis labels
  const yTicks = 4;
  const yLabels = Array.from({ length: yTicks + 1 }, (_, i) =>
    Math.round(minScore + (range / yTicks) * i)
  );

  // X-axis labels (show first, middle, last)
  const xIndices = [0, Math.floor(history.length / 2), history.length - 1];

  return (
    <div className={styles.chartWrapper}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        width="100%"
        className={styles.svg}
        role="img"
        aria-label="NexScore history chart"
        preserveAspectRatio="xMidYMid meet"
      >
        <defs>
          <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="rgba(0, 255, 157, 0.2)" />
            <stop offset="100%" stopColor="rgba(0, 255, 157, 0)" />
          </linearGradient>
          <linearGradient id="lineGrad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="rgba(0, 255, 157, 0.4)" />
            <stop offset="100%" stopColor="rgba(0, 255, 157, 1)" />
          </linearGradient>
        </defs>

        {/* Grid lines */}
        {yLabels.map((label) => {
          const y = toY(label);
          return (
            <g key={label}>
              <line
                x1={PAD.left}
                y1={y}
                x2={W - PAD.right}
                y2={y}
                stroke="rgba(255,255,255,0.04)"
                strokeWidth={1}
              />
              <text
                x={PAD.left - 6}
                y={y}
                textAnchor="end"
                dominantBaseline="middle"
                fill="rgba(100, 116, 139, 0.8)"
                fontSize="9"
                fontFamily="JetBrains Mono, monospace"
              >
                {label}
              </text>
            </g>
          );
        })}

        {/* X-axis labels */}
        {xIndices.map((idx) => (
          <text
            key={idx}
            x={toX(idx)}
            y={H - PAD.bottom + 14}
            textAnchor="middle"
            fill="rgba(100, 116, 139, 0.8)"
            fontSize="8"
            fontFamily="JetBrains Mono, monospace"
          >
            {history[idx]?.date?.slice(0, 10) ?? ''}
          </text>
        ))}

        {/* Area fill */}
        <path d={areaD} fill="url(#areaGrad)" />

        {/* Line */}
        <path
          ref={pathRef}
          d={d}
          fill="none"
          stroke="url(#lineGrad)"
          strokeWidth={2}
          strokeLinejoin="round"
          strokeLinecap="round"
          className={styles.line}
        />

        {/* Data points */}
        {points.map((p, i) => (
          <circle
            key={i}
            cx={p.x}
            cy={p.y}
            r={3}
            fill="var(--bg-surface)"
            stroke="#00ff9d"
            strokeWidth={1.5}
            className={styles.dot}
          >
            <title>{`${history[i].date}: ${history[i].nexscore.toFixed(1)}`}</title>
          </circle>
        ))}
      </svg>
    </div>
  );
}
