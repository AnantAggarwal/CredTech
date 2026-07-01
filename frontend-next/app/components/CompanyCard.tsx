'use client';

// app/components/CompanyCard.tsx
import { useEffect, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import type { Company } from '@/app/types';
import GradeBadge, { gradeColor } from './GradeBadge';
import SentimentPill, { getSentimentLabel } from './SentimentPill';
import styles from './CompanyCard.module.css';

interface CompanyCardProps {
  company: Company;
  animationDelay?: number;
}

function useCountUp(target: number, duration = 1200, delay = 0) {
  const [value, setValue] = useState(0);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    let startTime: number | null = null;
    const start = 0;

    const timeout = setTimeout(() => {
      const step = (timestamp: number) => {
        if (!startTime) startTime = timestamp;
        const elapsed = timestamp - startTime;
        const progress = Math.min(elapsed / duration, 1);
        // Ease out cubic
        const eased = 1 - Math.pow(1 - progress, 3);
        setValue(Math.round(start + (target - start) * eased));
        if (progress < 1) {
          rafRef.current = requestAnimationFrame(step);
        }
      };
      rafRef.current = requestAnimationFrame(step);
    }, delay);

    return () => {
      clearTimeout(timeout);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [target, duration, delay]);

  return value;
}

export default function CompanyCard({ company, animationDelay = 0 }: CompanyCardProps) {
  const router = useRouter();
  const animatedScore = useCountUp(Math.round(company.nexscore), 1000, animationDelay);
  const [isHovered, setIsHovered] = useState(false);

  const creditPct = Math.max(0, Math.min(100, company.creditworthiness_display));
  const sentimentPct = Math.max(0, Math.min(100, ((company.sentiment_score + 1) / 2) * 100));
  const sentimentLabel = getSentimentLabel(company.sentiment_display);
  const gradeCol = gradeColor(company.grade);

  const handleClick = () => {
    router.push(`/company/${company.ticker}`);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleClick();
    }
  };

  return (
    <article
      className={`${styles.card} ${isHovered ? styles.hovered : ''}`}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      role="button"
      tabIndex={0}
      aria-label={`View ${company.name} credit profile`}
      style={{ animationDelay: `${animationDelay}ms` }}
    >
      {/* Top row: ticker, name, grade */}
      <div className={styles.header}>
        <div className={styles.identity}>
          <span className={styles.ticker}>{company.ticker}</span>
          <span className={styles.name}>{company.name}</span>
        </div>
        <GradeBadge grade={company.grade} size="sm" />
      </div>

      {/* Score display */}
      <div className={styles.scoreSection}>
        <div className={styles.scoreWrapper}>
          <span
            className={styles.scoreNumber}
            style={{ color: gradeCol }}
            aria-label={`NexScore: ${Math.round(company.nexscore)} out of 100`}
          >
            {animatedScore}
          </span>
          <span className={styles.scoreMax}>/ 100</span>
        </div>
        <span className={styles.scoreLabel}>NEXSCORE™</span>
        <SentimentPill value={company.sentiment_display} label={sentimentLabel} size="sm" />
      </div>

      {/* Mini progress bars */}
      <div className={styles.metrics}>
        <div className={styles.metric}>
          <div className={styles.metricHeader}>
            <span className={styles.metricLabel}>CREDIT</span>
            <span className={styles.metricValue}>{creditPct.toFixed(1)}</span>
          </div>
          <div className={styles.bar} role="progressbar" aria-valuenow={creditPct} aria-valuemin={0} aria-valuemax={100}>
            <div
              className={styles.barFill}
              style={{
                width: `${creditPct}%`,
                backgroundColor: gradeCol,
                boxShadow: `0 0 8px ${gradeCol}60`,
              }}
            />
          </div>
        </div>

        <div className={styles.metric}>
          <div className={styles.metricHeader}>
            <span className={styles.metricLabel}>SENTIMENT</span>
            <span
              className={styles.metricValue}
              style={{
                color:
                  sentimentLabel === 'BULLISH'
                    ? 'var(--bullish)'
                    : sentimentLabel === 'BEARISH'
                    ? 'var(--bearish)'
                    : 'var(--neutral)',
              }}
            >
              {company.sentiment_display.toFixed(1)}
            </span>
          </div>
          <div className={styles.bar} role="progressbar" aria-valuenow={sentimentPct} aria-valuemin={0} aria-valuemax={100}>
            <div
              className={styles.barFill}
              style={{
                width: `${sentimentPct}%`,
                backgroundColor:
                  sentimentLabel === 'BULLISH'
                    ? 'var(--bullish)'
                    : sentimentLabel === 'BEARISH'
                    ? 'var(--bearish)'
                    : 'var(--neutral)',
              }}
            />
          </div>
        </div>
      </div>

      {/* Hover arrow */}
      <div className={styles.viewMore} aria-hidden="true">
        VIEW PROFILE →
      </div>
    </article>
  );
}
