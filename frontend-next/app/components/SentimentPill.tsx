// app/components/SentimentPill.tsx
import styles from './SentimentPill.module.css';

type Sentiment = 'BULLISH' | 'BEARISH' | 'NEUTRAL';

interface SentimentPillProps {
  value: number; // sentiment_display: sentiment_score * 100  (-100 to 100)
  label?: Sentiment;
  size?: 'sm' | 'md';
}

export function getSentimentLabel(value: number): Sentiment {
  if (value >= 15) return 'BULLISH';
  if (value <= -15) return 'BEARISH';
  return 'NEUTRAL';
}

export default function SentimentPill({ value, label, size = 'md' }: SentimentPillProps) {
  const sentiment = label ?? getSentimentLabel(value);
  return (
    <span
      className={`${styles.pill} ${styles[sentiment.toLowerCase()]} ${styles[size]}`}
      aria-label={`Market sentiment: ${sentiment}`}
    >
      <span className={styles.dot} aria-hidden="true" />
      {sentiment}
    </span>
  );
}
