// app/components/GradeBadge.tsx
import type { Grade } from '@/app/types';
import styles from './GradeBadge.module.css';

interface GradeBadgeProps {
  grade: Grade;
  size?: 'sm' | 'md' | 'lg';
  showDescription?: boolean;
}

const GRADE_DESCRIPTIONS: Record<Grade, string> = {
  AAA: 'Exceptional Creditworthiness',
  AA: 'Very Strong Creditworthiness',
  A: 'Strong Creditworthiness',
  BBB: 'Adequate Creditworthiness',
  BB: 'Speculative Grade',
  B: 'Highly Speculative',
  CCC: 'Substantial Credit Risk',
};

export function gradeColor(grade: Grade): string {
  const colors: Record<Grade, string> = {
    AAA: '#00ff9d',
    AA: '#00ff9d',
    A: '#4ade80',
    BBB: '#facc15',
    BB: '#fb923c',
    B: '#f87171',
    CCC: '#ef4444',
  };
  return colors[grade] ?? '#64748b';
}

export default function GradeBadge({ grade, size = 'md', showDescription = false }: GradeBadgeProps) {
  const color = gradeColor(grade);
  return (
    <span
      className={`${styles.badge} ${styles[size]}`}
      style={{
        color,
        borderColor: `${color}40`,
        backgroundColor: `${color}12`,
      }}
      aria-label={`Credit grade: ${grade} — ${GRADE_DESCRIPTIONS[grade]}`}
    >
      <span className={styles.gradeText}>{grade}</span>
      {showDescription && (
        <span className={styles.description}>{GRADE_DESCRIPTIONS[grade]}</span>
      )}
    </span>
  );
}

export { GRADE_DESCRIPTIONS };
