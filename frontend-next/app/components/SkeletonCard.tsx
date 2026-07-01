// app/components/SkeletonCard.tsx
import styles from './SkeletonCard.module.css';

export default function SkeletonCard() {
  return (
    <div className={styles.card} aria-hidden="true">
      <div className={styles.header}>
        <div className={styles.col}>
          <div className={`${styles.skel} ${styles.ticker}`} />
          <div className={`${styles.skel} ${styles.name}`} />
        </div>
        <div className={`${styles.skel} ${styles.badge}`} />
      </div>
      <div className={styles.scoreSection}>
        <div className={`${styles.skel} ${styles.score}`} />
        <div className={`${styles.skel} ${styles.label}`} />
        <div className={`${styles.skel} ${styles.pill}`} />
      </div>
      <div className={styles.metrics}>
        <div className={styles.metric}>
          <div className={`${styles.skel} ${styles.metricLabel}`} />
          <div className={`${styles.skel} ${styles.bar}`} />
        </div>
        <div className={styles.metric}>
          <div className={`${styles.skel} ${styles.metricLabel}`} />
          <div className={`${styles.skel} ${styles.bar}`} />
        </div>
      </div>
    </div>
  );
}
