'use client';

// app/company/[ticker]/page.tsx — Company Deep-Dive
import { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import type { NexScoreDetail, NewsItem, HistoryPoint } from '@/app/types';
import NexScoreGauge from '@/app/components/NexScoreGauge';
import GradeBadge, { GRADE_DESCRIPTIONS, gradeColor } from '@/app/components/GradeBadge';
import SentimentPill, { getSentimentLabel } from '@/app/components/SentimentPill';
import ScoreHistoryChart from '@/app/components/ScoreHistoryChart';
import styles from './page.module.css';

function NewsCard({ item }: { item: NewsItem }) {
  const sentimentColors = {
    BULLISH: { color: 'var(--bullish)', bg: 'rgba(0,255,157,0.08)', border: 'rgba(0,255,157,0.25)' },
    BEARISH: { color: 'var(--bearish)', bg: 'rgba(248,113,113,0.08)', border: 'rgba(248,113,113,0.25)' },
    NEUTRAL: { color: 'var(--neutral)', bg: 'rgba(250,204,21,0.08)', border: 'rgba(250,204,21,0.25)' },
  };
  const colors = sentimentColors[item.sentiment_label] ?? sentimentColors.NEUTRAL;
  const confidence = Math.round(Math.abs(item.sentiment_score) * 100);

  return (
    <a
      href={item.url || '#'}
      target="_blank"
      rel="noopener noreferrer"
      className={styles.newsCard}
      aria-label={`${item.title} — ${item.sentiment_label} sentiment`}
    >
      <div className={styles.newsTop}>
        <span className={styles.newsSource}>{item.source}</span>
        <span
          className={styles.newsBadge}
          style={{ color: colors.color, background: colors.bg, borderColor: colors.border }}
        >
          {item.sentiment_label}
        </span>
      </div>
      <p className={styles.newsTitle}>{item.title}</p>
      <div className={styles.newsBottom}>
        <span className={styles.newsConfidenceLabel}>CONFIDENCE</span>
        <div className={styles.newsConfBar}>
          <div
            className={styles.newsConfFill}
            style={{ width: `${confidence}%`, background: colors.color }}
          />
        </div>
        <span className={styles.newsConfValue} style={{ color: colors.color }}>
          {confidence}%
        </span>
      </div>
    </a>
  );
}

function StatCard({
  label,
  value,
  color,
  description,
}: {
  label: string;
  value: number;
  color: string;
  description: string;
}) {
  const pct = Math.max(0, Math.min(100, value));
  return (
    <div className={styles.statCard}>
      <div className={styles.statHeader}>
        <span className={styles.statLabel}>{label}</span>
        <span className={styles.statValue} style={{ color }}>
          {value.toFixed(1)}
        </span>
      </div>
      <div
        className={styles.statBar}
        role="progressbar"
        aria-valuenow={pct}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <div
          className={styles.statBarFill}
          style={{
            width: `${pct}%`,
            background: `linear-gradient(90deg, ${color}80, ${color})`,
            boxShadow: `0 0 12px ${color}40`,
          }}
        />
      </div>
      <p className={styles.statDesc}>{description}</p>
    </div>
  );
}

export default function CompanyDeepDivePage() {
  const params = useParams();
  const router = useRouter();
  const ticker = (params?.ticker as string)?.toUpperCase() ?? '';

  const [detail, setDetail] = useState<NexScoreDetail | null>(null);
  const [news, setNews] = useState<NewsItem[]>([]);
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!ticker) return;
    let cancelled = false;

    async function load() {
      setLoading(true);
      setError(null);
      try {
        const [detailRes, newsRes, histRes] = await Promise.allSettled([
          fetch(`http://localhost:8000/company/${ticker}/nexscore`, { cache: 'no-store' }),
          fetch(`http://localhost:8000/company/${ticker}/news`, { cache: 'no-store' }),
          fetch(`http://localhost:8000/company/${ticker}/history`, { cache: 'no-store' }),
        ]);

        if (cancelled) return;

        if (detailRes.status === 'fulfilled' && detailRes.value.ok) {
          setDetail(await detailRes.value.json());
        } else {
          setError('Company data unavailable');
        }

        if (newsRes.status === 'fulfilled' && newsRes.value.ok) {
          setNews(await newsRes.value.json());
        }

        if (histRes.status === 'fulfilled' && histRes.value.ok) {
          const data = await histRes.value.json();
          setHistory(data.score_history || []);
        }
      } catch {
        if (!cancelled) setError('Unable to connect to NexusCredit backend');
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => { cancelled = true; };
  }, [ticker]);

  const gradeCol = detail ? gradeColor(detail.grade) : 'var(--accent)';
  const sentimentLabel = detail ? getSentimentLabel(detail.sentiment_display) : 'NEUTRAL';

  return (
    <div className={styles.page}>
      {/* Header */}
      <header className={styles.header}>
        <button
          className={styles.backBtn}
          onClick={() => router.push('/')}
          aria-label="Back to dashboard"
        >
          ← DASHBOARD
        </button>
        <div className={styles.headerCenter}>
          {loading ? (
            <div className={`${styles.skel} ${styles.skelTitle}`} />
          ) : (
            <h1 className={styles.pageTitle}>
              <span className={styles.tickerTitle} style={{ color: gradeCol }}>
                {ticker}
              </span>
              <span className={styles.titleSep}>·</span>
              <span className={styles.titleSub}>CREDIT INTELLIGENCE REPORT</span>
            </h1>
          )}
        </div>
        <div className={styles.headerRight}>
          <a href="/" className={styles.navLink}>Dashboard</a>
          <a href="/compare" className={styles.navLink}>Compare</a>
        </div>
      </header>

      <main className={styles.main}>
        {/* Error state */}
        {error && (
          <div className={styles.errorBanner} role="alert">
            <span>⚠</span>
            <span>{error}</span>
          </div>
        )}

        {/* Top section: Gauge + Grade + Stats */}
        <section className={styles.heroSection} aria-label="Credit score overview">
          {/* Gauge panel */}
          <div className={styles.gaugePanel}>
            {loading ? (
              <div className={`${styles.skel} ${styles.skelGauge}`} />
            ) : detail ? (
              <NexScoreGauge score={detail.nexscore} grade={detail.grade} />
            ) : null}

            {detail && (
              <div className={styles.gradePanelRow}>
                <GradeBadge grade={detail.grade} size="lg" />
                <span className={styles.gradeDesc}>
                  {detail.grade} — {GRADE_DESCRIPTIONS[detail.grade]}
                </span>
              </div>
            )}

            {detail && (
              <SentimentPill value={detail.sentiment_display} label={sentimentLabel} size="md" />
            )}
          </div>

          {/* Stat cards */}
          <div className={styles.statsPanel}>
            {loading ? (
              <>
                <div className={`${styles.skel} ${styles.skelStat}`} />
                <div className={`${styles.skel} ${styles.skelStat}`} />
              </>
            ) : detail ? (
              <>
                <StatCard
                  label="CREDITWORTHINESS"
                  value={detail.credit_display}
                  color={gradeCol}
                  description="Raw credit risk assessment based on financial fundamentals and historical repayment behavior."
                />
                <StatCard
                  label="MARKET SENTIMENT"
                  value={Math.abs(detail.sentiment_display)}
                  color={
                    sentimentLabel === 'BULLISH'
                      ? 'var(--bullish)'
                      : sentimentLabel === 'BEARISH'
                      ? 'var(--bearish)'
                      : 'var(--neutral)'
                  }
                  description={`${sentimentLabel === 'BEARISH' ? 'Negative' : sentimentLabel === 'BULLISH' ? 'Positive' : 'Mixed'} market sentiment derived from news and social signal analysis.`}
                />
              </>
            ) : null}

            {/* AI Analyst Note */}
            {detail?.analyst_note && (
              <div className={styles.analystNote} aria-label="AI Analyst Note">
                <div className={styles.analystHeader}>
                  <span className={styles.analystIcon}>⬡</span>
                  <span className={styles.analystTitle}>AI ANALYST NOTE</span>
                </div>
                <p className={styles.analystText}>{detail.analyst_note}</p>
              </div>
            )}

            {loading && <div className={`${styles.skel} ${styles.skelNote}`} />}
          </div>
        </section>

        {/* News Feed */}
        <section className={styles.section} aria-label="Market intelligence feed">
          <div className={styles.sectionHeader}>
            <h2 className={styles.sectionTitle}>MARKET INTELLIGENCE FEED</h2>
            <span className={styles.sectionBadge}>{news.length} SIGNALS</span>
          </div>

          {loading ? (
            <div className={styles.newsGrid}>
              {Array.from({ length: 5 }).map((_, i) => (
                <div key={i} className={`${styles.skel} ${styles.skelNews}`} />
              ))}
            </div>
          ) : news.length > 0 ? (
            <div className={styles.newsGrid}>
              {news.map((item, i) => (
                <NewsCard key={i} item={item} />
              ))}
            </div>
          ) : (
            <div className={styles.emptyState}>No news signals available</div>
          )}
        </section>

        {/* Score History */}
        <section className={styles.section} aria-label="Score history">
          <div className={styles.sectionHeader}>
            <h2 className={styles.sectionTitle}>NEXSCORE™ HISTORY</h2>
          </div>
          {loading ? (
            <div className={`${styles.skel} ${styles.skelChart}`} />
          ) : (
            <div className={styles.chartContainer}>
              <ScoreHistoryChart history={history} />
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
