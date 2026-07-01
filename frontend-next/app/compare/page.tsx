'use client';

// app/compare/page.tsx — Company Comparison Page
import { useEffect, useState, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import type { Company, Grade } from '@/app/types';
import GradeBadge, { gradeColor } from '@/app/components/GradeBadge';
import { getSentimentLabel } from '@/app/components/SentimentPill';
import styles from './page.module.css';

const COMPARE_COLORS = ['#00ff9d', '#7c3aed', '#fb923c'];

interface RadarDataPoint {
  subject: string;
  fullMark: number;
  [key: string]: number | string;
}

function buildRadarData(selected: Company[]): RadarDataPoint[] {
  const metrics = [
    { key: 'nexscore', label: 'NexScore', scale: 1 },
    { key: 'creditworthiness_display', label: 'Creditworthiness', scale: 1 },
    { key: 'sentiment_normalized', label: 'Sentiment', scale: 1 },
  ];

  return metrics.map(({ key, label }) => {
    const point: RadarDataPoint = { subject: label, fullMark: 100 };
    selected.forEach((company) => {
      const val =
        key === 'sentiment_normalized'
          ? ((company.sentiment_score + 1) / 2) * 100
          : key === 'creditworthiness_display'
          ? company.creditworthiness_display
          : company.nexscore;
      point[company.ticker] = Math.round(val * 10) / 10;
    });
    return point;
  });
}

const GRADE_ORDER: Grade[] = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC'];

function gradeRank(g: Grade): number {
  return GRADE_ORDER.indexOf(g);
}

function CustomTooltip({ active, payload, label }: {
  active?: boolean;
  payload?: Array<{ name: string; value: number; color: string }>;
  label?: string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className={styles.tooltip}>
      <p className={styles.tooltipLabel}>{label}</p>
      {payload.map((entry) => (
        <div key={entry.name} className={styles.tooltipRow}>
          <span className={styles.tooltipDot} style={{ background: entry.color }} />
          <span className={styles.tooltipTicker}>{entry.name}</span>
          <span className={styles.tooltipValue} style={{ color: entry.color }}>
            {entry.value}
          </span>
        </div>
      ))}
    </div>
  );
}

export default function ComparePage() {
  const router = useRouter();
  const [companies, setCompanies] = useState<Company[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedTickers, setSelectedTickers] = useState<string[]>([]);

  const fetchCompanies = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('http://localhost:8000/companies?limit=100', { cache: 'no-store' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: Company[] = await res.json();
      setCompanies(data);
      // Auto-select first 2
      if (data.length >= 2) {
        setSelectedTickers([data[0].ticker, data[1].ticker]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchCompanies();
  }, [fetchCompanies]);

  const toggleTicker = (ticker: string) => {
    setSelectedTickers((prev) => {
      if (prev.includes(ticker)) {
        return prev.filter((t) => t !== ticker);
      }
      if (prev.length >= 3) {
        // Replace oldest
        return [...prev.slice(1), ticker];
      }
      return [...prev, ticker];
    });
  };

  const selectedCompanies = companies.filter((c) => selectedTickers.includes(c.ticker));
  const radarData = buildRadarData(selectedCompanies);

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
          <h1 className={styles.pageTitle}>COMPARE ENTITIES</h1>
        </div>
        <div className={styles.headerRight}>
          <a href="/" className={styles.navLink}>Dashboard</a>
        </div>
      </header>

      <main className={styles.main}>
        {/* Error */}
        {error && (
          <div className={styles.errorBanner} role="alert">
            <span>⚠</span>
            <span>{error}</span>
          </div>
        )}

        {/* Company selector */}
        <section className={styles.selectorSection} aria-label="Select companies to compare">
          <div className={styles.sectionHeader}>
            <h2 className={styles.sectionTitle}>SELECT ENTITIES (max 3)</h2>
            <span className={styles.sectionBadge}>{selectedTickers.length} / 3 SELECTED</span>
          </div>

          {loading ? (
            <div className={styles.selectorGrid}>
              {Array.from({ length: 6 }).map((_, i) => (
                <div key={i} className={`${styles.skel} ${styles.skelChip}`} />
              ))}
            </div>
          ) : (
            <div className={styles.selectorGrid} role="group" aria-label="Company selection chips">
              {companies.map((company, idx) => {
                const isSelected = selectedTickers.includes(company.ticker);
                const selIdx = selectedTickers.indexOf(company.ticker);
                const color = isSelected ? COMPARE_COLORS[selIdx] : undefined;
                return (
                  <button
                    key={company.ticker}
                    className={`${styles.chip} ${isSelected ? styles.chipSelected : ''}`}
                    onClick={() => toggleTicker(company.ticker)}
                    style={
                      isSelected
                        ? { borderColor: `${color}60`, background: `${color}12`, color }
                        : {}
                    }
                    aria-pressed={isSelected}
                    aria-label={`${isSelected ? 'Deselect' : 'Select'} ${company.name}`}
                  >
                    <span className={styles.chipTicker}>{company.ticker}</span>
                    <GradeBadge grade={company.grade} size="sm" />
                    {isSelected && (
                      <span
                        className={styles.chipNumber}
                        style={{ background: color, color: '#0a0a0f' }}
                      >
                        {selIdx + 1}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          )}
        </section>

        {/* Radar chart */}
        {selectedCompanies.length >= 2 && (
          <section className={styles.radarSection} aria-label="Radar comparison chart">
            <div className={styles.sectionHeader}>
              <h2 className={styles.sectionTitle}>MULTI-DIMENSIONAL ANALYSIS</h2>
            </div>
            <div className={styles.radarContainer}>
              <ResponsiveContainer width="100%" height={380}>
                <RadarChart data={radarData} margin={{ top: 20, right: 40, bottom: 20, left: 40 }}>
                  <PolarGrid stroke="rgba(255,255,255,0.06)" />
                  <PolarAngleAxis
                    dataKey="subject"
                    tick={{
                      fill: '#64748b',
                      fontFamily: 'JetBrains Mono, monospace',
                      fontSize: 10,
                      letterSpacing: '0.1em',
                    }}
                  />
                  <PolarRadiusAxis
                    angle={90}
                    domain={[0, 100]}
                    tick={{
                      fill: '#64748b',
                      fontFamily: 'JetBrains Mono, monospace',
                      fontSize: 8,
                    }}
                    tickCount={5}
                    stroke="rgba(255,255,255,0.04)"
                  />
                  {selectedCompanies.map((company, idx) => (
                    <Radar
                      key={company.ticker}
                      name={company.ticker}
                      dataKey={company.ticker}
                      stroke={COMPARE_COLORS[idx]}
                      fill={COMPARE_COLORS[idx]}
                      fillOpacity={0.12}
                      strokeWidth={2}
                      dot={{ fill: COMPARE_COLORS[idx], r: 3 }}
                    />
                  ))}
                  <Tooltip content={<CustomTooltip />} />
                  <Legend
                    wrapperStyle={{
                      fontFamily: 'JetBrains Mono, monospace',
                      fontSize: '10px',
                      letterSpacing: '0.1em',
                      color: '#94a3b8',
                    }}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </section>
        )}

        {selectedCompanies.length < 2 && !loading && (
          <div className={styles.promptBanner}>
            <span className={styles.promptIcon}>◎</span>
            <span>Select at least 2 companies above to begin comparison analysis</span>
          </div>
        )}

        {/* Grade comparison table */}
        {selectedCompanies.length >= 2 && (
          <section className={styles.tableSection} aria-label="Side-by-side grade comparison">
            <div className={styles.sectionHeader}>
              <h2 className={styles.sectionTitle}>SIDE-BY-SIDE COMPARISON</h2>
            </div>
            <div className={styles.tableWrapper} role="table" aria-label="Company credit comparison">
              {/* Header */}
              <div
                className={styles.tableHeader}
                role="row"
                style={{ gridTemplateColumns: `200px repeat(${selectedCompanies.length}, 1fr)` }}
              >
                <div role="columnheader" className={styles.thMetric}>METRIC</div>
                {selectedCompanies.map((c, idx) => (
                  <div key={c.ticker} role="columnheader" className={styles.thCompany}>
                    <span style={{ color: COMPARE_COLORS[idx] }}>{c.ticker}</span>
                    <span className={styles.thCompanyName}>{c.name}</span>
                  </div>
                ))}
              </div>

              {/* Rows */}
              {[
                { label: 'Grade', key: 'grade' },
                { label: 'NexScore', key: 'nexscore' },
                { label: 'Creditworthiness', key: 'creditworthiness' },
                { label: 'Sentiment', key: 'sentiment' },
                { label: 'Market Signal', key: 'signal' },
              ].map(({ label, key }) => (
                <div
                  key={key}
                  className={styles.tableRow}
                  role="row"
                  style={{ gridTemplateColumns: `200px repeat(${selectedCompanies.length}, 1fr)` }}
                >
                  <div role="cell" className={styles.tdMetric}>{label}</div>
                  {selectedCompanies.map((c, idx) => {
                    let content: React.ReactNode = '—';

                    if (key === 'grade') {
                      content = <GradeBadge grade={c.grade} size="sm" />;
                    } else if (key === 'nexscore') {
                      content = (
                        <span style={{ color: gradeColor(c.grade), fontFamily: 'var(--font-mono)', fontWeight: 700 }}>
                          {c.nexscore.toFixed(1)}
                        </span>
                      );
                    } else if (key === 'creditworthiness') {
                      content = (
                        <span style={{ fontFamily: 'var(--font-mono)' }}>
                          {c.creditworthiness_display.toFixed(1)}
                        </span>
                      );
                    } else if (key === 'sentiment') {
                      const s = getSentimentLabel(c.sentiment_display);
                      content = (
                        <span
                          style={{
                            fontFamily: 'var(--font-mono)',
                            color:
                              s === 'BULLISH' ? 'var(--bullish)' :
                              s === 'BEARISH' ? 'var(--bearish)' : 'var(--neutral)',
                          }}
                        >
                          {c.sentiment_display.toFixed(1)}
                        </span>
                      );
                    } else if (key === 'signal') {
                      const s = getSentimentLabel(c.sentiment_display);
                      content = (
                        <span
                          style={{
                            fontFamily: 'var(--font-mono)',
                            fontSize: '10px',
                            fontWeight: 700,
                            color:
                              s === 'BULLISH' ? 'var(--bullish)' :
                              s === 'BEARISH' ? 'var(--bearish)' : 'var(--neutral)',
                          }}
                        >
                          {s}
                        </span>
                      );
                    }

                    // Winner highlight (numeric comparisons)
                    let isWinner = false;
                    if (key === 'nexscore') {
                      isWinner = c.nexscore === Math.max(...selectedCompanies.map((x) => x.nexscore));
                    } else if (key === 'grade') {
                      isWinner = gradeRank(c.grade) === Math.min(...selectedCompanies.map((x) => gradeRank(x.grade)));
                    }

                    return (
                      <div
                        key={c.ticker}
                        role="cell"
                        className={`${styles.tdValue} ${isWinner ? styles.tdWinner : ''}`}
                      >
                        {content}
                        {isWinner && key === 'nexscore' && (
                          <span className={styles.winnerBadge} aria-label="Highest NexScore">★</span>
                        )}
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
