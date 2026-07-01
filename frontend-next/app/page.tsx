'use client';

// app/page.tsx — NexusCredit Dashboard
import { useEffect, useState, useCallback } from 'react';
import Link from 'next/link';
import type { Company, Grade } from '@/app/types';
import CompanyCard from '@/app/components/CompanyCard';
import SkeletonCard from '@/app/components/SkeletonCard';
import GradeBadge from '@/app/components/GradeBadge';
import styles from './page.module.css';

function formatDateTime(date: Date): string {
  return date.toLocaleString('en-US', {
    weekday: 'short',
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  });
}

function getAvgGrade(companies: Company[]): Grade {
  if (companies.length === 0) return 'BBB';
  const avg = companies.reduce((sum, c) => sum + c.nexscore, 0) / companies.length;
  if (avg >= 90) return 'AAA';
  if (avg >= 80) return 'AA';
  if (avg >= 70) return 'A';
  if (avg >= 60) return 'BBB';
  if (avg >= 50) return 'BB';
  if (avg >= 40) return 'B';
  return 'CCC';
}

export default function DashboardPage() {
  const [companies, setCompanies] = useState<Company[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [now, setNow] = useState(new Date());
  const [refreshing, setRefreshing] = useState(false);
  const [lastRefreshed, setLastRefreshed] = useState<Date | null>(null);
  const [mounted, setMounted] = useState(false);

  // Live clock
  useEffect(() => {
    setMounted(true);
    const interval = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(interval);
  }, []);

  const [offset, setOffset] = useState(0);
  const [hasMore, setHasMore] = useState(true);
  const LIMIT = 100;

  const fetchCompanies = useCallback(async (isRefresh = false, currentOffset = 0) => {
    if (isRefresh) setRefreshing(true);
    else setLoading(true);
    setError(null);
    try {
      const res = await fetch(`http://localhost:8000/companies?limit=${LIMIT}&offset=${currentOffset}`, { cache: 'no-store' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: Company[] = await res.json();
      
      if (currentOffset === 0) {
        setCompanies(data);
      } else {
        setCompanies(prev => [...prev, ...data]);
      }
      
      setHasMore(data.length === LIMIT);
      setLastRefreshed(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load companies');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetchCompanies(false, 0);
  }, [fetchCompanies]);

  const handleRefresh = () => {
    setOffset(0);
    fetchCompanies(true, 0);
  };

  const handleLoadMore = () => {
    const nextOffset = offset + LIMIT;
    setOffset(nextOffset);
    fetchCompanies(false, nextOffset);
  };

  const avgScore = companies.length
    ? companies.reduce((s, c) => s + c.nexscore, 0) / companies.length
    : 0;
  const avgGrade = getAvgGrade(companies);

  return (
    <div className={styles.page}>
      {/* Header */}
      <header className={styles.header}>
        <div className={styles.headerLeft}>
          <div className={styles.logoRow}>
            <span className={styles.pulseDot} aria-label="Live" />
            <h1 className={styles.logoText}>NEXUS CREDIT INTELLIGENCE</h1>
          </div>
          <p className={styles.tagline}>AI-Powered Corporate Credit Terminal</p>
        </div>
        <div className={styles.headerRight}>
          <div className={styles.clock} aria-live="polite" aria-atomic="true">
            <span className={styles.clockLabel}>LOCAL TIME</span>
            <span className={styles.clockTime} suppressHydrationWarning>{mounted ? formatDateTime(now) : ''}</span>
          </div>
          <nav className={styles.nav}>
            <Link href="/" className={`${styles.navLink} ${styles.navActive}`}>
              Dashboard
            </Link>
            <Link href="/compare" className={styles.navLink}>
              Compare
            </Link>
          </nav>
        </div>
      </header>

      {/* Scan line decoration */}
      <div className={styles.scanLine} aria-hidden="true" />

      <main className={styles.main}>
        {/* Portfolio Health Strip */}
        {!loading && companies.length > 0 && (
          <section className={styles.healthStrip} aria-label="Portfolio health summary">
            <div className={styles.healthContent}>
              <div className={styles.healthStat}>
                <span className={styles.healthLabel}>PORTFOLIO AVG. NEXSCORE</span>
                <div className={styles.healthScoreRow}>
                  <span className={styles.healthScore}>{avgScore.toFixed(1)}</span>
                  <GradeBadge grade={avgGrade} size="md" />
                </div>
              </div>
              <div className={styles.healthDivider} aria-hidden="true" />
              <div className={styles.healthStat}>
                <span className={styles.healthLabel}>COMPANIES TRACKED</span>
                <span className={styles.healthCount}>{companies.length}</span>
              </div>
              <div className={styles.healthDivider} aria-hidden="true" />
              <div className={styles.healthStat}>
                <span className={styles.healthLabel}>TOP NEXSCORE</span>
                <span className={styles.healthCount}>
                  {companies.length
                    ? Math.max(...companies.map((c) => c.nexscore)).toFixed(1)
                    : '—'}
                </span>
              </div>
              <div className={styles.healthDivider} aria-hidden="true" />
              <div className={styles.healthStat}>
                <span className={styles.healthLabel}>COVERAGE</span>
                <span className={styles.healthCount}>GLOBAL</span>
              </div>
            </div>
          </section>
        )}

        {/* Watchlist Section */}
        <section className={styles.watchlistSection}>
          <div className={styles.sectionHeader}>
            <div className={styles.sectionTitleRow}>
              <h2 className={styles.sectionTitle}>WATCHLIST</h2>
              <span className={styles.sectionCount}>
                {!loading ? `${companies.length} entities` : '— entities'}
              </span>
            </div>
            <div className={styles.controls}>
              {lastRefreshed && (
                <span className={styles.lastRefreshed}>
                  Updated {lastRefreshed.toLocaleTimeString('en-US', { hour12: false })}
                </span>
              )}
                <button
                className={`${styles.refreshBtn} ${refreshing ? styles.refreshing : ''}`}
                onClick={handleRefresh}
                disabled={refreshing || loading}
                aria-label="Refresh company data"
              >
                <span className={styles.refreshIcon} aria-hidden="true">⟳</span>
                {refreshing ? 'REFRESHING...' : 'REFRESH'}
              </button>
            </div>
          </div>

          {/* Error state */}
          {error && (
            <div className={styles.errorBanner} role="alert">
              <span className={styles.errorIcon}>⚠</span>
              <span>
                {error === 'Failed to fetch' || error.includes('HTTP')
                  ? 'Unable to connect to NexusCredit backend. Ensure the API server is running at localhost:8000.'
                  : error}
              </span>
            </div>
          )}

          {/* Grid */}
          <div className={styles.grid} aria-busy={loading}>
            {loading && offset === 0
              ? Array.from({ length: 6 }).map((_, i) => <SkeletonCard key={i} />)
              : companies.length === 0 && !error && !loading
              ? (
                <div className={styles.emptyState}>
                  <span className={styles.emptyIcon}>◎</span>
                  <p>No companies found. Add companies to your watchlist to get started.</p>
                </div>
              )
              : companies.map((company, idx) => (
                <CompanyCard
                  key={company.ticker}
                  company={company}
                  animationDelay={(idx % LIMIT) * 60}
                />
              ))}
          </div>

          {!loading && hasMore && companies.length > 0 && (
            <div style={{ textAlign: 'center', marginTop: '2rem' }}>
              <button
                className={styles.refreshBtn}
                onClick={handleLoadMore}
                disabled={loading}
              >
                LOAD MORE
              </button>
            </div>
          )}
          {loading && offset > 0 && (
            <div style={{ textAlign: 'center', marginTop: '2rem', color: 'var(--text-secondary)' }}>
              Loading...
            </div>
          )}
        </section>
      </main>

      <footer className={styles.footer}>
        <span>NEXUS CREDIT INTELLIGENCE</span>
        <span className={styles.footerDot} aria-hidden="true" />
        <span>Powered by AI · Real-time Analysis</span>
        <span className={styles.footerDot} aria-hidden="true" />
        <span suppressHydrationWarning>© {mounted ? now.getFullYear() : ''}</span>
      </footer>
    </div>
  );
}
