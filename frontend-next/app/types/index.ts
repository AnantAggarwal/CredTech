// types/index.ts — shared type definitions for NexusCredit

export interface Company {
  ticker: string;
  name: string;
  credit_score: number;      // raw 0–1
  sentiment_score: number;   // raw -1 to 1
  nexscore: number;          // 0–100
  grade: Grade;
  creditworthiness_display: number; // (1 - credit_score) * 100
  sentiment_display: number;        // sentiment_score * 100
}

export type Grade = 'AAA' | 'AA' | 'A' | 'BBB' | 'BB' | 'B' | 'CCC';

export interface NexScoreDetail {
  nexscore: number;
  grade: Grade;
  analyst_note: string;
  credit_display: number;
  sentiment_display: number;
}

export interface NewsItem {
  title: string;
  source: string;
  url: string;
  sentiment_label: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
  sentiment_score: number;
}

export interface HistoryPoint {
  date: string;
  nexscore: number;
}

export interface Stats {
  total_companies: number;
  avg_nexscore: number;
  avg_grade: Grade;
  top_ticker: string;
  bottom_ticker: string;
}
