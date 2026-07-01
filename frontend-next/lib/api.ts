// lib/api.ts — API client for NexusCredit backend

import type { Company, NexScoreDetail, NewsItem, HistoryPoint, Stats } from '@/app/types';

const BASE_URL = 'http://localhost:8000';

async function fetchAPI<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, { cache: 'no-store' });
  if (!res.ok) {
    throw new Error(`API error ${res.status}: ${path}`);
  }
  return res.json();
}

export async function getCompanies(): Promise<Company[]> {
  return fetchAPI<Company[]>('/companies');
}

export async function getLeaderboard(): Promise<Company[]> {
  return fetchAPI<Company[]>('/leaderboard');
}

export async function getNexScore(ticker: string): Promise<NexScoreDetail> {
  return fetchAPI<NexScoreDetail>(`/company/${ticker}/nexscore`);
}

export async function getCompanyNews(ticker: string): Promise<NewsItem[]> {
  return fetchAPI<NewsItem[]>(`/company/${ticker}/news`);
}

export async function getCompanyHistory(ticker: string): Promise<HistoryPoint[]> {
  return fetchAPI<HistoryPoint[]>(`/company/${ticker}/history`);
}

export async function getStats(): Promise<Stats> {
  return fetchAPI<Stats>('/stats');
}
