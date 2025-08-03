import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatDate(date: Date | string): string {
  if (typeof date === 'string') {
    date = new Date(date)
  }
  return date.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric'
  })
}

export function formatPercentage(value: number): string {
  return `${value.toFixed(2)}%`
}

export function getDeviationColor(value: number): string {
  const absValue = Math.abs(value)
  if (absValue <= 1) return 'text-success-600'
  if (absValue <= 3) return 'text-warning-600'
  if (absValue <= 5) return 'text-orange-600'
  return 'text-danger-600'
}

export function getDeviationBgColor(value: number): string {
  const absValue = Math.abs(value)
  if (absValue <= 1) return 'bg-success-50 border-success-200'
  if (absValue <= 3) return 'bg-warning-50 border-warning-200'
  if (absValue <= 5) return 'bg-orange-50 border-orange-200'
  return 'bg-danger-50 border-danger-200'
}

export function calculateNormalizedDeviation(
  inputsDeviated: number,
  totalInputs: number
): number {
  return totalInputs > 0 ? (inputsDeviated / totalInputs) * 100 : 0
}

export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout
  return (...args: Parameters<T>) => {
    clearTimeout(timeout)
    timeout = setTimeout(() => func(...args), wait)
  }
}