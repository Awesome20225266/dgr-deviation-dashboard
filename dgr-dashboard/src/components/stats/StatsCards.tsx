import React from 'react'
import { TrendingDown, TrendingUp, AlertTriangle, CheckCircle, Activity } from 'lucide-react'
import { Card } from '../ui/Card'
import { formatPercentage } from '../../lib/utils'

interface StatsData {
  totalRecords: number
  averageDeviation: number
  recordsBelowThreshold: number
  worstPerformingPlant: string
  bestPerformingPlant: string
  thresholdPercentage: number
}

interface StatsCardsProps {
  data: StatsData
  threshold: number
  isLoading?: boolean
}

export function StatsCards({ data, threshold, isLoading = false }: StatsCardsProps) {
  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6 mb-6">
        {[...Array(5)].map((_, i) => (
          <Card key={i} className="animate-pulse">
            <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
            <div className="h-8 bg-gray-200 rounded w-1/2"></div>
          </Card>
        ))}
      </div>
    )
  }

  const stats = [
    {
      title: 'Total Records',
      value: data.totalRecords.toLocaleString(),
      icon: Activity,
      color: 'text-primary-600',
      bgColor: 'bg-primary-100',
      description: 'Data points analyzed'
    },
    {
      title: 'Average Deviation',
      value: formatPercentage(data.averageDeviation),
      icon: data.averageDeviation < 0 ? TrendingDown : TrendingUp,
      color: Math.abs(data.averageDeviation) <= 3 ? 'text-success-600' : 'text-danger-600',
      bgColor: Math.abs(data.averageDeviation) <= 3 ? 'bg-success-100' : 'bg-danger-100',
      description: 'Overall performance'
    },
    {
      title: 'Below Threshold',
      value: data.recordsBelowThreshold.toLocaleString(),
      icon: AlertTriangle,
      color: 'text-warning-600',
      bgColor: 'bg-warning-100',
      description: `≤ ${threshold}% deviation`,
      subValue: `${formatPercentage(data.thresholdPercentage)} of total`
    },
    {
      title: 'Best Plant',
      value: data.bestPerformingPlant || 'N/A',
      icon: CheckCircle,
      color: 'text-success-600',
      bgColor: 'bg-success-100',
      description: 'Lowest deviation',
      isText: true
    },
    {
      title: 'Worst Plant',
      value: data.worstPerformingPlant || 'N/A',
      icon: AlertTriangle,
      color: 'text-danger-600',
      bgColor: 'bg-danger-100',
      description: 'Highest deviation',
      isText: true
    }
  ]

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6 mb-6">
      {stats.map((stat, index) => {
        const Icon = stat.icon
        return (
          <Card key={index} hover className="relative overflow-hidden">
            <div className="flex items-center justify-between">
              <div className="flex-1">
                <p className="text-sm font-medium text-gray-600 mb-1">
                  {stat.title}
                </p>
                <p className={`text-2xl font-bold ${stat.color} mb-1`}>
                  {stat.isText && stat.value.length > 12 
                    ? `${stat.value.substring(0, 12)}...`
                    : stat.value
                  }
                </p>
                <p className="text-xs text-gray-500">
                  {stat.description}
                </p>
                {stat.subValue && (
                  <p className="text-xs text-gray-400 mt-1">
                    {stat.subValue}
                  </p>
                )}
              </div>
              <div className={`${stat.bgColor} p-3 rounded-full`}>
                <Icon className={`w-6 h-6 ${stat.color}`} />
              </div>
            </div>
            
            {/* Animated background effect */}
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white to-transparent opacity-0 hover:opacity-10 transform -skew-x-12 -translate-x-full hover:translate-x-full transition-all duration-700"></div>
          </Card>
        )
      })}
    </div>
  )
}