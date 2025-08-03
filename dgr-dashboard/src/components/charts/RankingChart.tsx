import React from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell
} from 'recharts'
import { formatPercentage, getDeviationColor } from '../../lib/utils'

interface RankingData {
  name: string
  deviation: number
  rank: number
  inputsDeviated: number
}

interface RankingChartProps {
  data: RankingData[]
  height?: number
  title?: string
}

export function RankingChart({ data, height = 400, title = "Plant Rankings" }: RankingChartProps) {
  const getBarColor = (deviation: number) => {
    const absDeviation = Math.abs(deviation)
    if (absDeviation <= 1) return '#16a34a' // green
    if (absDeviation <= 3) return '#f59e0b' // yellow
    if (absDeviation <= 5) return '#f97316' // orange
    return '#ef4444' // red
  }

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload
      return (
        <div className="bg-white p-3 rounded-lg shadow-lg border border-gray-200">
          <p className="font-medium text-gray-900">{label}</p>
          <p className="text-primary-600">
            Rank: #{data.rank}
          </p>
          <p className="text-gray-600">
            Deviation: {formatPercentage(data.deviation)}
          </p>
          <p className="text-gray-600">
            Inputs Deviated: {data.inputsDeviated}
          </p>
        </div>
      )
    }
    return null
  }

  return (
    <div className="w-full" style={{ height }}>
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
        <p className="text-sm text-gray-500">Lower absolute deviation is better (green)</p>
      </div>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis 
            dataKey="name" 
            stroke="#6b7280"
            fontSize={12}
            tickLine={false}
            axisLine={false}
            angle={-45}
            textAnchor="end"
            height={80}
          />
          <YAxis 
            stroke="#6b7280"
            fontSize={12}
            tickLine={false}
            axisLine={false}
            tickFormatter={(value) => `${value}%`}
          />
          <Tooltip content={<CustomTooltip />} />
          <Bar dataKey="deviation" name="Deviation %" radius={[4, 4, 0, 0]}>
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={getBarColor(entry.deviation)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}