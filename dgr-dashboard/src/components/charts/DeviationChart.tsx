import React from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts'
import { formatDate, formatPercentage } from '../../lib/utils'

interface DeviationData {
  date: string
  value: number
  plant: string
  input_name: string
}

interface DeviationChartProps {
  data: DeviationData[]
  threshold?: number
  height?: number
}

export function DeviationChart({ data, threshold = -3, height = 400 }: DeviationChartProps) {
  // Group data by date and calculate average deviation
  const chartData = data.reduce((acc, item) => {
    const date = item.date
    if (!acc[date]) {
      acc[date] = { date, totalDeviation: 0, count: 0 }
    }
    acc[date].totalDeviation += item.value
    acc[date].count += 1
    return acc
  }, {} as Record<string, { date: string; totalDeviation: number; count: number }>)

  const processedData = Object.values(chartData).map(item => ({
    date: formatDate(item.date),
    avgDeviation: item.totalDeviation / item.count
  })).sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 rounded-lg shadow-lg border border-gray-200">
          <p className="font-medium text-gray-900">{label}</p>
          <p className="text-primary-600">
            Average Deviation: {formatPercentage(payload[0].value)}
          </p>
        </div>
      )
    }
    return null
  }

  return (
    <div className="w-full" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={processedData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis 
            dataKey="date" 
            stroke="#6b7280"
            fontSize={12}
            tickLine={false}
            axisLine={false}
          />
          <YAxis 
            stroke="#6b7280"
            fontSize={12}
            tickLine={false}
            axisLine={false}
            tickFormatter={(value) => `${value}%`}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          
          {/* Threshold line */}
          <ReferenceLine 
            y={threshold} 
            stroke="#ef4444" 
            strokeDasharray="5 5"
            label={{ value: `Threshold (${threshold}%)`, position: "insideTopRight" }}
          />
          
          <Line
            type="monotone"
            dataKey="avgDeviation"
            stroke="#2563eb"
            strokeWidth={3}
            dot={{ fill: '#2563eb', strokeWidth: 2, r: 4 }}
            activeDot={{ r: 6, stroke: '#2563eb', strokeWidth: 2 }}
            name="Average Deviation (%)"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}