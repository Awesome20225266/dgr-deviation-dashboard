import React from 'react'
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
  Legend
} from 'recharts'

interface ReasonData {
  reason: string
  count: number
  percentage: number
}

interface ReasonDistributionChartProps {
  data: ReasonData[]
  height?: number
}

const COLORS = [
  '#2563eb', '#dc2626', '#16a34a', '#f59e0b', '#7c3aed',
  '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#8b5cf6'
]

export function ReasonDistributionChart({ data, height = 400 }: ReasonDistributionChartProps) {
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload
      return (
        <div className="bg-white p-3 rounded-lg shadow-lg border border-gray-200">
          <p className="font-medium text-gray-900">{data.reason}</p>
          <p className="text-primary-600">Count: {data.count}</p>
          <p className="text-gray-600">Percentage: {data.percentage.toFixed(1)}%</p>
        </div>
      )
    }
    return null
  }

  const CustomLabel = ({ value, percentage }: any) => {
    if (percentage > 5) {
      return `${percentage.toFixed(1)}%`
    }
    return ''
  }

  return (
    <div className="w-full" style={{ height }}>
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-gray-900">Reason Distribution</h3>
        <p className="text-sm text-gray-500">Distribution of deviation reasons</p>
      </div>
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            labelLine={false}
            label={CustomLabel}
            outerRadius={120}
            fill="#8884d8"
            dataKey="count"
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip content={<CustomTooltip />} />
          <Legend 
            wrapperStyle={{ paddingTop: '20px' }}
            iconType="circle"
          />
        </PieChart>
      </ResponsiveContainer>
    </div>
  )
}