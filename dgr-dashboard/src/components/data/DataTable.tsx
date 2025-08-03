import React, { useState, useMemo } from 'react'
import { ChevronDown, ChevronUp, Filter, Download, Search } from 'lucide-react'
import { formatPercentage, getDeviationColor, getDeviationBgColor } from '../../lib/utils'
import { Button } from '../ui/Button'
import { Input } from '../ui/Input'

interface TableData {
  plant: string
  date: string
  input_name: string
  value: number
  reason?: string
}

interface DataTableProps {
  data: TableData[]
  isLoading?: boolean
}

export function DataTable({ data, isLoading = false }: DataTableProps) {
  const [sortConfig, setSortConfig] = useState<{
    key: keyof TableData
    direction: 'asc' | 'desc'
  } | null>(null)
  const [filterText, setFilterText] = useState('')
  const [currentPage, setCurrentPage] = useState(1)
  const itemsPerPage = 50

  // Filter data based on search text
  const filteredData = useMemo(() => {
    if (!filterText) return data
    return data.filter(item =>
      Object.values(item).some(value =>
        value?.toString().toLowerCase().includes(filterText.toLowerCase())
      )
    )
  }, [data, filterText])

  // Sort data
  const sortedData = useMemo(() => {
    if (!sortConfig) return filteredData
    
    return [...filteredData].sort((a, b) => {
      const aValue = a[sortConfig.key]
      const bValue = b[sortConfig.key]
      
      if (typeof aValue === 'number' && typeof bValue === 'number') {
        return sortConfig.direction === 'asc' ? aValue - bValue : bValue - aValue
      }
      
      const aString = String(aValue || '')
      const bString = String(bValue || '')
      
      if (sortConfig.direction === 'asc') {
        return aString.localeCompare(bString)
      } else {
        return bString.localeCompare(aString)
      }
    })
  }, [filteredData, sortConfig])

  // Paginate data
  const paginatedData = useMemo(() => {
    const startIndex = (currentPage - 1) * itemsPerPage
    return sortedData.slice(startIndex, startIndex + itemsPerPage)
  }, [sortedData, currentPage])

  const totalPages = Math.ceil(sortedData.length / itemsPerPage)

  const handleSort = (key: keyof TableData) => {
    setSortConfig(current => {
      if (current?.key === key) {
        return {
          key,
          direction: current.direction === 'asc' ? 'desc' : 'asc'
        }
      }
      return { key, direction: 'asc' }
    })
  }

  const handleExport = () => {
    const csv = [
      ['Plant', 'Date', 'Equipment', 'Deviation (%)', 'Reason'],
      ...sortedData.map(row => [
        row.plant,
        row.date,
        row.input_name,
        row.value,
        row.reason || ''
      ])
    ].map(row => row.join(',')).join('\n')

    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `dgr-data-${new Date().toISOString().split('T')[0]}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  const SortButton = ({ column, children }: { column: keyof TableData; children: React.ReactNode }) => (
    <button
      onClick={() => handleSort(column)}
      className="flex items-center space-x-1 font-medium text-gray-700 hover:text-gray-900"
    >
      <span>{children}</span>
      {sortConfig?.key === column ? (
        sortConfig.direction === 'asc' ? (
          <ChevronUp className="w-4 h-4" />
        ) : (
          <ChevronDown className="w-4 h-4" />
        )
      ) : (
        <ChevronDown className="w-4 h-4 opacity-30" />
      )}
    </button>
  )

  if (isLoading) {
    return (
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-1/4 mb-6"></div>
          <div className="space-y-3">
            {[...Array(10)].map((_, i) => (
              <div key={i} className="h-4 bg-gray-200 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200">
      {/* Header */}
      <div className="p-6 border-b border-gray-200">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between space-y-4 sm:space-y-0">
          <div>
            <h2 className="text-xl font-semibold text-gray-900">Deviation Data</h2>
            <p className="text-sm text-gray-500">
              Showing {paginatedData.length} of {sortedData.length} records
            </p>
          </div>
          
          <div className="flex items-center space-x-3">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
              <Input
                placeholder="Search..."
                value={filterText}
                onChange={(e) => setFilterText(e.target.value)}
                className="pl-10 w-64"
              />
            </div>
            <Button onClick={handleExport} variant="secondary" size="sm">
              <Download className="w-4 h-4 mr-2" />
              Export
            </Button>
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left">
                <SortButton column="plant">Plant</SortButton>
              </th>
              <th className="px-6 py-3 text-left">
                <SortButton column="date">Date</SortButton>
              </th>
              <th className="px-6 py-3 text-left">
                <SortButton column="input_name">Equipment</SortButton>
              </th>
              <th className="px-6 py-3 text-left">
                <SortButton column="value">Deviation</SortButton>
              </th>
              <th className="px-6 py-3 text-left">
                <SortButton column="reason">Reason</SortButton>
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {paginatedData.map((row, index) => (
              <tr key={index} className="hover:bg-gray-50">
                <td className="px-6 py-4 text-sm text-gray-900">{row.plant}</td>
                <td className="px-6 py-4 text-sm text-gray-900">{row.date}</td>
                <td className="px-6 py-4 text-sm text-gray-900">{row.input_name}</td>
                <td className="px-6 py-4 text-sm">
                  <span
                    className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${getDeviationBgColor(row.value)} ${getDeviationColor(row.value)}`}
                  >
                    {formatPercentage(row.value)}
                  </span>
                </td>
                <td className="px-6 py-4 text-sm text-gray-900">
                  {row.reason || (
                    <span className="text-gray-400 italic">Not specified</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="px-6 py-4 border-t border-gray-200">
          <div className="flex items-center justify-between">
            <p className="text-sm text-gray-700">
              Page {currentPage} of {totalPages}
            </p>
            <div className="flex space-x-2">
              <Button
                onClick={() => setCurrentPage(page => Math.max(1, page - 1))}
                disabled={currentPage === 1}
                variant="secondary"
                size="sm"
              >
                Previous
              </Button>
              <Button
                onClick={() => setCurrentPage(page => Math.min(totalPages, page + 1))}
                disabled={currentPage === totalPages}
                variant="secondary"
                size="sm"
              >
                Next
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}