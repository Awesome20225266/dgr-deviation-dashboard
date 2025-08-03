import React from 'react'
import { Calendar, Filter, RotateCcw } from 'lucide-react'
import { Button } from '../ui/Button'
import { Input } from '../ui/Input'
import { Select } from '../ui/Input'
import { Card } from '../ui/Card'

interface FilterPanelProps {
  plants: string[]
  selectedPlants: string[]
  onPlantsChange: (plants: string[]) => void
  dateRange: { start: string; end: string }
  onDateRangeChange: (range: { start: string; end: string }) => void
  threshold: number
  onThresholdChange: (threshold: number) => void
  onApplyFilters: () => void
  onResetFilters: () => void
  isLoading?: boolean
}

export function FilterPanel({
  plants,
  selectedPlants,
  onPlantsChange,
  dateRange,
  onDateRangeChange,
  threshold,
  onThresholdChange,
  onApplyFilters,
  onResetFilters,
  isLoading = false
}: FilterPanelProps) {
  const handlePlantChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const selected = Array.from(e.target.selectedOptions, option => option.value)
    onPlantsChange(selected)
  }

  const handleSelectAllPlants = () => {
    onPlantsChange(plants)
  }

  const handleDeselectAllPlants = () => {
    onPlantsChange([])
  }

  return (
    <Card className="mb-6">
      <div className="flex items-center space-x-2 mb-4">
        <Filter className="w-5 h-5 text-primary-600" />
        <h2 className="text-lg font-semibold text-gray-900">Filters</h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Plant Selection */}
        <div className="space-y-2">
          <label className="text-sm font-medium text-gray-700">Plants</label>
          <select
            multiple
            value={selectedPlants}
            onChange={handlePlantChange}
            className="select h-32"
            disabled={isLoading}
          >
            {plants.map(plant => (
              <option key={plant} value={plant}>
                {plant}
              </option>
            ))}
          </select>
          <div className="flex space-x-2">
            <Button
              size="sm"
              variant="ghost"
              onClick={handleSelectAllPlants}
              disabled={isLoading}
            >
              Select All
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={handleDeselectAllPlants}
              disabled={isLoading}
            >
              Clear
            </Button>
          </div>
          <p className="text-xs text-gray-500">
            {selectedPlants.length} of {plants.length} selected
          </p>
        </div>

        {/* Date Range */}
        <div className="space-y-4">
          <div>
            <label className="text-sm font-medium text-gray-700 mb-2 flex items-center">
              <Calendar className="w-4 h-4 mr-1" />
              Start Date
            </label>
            <Input
              type="date"
              value={dateRange.start}
              onChange={(e) => onDateRangeChange({ ...dateRange, start: e.target.value })}
              disabled={isLoading}
            />
          </div>
          <div>
            <label className="text-sm font-medium text-gray-700 mb-2 flex items-center">
              <Calendar className="w-4 h-4 mr-1" />
              End Date
            </label>
            <Input
              type="date"
              value={dateRange.end}
              onChange={(e) => onDateRangeChange({ ...dateRange, end: e.target.value })}
              disabled={isLoading}
            />
          </div>
        </div>

        {/* Threshold */}
        <div>
          <Input
            type="number"
            label="Deviation Threshold (%)"
            value={threshold}
            onChange={(e) => onThresholdChange(Number(e.target.value))}
            step="0.1"
            placeholder="-3.0"
            disabled={isLoading}
          />
          <p className="text-xs text-gray-500 mt-1">
            Values below this threshold will be highlighted
          </p>
        </div>

        {/* Actions */}
        <div className="flex flex-col justify-end space-y-2">
          <Button
            onClick={onApplyFilters}
            disabled={isLoading || selectedPlants.length === 0}
            className="w-full"
          >
            {isLoading ? 'Loading...' : 'Apply Filters'}
          </Button>
          <Button
            onClick={onResetFilters}
            variant="secondary"
            disabled={isLoading}
            className="w-full"
          >
            <RotateCcw className="w-4 h-4 mr-2" />
            Reset
          </Button>
        </div>
      </div>
    </Card>
  )
}