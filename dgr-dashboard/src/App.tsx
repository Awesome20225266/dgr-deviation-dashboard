import React, { useState, useEffect, useMemo } from 'react'
import { Header } from './components/layout/Header'
import { FilterPanel } from './components/filters/FilterPanel'
import { StatsCards } from './components/stats/StatsCards'
import { DataTable } from './components/data/DataTable'
import { DeviationChart } from './components/charts/DeviationChart'
import { RankingChart } from './components/charts/RankingChart'
import { ReasonDistributionChart } from './components/charts/ReasonDistributionChart'
import { Card, CardHeader, CardTitle, CardContent } from './components/ui/Card'
import { LoadingOverlay } from './components/ui/LoadingSpinner'
import { supabase, DGRData } from './lib/supabase'
import { calculateNormalizedDeviation } from './lib/utils'
import { ReasonTracker } from './components/reasons/ReasonTracker'

// Mock data for development (replace with actual Supabase queries)
const mockData: DGRData[] = [
  { plant: "Plant A", date: "2024-01-01", input_name: "Inverter 1", value: -2.5 },
  { plant: "Plant A", date: "2024-01-01", input_name: "Inverter 2", value: -4.2 },
  { plant: "Plant B", date: "2024-01-01", input_name: "Inverter 1", value: -1.8 },
  { plant: "Plant B", date: "2024-01-01", input_name: "Inverter 2", value: -3.5 },
  { plant: "Plant C", date: "2024-01-01", input_name: "Inverter 1", value: -0.9 },
  // Add more mock data as needed
]

const mockPlants = ["Plant A", "Plant B", "Plant C", "Plant D", "Plant E"]

function App() {
  const [activeTab, setActiveTab] = useState('overview')
  const [isLoading, setIsLoading] = useState(false)
  const [data, setData] = useState<DGRData[]>(mockData)
  
  // Filter states
  const [selectedPlants, setSelectedPlants] = useState<string[]>([])
  const [dateRange, setDateRange] = useState({
    start: '2024-01-01',
    end: '2024-12-31'
  })
  const [threshold, setThreshold] = useState(-3)

  // Load initial data
  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    setIsLoading(true)
    try {
      // In a real implementation, this would query Supabase
      // const { data, error } = await supabase
      //   .from('dgr_data')
      //   .select('*')
      //   .in('plant', selectedPlants)
      //   .gte('date', dateRange.start)
      //   .lte('date', dateRange.end)
      
      // For now, use mock data
      await new Promise(resolve => setTimeout(resolve, 1000)) // Simulate loading
      setData(mockData)
    } catch (error) {
      console.error('Error loading data:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleApplyFilters = () => {
    loadData()
  }

  const handleResetFilters = () => {
    setSelectedPlants([])
    setDateRange({ start: '2024-01-01', end: '2024-12-31' })
    setThreshold(-3)
  }

  // Filtered data based on current filters
  const filteredData = useMemo(() => {
    return data.filter(item => {
      const plantMatch = selectedPlants.length === 0 || selectedPlants.includes(item.plant)
      const dateMatch = item.date >= dateRange.start && item.date <= dateRange.end
      return plantMatch && dateMatch
    })
  }, [data, selectedPlants, dateRange])

  // Statistics calculation
  const stats = useMemo(() => {
    const totalRecords = filteredData.length
    const averageDeviation = totalRecords > 0 
      ? filteredData.reduce((sum, item) => sum + item.value, 0) / totalRecords 
      : 0
    
    const recordsBelowThreshold = filteredData.filter(item => item.value <= threshold).length
    const thresholdPercentage = totalRecords > 0 ? (recordsBelowThreshold / totalRecords) * 100 : 0

    // Plant performance calculations
    const plantStats = filteredData.reduce((acc, item) => {
      if (!acc[item.plant]) {
        acc[item.plant] = { total: 0, count: 0 }
      }
      acc[item.plant].total += item.value
      acc[item.plant].count += 1
      return acc
    }, {} as Record<string, { total: number; count: number }>)

    const plantAverages = Object.entries(plantStats).map(([plant, stats]) => ({
      plant,
      average: stats.total / stats.count
    }))

    const bestPerformingPlant = plantAverages.length > 0
      ? plantAverages.sort((a, b) => Math.abs(a.average) - Math.abs(b.average))[0]?.plant
      : ''

    const worstPerformingPlant = plantAverages.length > 0
      ? plantAverages.sort((a, b) => Math.abs(b.average) - Math.abs(a.average))[0]?.plant
      : ''

    return {
      totalRecords,
      averageDeviation,
      recordsBelowThreshold,
      worstPerformingPlant,
      bestPerformingPlant,
      thresholdPercentage
    }
  }, [filteredData, threshold])

  // Ranking data
  const rankingData = useMemo(() => {
    const plantGroups = filteredData.reduce((acc, item) => {
      if (!acc[item.plant]) {
        acc[item.plant] = []
      }
      acc[item.plant].push(item)
      return acc
    }, {} as Record<string, DGRData[]>)

    return Object.entries(plantGroups).map(([plant, records]) => {
      const totalRecords = records.length
      const recordsBelowThreshold = records.filter(r => r.value <= threshold).length
      const normalizedDeviation = calculateNormalizedDeviation(recordsBelowThreshold, totalRecords)
      
      return {
        name: plant,
        deviation: normalizedDeviation,
        rank: 0, // Will be set after sorting
        inputsDeviated: recordsBelowThreshold
      }
    }).sort((a, b) => Math.abs(a.deviation) - Math.abs(b.deviation))
      .map((item, index) => ({ ...item, rank: index + 1 }))
  }, [filteredData, threshold])

  // Reason distribution (mock data for now)
  const reasonData = [
    { reason: 'Soiling', count: 45, percentage: 30 },
    { reason: 'Shadow', count: 30, percentage: 20 },
    { reason: 'Module Damage', count: 22, percentage: 15 },
    { reason: 'Grid Outage', count: 15, percentage: 10 },
    { reason: 'Others', count: 38, percentage: 25 }
  ]

  const renderTabContent = () => {
    switch (activeTab) {
      case 'overview':
        return (
          <div className="space-y-6">
            <StatsCards data={stats} threshold={threshold} isLoading={isLoading} />
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Deviation Trend</CardTitle>
                </CardHeader>
                <CardContent>
                  <DeviationChart data={filteredData} threshold={threshold} />
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader>
                  <CardTitle>Plant Rankings</CardTitle>
                </CardHeader>
                <CardContent>
                  <RankingChart data={rankingData.slice(0, 10)} />
                </CardContent>
              </Card>
            </div>
          </div>
        )

      case 'table':
        return <DataTable data={filteredData} isLoading={isLoading} />

      case 'ranking':
        return (
          <Card>
            <CardHeader>
              <CardTitle>Plant Performance Rankings</CardTitle>
            </CardHeader>
            <CardContent>
              <RankingChart data={rankingData} height={600} />
            </CardContent>
          </Card>
        )

      case 'analytics':
        return (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Deviation Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                <DeviationChart data={filteredData} threshold={threshold} height={500} />
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Failure Reasons</CardTitle>
              </CardHeader>
              <CardContent>
                <ReasonDistributionChart data={reasonData} height={500} />
              </CardContent>
            </Card>
          </div>
        )

      case 'reasons':
        return (
          <ReasonTracker
            plants={mockPlants}
            equipment={['Inverter 1', 'Inverter 2', 'Inverter 3', 'Block A', 'Block B', 'Block C']}
            existingReasons={['Custom Reason 1', 'Custom Reason 2']}
            onAddReason={async (reason) => {
              // In a real implementation, this would save to Supabase
              console.log('Adding reason:', reason)
              // await supabase.from('reasons').insert(reason)
            }}
          />
        )

      default:
        return <div>Tab not implemented</div>
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Header activeTab={activeTab} onTabChange={setActiveTab} />
      
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <FilterPanel
          plants={mockPlants}
          selectedPlants={selectedPlants}
          onPlantsChange={setSelectedPlants}
          dateRange={dateRange}
          onDateRangeChange={setDateRange}
          threshold={threshold}
          onThresholdChange={setThreshold}
          onApplyFilters={handleApplyFilters}
          onResetFilters={handleResetFilters}
          isLoading={isLoading}
        />

        {renderTabContent()}
      </main>
    </div>
  )
}

export default App
