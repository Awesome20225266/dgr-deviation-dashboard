import React, { useState } from 'react'
import { Plus, MessageSquare, Save, X } from 'lucide-react'
import { Button } from '../ui/Button'
import { Input, Select } from '../ui/Input'
import { Card, CardHeader, CardTitle, CardContent } from '../ui/Card'

interface ReasonEntry {
  id: string
  plant: string
  date: string
  equipment: string
  reason: string
  comment: string
  createdAt: string
}

interface ReasonTrackerProps {
  plants: string[]
  equipment: string[]
  existingReasons: string[]
  onAddReason: (reason: ReasonEntry) => Promise<void>
}

const defaultReasons = [
  'Soiling', 'Shadow', 'Disconnected String', 'Connector Burn', 'Fuse Failure',
  'IGBT Failure', 'Module Damage', 'Power Clipping', 'Vegetation Growth',
  'Bypass diode', 'Degradation', 'Temperature Loss', 'RISO Fault',
  'MPPT Malfunction', 'Grid Outage', 'Load Curtailment', 'Efficiency loss',
  'Ground Fault', 'Module Mismatch', 'Array Misalignment', 'Tracker Failure',
  'Inverter Fan Issue', 'Bifacial factor Loss', 'Power Limitation'
]

export function ReasonTracker({ plants, equipment, existingReasons, onAddReason }: ReasonTrackerProps) {
  const [isAdding, setIsAdding] = useState(false)
  const [formData, setFormData] = useState({
    plant: '',
    date: '',
    equipment: '',
    reason: '',
    comment: ''
  })
  const [recentEntries, setRecentEntries] = useState<ReasonEntry[]>([])

  const allReasons = [...new Set([...defaultReasons, ...existingReasons])].sort()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!formData.plant || !formData.date || !formData.equipment || !formData.reason) {
      return
    }

    const newEntry: ReasonEntry = {
      id: Date.now().toString(),
      ...formData,
      createdAt: new Date().toISOString()
    }

    try {
      await onAddReason(newEntry)
      setRecentEntries(prev => [newEntry, ...prev.slice(0, 9)]) // Keep last 10 entries
      setFormData({ plant: '', date: '', equipment: '', reason: '', comment: '' })
      setIsAdding(false)
    } catch (error) {
      console.error('Error adding reason:', error)
    }
  }

  const handleCancel = () => {
    setFormData({ plant: '', date: '', equipment: '', reason: '', comment: '' })
    setIsAdding(false)
  }

  return (
    <div className="space-y-6">
      {/* Add Reason Form */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center space-x-2">
              <MessageSquare className="w-5 h-5 text-primary-600" />
              <span>Add Deviation Reason</span>
            </CardTitle>
            {!isAdding && (
              <Button onClick={() => setIsAdding(true)}>
                <Plus className="w-4 h-4 mr-2" />
                Add Reason
              </Button>
            )}
          </div>
        </CardHeader>
        
        {isAdding && (
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <Select
                  label="Plant"
                  value={formData.plant}
                  onChange={(e) => setFormData(prev => ({ ...prev, plant: e.target.value }))}
                  required
                >
                  <option value="">Select Plant</option>
                  {plants.map(plant => (
                    <option key={plant} value={plant}>{plant}</option>
                  ))}
                </Select>

                <Input
                  type="date"
                  label="Date"
                  value={formData.date}
                  onChange={(e) => setFormData(prev => ({ ...prev, date: e.target.value }))}
                  required
                />

                <Select
                  label="Equipment"
                  value={formData.equipment}
                  onChange={(e) => setFormData(prev => ({ ...prev, equipment: e.target.value }))}
                  required
                >
                  <option value="">Select Equipment</option>
                  {equipment.map(eq => (
                    <option key={eq} value={eq}>{eq}</option>
                  ))}
                </Select>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Select
                  label="Reason"
                  value={formData.reason}
                  onChange={(e) => setFormData(prev => ({ ...prev, reason: e.target.value }))}
                  required
                >
                  <option value="">Select Reason</option>
                  {allReasons.map(reason => (
                    <option key={reason} value={reason}>{reason}</option>
                  ))}
                </Select>

                <Input
                  label="Additional Comment (Optional)"
                  value={formData.comment}
                  onChange={(e) => setFormData(prev => ({ ...prev, comment: e.target.value }))}
                  placeholder="Any additional details..."
                />
              </div>

              <div className="flex space-x-3 pt-4">
                <Button type="submit">
                  <Save className="w-4 h-4 mr-2" />
                  Save Reason
                </Button>
                <Button type="button" variant="secondary" onClick={handleCancel}>
                  <X className="w-4 h-4 mr-2" />
                  Cancel
                </Button>
              </div>
            </form>
          </CardContent>
        )}
      </Card>

      {/* Recent Entries */}
      {recentEntries.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Recent Entries</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {recentEntries.map(entry => (
                <div
                  key={entry.id}
                  className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                >
                  <div className="flex-1">
                    <div className="flex items-center space-x-4 text-sm">
                      <span className="font-medium text-primary-600">{entry.plant}</span>
                      <span className="text-gray-600">{entry.date}</span>
                      <span className="text-gray-600">{entry.equipment}</span>
                    </div>
                    <div className="mt-1">
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-warning-100 text-warning-800">
                        {entry.reason}
                      </span>
                      {entry.comment && (
                        <span className="ml-2 text-sm text-gray-500">
                          "{entry.comment}"
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="text-xs text-gray-400">
                    {new Date(entry.createdAt).toLocaleDateString()}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Quick Stats */}
      <Card>
        <CardHeader>
          <CardTitle>Reason Statistics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-3 bg-primary-50 rounded-lg">
              <div className="text-2xl font-bold text-primary-600">{recentEntries.length}</div>
              <div className="text-sm text-gray-600">Today's Entries</div>
            </div>
            <div className="text-center p-3 bg-success-50 rounded-lg">
              <div className="text-2xl font-bold text-success-600">
                {new Set(recentEntries.map(e => e.plant)).size}
              </div>
              <div className="text-sm text-gray-600">Plants Covered</div>
            </div>
            <div className="text-center p-3 bg-warning-50 rounded-lg">
              <div className="text-2xl font-bold text-warning-600">
                {new Set(recentEntries.map(e => e.reason)).size}
              </div>
              <div className="text-sm text-gray-600">Unique Reasons</div>
            </div>
            <div className="text-center p-3 bg-danger-50 rounded-lg">
              <div className="text-2xl font-bold text-danger-600">
                {recentEntries.filter(e => e.comment).length}
              </div>
              <div className="text-sm text-gray-600">With Comments</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}