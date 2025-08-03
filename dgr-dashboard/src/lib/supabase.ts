import { createClient } from '@supabase/supabase-js'

const supabaseUrl = 'https://ubkcxehguactwwcarkae.supabase.co'
const supabaseKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVia2N4ZWhndWFjdHd3Y2Fya2FlIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTIyMTU3OTYsImV4cCI6MjA2Nzc5MTc5Nn0.NPiJj_o-YervOE1dPxWRJhEI1fUwxT3Dptz-JszChLo'

export const supabase = createClient(supabaseUrl, supabaseKey)

// Types for the database
export interface DGRData {
  id?: number
  plant: string
  date: string
  input_name: string
  value: number
  created_at?: string
  reason?: string
}

export interface Reason {
  id?: number
  reason_name: string
  created_at?: string
}

export interface PlantMapping {
  id?: number
  Plant_Name: string
  Data_Start_Col: string
  Data_End_Col: string
}