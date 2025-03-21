/**
 * File system utility for browser environment
 */
declare global {
  interface Window {
    fs: {
      readFile: (path: string, options?: { encoding?: string }) => Promise<Uint8Array | string>;
    };
  }
}

export interface AirQualityDataItem {
  Temperature: number;
  Humidity: number;
  'PM2.5': number;
  PM10: number;
  NO2: number;
  SO2: number;
  CO: number;
  Proximity_to_Industrial_Areas: number;
  Population_Density: number;
  'Air Quality': string;
}

export interface StatItem {
  min: number;
  max: number;
  avg: number;
  unit: string;
}

export interface StatsObject {
  [key: string]: StatItem;
}

export interface DistributionItem {
  name: string;
  value: number;
  color: string;
}

export interface PollutantAverageItem {
  name: string;
  'PM2.5': number;
  PM10: number;
  NO2: number;
  SO2: number;
  CO: number;
}

export interface CorrelationItem {
  x: number;
  y: number;
  z: number;
  airQuality: string;
}

export const AIR_QUALITY_COLORS = {
  'Good': '#4caf50',
  'Moderate': '#ff9800',
  'Poor': '#f44336',
  'Hazardous': '#9c27b0'
};

export const getUnit = (column: string): string => {
  switch(column) {
    case 'Temperature': return '°C';
    case 'Humidity': return '%';
    case 'PM2.5': 
    case 'PM10': 
    case 'NO2': 
    case 'SO2': return 'μg/m³';
    case 'CO': return 'mg/m³';
    case 'Proximity_to_Industrial_Areas': return 'km';
    case 'Population_Density': return '/km²';
    default: return '';
  }
};
