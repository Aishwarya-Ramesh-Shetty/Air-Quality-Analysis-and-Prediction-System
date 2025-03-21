import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, 
         LineChart, Line, PieChart, Pie, Cell, ScatterChart, Scatter, ZAxis } from 'recharts';
import { AlertCircle, Droplets, Thermometer, Wind, Activity, BarChart2 } from 'lucide-react';
import Papa from 'papaparse';
import { 
  AirQualityDataItem, 
  StatItem, 
  StatsObject, 
  DistributionItem, 
  PollutantAverageItem, 
  CorrelationItem,
  AIR_QUALITY_COLORS,
  getUnit
} from '../utils/dataUtils';

const AirQualityDashboard: React.FC = () => {
  const [data, setData] = useState<AirQualityDataItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [stats, setStats] = useState<StatsObject>({});
  const [airQualityDistribution, setAirQualityDistribution] = useState<DistributionItem[]>([]);
  const [pollutantAverages, setPollutantAverages] = useState<PollutantAverageItem[]>([]);
  const [correlationData, setCorrelationData] = useState<CorrelationItem[]>([]);
  const [selectedFeature, setSelectedFeature] = useState<string>('Temperature');

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await window.fs.readFile('TechBlitz DataScience Dataset.csv', { encoding: 'utf8' });
        const text = response as string;
        
        Papa.parse(text, {
          header: true,
          dynamicTyping: true,
          skipEmptyLines: true,
          complete: (results) => {
            setData(results.data as AirQualityDataItem[]);
            
            // Calculate statistics
            calculateStats(results.data as AirQualityDataItem[]);
            
            // Calculate air quality distribution
            calculateAirQualityDistribution(results.data as AirQualityDataItem[]);
            
            // Calculate pollutant averages by air quality
            calculatePollutantAverages(results.data as AirQualityDataItem[]);
            
            // Calculate correlation data
            calculateCorrelationData(results.data as AirQualityDataItem[]);
            
            setLoading(false);
          }
        });
      } catch (error) {
        console.error('Error reading file:', error);
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const calculateStats = (data: AirQualityDataItem[]) => {
    const numericalColumns = ['Temperature', 'Humidity', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Proximity_to_Industrial_Areas', 'Population_Density'];
    const calculatedStats: StatsObject = {};

    numericalColumns.forEach(column => {
      const values = data
        .map(row => row[column as keyof AirQualityDataItem] as number)
        .filter(val => val !== null && val !== undefined);
      
      calculatedStats[column] = {
        min: Math.min(...values),
        max: Math.max(...values),
        avg: values.reduce((sum, val) => sum + val, 0) / values.length,
        unit: getUnit(column)
      };
    });

    setStats(calculatedStats);
  };

  const calculateAirQualityDistribution = (data: AirQualityDataItem[]) => {
    const distribution: Record<string, number> = data.reduce((acc, row) => {
      const quality = row['Air Quality'];
      acc[quality] = (acc[quality] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    const chartData: DistributionItem[] = Object.keys(distribution).map(quality => ({
      name: quality,
      value: distribution[quality],
      color: AIR_QUALITY_COLORS[quality as keyof typeof AIR_QUALITY_COLORS]
    }));

    setAirQualityDistribution(chartData);
  };

  const calculatePollutantAverages = (data: AirQualityDataItem[]) => {
    const pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO'];
    const averagesByQuality: Record<string, Record<string, number>> = {};
    
    // Group by air quality
    data.forEach(row => {
      const quality = row['Air Quality'];
      if (!averagesByQuality[quality]) {
        averagesByQuality[quality] = { count: 0 };
        pollutants.forEach(pollutant => {
          averagesByQuality[quality][pollutant] = 0;
        });
      }
      
      averagesByQuality[quality].count++;
      pollutants.forEach(pollutant => {
        averagesByQuality[quality][pollutant] += row[pollutant as keyof AirQualityDataItem] as number || 0;
      });
    });
    
    // Calculate averages
    const chartData = Object.keys(averagesByQuality).map(quality => {
      const result: Record<string, string | number> = { name: quality };
      pollutants.forEach(pollutant => {
        result[pollutant] = averagesByQuality[quality][poll