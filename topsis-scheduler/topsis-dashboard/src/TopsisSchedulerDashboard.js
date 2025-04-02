import React, { useState } from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  LineChart, Line, Area, AreaChart,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts';

const TopsisSchedulerDashboard = () => {
  const [activeTab, setActiveTab] = useState('nodeDistribution');
  
  // Node distribution data for different weighting strategies
  const nodeDistributionData = [
    {
      name: 'Balanced',
      topsisA: 27.3,
      topsisB: 13.6,
      topsisC: 50.0,
      topsisDefault: 9.1,
      defaultA: 0,
      defaultB: 31.9,
      defaultC: 68.2,
      defaultDefault: 0,
    },
    {
      name: 'Energy-centric',
      topsisA: 68.2,
      topsisB: 13.6,
      topsisC: 0,
      topsisDefault: 18.2,
      defaultA: 0,
      defaultB: 31.9,
      defaultC: 68.2,
      defaultDefault: 0,
    },
    {
      name: 'Performance-centric',
      topsisA: 36.4,
      topsisB: 18.2,
      topsisC: 41.0,
      topsisDefault: 4.6,
      defaultA: 0,
      defaultB: 31.9,
      defaultC: 68.2,
      defaultDefault: 0,
    },
    {
      name: 'Memory-optimized',
      topsisA: 0,
      topsisB: 0,
      topsisC: 81.9,
      topsisDefault: 18.2,
      defaultA: 0,
      defaultB: 31.9,
      defaultC: 68.2,
      defaultDefault: 0,
    },
    {
      name: 'Resource-efficiency',
      topsisA: 41.0,
      topsisB: 31.9,
      topsisC: 27.3,
      topsisDefault: 0,
      defaultA: 0,
      defaultB: 27.3,
      defaultC: 72.8,
      defaultDefault: 0,
    }
  ];

  // Workload distribution for balanced scenario
  const workloadDistributionData = [
    {
      name: 'Light (TOPSIS)',
      categoryA: 33.3,
      categoryB: 8.3,
      categoryC: 41.7,
      defaultPool: 16.7,
    },
    {
      name: 'Light (Default)',
      categoryA: 0,
      categoryB: 41.7,
      categoryC: 58.3,
      defaultPool: 0,
    },
    {
      name: 'Medium (TOPSIS)',
      categoryA: 16.6,
      categoryB: 0,
      categoryC: 83.3,
      defaultPool: 0,
    },
    {
      name: 'Medium (Default)',
      categoryA: 0,
      categoryB: 16.6,
      categoryC: 83.3,
      defaultPool: 0,
    },
    {
      name: 'Heavy (TOPSIS)',
      categoryA: 25.0,
      categoryB: 50.0,
      categoryC: 25.0,
      defaultPool: 0,
    },
    {
      name: 'Heavy (Default)',
      categoryA: 0,
      categoryB: 25.0,
      categoryC: 75.0,
      defaultPool: 0,
    },
  ];

  // Energy utilization comparison for different workload scenarios
  const energyUtilizationData = [
    {
      name: 'Low Competition',
      topsis: 65,
      default: 85,
    },
    {
      name: 'Medium Competition',
      topsis: 72,
      default: 89,
    },
    {
      name: 'High Competition',
      topsis: 78,
      default: 95,
    },
  ];

  // Radar chart data for comparing scheduling strategies across metrics
  const radarData = [
    {
      subject: 'Execution Time',
      topsis: 75,
      default: 85,
      fullMark: 100,
    },
    {
      subject: 'Energy Efficiency',
      topsis: 80,
      default: 55,
      fullMark: 100,
    },
    {
      subject: 'Core Utilization',
      topsis: 85,
      default: 70,
      fullMark: 100,
    },
    {
      subject: 'Memory Utilization',
      topsis: 70,
      default: 75,
      fullMark: 100,
    },
    {
      subject: 'Resource Balance',
      topsis: 90,
      default: 65,
      fullMark: 100,
    },
  ];

  // Heat map data representing workload efficiency
  const getHeatMapSection = (title, description) => {
    return (
      <div className="mt-4 mb-8">
        <h3 className="text-lg font-bold">{title}</h3>
        <p className="mb-4 text-gray-700">{description}</p>
        <div className="grid grid-cols-7 gap-1">
          <div className="col-span-1"></div>
          <div className="text-center text-sm font-medium">Cat A</div>
          <div className="text-center text-sm font-medium">Cat B</div>
          <div className="text-center text-sm font-medium">Cat C</div>
          <div className="text-center text-sm font-medium">Default</div>
          <div className="text-center text-sm font-medium">Total</div>
          <div className="col-span-1"></div>
          
          {/* TOPSIS Small Workloads */}
          <div className="text-right pr-2 text-sm font-medium">TOPSIS Small</div>
          <div className="bg-blue-200 p-2 text-center">33.3%</div>
          <div className="bg-blue-300 p-2 text-center">8.3%</div>
          <div className="bg-blue-600 p-2 text-center text-white">41.7%</div>
          <div className="bg-blue-400 p-2 text-center">16.7%</div>
          <div className="bg-gray-100 p-2 text-center font-bold">100%</div>
          <div className="col-span-1"></div>
          
          {/* Default Small Workloads */}
          <div className="text-right pr-2 text-sm font-medium">Default Small</div>
          <div className="bg-gray-100 p-2 text-center">0%</div>
          <div className="bg-blue-500 p-2 text-center text-white">41.7%</div>
          <div className="bg-blue-700 p-2 text-center text-white">58.3%</div>
          <div className="bg-gray-100 p-2 text-center">0%</div>
          <div className="bg-gray-100 p-2 text-center font-bold">100%</div>
          <div className="col-span-1"></div>
          
          {/* TOPSIS Medium Workloads */}
          <div className="text-right pr-2 text-sm font-medium">TOPSIS Medium</div>
          <div className="bg-blue-200 p-2 text-center">16.6%</div>
          <div className="bg-gray-100 p-2 text-center">0%</div>
          <div className="bg-blue-800 p-2 text-center text-white">83.3%</div>
          <div className="bg-gray-100 p-2 text-center">0%</div>
          <div className="bg-gray-100 p-2 text-center font-bold">100%</div>
          <div className="col-span-1"></div>
          
          {/* Default Medium Workloads */}
          <div className="text-right pr-2 text-sm font-medium">Default Medium</div>
          <div className="bg-gray-100 p-2 text-center">0%</div>
          <div className="bg-blue-200 p-2 text-center">16.6%</div>
          <div className="bg-blue-800 p-2 text-center text-white">83.3%</div>
          <div className="bg-gray-100 p-2 text-center">0%</div>
          <div className="bg-gray-100 p-2 text-center font-bold">100%</div>
          <div className="col-span-1"></div>
          
          {/* TOPSIS Heavy Workloads */}
          <div className="text-right pr-2 text-sm font-medium">TOPSIS Heavy</div>
          <div className="bg-blue-400 p-2 text-center">25.0%</div>
          <div className="bg-blue-600 p-2 text-center text-white">50.0%</div>
          <div className="bg-blue-400 p-2 text-center">25.0%</div>
          <div className="bg-gray-100 p-2 text-center">0%</div>
          <div className="bg-gray-100 p-2 text-center font-bold">100%</div>
          <div className="col-span-1"></div>
          
          {/* Default Heavy Workloads */}
          <div className="text-right pr-2 text-sm font-medium">Default Heavy</div>
          <div className="bg-gray-100 p-2 text-center">0%</div>
          <div className="bg-blue-400 p-2 text-center">25.0%</div>
          <div className="bg-blue-700 p-2 text-center text-white">75.0%</div>
          <div className="bg-gray-100 p-2 text-center">0%</div>
          <div className="bg-gray-100 p-2 text-center font-bold">100%</div>
        </div>
      </div>
    );
  };

  return (
    <div className="p-4 max-w-6xl mx-auto bg-white rounded-lg shadow">
      <h1 className="text-2xl font-bold mb-4">TOPSIS vs Default Scheduler Dashboard</h1>
      
      <div className="mb-6 border-b">
        <div className="flex space-x-4">
          <button 
            className={`py-2 px-4 ${activeTab === 'nodeDistribution' ? 'border-b-2 border-blue-500 font-medium' : 'text-gray-500'}`}
            onClick={() => setActiveTab('nodeDistribution')}
          >
            Node Distribution
          </button>
          <button 
            className={`py-2 px-4 ${activeTab === 'workloadDistribution' ? 'border-b-2 border-blue-500 font-medium' : 'text-gray-500'}`}
            onClick={() => setActiveTab('workloadDistribution')}
          >
            Workload Distribution
          </button>
          <button 
            className={`py-2 px-4 ${activeTab === 'energyUtilization' ? 'border-b-2 border-blue-500 font-medium' : 'text-gray-500'}`}
            onClick={() => setActiveTab('energyUtilization')}
          >
            Energy Utilization
          </button>
          <button 
            className={`py-2 px-4 ${activeTab === 'radarComparison' ? 'border-b-2 border-blue-500 font-medium' : 'text-gray-500'}`}
            onClick={() => setActiveTab('radarComparison')}
          >
            Metric Comparison
          </button>
          <button 
            className={`py-2 px-4 ${activeTab === 'heatmap' ? 'border-b-2 border-blue-500 font-medium' : 'text-gray-500'}`}
            onClick={() => setActiveTab('heatmap')}
          >
            Workload Heatmap
          </button>
        </div>
      </div>
      
      {activeTab === 'nodeDistribution' && (
        <div>
          <h2 className="text-xl font-semibold mb-4">Node Distribution By Weighting Strategy</h2>
          <div className="mb-4">
            <p className="text-gray-700">
              This chart compares how TOPSIS and the Default scheduler distribute workloads across different node types under various weighting strategies.
            </p>
          </div>
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={nodeDistributionData}
                margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis label={{ value: 'Percentage', angle: -90, position: 'insideLeft' }} />
                <Tooltip formatter={(value) => `${value}%`} />
                <Legend />
                <Bar dataKey="topsisA" name="TOPSIS - Cat A" fill="#8884d8" />
                <Bar dataKey="topsisB" name="TOPSIS - Cat B" fill="#82ca9d" />
                <Bar dataKey="topsisC" name="TOPSIS - Cat C" fill="#ffc658" />
                <Bar dataKey="topsisDefault" name="TOPSIS - Default Pool" fill="#ff8042" />
                <Bar dataKey="defaultA" name="Default - Cat A" fill="#8884d8" stackId="a" />
                <Bar dataKey="defaultB" name="Default - Cat B" fill="#82ca9d" stackId="a" />
                <Bar dataKey="defaultC" name="Default - Cat C" fill="#ffc658" stackId="a" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
      
      {activeTab === 'workloadDistribution' && (
        <div>
          <h2 className="text-xl font-semibold mb-4">Workload Distribution (Balanced Scenario)</h2>
          <div className="mb-4">
            <p className="text-gray-700">
              This chart shows how different workload types are distributed across node categories in the balanced weighting scenario.
            </p>
          </div>
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={workloadDistributionData}
                margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis label={{ value: 'Percentage', angle: -90, position: 'insideLeft' }} />
                <Tooltip formatter={(value) => `${value}%`} />
                <Legend />
                <Bar dataKey="categoryA" name="Category A" fill="#8884d8" />
                <Bar dataKey="categoryB" name="Category B" fill="#82ca9d" />
                <Bar dataKey="categoryC" name="Category C" fill="#ffc658" />
                <Bar dataKey="defaultPool" name="Default Pool" fill="#ff8042" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
      
      {activeTab === 'energyUtilization' && (
        <div>
          <h2 className="text-xl font-semibold mb-4">Energy Utilization by Competition Level</h2>
          <div className="mb-4">
            <p className="text-gray-700">
              This chart compares the energy utilization between TOPSIS and Default scheduler across different workload competition levels.
            </p>
          </div>
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart
                data={energyUtilizationData}
                margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Area type="monotone" dataKey="topsis" name="TOPSIS Scheduler" stroke="#8884d8" fill="#8884d8" />
                <Area type="monotone" dataKey="default" name="Default Scheduler" stroke="#82ca9d" fill="#82ca9d" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
      
      {activeTab === 'radarComparison' && (
        <div>
          <h2 className="text-xl font-semibold mb-4">Balanced Strategy Metric Comparison</h2>
          <div className="mb-4">
            <p className="text-gray-700">
              This radar chart compares TOPSIS and Default scheduler performance across multiple metrics in the balanced weighting scenario.
            </p>
          </div>
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="subject" />
                <PolarRadiusAxis angle={30} domain={[0, 100]} />
                <Radar name="TOPSIS Scheduler" dataKey="topsis" stroke="#8884d8" fill="#8884d8" fillOpacity={0.6} />
                <Radar name="Default Scheduler" dataKey="default" stroke="#82ca9d" fill="#82ca9d" fillOpacity={0.6} />
                <Legend />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
      
      {activeTab === 'heatmap' && (
        <div>
          <h2 className="text-xl font-semibold mb-4">Workload Distribution Heatmap (Balanced)</h2>
          {getHeatMapSection(
            "Balanced Workload Distribution Heatmap", 
            "This heatmap shows the percentage distribution of different workload types across node categories for both TOPSIS and Default schedulers in the balanced weighting scenario. Darker colors indicate higher percentages."
          )}
        </div>
      )}
    </div>
  );
};

export default TopsisSchedulerDashboard;