import React, { useState, useEffect, useRef } from 'react';
import { Chart as ChartJS, registerables } from 'chart.js';
import { Chart } from 'react-chartjs-2';
import 'chart.js/auto';
import zoomPlugin from 'chartjs-plugin-zoom';
import axios from 'axios';
import { create } from 'zustand';


ChartJS.register(...registerables, zoomPlugin);

// 1. State Management with Zustand
const useStore = create((set) => ({
  nodes: [],
  edges: [],
  fileNames: [],
  loading: true, // Add a loading state

  // Actions
  fetchData: async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/data');
      set({ nodes: response.data.nodes, edges: response.data.edges, fileNames: response.data.file_names, loading: false });
    } catch (error) {
      console.error("Error fetching data:", error);
      set({ loading: false }); // Ensure loading is false even on error
    }
  },

  setMode: (newMode) => set({ mode: newMode }),
}));

// 2. Sidebar Component
const Sidebar = () => {
  const { setMode } = useStore();

  return (
    <div style={{ padding: '10px', width: '200px', backgroundColor: '#f0f0f0' }}>
      <h3>Controls</h3>
      <button onClick={() => setMode('select')}>Select</button>
      <button onClick={() => setMode('draw')}>Draw</button>
      <button onClick={() => setMode('smooth_start')}>Smooth</button>
      <button onClick={() => setMode('connect')}>Connect Nodes</button>
    </div>
  );
};

// 3. Plot Component
const Plot = () => {
  const { nodes, edges, loading, fetchData } = useStore();
  const chartRef = useRef(null);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  if (loading) {
    return <div>Loading...</div>;
  }

  const data = {
    datasets: [
      {
        label: 'Nodes',
        data: nodes.map(node => ({ x: node[1], y: node[2] })),
        backgroundColor: 'rgba(255, 99, 132, 1)',
        pointRadius: 5,
        type: 'scatter',
      },
    ],
  };

  const options = {
    plugins: {
      zoom: {
        zoom: {
          wheel: {
            enabled: true,
          },
          pinch: {
            enabled: true,
          },
          mode: 'xy',
        },
        pan: {
          enabled: true,
          mode: 'xy',
        },
      },
    },
    onClick: (event, elements) => {
      if (elements.length > 0) {
        const elementIndex = elements[0].index;
        console.log('Clicked node:', nodes[elementIndex]);
      }
    }
  };

  return <Chart ref={chartRef} type='scatter' data={data} options={options} />;
};


// 4. Main App Component
function App() {
  return (
    <div style={{ display: 'flex' }}>
      <Sidebar />
      <div style={{ flex: 1 }}>
        <Plot />
      </div>
    </div>
  );
}

export default App;
