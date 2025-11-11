import React, { useState, useEffect, useRef } from 'react';
import { Chart as ChartJS, registerables } from 'chart.js';
import { Chart } from 'react-chartjs-2';
import 'chart.js/auto';
import zoomPlugin from 'chartjs-plugin-zoom';

ChartJS.register(...registerables, zoomPlugin);
import axios from 'axios';
import create from 'zustand';

// 1. State Management with Zustand
const useStore = create((set) => ({
  nodes: [],
  edges: [],
  fileNames: [],
  selectedNodes: [],
  mode: 'select', // select, draw, smooth_start, smooth_end, etc.

  // Actions
  fetchData: async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/data');
      set({ nodes: response.data.nodes, edges: response.data.edges, fileNames: response.data.file_names });
    } catch (error) {
      console.error("Error fetching data:", error);
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
      {/* Add more buttons for other modes as needed */}
    </div>
  );
};

// 3. Plot Component
const Plot = () => {
  const { nodes, edges, fetchData } = useStore();
  const chartRef = useRef(null);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const data = {
    datasets: [
      {
        label: 'Nodes',
        data: nodes.map(node => ({ x: node[1], y: node[2] })),
        backgroundColor: 'rgba(255, 99, 132, 1)',
        pointRadius: 5,
        type: 'scatter',
      },
      // Edges will be drawn on the canvas directly
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
      // Handle click events on the chart
      if (elements.length > 0) {
        const elementIndex = elements[0].index;
        // You can now access the clicked node's data
        console.log('Clicked node:', nodes[elementIndex]);
      }
    }
  };

  useEffect(() => {
    const chart = chartRef.current;
    if (chart) {
      // const ctx = chart.ctx;

      // // Custom drawing for edges
      // const drawEdges = () => {
      //   edges.forEach(edge => {
      //     const fromNode = nodes.find(n => n[0] === edge[0]);
      //     const toNode = nodes.find(n => n[0] === edge[1]);

      //     if (fromNode && toNode) {
      //       ctx.beginPath();
      //       ctx.moveTo(chart.scales.x.getPixelForValue(fromNode[1]), chart.scales.y.getPixelForValue(fromNode[2]));
      //       ctx.lineTo(chart.scales.x.getPixelForValue(toNode[1]), chart.scales.y.getPixelForValue(toNode[2]));
      //       ctx.stroke();
      //     }
      //   });
      // };

      // // Clear and redraw edges on each render
      // chart.clear();
      // chart.update();
      // drawEdges();
    }
  }, [nodes, edges]);


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
