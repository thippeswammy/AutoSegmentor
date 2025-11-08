import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Scatter } from 'react-chartjs-2';
import { Chart, LinearScale, PointElement } from 'chart.js';
import zoomPlugin from 'chartjs-plugin-zoom';

import './App.css';

Chart.register(zoomPlugin, LinearScale, PointElement);

function App() {
  const [data, setData] = useState({ nodes: [], edges: [] });
  const [smoothness, setSmoothness] = useState(0.5);
  const [smoothingWeight, setSmoothingWeight] = useState(50);
  const [pointSize, setPointSize] = useState(1);
  const [smoothedData, setSmoothedData] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
    const chartRef = useRef(null);

  useEffect(() => {
    axios.get('http://127.0.0.1:5000/api/data')
      .then(response => {
        setData(response.data);
      })
      .catch(error => {
        console.error('Error fetching data:', error);
      });
  }, []);

  
  const handleSaveClick = () => {
    axios.post('http://127.0.0.1:5000/api/save')
      .then(response => {
        alert(response.data.message);
      })
      .catch(error => {
        console.error('Error saving data:', error);
        alert('Error saving data. See console for details.');
      });
  };

  const handleChartRightClick = (event) => {
    event.preventDefault();
    const chart = chartRef.current;
    if (!chart) {
      return;
    }
    const elements = chart.getElementsAtEventForMode(event, 'point', { intersect: true, tolerance: 1 }, true);

    if (elements.length > 0) {
      const firstElement = elements[0];
      const { index } = firstElement;
      const clickedNode = data.nodes[index];
      const nodeId = clickedNode[0];

      axios.post('http://127.0.0.1:5000/api/delete_node', { node_id: nodeId })
        .then(() => {
          // Refetch data to reflect the deletion
          axios.get('http://127.0.0.1:5000/api/data').then(response => setData(response.data));
        })
        .catch(error => console.error('Error deleting node:', error));
    }
  };

  
    const onChartClick = (event) => {
    const chart = chartRef.current;
    if (!chart) return;

    const elements = chart.getElementsAtEventForMode(event, 'point', { intersect: true, tolerance: 1 }, true);

    if (elements.length > 0) {
      const firstElement = elements[0];
      const { index } = firstElement;
      const clickedNode = data.nodes[index];
      const nodeId = clickedNode[0];

      if (selectedNode && selectedNode.id === nodeId) {
        setSelectedNode(null); // Deselect if clicking the same node
      } else {
        setSelectedNode({ id: nodeId, index: index });
      }
    } else {
      setSelectedNode(null); // Deselect if clicking away from any node
    }
  };

  const datasets = [
    {
      label: 'Lane Data',
      data: data.nodes.map(node => ({ x: node[1], y: node[2] })),
      backgroundColor: (context) => {
        if (selectedNode && context.dataIndex === selectedNode.index) {
          return 'yellow'; // Highlight selected point
        }
        return 'rgba(75,192,192,0.4)';
      },
      pointBorderColor: 'rgba(75,192,192,1)',
      pointRadius: pointSize,
    },
  ];

  if (smoothedData) {
    datasets.push({
      label: 'Smoothed Path',
      data: smoothedData.map(point => ({ x: point[0], y: point[1] })),
      borderColor: 'rgba(255, 99, 132, 1)',
      backgroundColor: 'transparent',
      showLine: true,
      pointRadius: 0,
    });
  }

  const chartOptions = {
    interaction: {
      mode: 'point',
      intersect: true,
    },
    plugins: {
      tooltip: {
        callbacks: {
          label: function(context) {
            const point = data.nodes[context.dataIndex];
            if (!point) return '';
            const nodeId = point[0];
            const x = point[1].toFixed(2);
            const y = point[2].toFixed(2);
            return `ID: ${nodeId} (x: ${x}, y: ${y})`;
          }
        }
      },
      zoom: {
        pan: {
          enabled: true,
          mode: 'xy',
        },
        zoom: {
          wheel: {
            enabled: true,
          },
          pinch: {
            enabled: true
          },
          mode: 'xy',
        }
      }
    }
  };

  const chartData = { datasets };

  return (
    <div className="App">
      <div className="sidebar">
        <div className="controls-section">
          <h3>Editing Tools</h3>
          <button className="button">Draw</button>
          <button className="button">Line</button>
          <button className="button">
            Smooth
          </button>
        </div>
        <div className="controls-section">
          <h3>File Operations</h3>
          <button className="button">Cancel Operation</button>
          <button className="button">Clear Selection</button>
          <button className="button" onClick={handleSaveClick}>Save</button>
          <button className="button">Export Selected</button>
        </div>
        <div className="controls-section">
          <h3>View Options</h3>
          <button className="button">Toggle Grid</button>
          <button className="button">Remove Between</button>
          <button className="button">Reverse Path</button>
        </div>
        <div className="controls-section">
          <h3>Adjustments</h3>
          <div className="slider-container">
            <label>Smoothness: {smoothness}</label>
            <input 
              type="range" 
              min="0" 
              max="1" 
              step="0.1" 
              value={smoothness}
              onChange={(e) => setSmoothness(e.target.value)} 
              className="slider" 
            />
          </div>
          <div className="slider-container">
            <label>Smoothing Weight: {smoothingWeight}</label>
            <input 
              type="range" 
              min="1" 
              max="100" 
              value={smoothingWeight}
              onChange={(e) => setSmoothingWeight(e.target.value)} 
              className="slider" 
            />
          </div>
          <div className="slider-container">
            <label>Point Size: {pointSize}</label>
            <input 
              type="range" 
              min="1" 
              max="20" 
              value={pointSize}
              onChange={(e) => setPointSize(e.target.value)} 
              className="slider" 
            />
          </div>
        </div>
      </div>
      <div className="main-content">
                <h2>Lane Data Visualization</h2>
        <div onContextMenu={handleChartRightClick}>
          <Scatter ref={chartRef} data={chartData} options={chartOptions} onClick={onChartClick} />
        </div>
      </div>
    </div>
  );
}

export default App;
