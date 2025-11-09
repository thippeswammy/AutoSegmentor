import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import Plot from './Plot';

function App() {
  const [selectedNodes, setSelectedNodes] = useState(new Set());
  const [gridVisible, setGridVisible] = useState(false);
  const [pointSize, setPointSize] = useState(5);
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);
  const [fileNames, setFileNames] = useState([]);

  useEffect(() => {
    axios.get('/api/data')
        .then(response => {
            setNodes(response.data.nodes);
            setEdges(response.data.edges);
            setFileNames(response.data.file_names);
        })
        .catch(error => {
            console.error('Error fetching data:', error);
        });
  }, []);

  const handleConnectNodes = () => {
    const newEdges = [...edges];
    const selectedNodesArray = Array.from(selectedNodes);
    for (let i = 0; i < selectedNodesArray.length - 1; i++) {
      newEdges.push([selectedNodesArray[i], selectedNodesArray[i + 1]]);
    }
    setEdges(newEdges);
    setSelectedNodes(new Set());
  };

  const handleSave = () => {
    axios.post('/api/save', { nodes, edges })
        .then(response => {
            console.log('Data saved successfully');
        })
        .catch(error => {
            console.error('Error saving data:', error);
        });
  };

  return (
    <div className="App">
      <div className="sidebar">
        <button>Draw</button>
        <button>Line</button>
        <button>Smooth</button>
        <hr />
        <button>Cancel Operation</button>
        <button onClick={() => setSelectedNodes(new Set())}>Clear Selection</button>
        <button onClick={handleSave}>Save</button>
        <button onClick={handleConnectNodes}>Connect Nodes</button>
        <button>Export Selected</button>
        <button onClick={() => setGridVisible(!gridVisible)}>Toggle Grid</button>
        <button>Remove Between</button>
        <button>Reverse Path</button>
      </div>
      <div className="main-content">
        <h1>Lane Data Visualization</h1>
        <Plot
          selectedNodes={selectedNodes}
          setSelectedNodes={setSelectedNodes}
          gridVisible={gridVisible}
          pointSize={pointSize}
          nodes={nodes}
          setNodes={setNodes}
          edges={edges}
          setEdges={setEdges}
          fileNames={fileNames}
        />
        <div className="sliders">
          <label>
            Smoothness:
            <input type="range" min="0" max="100" />
          </label>
          <label>
            Smoothing Weight:
            <input type="range" min="0" max="100" />
          </label>
          <label>
            Point Size:
            <input type="range" min="1" max="100" value={pointSize} onChange={(e) => setPointSize(e.target.value)} />
          </label>
        </div>
      </div>
    </div>
  );
}

export default App;
