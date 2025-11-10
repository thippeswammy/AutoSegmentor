import React, { useState } from 'react';
import './App.css';
import Sidebar from './Sidebar';
import Plot from './Plot';

function App() {
  const [drawMode, setDrawMode] = useState(false);
  const [connectMode, setConnectMode] = useState(false);
  const [data, setData] = useState(null);
  const [selectedNodes, setSelectedNodes] = useState([]);

  const toggleDrawMode = () => {
    setDrawMode(!drawMode);
    setConnectMode(false);
    setSelectedNodes([]);
  };

  const toggleConnectMode = () => {
    setConnectMode(!connectMode);
    setDrawMode(false);
    setSelectedNodes([]);
  };

  const handleNodeSelect = (nodeId) => {
    setSelectedNodes(prevSelected => {
      if (connectMode) {
        if (prevSelected.length < 2) {
          return [...prevSelected, nodeId];
        } else {
          return [nodeId];
        }
      }
      if (prevSelected.includes(nodeId)) {
        return prevSelected.filter(id => id !== nodeId);
      } else {
        return [...prevSelected, nodeId];
      }
    });
  };

  const handleConnectNodes = () => {
    if (selectedNodes.length === 2) {
      const newEdge = [selectedNodes[0], selectedNodes[1]];
      setData(prevData => ({
        ...prevData,
        edges: [...prevData.edges, newEdge]
      }));
      setSelectedNodes([]);
    }
  };

  const clearSelection = () => {
    setSelectedNodes([]);
  };

  return (
    <div className="App">
      <Sidebar
        drawMode={drawMode}
        toggleDrawMode={toggleDrawMode}
        connectMode={connectMode}
        toggleConnectMode={toggleConnectMode}
        data={data}
        selectedNodes={selectedNodes}
        clearSelection={clearSelection}
        handleConnectNodes={handleConnectNodes}
      />
      <Plot
        drawMode={drawMode}
        connectMode={connectMode}
        data={data}
        setData={setData}
        selectedNodes={selectedNodes}
        handleNodeSelect={handleNodeSelect}
      />
    </div>
  );
}

export default App;
