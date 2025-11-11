import React from 'react';
import axios from 'axios';
import './Sidebar.css';

const Sidebar = ({
  drawMode, toggleDrawMode,
  lineMode, toggleLineMode,
  connectMode, toggleConnectMode,
  removeBetweenMode, toggleRemoveBetweenMode,
  reversePathMode, toggleReversePathMode,
  smoothMode, toggleSmoothMode,
  data, clearSelection,
  selectedNodes,
  toggleGrid, pointSize, handlePointSizeChange,
  smoothness, handleSmoothnessChange,
  smoothingWeight, handleSmoothingWeightChange,
  cancelOperation
}) => {
  const handleSave = () => {
    if (data) {
      axios.post('/api/save', { nodes: data.nodes, edges: data.edges })
        .then(response => {
          console.log('Data saved successfully:', response.data);
          alert('Data saved!');
        })
        .catch(error => {
          console.error('Error saving data:', error);
          alert('Error saving data.');
        });
    }
  };

  const handleExport = () => {
    if (data && selectedNodes.length > 0) {
      const nodesToExport = data.nodes.filter(node => selectedNodes.includes(node[0]));
      const csvContent = "data:text/csv;charset=utf-8,"
        + "node_id,x,y,yaw,lane_id\n"
        + nodesToExport.map(e => e.join(",")).join("\n");
      const encodedUri = encodeURI(csvContent);
      const link = document.createElement("a");
      link.setAttribute("href", encodedUri);
      link.setAttribute("download", "selected_nodes.csv");
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  return (
    <div className="sidebar">
      <div className="sidebar-section">
        <button
          onClick={toggleDrawMode}
          className={drawMode ? 'active' : ''}
        >
          Draw
        </button>
        <button
          onClick={toggleLineMode}
          className={lineMode ? 'active' : ''}
        >
          Line
        </button>
        <button
          onClick={toggleSmoothMode}
          className={smoothMode ? 'active' : ''}
        >
          Smooth
        </button>
      </div>
      <div className="sidebar-section">
        <button onClick={cancelOperation}>Cancel Operation</button>
        <button onClick={clearSelection}>Clear Selection</button>
        <button onClick={handleSave}>Save</button>
        <button
          onClick={toggleConnectMode}
          className={connectMode ? 'active' : ''}
        >
          Connect Nodes
        </button>
        <button onClick={handleExport}>Export Selected</button>
        <button onClick={toggleGrid}>Toggle Grid</button>
        <button
          onClick={toggleRemoveBetweenMode}
          className={removeBetweenMode ? 'active' : ''}
        >
          Remove Between
        </button>
        <button
          onClick={toggleReversePathMode}
          className={reversePathMode ? 'active' : ''}
        >
          Reverse Path
        </button>
      </div>
      <div className="sidebar-section">
        <label>Smoothness</label>
        <input
          type="range"
          min="0.1"
          max="30"
          step="0.1"
          value={smoothness}
          onChange={(e) => handleSmoothnessChange(Number(e.target.value))}
        />
        <label>Smoothing Weight</label>
        <input
          type="range"
          min="1"
          max="100"
          value={smoothingWeight}
          onChange={(e) => handleSmoothingWeightChange(Number(e.target.value))}
        />
        <label>Point Size</label>
        <input
          type="range"
          min="1"
          max="10"
          value={pointSize}
          onChange={(e) => handlePointSizeChange(Number(e.target.value))}
        />
      </div>
    </div>
  );
};

export default Sidebar;
