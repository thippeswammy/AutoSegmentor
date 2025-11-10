import React from 'react';
import axios from 'axios';
import './Sidebar.css';

const Sidebar = ({ drawMode, toggleDrawMode, data }) => {
  const handleSave = () => {
    if (data) {
      axios.post('/api/save', { nodes: data.nodes })
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

  return (
    <div className="sidebar">
      <div className="sidebar-section">
        <button
          onClick={toggleDrawMode}
          className={drawMode ? 'active' : ''}
        >
          Draw
        </button>
        <button>Line</button>
        <button>Smooth</button>
      </div>
      <div className="sidebar-section">
        <button>Cancel Operation</button>
        <button>Clear Selection</button>
        <button onClick={handleSave}>Save</button>
        <button>Connect Nodes</button>
        <button>Export Selected</button>
        <button>Toggle Grid</button>
        <button>Remove Between</button>
        <button>Reverse Path</button>
      </div>
      <div className="sidebar-section">
        <label>Smoothness</label>
        <input type="range" />
        <label>Smoothing Weight</label>
        <input type="range" />
        <label>Point Size</label>
        <input type="range" />
      </div>
    </div>
  );
};

export default Sidebar;
