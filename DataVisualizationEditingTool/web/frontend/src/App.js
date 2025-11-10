import React, { useState } from 'react';
import './App.css';
import Sidebar from './Sidebar';
import Plot from './Plot';

function App() {
  const [drawMode, setDrawMode] = useState(false);
  const [data, setData] = useState(null);

  const toggleDrawMode = () => {
    setDrawMode(!drawMode);
  };

  return (
    <div className="App">
      <Sidebar
        drawMode={drawMode}
        toggleDrawMode={toggleDrawMode}
        data={data}
      />
      <Plot
        drawMode={drawMode}
        setData={setData}
      />
    </div>
  );
}

export default App;
