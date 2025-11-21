import React from 'react';
import Sidebar from './components/Sidebar';
import CanvasArea from './components/CanvasArea';

function App() {
  return (
    <div style={{ display: 'flex' }}>
      <Sidebar />
      <CanvasArea />
    </div>
  );
}

export default App;
