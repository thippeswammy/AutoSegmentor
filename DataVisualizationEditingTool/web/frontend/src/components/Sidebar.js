import React from 'react';
import { useStore } from '../store';

const Sidebar = () => {
  const {
    mode, setMode,
    smoothness, weight, setSmoothParams, confirmSmooth,
    undo, redo, save,
    operationStep
  } = useStore();

  const modes = [
    { id: 'SELECT', label: 'Select / Nav' },
    { id: 'DRAW', label: 'Draw (Mock)' },
    { id: 'SMOOTH', label: 'Smooth Path' },
    { id: 'CONNECT', label: 'Connect Nodes' },
    { id: 'REMOVE_BETWEEN', label: 'Remove Between' },
    { id: 'REVERSE_PATH', label: 'Reverse Path' },
  ];

  return (
    <div style={{
      width: '300px',
      height: '100vh',
      background: '#333',
      color: 'white',
      padding: '20px',
      boxSizing: 'border-box',
      display: 'flex',
      flexDirection: 'column',
      gap: '10px'
    }}>
      <h2>Data Editor</h2>

      <div style={{ display: 'flex', gap: '10px', marginBottom: '20px' }}>
        <button onClick={save}>Save</button>
        <button onClick={undo}>Undo</button>
        <button onClick={redo}>Redo</button>
      </div>

      <h3>Modes</h3>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
        {modes.map(m => (
          <button
            key={m.id}
            onClick={() => setMode(m.id)}
            style={{
               background: mode === m.id ? '#4CAF50' : '#555',
               color: 'white',
               border: 'none',
               padding: '10px',
               cursor: 'pointer',
               textAlign: 'left'
            }}
          >
            {m.label}
          </button>
        ))}
      </div>

      {mode === 'SMOOTH' && (
        <div style={{ marginTop: '20px', borderTop: '1px solid #555', paddingTop: '10px' }}>
            <h4>Smooth Settings</h4>
            <label>
                Smoothness ({smoothness})
                <input
                    type="range" min="0.1" max="30" step="0.1"
                    value={smoothness}
                    onChange={(e) => setSmoothParams(parseFloat(e.target.value), weight)}
                    style={{ width: '100%' }}
                />
            </label>
            <label>
                Weight ({weight})
                <input
                    type="range" min="1" max="100" step="1"
                    value={weight}
                    onChange={(e) => setSmoothParams(smoothness, parseInt(e.target.value))}
                    style={{ width: '100%' }}
                />
            </label>

            {operationStep === 2 && (
                <button
                    onClick={confirmSmooth}
                    style={{ marginTop: '10px', background: '#2196F3', color: 'white', border: 'none', padding: '10px', width: '100%' }}
                >
                    Confirm Smooth
                </button>
            )}
             <div style={{fontSize: '0.8em', color: '#aaa', marginTop: '5px'}}>
                {operationStep === 0 && "Click START node"}
                {operationStep === 1 && "Click END node"}
                {operationStep === 2 && "Adjust & Confirm"}
            </div>
        </div>
      )}

      <div style={{ marginTop: 'auto', fontSize: '0.8em', color: '#aaa' }}>
        <p>Ctrl+Click: Add Node</p>
        <p>Delete: Remove Selected</p>
        <p>Wheel: Zoom</p>
      </div>
    </div>
  );
};

export default Sidebar;
