import React from 'react';

const Toolbar = ({
    onDelete,
    onUndo,
    onRedo,
    onSave,
    onAddEdge,
    onReverse,
    onSmooth,
    selectionCount,
    pointSize,
    setPointSize,
    smoothness,
    setSmoothness,
    smoothWeight,
    setSmoothWeight
}) => {
    return (
        <>
            <div className="section">
                <h4>History</h4>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                    <button onClick={onUndo}>Undo</button>
                    <button onClick={onRedo}>Redo</button>
                </div>
            </div>

            <div className="section">
                <h4>File</h4>
                <button className="primary" onClick={onSave}>Save Changes</button>
            </div>

            <div className="section">
                <h4>Edit</h4>
                <button
                    className="danger"
                    onClick={onDelete}
                    disabled={selectionCount === 0}
                >
                    Delete Selected ({selectionCount})
                </button>

                <button
                    onClick={onAddEdge}
                    disabled={selectionCount !== 2}
                    title="Select exactly 2 nodes to connect"
                >
                    Add Edge
                </button>

                <button
                    onClick={onReverse}
                    disabled={selectionCount !== 2}
                    title="Select 2 nodes to reverse path between them"
                >
                    Reverse Path
                </button>

                <button
                    onClick={onSmooth}
                    disabled={selectionCount !== 2}
                    title="Select 2 nodes to smooth path between them"
                >
                    Smooth Path
                </button>
            </div>

            <div className="section">
                <h4>View Settings</h4>
                <div className="slider-container">
                    <label>
                        Point Size
                        <span>{pointSize}px</span>
                    </label>
                    <input
                        type="range"
                        min="4"
                        max="20"
                        value={pointSize}
                        onChange={(e) => setPointSize(parseInt(e.target.value))}
                    />
                </div>
                <div className="slider-container">
                    <label>
                        Smoothness
                        <span>{smoothness.toFixed(1)}</span>
                    </label>
                    <input
                        type="range"
                        min="0.1"
                        max="10.0"
                        step="0.1"
                        value={smoothness}
                        onChange={(e) => setSmoothness(parseFloat(e.target.value))}
                    />
                </div>
                <div className="slider-container">
                    <label>
                        Smooth Weight
                        <span>{smoothWeight.toFixed(0)}</span>
                    </label>
                    <input
                        type="range"
                        min="1"
                        max="100"
                        step="1"
                        value={smoothWeight}
                        onChange={(e) => setSmoothWeight(parseFloat(e.target.value))}
                    />
                </div>
            </div>

            <div className="section">
                <h4>Instructions</h4>
                <div className="instructions">
                    <div><b>Pan:</b> Click & Drag</div>
                    <div><b>Zoom:</b> Scroll</div>
                    <div><b>Select:</b> Click node</div>
                    <div><b>Multi:</b> Shift+Click</div>
                    <div><b>Right-Click:</b> Delete point</div>
                    <div><b>Delete:</b> Del/Backspace</div>
                    <div><b>Undo:</b> Ctrl+Z</div>
                </div>
            </div>
        </>
    );
};

export default Toolbar;
