import React, { useEffect, useRef, useState } from 'react';
import { Stage, Layer, Circle, Line, Text } from 'react-konva';
import { useStore } from '../store';

// Colors for lanes (simple palette)
const COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'];

const CanvasArea = () => {
  const {
    nodes, edges, mode,
    fetchData, handleNodeClick, addNode, deleteSelected,
    selectedNodeIds, operationStartId, operationEndId, previewPoints,
    hoveredNodeId
  } = useStore();

  const stageRef = useRef(null);
  const [scale, setScale] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Keyboard Shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Delete') deleteSelected();
      // Add Undo/Redo etc here
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [deleteSelected]);


  const handleWheel = (e) => {
    e.evt.preventDefault();
    const scaleBy = 1.1;
    const stage = stageRef.current;
    const oldScale = stage.scaleX();
    const mousePointTo = {
      x: stage.getPointerPosition().x / oldScale - stage.x() / oldScale,
      y: stage.getPointerPosition().y / oldScale - stage.y() / oldScale,
    };

    const newScale = e.evt.deltaY < 0 ? oldScale * scaleBy : oldScale / scaleBy;
    setScale(newScale);
    setPosition({
      x: -(mousePointTo.x - stage.getPointerPosition().x / newScale) * newScale,
      y: -(mousePointTo.y - stage.getPointerPosition().y / newScale) * newScale,
    });
  };

  const handleStageClick = (e) => {
    // If clicked on empty space
    if (e.target === stageRef.current) {
      // Handle Ctrl+Click for adding node
      if (e.evt.ctrlKey) {
         // Transform pointer pos to local coords
         const stage = stageRef.current;
         const transform = stage.getAbsoluteTransform().copy();
         transform.invert();
         const pos = transform.point(stage.getPointerPosition());

         // Find nearest node to connect to if any (simple heuristic or reuse backend logic)
         // For now, passing null for connection, or we could track 'last selected'
         // The original app connects to the "nearest existing node".
         // We can implement that logic on backend or simple frontend distance check.
         // Let's just add a node for now.
         const defaultLaneId = 0; // Or selected lane
         addNode(pos.x, pos.y, defaultLaneId);
      } else {
        // Deselect
        useStore.getState().selectNode(null);
      }
    }
  };

  // --- Rendering Helpers ---

  // Prepare edges for rendering
  // Map edge [from, to] to coordinates
  const renderedEdges = edges.map((edge, i) => {
      const fromNode = nodes.find(n => n[0] === edge[0]);
      const toNode = nodes.find(n => n[0] === edge[1]);
      if (!fromNode || !toNode) return null;
      return (
          <Line
            key={`edge-${i}`}
            points={[fromNode[1], fromNode[2], toNode[1], toNode[2]]}
            stroke="black"
            strokeWidth={1 / scale} // Keep lines thin when zoomed
            opacity={0.3}
            listening={false} // Pass through clicks
          />
      );
  });

  // Prepare nodes
  // Node: [id, x, y, yaw, lane_id]
  const renderedNodes = nodes.map((node) => {
      const id = node[0];
      const x = node[1];
      const y = node[2];
      const laneId = node[4];
      const isSelected = selectedNodeIds.includes(id);
      const isStart = id === operationStartId;
      const isEnd = id === operationEndId;

      let radius = 4 / scale; // Base size
      let fill = COLORS[laneId % COLORS.length];

      if (isSelected) { radius = 8 / scale; fill = 'red'; }
      if (isStart) { radius = 10 / scale; fill = 'green'; }
      if (isEnd) { radius = 10 / scale; fill = 'orange'; }

      return (
          <Circle
            key={`node-${id}`}
            x={x} y={y}
            radius={radius}
            fill={fill}
            onClick={(e) => {
                e.cancelBubble = true; // Stop propagation to Stage
                handleNodeClick(id);
            }}
            onTap={(e) => {
                e.cancelBubble = true;
                handleNodeClick(id);
            }}
            // Tooltip logic handled via store or simple local state
          />
      );
  });

  return (
    <div style={{ width: '100%', height: '100vh', background: '#f0f0f0' }}>
      <Stage
        width={window.innerWidth - 300} // Subtract sidebar width
        height={window.innerHeight}
        onWheel={handleWheel}
        scaleX={scale}
        scaleY={scale}
        x={position.x}
        y={position.y}
        draggable
        ref={stageRef}
        onClick={handleStageClick}
      >
        <Layer>
            {/* Edges First (Bottom Layer) */}
            {renderedEdges}

            {/* Preview Line for Smoothing */}
            {previewPoints.length > 0 && (
                <Line
                    points={previewPoints.flat()}
                    stroke="blue"
                    strokeWidth={2 / scale}
                    dash={[10, 5]}
                />
            )}

            {/* Nodes (Top Layer) */}
            {renderedNodes}
        </Layer>
      </Stage>
    </div>
  );
};

export default CanvasArea;
