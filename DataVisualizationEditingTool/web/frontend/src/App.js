import React, { useState, useEffect } from 'react';
import './App.css';
import Sidebar from './Sidebar';
import Plot from './Plot';
import * as d3 from 'd3';

function App() {
  const [drawMode, setDrawMode] = useState(false);
  const [lineMode, setLineMode] = useState(false);
  const [connectMode, setConnectMode] = useState(false);
  const [removeBetweenMode, setRemoveBetweenMode] = useState(false);
  const [reversePathMode, setReversePathMode] = useState(false);
  const [smoothMode, setSmoothMode] = useState(false);
  const [data, setData] = useState(null);
  const [selectedNodes, setSelectedNodes] = useState([]);
  const [gridVisible, setGridVisible] = useState(false);
  const [pointSize, setPointSize] = useState(5);
  const [smoothness, setSmoothness] = useState(1);
  const [smoothingWeight, setSmoothingWeight] = useState(20);
  const [highlightedLane, setHighlightedLane] = useState(null);

  const toggleDrawMode = () => {
    setDrawMode(!drawMode);
    setLineMode(false);
    setConnectMode(false);
    setRemoveBetweenMode(false);
    setReversePathMode(false);
    setSmoothMode(false);
    setSelectedNodes([]);
  };

  const toggleLineMode = () => {
    setLineMode(!lineMode);
    setDrawMode(false);
    setConnectMode(false);
    setRemoveBetweenMode(false);
    setReversePathMode(false);
    setSmoothMode(false);
    setSelectedNodes([]);
  };

  const toggleConnectMode = () => {
    setConnectMode(!connectMode);
    setDrawMode(false);
    setLineMode(false);
    setRemoveBetweenMode(false);
    setReversePathMode(false);
    setSmoothMode(false);
    setSelectedNodes([]);
  };

  const toggleRemoveBetweenMode = () => {
    setRemoveBetweenMode(!removeBetweenMode);
    setDrawMode(false);
    setLineMode(false);
    setConnectMode(false);
    setReversePathMode(false);
    setSmoothMode(false);
    setSelectedNodes([]);
  };

  const toggleReversePathMode = () => {
    setReversePathMode(!reversePathMode);
    setDrawMode(false);
    setLineMode(false);
    setConnectMode(false);
    setRemoveBetweenMode(false);
    setSmoothMode(false);
    setSelectedNodes([]);
  };

  const toggleSmoothMode = () => {
    setSmoothMode(!smoothMode);
    setDrawMode(false);
    setLineMode(false);
    setConnectMode(false);
    setRemoveBetweenMode(false);
    setReversePathMode(false);
    setSelectedNodes([]);
  };

  const toggleGrid = () => {
    setGridVisible(!gridVisible);
  };

  const handlePointSizeChange = (newSize) => {
    setPointSize(newSize);
  };

  const handleSmoothnessChange = (newSmoothness) => {
    setSmoothness(newSmoothness);
  };

  const handleSmoothingWeightChange = (newWeight) => {
    setSmoothingWeight(newWeight);
  };

  const handleHighlightLane = (laneId) => {
    setHighlightedLane(prevLane => (prevLane === laneId ? null : laneId));
  };

  const cancelOperation = () => {
    setDrawMode(false);
    setLineMode(false);
    setConnectMode(false);
    setRemoveBetweenMode(false);
    setReversePathMode(false);
    setSmoothMode(false);
    setSelectedNodes([]);
  };

  const findPath = (startNodeId, endNodeId, edges) => {
    const adj = new Map();
    for (const [u, v] of edges) {
      if (!adj.has(u)) adj.set(u, []);
      if (!adj.has(v)) adj.set(v, []);
      adj.get(u).push(v);
    }

    const queue = [[startNodeId]];
    const visited = new Set([startNodeId]);

    while (queue.length > 0) {
      const path = queue.shift();
      const node = path[path.length - 1];

      if (node === endNodeId) {
        return path;
      }

      for (const neighbor of adj.get(node) || []) {
        if (!visited.has(neighbor)) {
          visited.add(neighbor);
          const newPath = [...path, neighbor];
          queue.push(newPath);
        }
      }
    }
    return null;
  };

  const handleNodeSelect = (nodeId) => {
    setSelectedNodes(prevSelected => {
      if (connectMode || removeBetweenMode || reversePathMode || smoothMode || lineMode) {
        if (prevSelected.length === 1) {
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

  const handleConnectNodes = (nodeIds) => {
    if (nodeIds.length === 2) {
      const newEdge = [nodeIds[0], nodeIds[1]];
      setData(prevData => ({
        ...prevData,
        edges: [...prevData.edges, newEdge]
      }));
      setSelectedNodes([]);
    }
  };

  const handleRemoveBetween = (nodeIds) => {
    if (nodeIds.length === 2) {
      const path = findPath(nodeIds[0], nodeIds[1], data.edges);
      if (path) {
        const nodesToRemove = path.slice(1, -1);
        const nodeIdsToRemove = new Set(nodesToRemove);
        const newNodes = data.nodes.filter(node => !nodeIdsToRemove.has(node[0]));
        const newEdges = data.edges.filter(edge => !nodeIdsToRemove.has(edge[0]) && !nodeIdsToRemove.has(edge[1]));
        setData({ ...data, nodes: newNodes, edges: newEdges });
      }
      setSelectedNodes([]);
    }
  };

  const handleReversePath = (nodeIds) => {
    if (nodeIds.length === 2) {
      const path = findPath(nodeIds[0], nodeIds[1], data.edges);
      if (path) {
        const pathEdges = [];
        for (let i = 0; i < path.length - 1; i++) {
          pathEdges.push([path[i], path[i+1]]);
        }

        const newEdges = data.edges.filter(edge => !pathEdges.some(pathEdge => pathEdge[0] === edge[0] && pathEdge[1] === edge[1]));

        for (const edge of pathEdges) {
          newEdges.push([edge[1], edge[0]]);
        }

        setData({ ...data, edges: newEdges });
      }
      setSelectedNodes([]);
    }
  };

  const handleSmooth = (nodeIds) => {
    if (nodeIds.length === 2) {
      const path = findPath(nodeIds[0], nodeIds[1], data.edges);
      if (path) {
        const pathNodes = path.map(nodeId => data.nodes.find(n => n[0] === nodeId));
        const points = pathNodes.map(node => [node[1], node[2]]);

        const lineGenerator = d3.line().curve(d3.curveCatmullRom.alpha(smoothness / 30));
        const pathData = lineGenerator(points);

        const numNewPoints = Math.floor(points.length * (smoothingWeight / 20));
        const svgPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
        svgPath.setAttribute("d", pathData);
        const totalLength = svgPath.getTotalLength();

        const newNodes = [];
        let maxNodeId = Math.max(...data.nodes.map(n => n[0]));

        for (let i = 0; i < numNewPoints; i++) {
          const point = svgPath.getPointAtLength((i / (numNewPoints - 1)) * totalLength);
          newNodes.push([
            ++maxNodeId,
            point.x,
            point.y,
            0,
            pathNodes[0][4] // Lane ID
          ]);
        }

        const nodeIdsToRemove = new Set(path.slice(1, -1));
        const filteredNodes = data.nodes.filter(node => !nodeIdsToRemove.has(node[0]));
        const finalNodes = [...filteredNodes, ...newNodes];

        const edgeIdsToRemove = new Set(path);
        const filteredEdges = data.edges.filter(edge => !edgeIdsToRemove.has(edge[0]) || !edgeIdsToRemove.has(edge[1]));
        const newEdges = [...filteredEdges];

        let prevNodeId = path[0];
        for (const newNode of newNodes) {
          newEdges.push([prevNodeId, newNode[0]]);
          prevNodeId = newNode[0];
        }
        newEdges.push([prevNodeId, path[path.length - 1]]);

        setData({ nodes: finalNodes, edges: newEdges });
      }
      setSelectedNodes([]);
    }
  };

  useEffect(() => {
    if (selectedNodes.length === 2) {
      if (connectMode) handleConnectNodes(selectedNodes);
      if (removeBetweenMode) handleRemoveBetween(selectedNodes);
      if (reversePathMode) handleReversePath(selectedNodes);
      if (smoothMode) handleSmooth(selectedNodes);
    }
  }, [selectedNodes, connectMode, removeBetweenMode, reversePathMode, smoothMode, handleConnectNodes, handleRemoveBetween, handleReversePath, handleSmooth]);

  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.key === 'Escape') {
        cancelOperation();
      } else if (event.key === 'Delete') {
        if (selectedNodes.length > 0) {
          const newNodes = data.nodes.filter(node => !selectedNodes.includes(node[0]));
          const newEdges = data.edges.filter(edge => !selectedNodes.includes(edge[0]) && !selectedNodes.includes(edge[1]));
          setData({ ...data, nodes: newNodes, edges: newEdges });
          setSelectedNodes([]);
        }
      } else if (event.key === 'd') {
        toggleDrawMode();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [selectedNodes, data, cancelOperation, toggleDrawMode]);

  const clearSelection = () => {
    setSelectedNodes([]);
  };

  return (
    <div className="App">
      <Sidebar
        drawMode={drawMode}
        toggleDrawMode={toggleDrawMode}
        lineMode={lineMode}
        toggleLineMode={toggleLineMode}
        connectMode={connectMode}
        toggleConnectMode={toggleConnectMode}
        removeBetweenMode={removeBetweenMode}
        toggleRemoveBetweenMode={toggleRemoveBetweenMode}
        reversePathMode={reversePathMode}
        toggleReversePathMode={toggleReversePathMode}
        smoothMode={smoothMode}
        toggleSmoothMode={toggleSmoothMode}
        data={data}
        selectedNodes={selectedNodes}
        clearSelection={clearSelection}
        toggleGrid={toggleGrid}
        pointSize={pointSize}
        handlePointSizeChange={handlePointSizeChange}
        smoothness={smoothness}
        handleSmoothnessChange={handleSmoothnessChange}
        smoothingWeight={smoothingWeight}
        handleSmoothingWeightChange={handleSmoothingWeightChange}
        cancelOperation={cancelOperation}
      />
      <Plot
        drawMode={drawMode}
        lineMode={lineMode}
        connectMode={connectMode}
        removeBetweenMode={removeBetweenMode}
        reversePathMode={reversePathMode}
        smoothMode={smoothMode}
        data={data}
        setData={setData}
        selectedNodes={selectedNodes}
        handleNodeSelect={handleNodeSelect}
        gridVisible={gridVisible}
        pointSize={pointSize}
        highlightedLane={highlightedLane}
        handleHighlightLane={handleHighlightLane}
      />
    </div>
  );
}

export default App;
