import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';

const PlotComponent = ({
    data,
    selectedPoints,
    setSelectedPoints,
    onRightClickDelete,
    pointSize
}) => {
    const { nodes, edges, file_names } = data;

    // Define distinct colors for lanes
    const laneColors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ];

    const traces = useMemo(() => {
        const plotTraces = [];
        const nodeMap = new Map();
        nodes.forEach(n => nodeMap.set(n.id, n));

        // Identify start nodes
        const toIds = new Set(edges.map(e => e.to_id));
        const startNodes = nodes.filter(n => !toIds.has(n.id));

        // 1. Edges
        const edgeX = [];
        const edgeY = [];
        edges.forEach(edge => {
            const fromNode = nodeMap.get(edge.from_id);
            const toNode = nodeMap.get(edge.to_id);
            if (fromNode && toNode) {
                edgeX.push(fromNode.x, toNode.x, null);
                edgeY.push(fromNode.y, toNode.y, null);
            }
        });

        plotTraces.push({
            x: edgeX,
            y: edgeY,
            mode: 'lines',
            type: 'scatter',
            line: { color: '#888', width: 1 },
            hoverinfo: 'none',
            name: 'Edges',
            showlegend: false
        });

        // 2. Nodes (by lane)
        const uniqueLanes = [...new Set(nodes.map(n => n.lane_id))];

        uniqueLanes.forEach(laneId => {
            const laneNodes = nodes.filter(n => n.lane_id === laneId);
            const isSelected = (id) => selectedPoints.includes(id);
            const laneColor = laneColors[laneId % laneColors.length];

            plotTraces.push({
                x: laneNodes.map(n => n.x),
                y: laneNodes.map(n => n.y),
                customdata: laneNodes.map(n => ({ id: n.id, lane: n.lane_id })),
                mode: 'markers',
                type: 'scatter',
                marker: {
                    size: laneNodes.map(n => isSelected(n.id) ? pointSize + 4 : pointSize),
                    color: laneNodes.map(n => isSelected(n.id) ? '#ff4444' : laneColor),
                    line: {
                        color: '#000',
                        width: laneNodes.map(n => isSelected(n.id) ? 2 : 0.5)
                    }
                },
                name: file_names[laneId] || `Lane ${laneId}`,
                text: laneNodes.map(n => `ID: ${n.id}<br>Lane: ${n.lane_id}<br>X: ${n.x.toFixed(2)}<br>Y: ${n.y.toFixed(2)}`),
                hoverinfo: 'text'
            });
        });

        // 3. Start Points
        if (startNodes.length > 0) {
            const startColors = startNodes.map(n => laneColors[n.lane_id % laneColors.length]);

            plotTraces.push({
                x: startNodes.map(n => n.x),
                y: startNodes.map(n => n.y),
                customdata: startNodes.map(n => ({ id: n.id, lane: n.lane_id })),
                mode: 'markers',
                type: 'scatter',
                marker: {
                    symbol: 'square',
                    size: pointSize + 6,
                    color: startColors,
                    line: { color: '#000', width: 2 }
                },
                name: 'Start Points',
                hoverinfo: 'none',
                showlegend: false
            });
        }

        return plotTraces;
    }, [nodes, edges, file_names, selectedPoints, pointSize, laneColors]);

    const layout = {
        autosize: true,
        hovermode: 'closest',
        dragmode: 'pan',
        showlegend: true,
        legend: {
            x: 1,
            xanchor: 'right',
            y: 1,
            font: { color: '#333' },
            bgcolor: 'rgba(255,255,255,0.8)',
            bordercolor: '#ddd',
            borderwidth: 1
        },
        xaxis: {
            title: 'X',
            scaleanchor: 'y',
            scaleratio: 1,
            gridcolor: '#e0e0e0',
            zerolinecolor: '#999',
            tickfont: { color: '#555' },
            titlefont: { color: '#333' }
        },
        yaxis: {
            title: 'Y',
            gridcolor: '#e0e0e0',
            zerolinecolor: '#999',
            tickfont: { color: '#555' },
            titlefont: { color: '#333' }
        },
        margin: { t: 20, l: 50, r: 20, b: 50 },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        clickmode: 'event+select'
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToAdd: ['select2d', 'lasso2d'],
        displaylogo: false
    };

    const handleClick = (event) => {
        if (event.points && event.points[0] && event.points[0].data.type === 'scatter' && event.points[0].data.mode === 'markers') {
            const point = event.points[0];
            const id = point.customdata ? point.customdata.id : null;

            if (id !== null) {
                if (event.event.ctrlKey || event.event.shiftKey) {
                    if (selectedPoints.includes(id)) {
                        setSelectedPoints(selectedPoints.filter(p => p !== id));
                    } else {
                        setSelectedPoints([...selectedPoints, id]);
                    }
                } else {
                    setSelectedPoints([id]);
                }
            }
        }
    };

    // Handle right-click via DOM event (Plotly doesn't support onContextMenu directly)
    React.useEffect(() => {
        const plotDiv = document.getElementById('plotly-graph');
        if (!plotDiv) return;

        const handleContextMenu = (e) => {
            e.preventDefault();

            // We need to find the point at this location
            // This is tricky with Plotly. We can use the current hover state or selected
            // For simplicity, we'll delete the last selected point if there's only one selected
            if (selectedPoints.length === 1) {
                onRightClickDelete(selectedPoints[0]);
            }
        };

        plotDiv.addEventListener('contextmenu', handleContextMenu);
        return () => plotDiv.removeEventListener('contextmenu', handleContextMenu);
    }, [selectedPoints, onRightClickDelete]);

    return (
        <Plot
            data={traces}
            layout={layout}
            config={config}
            style={{ width: '100%', height: '100%' }}
            onClick={handleClick}
            onSelected={(e) => {
                if (e && e.points) {
                    const ids = e.points.map(p => p.customdata ? p.customdata.id : null).filter(id => id !== null);
                    setSelectedPoints(ids);
                }
            }}
            useResizeHandler={true}
            divId="plotly-graph"
        />
    );
};

export default PlotComponent;
