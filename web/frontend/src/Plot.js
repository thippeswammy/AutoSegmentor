import React, { useEffect, useState, useRef } from 'react';
import axios from 'axios';
import * as d3 from 'd3';

const Plot = () => {
    const [data, setData] = useState(null);
    const [nodes, setNodes] = useState([]);
    const [selectedNodes, setSelectedNodes] = useState(new Set());
    const svgRef = useRef();
    const gRef = useRef();
    const tooltipRef = useRef();

    useEffect(() => {
        axios.get('http://127.0.0.1:5000/api/data')
            .then(response => {
                setData(response.data);
                setNodes(response.data.nodes);
            })
            .catch(error => {
                console.error('Error fetching data:', error);
            });
    }, []);

    useEffect(() => {
        if (data) {
            const { edges } = data;
            const svg = d3.select(svgRef.current);
            const g = d3.select(gRef.current);
            const tooltip = d3.select(tooltipRef.current);
            g.selectAll('*').remove();

            const width = 800;
            const height = 600;
            svg.attr('width', width).attr('height', height);

            const xExtent = d3.extent(nodes, d => d[1]) || [0, 1];
            const yExtent = d3.extent(nodes, d => d[2]) || [0, 1];

            const xScale = d3.scaleLinear().domain(xExtent).range([50, width - 50]);
            const yScale = d3.scaleLinear().domain(yExtent).range([height - 50, 50]);

            const nodeCoords = {};
            nodes.forEach(node => {
                nodeCoords[node[0]] = { x: xScale(node[1]), y: yScale(node[2]) };
            });

            g.selectAll('line')
                .data(edges)
                .enter()
                .append('line')
                .attr('x1', d => nodeCoords[d[0]]?.x)
                .attr('y1', d => nodeCoords[d[0]]?.y)
                .attr('x2', d => nodeCoords[d[1]]?.x)
                .attr('y2', d => nodeCoords[d[1]]?.y)
                .attr('stroke', 'black')
                .attr('stroke-width', 1);

            const drag = d3.drag()
                .on('start', (event, d) => {
                    d3.select(event.sourceEvent.target).raise().attr('stroke', 'black');
                })
                .on('drag', (event, d) => {
                    const newX = xScale.invert(event.x);
                    const newY = yScale.invert(event.y);
                    const newNodes = nodes.map(node => {
                        if (node[0] === d[0]) {
                            return [node[0], newX, newY, node[3], node[4]];
                        }
                        return node;
                    });
                    setNodes(newNodes);
                })
                .on('end', (event, d) => {
                    d3.select(event.sourceEvent.target).attr('stroke', null);
                });

            g.selectAll('circle')
                .data(nodes)
                .enter()
                .append('circle')
                .attr('cx', d => xScale(d[1]))
                .attr('cy', d => yScale(d[2]))
                .attr('r', 5)
                .attr('fill', d => selectedNodes.has(d[0]) ? 'red' : 'blue')
                .on('click', (event, d) => {
                    const newSelectedNodes = new Set(selectedNodes);
                    if (newSelectedNodes.has(d[0])) {
                        newSelectedNodes.delete(d[0]);
                    } else {
                        newSelectedNodes.add(d[0]);
                    }
                    setSelectedNodes(newSelectedNodes);
                })
                .on('mouseover', (event, d) => {
                    tooltip.style('visibility', 'visible')
                           .html(`Node ID: ${d[0]}<br>X: ${d[1].toFixed(2)}<br>Y: ${d[2].toFixed(2)}`)
                           .style('left', `${event.pageX + 10}px`)
                           .style('top', `${event.pageY + 10}px`);
                })
                .on('mouseout', () => {
                    tooltip.style('visibility', 'hidden');
                })
                .call(drag);

            const zoom = d3.zoom()
                .scaleExtent([0.1, 10])
                .on('zoom', (event) => {
                    g.attr('transform', event.transform);
                });

            svg.call(zoom);
        }
    }, [data, nodes, selectedNodes]);

    const handleSave = () => {
        const newEdges = data.edges.filter(edge => {
            const fromNodeExists = nodes.some(node => node[0] === edge[0]);
            const toNodeExists = nodes.some(node => node[0] === edge[1]);
            return fromNodeExists && toNodeExists;
        });

        axios.post('http://127.0.0.1:5000/api/save', { nodes: nodes, edges: newEdges })
            .then(response => {
                console.log('Data saved successfully');
            })
            .catch(error => {
                console.error('Error saving data:', error);
            });
    };

    const handleDelete = () => {
        const newNodes = nodes.filter(node => !selectedNodes.has(node[0]));
        const newEdges = data.edges.filter(edge => !selectedNodes.has(edge[0]) && !selectedNodes.has(edge[1]));

        setNodes(newNodes);
        setData({ ...data, edges: newEdges });
        setSelectedNodes(new Set());
    };

    useEffect(() => {
        const handleKeyDown = (event) => {
            if (event.key === 'Delete') {
                handleDelete();
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => {
            window.removeEventListener('keydown', handleKeyDown);
        };
    }, [selectedNodes, nodes, data]);

    return (
        <div>
            <h1>Lane Visualization</h1>
            <button onClick={handleSave}>Save</button>
            <button onClick={handleDelete}>Delete Selected</button>
            <div ref={tooltipRef} style={{ position: 'absolute', visibility: 'hidden', backgroundColor: 'white', border: '1px solid black', padding: '5px' }}></div>
            <svg ref={svgRef}>
                <g ref={gRef}></g>
            </svg>
        </div>
    );
};

export default Plot;
