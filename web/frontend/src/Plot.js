import React, { useEffect, useState, useRef } from 'react';
import axios from 'axios';
import * as d3 from 'd3';

const Plot = ({ selectedNodes, setSelectedNodes, gridVisible, pointSize, nodes, setNodes, edges, setEdges, fileNames }) => {
    const svgRef = useRef();
    const gRef = useRef();
    const tooltipRef = useRef();

    useEffect(() => {
        if (nodes) {
            const svg = d3.select(svgRef.current);
            const g = d3.select(gRef.current);
            const tooltip = d3.select(tooltipRef.current);
            g.selectAll('*').remove();

            const width = 800;
            const height = 600;
            const legendWidth = 150;
            svg.attr('width', width + legendWidth).attr('height', height);

            const xExtent = d3.extent(nodes, d => d[1]) || [0, 1];
            const yExtent = d3.extent(nodes, d => d[2]) || [0, 1];

            const xScale = d3.scaleLinear().domain(xExtent).range([50, width - 50]);
            const yScale = d3.scaleLinear().domain(yExtent).range([height - 50, 50]);

            const colorScale = d3.scaleOrdinal(d3.schemeCategory10);

            if (gridVisible) {
                const xAxis = d3.axisBottom(xScale);
                const yAxis = d3.axisLeft(yScale);
                g.append('g').attr('transform', `translate(0, ${height - 50})`).call(xAxis);
                g.append('g').attr('transform', `translate(50, 0)`).call(yAxis);
            }

            const nodeCoords = {};
            nodes.forEach(node => {
                nodeCoords[node[0]] = { x: xScale(node[1]), y: yScale(node[2]), lane: node[4] };
            });

            g.selectAll('line')
                .data(edges)
                .enter()
                .append('line')
                .attr('x1', d => nodeCoords[d[0]]?.x)
                .attr('y1', d => nodeCoords[d[0]]?.y)
                .attr('x2', d => nodeCoords[d[1]]?.x)
                .attr('y2', d => nodeCoords[d[1]]?.y)
                .attr('stroke', d => colorScale(nodeCoords[d[0]]?.lane))
                .attr('stroke-width', 2);

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
                .attr('r', pointSize)
                .attr('fill', d => selectedNodes.has(d[0]) ? 'red' : colorScale(d[4]))
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

            const toIds = new Set(edges.map(edge => edge[1]));
            const startNodes = nodes.filter(node => !toIds.has(node[0]));

            g.selectAll('rect.start-node')
                .data(startNodes)
                .enter()
                .append('rect')
                .attr('class', 'start-node')
                .attr('x', d => xScale(d[1]) - pointSize)
                .attr('y', d => yScale(d[2]) - pointSize)
                .attr('width', pointSize * 2)
                .attr('height', pointSize * 2)
                .attr('fill', d => colorScale(d[4]));

            const legend = g.append('g')
                .attr('transform', `translate(${width}, 20)`);

            fileNames.forEach((name, i) => {
                const legendRow = legend.append('g')
                    .attr('transform', `translate(0, ${i * 20})`);

                legendRow.append('rect')
                    .attr('width', 10)
                    .attr('height', 10)
                    .attr('fill', colorScale(i));

                legendRow.append('text')
                    .attr('x', 20)
                    .attr('y', 10)
                    .text(name);
            });

            const zoom = d3.zoom()
                .scaleExtent([0.1, 10])
                .on('zoom', (event) => {
                    g.attr('transform', event.transform);
                });

            svg.call(zoom);
        }
    }, [nodes, edges, selectedNodes, fileNames, setSelectedNodes, gridVisible, pointSize, setNodes]);

    return (
        <div>
            <div ref={tooltipRef} style={{ position: 'absolute', visibility: 'hidden', backgroundColor: 'white', border: '1px solid black', padding: '5px' }}></div>
            <svg ref={svgRef}>
                <g ref={gRef}></g>
            </svg>
        </div>
    );
};

export default Plot;
